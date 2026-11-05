"""Phase 0.2: hardware-free frame metadata, replay, timing and real Gateway tests."""
import base64
import contextlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import platform
from queue import Queue, Empty
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'notebooks'))
from src.utils.frame_packet import (make_frame_packet, make_capture_packet, validate_frame_packet,
    validate_capture_latency_offset, FramePacketOrder, FrameDecodeError, FrameShapeError,
    preprocess_frame_packet, LatencyStatistics)
from src.algorithms.LaneDetection.utils.preprocessing import PreprocessingMixin
from src.algorithms.LaneDetection.utils.img_codec import ImgCodecMixin
from src.algorithms.LaneDetection.utils import tuning
from src.algorithms.LaneDetection.threads.threadLaneDetection import threadLaneDetection
from src.hardware.camera.threads.threadCamera import threadCamera
from src.gateway.processGateway import processGateway
from src.utils.messages.allMessages import serialCamera, LaneControl, LaneActuation
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from helpers.frame_replay import iter_replay_packets, replay_timing_sample


def image_payload(index=0):
    image = np.full((270, 512, 3), 20, np.uint8)
    cv2.line(image, (170 + index, 185), (85 + index, 269), (250, 250, 250), 5)
    cv2.line(image, (342 + index, 185), (427 + index, 269), (250, 250, 250), 5)
    ok, jpg = cv2.imencode('.jpg', image)
    assert ok
    return image, base64.b64encode(jpg).decode('ascii')


def packet(index=0, session='session', capture=None, payload=None):
    return make_frame_packet(session, index, time.monotonic_ns() if capture is None else capture,
                             image_payload(index % 5)[1] if payload is None else payload)


def local_lane():
    return threadLaneDetection({name: Queue() for name in ('Config', 'Critical', 'Warning', 'General')})


def close_lane(lane):
    lane._rx_stop.set()
    if lane._rx_thread.ident is not None:
        lane._rx_thread.join(2)
        assert not lane._rx_thread.is_alive()
    for subscriber in (lane.cameraSubscriber, lane.controlSubscriber):
        subscriber._pipeRecv.close()
        subscriber._pipeSend.close()


def wait_until(condition, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError('Timed out waiting for expected frame/result')


class Phase02Tests(unittest.TestCase):
    def test_A_packet_creation_and_jpeg_compatibility(self):
        payload = image_payload()[1]
        p = make_capture_packet('camera-A', 0, payload, 1_000, 50, 1_200)
        self.assertEqual(p['t_capture_ns'], 950)
        self.assertNotIn('timestamp_source', p)
        self.assertEqual(p['capture_latency_offset_ns'], 50)
        self.assertEqual(serialCamera.msgType.value, 'dict')
        decoded = ImgCodecMixin()._decode_b64_jpg_to_bgr(p['image_b64'])
        self.assertEqual(decoded.shape, (270, 512, 3))
        for bad in (-1, True, 2.5, None):
            with self.assertRaises(ValueError):
                validate_capture_latency_offset(bad)
        for bad in (1_000, 1_001):
            with self.assertRaises(ValueError):
                make_capture_packet('s', 0, payload, 1_000, bad, 1_200)

    def test_B_camera_capture_timestamp_precedes_encoding_and_drop_ids_are_not_reused(self):
        image, _ = image_payload()
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
        events = []
        class FakePicamera:
            @staticmethod
            def global_camera_info(): return [{}]
            def create_preview_configuration(self, **kwargs): return kwargs
            def configure(self, config): pass
            def start(self): pass
            def close(self): pass
            def capture_array(self, stream):
                events.append('capture')
                return yuv.copy()
        queues = {name: Queue(maxsize=1 if name == 'General' else 0)
                  for name in ('Config', 'Critical', 'Warning', 'General')}
        with patch.dict(sys.modules, picamera2=SimpleNamespace(Picamera2=FakePicamera)):
            camera = threadCamera(queues, capture_latency_offset_ns=10)
            other = threadCamera(queues)
        self.assertNotEqual(camera.session_id, other.session_id)
        self.assertEqual(camera.frame_rate, 5)
        ticks = iter((1_000, 1_100, 2_000, 2_100, 3_000, 3_100))
        def clock():
            events.append('timestamp')
            return next(ticks)
        convert, encode = cv2.cvtColor, cv2.imencode
        def cvt(*args, **kwargs):
            events.append('convert')
            return convert(*args, **kwargs)
        def enc(*args, **kwargs):
            events.append('encode')
            return encode(*args, **kwargs)
        try:
            with patch('src.hardware.camera.threads.threadCamera.time.monotonic_ns', side_effect=clock), patch.object(cv2, 'cvtColor', side_effect=cvt), patch.object(cv2, 'imencode', side_effect=enc):
                for _ in range(2):
                    camera._next_frame_t = 0
                    camera.thread_work()
                first = queues['General'].get_nowait()['msgValue']
                camera._next_frame_t = 0
                camera.thread_work()
                third = queues['General'].get_nowait()['msgValue']
            self.assertEqual(events[:5], ['capture', 'timestamp', 'convert', 'encode', 'timestamp'])
            self.assertEqual((first['frame_id'], third['frame_id']), (0, 2))
            self.assertEqual((first['t_capture_ns'], third['t_capture_ns']), (990, 2_990))
        finally:
            camera._close_camera()
            other._close_camera()

    def test_C_ordering_duplicates_invalid_packets_and_restart(self):
        order = FramePacketOrder()
        p = packet(0, capture=0)
        order.accept(p)
        with self.assertRaises(ValueError): order.accept(p)
        for altered in (dict(p, frame_id=True), dict(p, frame_id=-1), dict(p, t_capture_ns='5'),
                        dict(p, session_id=''), dict(p, image_b64=None)):
            with self.assertRaises(ValueError): order.accept(altered)
        order.accept(packet(2, capture=20))
        for bad in (packet(1, capture=30),
                    packet(3, capture=19)):
            with self.assertRaises(ValueError): order.accept(bad)
        order.accept(packet(0, session='restarted', capture=0))
        self.assertEqual(order.frame_id, 0)
        with self.assertRaises(ValueError): order.accept(packet(5, capture=40))
        self.assertEqual(order.session_id, 'restarted')
        for malformed in (None, 'old base64 protocol', {}, dict(p, session_id=['bad'])):
            with self.assertRaises(ValueError): validate_frame_packet(malformed)

    def test_D_latest_packet_atomicity_no_backlog_and_bad_packet_recovery(self):
        lane = local_lane()
        lane.lane_enabled = True
        try:
            lane._rx_thread.start()
            final = None
            for index in range(20):
                final = packet(index, capture=index)
                lane.cameraSubscriber._pipeSend.send({'Type': 'dict', 'value': final})
            wait_until(lambda: lane._latest_packet is not None and lane._latest_packet['frame_id'] == 19)
            self.assertIsNone(lane.latest_preprocessed_frame)
            lane.thread_work()
            result = lane.latest_preprocessed_frame
            self.assertEqual(result['frame_id'], 19)
            self.assertEqual(result['t_capture_ns'], 19)
            decoded = ImgCodecMixin()._decode_b64_jpg_to_bgr(final['image_b64'])
            self.assertTrue(np.array_equal(result['binary_bev'], PreprocessingMixin().preprocess_frame_bgr(decoded)))
            lane.thread_work()
            self.assertIs(lane.latest_preprocessed_frame, result)
            self.assertFalse(lane._store_camera_packet(final))
            self.assertFalse(lane._store_camera_packet({'frame_id': []}))
            self.assertTrue(lane._store_camera_packet(packet(20, capture=20)))
            lane.thread_work()
            self.assertEqual(lane.latest_preprocessed_frame['frame_id'], 20)
            self.assertTrue(lane._store_camera_packet(packet(0, session='new-session', capture=0)))
            lane.thread_work()
            self.assertEqual(lane.latest_preprocessed_frame['session_id'], 'new-session')
            self.assertEqual(lane.latest_preprocessed_frame['frame_id'], 0)
            self.assertFalse(lane._store_camera_packet(packet(21, capture=21)))
            restart = packet(1, session='new-session', capture=1)
            self.assertTrue(lane._store_camera_packet(restart))
            restart['frame_id'] = 999  # Sender mutation cannot change the stored snapshot.
            lane.thread_work()
            self.assertEqual(lane.latest_preprocessed_frame['frame_id'], 1)
            while not lane.laneActuationSender.queuesList['Critical'].empty():
                self.assertEqual(lane.laneActuationSender.queuesList['Critical'].get()['msgValue'], {'steer': 0, 'speed': 0, 'valid': False})
        finally:
            close_lane(lane)

    def test_E_metadata_and_processing_timing_preservation(self):
        p = packet(capture=10)
        p['t_publish_ns'] = 20
        ticks = iter((40, 50, 70))
        result = preprocess_frame_packet(p, PreprocessingMixin(), t_received_ns=30, clock=lambda: next(ticks))
        for field in ('session_id', 'frame_id', 't_capture_ns'):
            self.assertEqual(result[field], p[field])
        self.assertEqual((result['t_received_ns'], result['t_process_start_ns'], result['t_preprocess_start_ns'], result['t_done_ns']), (30, 40, 50, 70))
        decoded = ImgCodecMixin()._decode_b64_jpg_to_bgr(p['image_b64'])
        self.assertTrue(np.array_equal(result['binary_bev'], PreprocessingMixin().preprocess_frame_bgr(decoded)))
        self.assertEqual(result['binary_bev'].shape, (tuning.BEV_OUTPUT_HEIGHT, tuning.BEV_OUTPUT_WIDTH))
        self.assertEqual(result['binary_bev'].dtype, np.uint8)
        self.assertTrue(set(np.unique(result['binary_bev'])) <= {0, 1})
        stats = LatencyStatistics(max_samples=5)
        stats.add(result)
        self.assertFalse(stats.summary()['ready'])
        for _ in range(10): stats.add(result)
        summary = stats.summary()
        self.assertEqual(summary['samples'], 5)
        self.assertAlmostEqual(summary['metrics_ms']['estimated_frame_age']['median'], 60 / 1_000_000)
        self.assertAlmostEqual(summary['metrics_ms']['preprocessing']['p95'], 20 / 1_000_000)
        replay = dict(result, t_capture_ns=10**30)
        replay_stats = LatencyStatistics(min_samples=1)
        replay_stats.add(replay_timing_sample(replay))
        self.assertNotIn('estimated_frame_age', replay_stats.summary()['metrics_ms'])
        with self.assertRaises(FrameDecodeError):
            preprocess_frame_packet(dict(p, image_b64='bad jpeg'), PreprocessingMixin())
        ok, small = cv2.imencode('.jpg', decoded[:100])
        self.assertTrue(ok)
        with self.assertRaises(FrameShapeError):
            preprocess_frame_packet(dict(p, image_b64=base64.b64encode(small).decode('ascii')), PreprocessingMixin())

    def test_F_replay_numeric_order_recorded_times_and_new_sessions(self):
        image, _ = image_payload()
        with tempfile.TemporaryDirectory(prefix='scr-phase02-replay-') as temporary:
            directory = Path(temporary)
            for index in (10, 2, 1):
                self.assertTrue(cv2.imwrite(str(directory / f'frame{index}.png'), image))
            packets = list(iter_replay_packets(directory, assumed_fps=5.0))
            self.assertEqual([p['image_name'] for p in packets], ['frame1.png', 'frame2.png', 'frame10.png'])
            self.assertEqual([p['t_capture_ns'] for p in packets], [0, 200_000_000, 400_000_000])
            self.assertTrue(all('timestamp_source' not in p for p in packets))
            self.assertNotEqual(packets[0]['session_id'], next(iter_replay_packets(directory))['session_id'])
            manifest = directory / 'frames.jsonl'
            rows = [dict(image_path='frame1.png', session_id='Pi-A', frame_id=125, t_capture_ns=12345),
                    dict(image_path='frame2.png', session_id='Pi-A', frame_id=130, t_capture_ns=34567),
                    dict(image_path='frame10.png', session_id='Pi-B', frame_id=0, t_capture_ns=45678)]
            manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            recorded = list(iter_replay_packets(directory, mode='recorded', metadata_path=manifest))
            manifest_json = directory / 'frames.json'
            manifest_json.write_text(json.dumps({'frames': rows}), encoding='utf-8')
            recorded_json = list(iter_replay_packets(directory, mode='recorded', metadata_path=manifest_json))
            self.assertEqual([p['t_capture_ns'] for p in recorded_json], [r['t_capture_ns'] for r in rows])
            self.assertTrue(all('timestamp_source' not in p for p in recorded_json))
            for row, p in zip(rows, recorded):
                for field in ('session_id', 'frame_id', 't_capture_ns'):
                    self.assertEqual(row[field], p[field])
                result = preprocess_frame_packet(p, PreprocessingMixin())
                self.assertEqual(result['t_capture_ns'], row['t_capture_ns'])
        first = list(iter_replay_packets(assumed_fps=7.0, session_id='demo'))
        time.sleep(0.01)
        second = list(iter_replay_packets(assumed_fps=7.0, session_id='demo'))
        self.assertEqual([p['t_capture_ns'] for p in first], [p['t_capture_ns'] for p in second])
        self.assertTrue(all(p['t_capture_ns'] == round(p['frame_id'] * 1e9 / 7.0) for p in first))

    def test_G_real_gateway_preserves_packet_and_safe_lane_result(self):
        queues = {name: mp.Queue(maxsize=4 if name == 'General' else 0)
                  for name in ('Critical', 'Warning', 'General', 'Config')}
        ready = mp.Event()
        observer = messageHandlerSubscriber(queues, LaneActuation, 'fifo', True)
        lane = threadLaneDetection(queues)
        lane.daemon = True
        gateway = processGateway(queues, ready, mp.Event())
        try:
            gateway.start()
            lane.start()
            self.assertTrue(ready.wait(5))
            control = messageHandlerSender(queues, LaneControl)
            control.send({'enable': True})
            wait_until(observer.is_data_in_pipe)
            self.assertEqual(observer.receive(), {'steer': 0, 'speed': 0, 'valid': False})
            payload = image_payload()[1]
            captured = time.monotonic_ns()
            p = make_frame_packet('gateway-session', 125, captured, payload, t_publish_ns=time.monotonic_ns())
            messageHandlerSender(queues, serialCamera).send(p)
            wait_until(lambda: lane.latest_preprocessed_frame is not None)
            result = lane.latest_preprocessed_frame
            for field in ('session_id', 'frame_id', 't_capture_ns', 't_publish_ns'):
                self.assertEqual(result[field], p[field])
            self.assertLessEqual(result['t_publish_ns'], result['t_received_ns'])
            self.assertLessEqual(result['t_received_ns'], result['t_process_start_ns'])
            self.assertLessEqual(result['t_process_start_ns'], result['t_done_ns'])
            control.send({'enable': False})
            wait_until(observer.is_data_in_pipe)
            self.assertEqual(observer.receive(), {'steer': 0, 'speed': 0, 'valid': False})
        finally:
            lane.stop()
            if lane.ident is not None: lane.join(5)
            self.assertFalse(lane.is_alive())
            if gateway.pid is not None:
                gateway.stop()
                gateway.join(5)
                if gateway.is_alive():
                    gateway.terminate()
                    gateway.join(2)
                    self.fail('Gateway required forced shutdown')
                self.assertEqual(gateway.exitcode, 0)
            close_lane(lane)
            observer._pipeRecv.close()
            observer._pipeSend.close()
            for q in queues.values():
                q.close()
                q.cancel_join_thread()

    def test_H_phase01_preserved(self):
        self.assertFalse(hasattr(tuning, 'THRESH_OPEN_ENABLE'))
        self.assertEqual(tuning.THRESH_OPEN_ITER, 0)
        self.assertEqual(tuning.THRESH_CLOSE_ITER, 0)
        image = np.zeros((tuning.BEV_OUTPUT_HEIGHT, tuning.BEV_OUTPUT_WIDTH), np.uint8)
        with patch.object(cv2, 'morphologyEx', wraps=cv2.morphologyEx) as morph:
            PreprocessingMixin()._apply_threshold_morphology(image)
        self.assertEqual(morph.call_count, 0)


def benchmark_development_host():
    import platform
    payload = image_payload()[1]
    processor = PreprocessingMixin()
    stats = LatencyStatistics()
    for frame_id in range(10):
        p = make_capture_packet('synthetic-benchmark', frame_id, payload, time.monotonic_ns())
        stats.add(preprocess_frame_packet(p, processor))
    print('DEVELOPMENT HOST ONLY, synthetic direct calls; NOT Raspberry Pi capture latency:',
          platform.platform(), sys.version.split()[0], flush=True)
    print('Monotonic clock resolution (ns):', round(time.get_clock_info('monotonic').resolution * 1e9), flush=True)
    print('Zero durations may be below clock resolution, not zero processing cost.', flush=True)
    print(json.dumps(stats.summary(), indent=2), flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    result = unittest.main(verbosity=2, exit=False).result
    if not result.wasSuccessful():
        sys.exit(1)
    benchmark_development_host()
