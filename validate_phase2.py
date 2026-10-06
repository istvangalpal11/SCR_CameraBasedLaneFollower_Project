"""Hardware-free codec, frame lifecycle and real spawned transport validation."""
import base64
from contextlib import redirect_stdout
import io
import multiprocessing as mp
from pathlib import Path
from queue import Queue
import subprocess
import sys
import time
import cv2
import numpy as np

from src.algorithms.LaneDetection.utils.img_codec import ImgCodecMixin
from src.algorithms.LaneDetection.threads.threadLaneDetection import threadLaneDetection
from src.algorithms.LaneDetection.processLaneDetection import processLaneDetection
from src.gateway.processGateway import processGateway
from src.utils.messages.allMessages import serialCamera, LaneControl, LaneActuation
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber

SAFE_STOP = {'steer': 0, 'speed': 0, 'valid': False}


def test_image():
    image = np.zeros((270, 512, 3), dtype=np.uint8)
    cv2.line(image, (100, 20), (100, 250), (255, 255, 255), 8)
    cv2.line(image, (400, 20), (400, 250), (255, 255, 255), 8)
    cv2.rectangle(image, (200, 90), (300, 170), (30, 140, 240), -1)
    ok, jpg = cv2.imencode('.jpg', image)
    assert ok
    return image, base64.b64encode(jpg).decode('ascii')


def validate_codec_and_lifecycle():
    image, payload = test_image()
    codec = ImgCodecMixin()
    decoded = codec._decode_b64_jpg_to_bgr(payload)
    assert decoded is not None and decoded.shape == (270, 512, 3)
    assert decoded.dtype == np.uint8
    assert np.abs(decoded.astype(float) - image.astype(float)).mean() < 5
    for bad in ('this_is_not_a_valid_jpeg', '', '%%%%',
                base64.b64encode(b'not jpeg').decode('ascii'), None, 123):
        assert codec._decode_b64_jpg_to_bgr(bad) is None
    print('PASS: JPEG/base64 codec and 512x270 BGR content', flush=True)

    # Exercise the real lane methods deterministically, without running a camera.
    queues = {name: Queue() for name in ('Config', 'Critical', 'General', 'Warning')}
    lane = threadLaneDetection(queues)
    class Controls:
        value = None

        def receive(self):
            value, self.value = self.value, None
            return value

    controls = Controls()
    original_subscriber = lane.controlSubscriber
    lane.controlSubscriber = controls
    calls = []
    def decode(value):
        calls.append(value)
        return codec._decode_b64_jpg_to_bgr(value)
    lane._decode_b64_jpg_to_bgr = decode
    def frame(value):
        with lane._rx_lock:
            lane._latest_b64 = value
            lane._rx_count += 1
    def control(value):
        controls.value = {'enable': value}
        lane.state_change_handler()

    logs = io.StringIO()
    try:
        with redirect_stdout(logs):
            frame(payload)
            lane.thread_work()
            assert not calls, 'Decoded while disabled'
            control(True)
            lane.thread_work()
            assert not calls, 'Decoded stale frame on enable'
            frame('older malformed frame')
            frame(payload)
            lane.thread_work()
            lane.thread_work()
            assert calls == [payload], 'Did not select newest frame, or decoded twice'
            for _ in range(3):
                frame('bad jpeg')
                lane.thread_work()
            frame(payload)
            lane.thread_work()
            frame('bad jpeg')
            lane.thread_work()
            ok, wrong_size = cv2.imencode('.jpg', image[:100])
            assert ok
            wrong = base64.b64encode(wrong_size).decode('ascii')
            for _ in range(3):
                frame(wrong)
                lane.thread_work()
            control(False)
            count = len(calls)
            frame(payload)
            lane.thread_work()
            assert len(calls) == count
            control(True)
            lane.thread_work()
            assert len(calls) == count, 'Restart decoded disabled-period frame'
            frame(payload)
            lane.thread_work()
            control('false')
            assert lane.lane_enabled is False
            count = len(calls)
            lane.thread_work()
            assert len(calls) == count
        output = logs.getvalue()
        assert output.count('First camera frame decoded successfully') == 1
        assert output.count('Camera frame decode failed') == 2
        assert output.count('Unexpected camera frame shape') == 1
        assert output.count('LaneControl requires a boolean enable') == 1
        while not queues['Critical'].empty():
            assert queues['Critical'].get_nowait()['msgValue'] == SAFE_STOP
    finally:
        original_subscriber._pipeRecv.close()
        original_subscriber._pipeSend.close()
        lane.cameraSubscriber._pipeRecv.close()
        lane.cameraSubscriber._pipeSend.close()
    print('PASS: newest frame only, no disabled/stale/repeated decode, strict control, quiet errors', flush=True)


def transport_child():
    queues = {name: mp.Queue(maxsize=4 if name == 'General' else 0)
              for name in ('Critical', 'Warning', 'General', 'Config')}
    registered = mp.Event()
    observer = messageHandlerSubscriber(queues, LaneActuation, 'fifo', True)
    gateway = processGateway(queues, registered, mp.Event())
    lane = processLaneDetection(queues, mp.Event())
    started = []
    actuation_count = 0
    def collect(wait_for_one=False):
        nonlocal actuation_count
        deadline = time.monotonic() + (5 if wait_for_one else 0.1)
        received = False
        while time.monotonic() < deadline:
            value = observer.receive()
            if value is not None:
                assert value == SAFE_STOP, f'Unsafe actuation: {value}'
                actuation_count += 1
                received = True
            elif received or not wait_for_one:
                break
            time.sleep(0.01)
        assert not wait_for_one or received, 'Safe-stop acknowledgement missing'
    try:
        for process in (gateway, lane):
            process.start()
            started.append(process)
            assert process.ready_event.wait(5)
        assert registered.wait(5)
        camera = messageHandlerSender(queues, serialCamera)
        control = messageHandlerSender(queues, LaneControl)
        image, payload = test_image()
        ok, jpg = cv2.imencode('.jpg', image[:100])
        assert ok
        wrong = base64.b64encode(jpg).decode('ascii')
        camera.send(wrong)  # Must not decode while initially disabled.
        time.sleep(0.2)
        control.send({'enable': True})
        collect(True)
        time.sleep(0.2)  # No new frame yet: stale wrong-size image must be skipped.
        for _ in range(5):
            camera.send(payload)
            time.sleep(0.05)
        for value in ('bad jpeg', 'bad jpeg', payload, 'bad jpeg', payload):
            camera.send(value)
            time.sleep(0.2)
            assert lane.is_alive() and gateway.is_alive()
            collect()
        control.send({'enable': False})
        collect(True)
        camera.send(wrong)
        time.sleep(0.2)
        control.send({'enable': True})
        collect(True)
        time.sleep(0.2)
        camera.send(payload)
        time.sleep(0.2)
        control.send({'enable': False})
        collect(True)
        assert lane.is_alive() and gateway.is_alive()
    finally:
        for process in reversed(started):
            process.stop()
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(2)
                raise AssertionError('Phase 2 process required forced termination')
            assert process.exitcode == 0
            collect()
        assert actuation_count >= 4
        for queue in queues.values():
            queue.close()
            queue.cancel_join_thread()
    print('PASS: safe transport child and clean shutdown', flush=True)


def validate():
    root = Path(__file__).parent
    regression = subprocess.run([sys.executable, str(root / 'validate_phase1.py')],
                                capture_output=True, text=True, timeout=30)
    print(regression.stdout, end='')
    assert regression.returncode == 0, regression.stderr
    print('PASS: Phase 1 regression validation', flush=True)
    validate_codec_and_lifecycle()
    transport = subprocess.run([sys.executable, str(Path(__file__)), '--transport-child'],
                               capture_output=True, text=True, timeout=30)
    print(transport.stdout, end='')
    assert transport.returncode == 0, transport.stderr
    assert transport.stdout.count('First camera frame decoded successfully: 512x270 BGR') == 1
    # Two failure episodes separated by a valid image prove real-process recovery.
    assert transport.stdout.count('Camera frame decode failed') == 2, transport.stdout
    assert 'Unexpected camera frame shape' not in transport.stdout
    assert 'PASS: safe transport child and clean shutdown' in transport.stdout
    print('PASS: malformed frame handled safely and decoding recovers', flush=True)
    print('PASS: Gateway -> LaneDetection camera transport', flush=True)
    print('PASS: LaneDetection remains safe: no valid movement command', flush=True)
    print('PASS: clean Phase 2 process shutdown', flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    if '--transport-child' in sys.argv:
        transport_child()
    else:
        validate()
