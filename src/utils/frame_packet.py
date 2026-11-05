"""Camera frame dictionaries and orchestration-level timing (no image algorithms)."""
from collections import deque
import time

FRAME_METADATA_FIELDS = ('session_id', 'frame_id', 't_capture_ns')


def validate_capture_latency_offset(offset):
    if type(offset) is not int or offset < 0:
        raise ValueError('capture_latency_offset_ns must be a nonnegative integer')
    return offset


def validate_frame_packet(packet):
    """Validate metadata, not JPEG contents; return a private shallow snapshot."""
    if not isinstance(packet, dict):
        raise ValueError('Camera payload must be a frame-packet dictionary')
    if not isinstance(packet.get('session_id'), str) or not packet['session_id'].strip():
        raise ValueError('session_id must be a nonempty string')
    for name in ('frame_id', 't_capture_ns'):
        if type(packet.get(name)) is not int or packet[name] < 0:
            raise ValueError(name + ' must be a nonnegative integer')
    if not isinstance(packet.get('image_b64'), str) or not packet['image_b64']:
        raise ValueError('image_b64 must be a nonempty base64/JPEG string')
    for name in ('t_publish_ns', 't_capture_return_ns', 'capture_latency_offset_ns'):
        if name in packet and (type(packet[name]) is not int or packet[name] < 0):
            raise ValueError(name + ' must be a nonnegative integer')
    if 't_publish_ns' in packet and packet['t_publish_ns'] < packet['t_capture_ns']:
        raise ValueError('Publish time precedes capture time')
    return dict(packet)


def make_frame_packet(session_id, frame_id, t_capture_ns, image_b64, **diagnostics):
    packet = dict(diagnostics)
    packet.update(session_id=session_id, frame_id=frame_id, t_capture_ns=t_capture_ns,
                  image_b64=image_b64)
    return validate_frame_packet(packet)


def make_capture_packet(session_id, frame_id, image_b64, t_capture_return_ns,
                        capture_latency_offset_ns=0, t_publish_ns=None):
    """Approximate acquisition time; offset is uncalibrated unless measured on hardware."""
    offset = validate_capture_latency_offset(capture_latency_offset_ns)
    if type(t_capture_return_ns) is not int or t_capture_return_ns <= offset:
        raise ValueError('Capture-return time must exceed the configured latency offset')
    if t_publish_ns is None:
        t_publish_ns = time.monotonic_ns()
    if t_publish_ns < t_capture_return_ns:
        raise ValueError('Publish time precedes capture return')
    return make_frame_packet(
        session_id, frame_id, t_capture_return_ns - offset, image_b64,
        t_capture_return_ns=t_capture_return_ns, capture_latency_offset_ns=offset,
        t_publish_ns=t_publish_ns)


class FramePacketOrder:
    """Strict per-session ordering; bounded restart history rejects recently retired sessions."""
    def __init__(self):
        self.session_id = None
        self.frame_id = -1
        self.t_capture_ns = -1
        self._retired_sessions = deque(maxlen=16)

    def accept(self, packet):
        packet = validate_frame_packet(packet)
        session = packet['session_id']
        if session != self.session_id:
            if session in self._retired_sessions:
                raise ValueError('Packet belongs to a retired camera session')
        elif packet['frame_id'] <= self.frame_id or packet['t_capture_ns'] <= self.t_capture_ns:
            raise ValueError('Duplicate or out-of-order camera frame')
        # Never update ordering state until every check has passed.
        if session != self.session_id and self.session_id is not None:
            self._retired_sessions.append(self.session_id)
        self.session_id, self.frame_id, self.t_capture_ns = (
            session, packet['frame_id'], packet['t_capture_ns'])
        return packet


class FrameDecodeError(ValueError):
    pass


class FrameShapeError(ValueError):
    pass


def preprocess_frame_packet(packet, processor, *, t_received_ns=None,
                            expected_frame_shape=(270, 512, 3),
                            clock=None):
    """Decode + unchanged preprocessing, returning one BEV with its original metadata."""
    from src.algorithms.LaneDetection.utils.img_codec import ImgCodecMixin
    from src.algorithms.LaneDetection.utils import tuning
    import numpy as np
    packet = validate_frame_packet(packet)
    clock = time.monotonic_ns if clock is None else clock
    if t_received_ns is None:
        t_received_ns = clock()
    t_process_start_ns = clock()
    decoder = getattr(processor, '_decode_b64_jpg_to_bgr', ImgCodecMixin()._decode_b64_jpg_to_bgr)
    frame = decoder(packet['image_b64'])
    if frame is None:
        raise FrameDecodeError('Camera frame decode failed')
    if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
            or frame.dtype != np.uint8
            or (expected_frame_shape is not None and frame.shape != expected_frame_shape)):
        raise FrameShapeError(f'Unexpected camera frame shape: {getattr(frame, "shape", None)}, expected {expected_frame_shape}')
    t_preprocess_start_ns = clock()
    binary = processor.preprocess_frame_bgr(frame)
    t_done_ns = clock()
    if (not isinstance(binary, np.ndarray)
            or binary.shape != (tuning.BEV_OUTPUT_HEIGHT, tuning.BEV_OUTPUT_WIDTH)
            or binary.dtype != np.uint8 or not np.all((binary == 0) | (binary == 1))):
        raise ValueError('Invalid binary BEV shape, dtype or values')
    if not t_received_ns <= t_process_start_ns <= t_preprocess_start_ns <= t_done_ns:
        raise ValueError('Diagnostic monotonic timestamps are out of order')
    result = {key: value for key, value in packet.items() if key != 'image_b64'}
    result.update(binary_bev=binary, t_received_ns=t_received_ns,
                  t_process_start_ns=t_process_start_ns,
                  t_preprocess_start_ns=t_preprocess_start_ns, t_done_ns=t_done_ns)
    return result


class LatencyStatistics:
    """Bounded host-monotonic diagnostics; never infer Pi performance or exposure precision."""
    def __init__(self, max_samples=256, min_samples=5):
        if not 1 <= min_samples <= max_samples:
            raise ValueError('Require 1 <= min_samples <= max_samples')
        self._samples = deque(maxlen=max_samples)
        self.min_samples = min_samples

    def add(self, result):
        pairs = {
            'wait_before_processing': ('t_received_ns', 't_process_start_ns'),
            'processing_including_decode': ('t_process_start_ns', 't_done_ns'),
            'preprocessing': ('t_preprocess_start_ns', 't_done_ns'),
        }
        if 't_capture_ns' in result:
            pairs.update(estimated_frame_age=('t_capture_ns', 't_done_ns'))
            if 't_publish_ns' in result:
                pairs.update(capture_to_publish=('t_capture_ns', 't_publish_ns'),
                             publish_to_receive=('t_publish_ns', 't_received_ns'))
        sample = {}
        for name, (start, end) in pairs.items():
            duration = result[end] - result[start]
            if duration < 0:
                raise ValueError('Negative duration for ' + name + ': check clock domains')
            sample[name] = duration / 1_000_000
        self._samples.append(sample)

    def summary(self):
        import numpy as np
        metrics = {}
        for key in sorted({k for sample in self._samples for k in sample}):
            values = [s[key] for s in self._samples if key in s]
            if len(values) >= self.min_samples:
                metrics[key] = dict(samples=len(values), median=float(np.median(values)),
                                    p95=float(np.percentile(values, 95)))
        return dict(host_clock_resolution_ns=round(time.get_clock_info('monotonic').resolution * 1e9),
                    samples=len(self._samples), required_samples=self.min_samples,
                    ready=len(self._samples) >= self.min_samples, metrics_ms=metrics)
