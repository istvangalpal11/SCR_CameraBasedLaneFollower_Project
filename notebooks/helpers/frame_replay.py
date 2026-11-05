"""Small saved-image replay adapter; uses the same frame packet as the live camera."""
import base64
import json
import math
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np
from .image_loader import natural_key
from src.utils.frame_packet import make_frame_packet, FramePacketOrder


def _jpeg_payload(path):
    if path.suffix.lower() in ('.jpg', '.jpeg'):
        return base64.b64encode(path.read_bytes()).decode('ascii')
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot read replay image: ' + str(path))
    ok, jpg = cv2.imencode('.jpg', image)
    if not ok:
        raise ValueError('Cannot encode replay image: ' + str(path))
    return base64.b64encode(jpg).decode('ascii')


def _demo_payload(index):
    image = np.full((270, 512, 3), 20, np.uint8)
    cv2.line(image, (170 + index, 185), (85 + index, 269), (250, 250, 250), 5)
    cv2.line(image, (342 + index, 185), (427 + index, 269), (250, 250, 250), 5)
    ok, jpg = cv2.imencode('.jpg', image)
    if not ok:
        raise ValueError('Cannot encode demo image')
    return base64.b64encode(jpg).decode('ascii')


def iter_replay_packets(image_directory=None, *, mode='simulated', assumed_fps=5.0,
                        metadata_path=None, session_id=None, max_frames=20):
    """One call is one simulated sequence/session; recorded JSON/JSONL preserves original IDs/times.

    Records: image_path, session_id, frame_id, t_capture_ns,
    plus optional original diagnostics. Relative image paths resolve in image_directory
    (or beside the manifest). No packet timing depends on replay execution speed.
    Images are not resized/calibrated; the notebook can explicitly permit other sizes.
    """
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError('max_frames must be a positive integer or None')
    if mode == 'recorded':
        if metadata_path is None:
            raise ValueError('Recorded mode requires a JSON or JSONL metadata manifest')
        manifest = Path(metadata_path).expanduser().resolve()
        directory = Path(image_directory).expanduser().resolve() if image_directory is not None else manifest.parent
        order = FramePacketOrder()
        emitted = 0
        with manifest.open(encoding='utf-8-sig') as stream:
            if manifest.suffix.lower() == '.json':
                data = json.load(stream)
                records = data if isinstance(data, list) else data.get('frames', [data])
            else:
                records = (json.loads(line) for line in stream if line.strip())
            for line_number, record in enumerate(records, 1):
                image_path = directory / record['image_path']
                diagnostics = {key: record[key] for key in
                               ('t_publish_ns', 't_capture_return_ns', 'capture_latency_offset_ns')
                               if key in record}
                packet = make_frame_packet(
                    record['session_id'], record['frame_id'], record['t_capture_ns'],
                    _jpeg_payload(image_path), image_name=image_path.name, **diagnostics)
                try:
                    packet = order.accept(packet)
                except ValueError as exc:
                    raise ValueError(f'{manifest}:{line_number}: {exc}') from exc
                yield packet
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
        return
    if mode != 'simulated':
        raise ValueError('Replay mode must be simulated or recorded')
    if isinstance(assumed_fps, bool):
        raise ValueError('assumed_fps must be a finite positive number')
    fps = float(assumed_fps)
    if not math.isfinite(fps) or not 0 < fps <= 1_000_000_000:
        raise ValueError('assumed_fps must produce distinct nanosecond timestamps')
    session = uuid4().hex if session_id is None else session_id
    if image_directory is None:
        paths = [None] * 5  # Five tiny deterministic demo frames; no files generated.
    else:
        directory = Path(image_directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError('Replay image directory does not exist: ' + str(directory))
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        paths = sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions), key=natural_key)
        if not paths:
            raise ValueError('No replay images found in ' + str(directory))
    for index, path in enumerate(paths):
        if max_frames is not None and index >= max_frames:
            break
        payload = _demo_payload(index) if path is None else _jpeg_payload(path)
        yield make_frame_packet(session, index, round(index * 1_000_000_000 / fps),
                                payload,
                                image_name=f'demo{index}.jpg' if path is None else path.name)


def replay_timing_sample(result):
    """Only same-host stage clocks go to diagnostics, never the replay capture timeline."""
    return {key: result[key] for key in
            ('t_received_ns', 't_process_start_ns', 't_preprocess_start_ns', 't_done_ns')}


def adapt_replay_packet(packet, *, enabled=False, bottom_crop_enable=False, bottom_crop_px=110):
    """Optional offline screenshot preparation; identity/timing fields are never changed.

    Disabled returns the exact packet without any decode/re-encode. Enabled reuses
    the preprocessing notebook normalizer with a scoped camera-size target, followed
    by its existing crop/height-resize/center-crop sequence on BGR rather than gray.
    Returns (packet, original BGR, prepared BGR) for notebook visualization.
    """
    if not enabled:
        return packet, None, None
    from unittest.mock import patch
    from . import test_image_utils
    from src.algorithms.LaneDetection.utils.img_codec import ImgCodecMixin
    original = ImgCodecMixin()._decode_b64_jpg_to_bgr(packet['image_b64'])
    if original is None:
        raise ValueError('Cannot decode replay image for optional adaptation')
    width, height = 512, 270  # Offline target is the production camera INPUT, not BEV.
    with patch.multiple(test_image_utils, TEST_TARGET_WIDTH=width, TEST_TARGET_HEIGHT=height):
        adapted = test_image_utils.normalize_notebook_image(original)
    if bottom_crop_enable:
        if type(bottom_crop_px) is not int or not 0 <= bottom_crop_px < adapted.shape[0]:
            print(f'WARNING: Offline bottom crop must be an integer in [0, {height}); correction skipped.')
        elif bottom_crop_px > 0:
            cropped = adapted[:height - bottom_crop_px]
            scale = height / cropped.shape[0]
            new_width = max(1, round(width * scale))
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            resized = cv2.resize(cropped, (new_width, height), interpolation=interpolation)
            left = (new_width - width) // 2
            adapted = resized[:, left:left + width].copy()
    assert adapted.shape == (height, width, 3) and adapted.dtype == np.uint8
    ok, jpg = cv2.imencode('.jpg', adapted)
    if not ok:
        raise ValueError('Cannot encode adapted replay image')
    return dict(packet, image_b64=base64.b64encode(jpg).decode('ascii')), original, adapted
