"""Phase 3.1 preprocessing and spawned Gateway integration, without hardware."""
import base64
from contextlib import redirect_stdout
import importlib.util
import io
import multiprocessing as mp
from pathlib import Path
from queue import Queue
import subprocess
import sys
import cv2
import numpy as np

from src.algorithms.LaneDetection.utils.preprocessing import PreprocessingMixin
from src.algorithms.LaneDetection.utils import tuning
from src.algorithms.LaneDetection.threads.threadLaneDetection import threadLaneDetection


def road_image(bright=False):
    image = np.full((270, 512, 3), 155 if bright else 20, dtype=np.uint8)
    cv2.line(image, (170, 185), (85, 269), (250, 250, 250), 5)
    cv2.line(image, (342, 185), (427, 269), (250, 250, 250), 5)
    return image


def assert_binary(binary):
    assert isinstance(binary, np.ndarray)
    assert binary.shape == (270, 512) and binary.ndim == 2
    assert binary.dtype == np.uint8
    assert np.all((binary == 0) | (binary == 1))


def reference_check(images):
    # Optional read-only comparison when the original repository is alongside SCR.
    directory = Path(__file__).parent.parent / 'brain/src/algorithms/LaneDetection/utils'
    if not (directory / 'preprocessing.py').exists():
        print('INFO: BFMC reference comparison skipped (reference not installed)', flush=True)
        return
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    old_tuning = load('bfmc_reference_tuning', directory / 'tuning.py')
    old_code = load('bfmc_reference_preprocessing', directory / 'preprocessing.py')
    old_code.lane_tuning = old_tuning
    original = old_code.PreprocessingMixin()
    current = PreprocessingMixin()
    for name, value in vars(tuning).items():
        if name.startswith(('BEV_', 'THRESH_')):
            assert value == getattr(old_tuning, name), name
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bev, _, _ = original.get_birdseye_responsive_gray(
            gray, top_width_factor=old_tuning.BEV_SRC_TOP_WIDTH_FACTOR,
            bottom_width_factor=old_tuning.BEV_SRC_BOTTOM_WIDTH_FACTOR,
            bottom_offset_factor=old_tuning.BEV_SRC_BOTTOM_OFFSET_FACTOR,
            horizon_factor=old_tuning.BEV_SRC_HORIZON_FACTOR,
            empty_margins=old_tuning.BEV_DST_EMPTY_MARGINS)
        if old_tuning.BEV_BLACK_TRAPEZOID_ENABLE:
            bev = original.apply_bev_black_trapezoid(
                bev, top_width_frac=old_tuning.BEV_BLACK_TRAPEZOID_TOP_WIDTH_FRAC,
                bottom_width_frac=old_tuning.BEV_BLACK_TRAPEZOID_BOTTOM_WIDTH_FRAC,
                top_y_frac=old_tuning.BEV_BLACK_TRAPEZOID_TOP_Y_FRAC,
                bottom_y_frac=old_tuning.BEV_BLACK_TRAPEZOID_BOTTOM_Y_FRAC,
                center_x_frac=old_tuning.BEV_BLACK_TRAPEZOID_CENTER_X_FRAC)
        current_bev, _, _ = current.get_birdseye_responsive_gray(
            gray, top_width_factor=tuning.BEV_SRC_TOP_WIDTH_FACTOR,
            bottom_width_factor=tuning.BEV_SRC_BOTTOM_WIDTH_FACTOR,
            bottom_offset_factor=tuning.BEV_SRC_BOTTOM_OFFSET_FACTOR,
            horizon_factor=tuning.BEV_SRC_HORIZON_FACTOR,
            empty_margins=tuning.BEV_DST_EMPTY_MARGINS)
        if tuning.BEV_BLACK_TRAPEZOID_ENABLE:
            current_bev = current.apply_bev_black_trapezoid(
                current_bev, top_width_frac=tuning.BEV_BLACK_TRAPEZOID_TOP_WIDTH_FRAC,
                bottom_width_frac=tuning.BEV_BLACK_TRAPEZOID_BOTTOM_WIDTH_FRAC,
                top_y_frac=tuning.BEV_BLACK_TRAPEZOID_TOP_Y_FRAC,
                bottom_y_frac=tuning.BEV_BLACK_TRAPEZOID_BOTTOM_Y_FRAC,
                center_x_frac=tuning.BEV_BLACK_TRAPEZOID_CENTER_X_FRAC)
        assert np.array_equal(current_bev, bev)
        assert np.array_equal(current._apply_tophat_prethreshold(bev),
                              original._apply_tophat_prethreshold(bev))
        probe = (bev > 127).astype(np.uint8)
        assert np.array_equal(current._apply_threshold_morphology(probe),
                              original._apply_threshold_morphology(probe))
    print('PASS: Unchanged BFMC BEV, Top-Hat, morphology and active tuning match reference', flush=True)


def validate_top_auto_adjust():
    from unittest.mock import patch

    class ThresholdProbe(PreprocessingMixin):
        def _apply_tophat_prethreshold(self, image):
            return image  # Isolate the threshold stage with deterministic prepared data.

        def _apply_threshold_morphology(self, binary):
            self.raw = binary.copy()
            self.calls = getattr(self, 'calls', 0) + 1
            return super()._apply_threshold_morphology(binary)

    def run(image, **settings):
        probe = ThresholdProbe()
        options = dict(OTSU_VERTICAL_SPLIT_FRAC=0.25, OTSU_TOP_OFFSET=0,
                       OTSU_BOTTOM_OFFSET=0, OTSU_TOP_AUTO_ADJUST_ENABLE=True,
                       OTSU_TOP_MAX_TO_BOTTOM_RATIO=2.0,
                       OTSU_MIN_BOTTOM_FOREGROUND_RATIO=0.005,
                       OTSU_TOP_ADJUST_PERCENT=0.05, OTSU_TOP_ADJUST_MAX_STEPS=4)
        options.update(settings)
        with patch.multiple(tuning, **options), patch.object(cv2, 'threshold', wraps=cv2.threshold) as otsu:
            result = probe._otsu_binary(image)
        debug = probe._last_threshold_debug
        assert otsu.call_count == 2  # Once per region, never inside the loop.
        assert probe.calls == 1
        assert result.shape == image.shape and result.dtype == np.uint8
        assert set(np.unique(result)) <= {0, 1}
        assert 0 <= debug['top_auto_steps'] <= options['OTSU_TOP_ADJUST_MAX_STEPS']
        assert debug['top_t'] >= debug['top_threshold_before_auto']
        assert debug['bottom_t'] == np.clip(debug['bottom_otsu_raw'] + options['OTSU_BOTTOM_OFFSET'], 0, 255)
        return probe, debug

    # Unequal regions: top has fewer white pixels but higher foreground density.
    image = np.zeros((100, 100), np.uint8)
    image[:25, :50] = 10
    image[:25, 50:] = 11
    image[25:, :20] = 200
    probe, debug = run(image)
    assert debug['top_auto_steps'] == 1
    assert debug['top_density_initial'] == 0.5 and debug['bottom_density'] == 0.2
    assert debug['top_t'] == 11 and debug['top_density_final'] == 0
    # Positive threshold: 5% of 100 gives an increment of five.
    image[:25, :50] = 100
    image[:25, 50:] = 101
    _, debug = run(image)
    assert debug['top_t'] == 105 and debug['top_auto_added_total'] == 5
    # Threshold zero still advances by one and respects the hard four-step cap.
    image[:25] = 100
    _, debug = run(image)
    assert debug['top_auto_steps'] == 4 and debug['top_t'] == 4
    for settings in ({'OTSU_TOP_AUTO_ADJUST_ENABLE': False},
                     {'OTSU_TOP_ADJUST_MAX_STEPS': 0},
                     {'OTSU_MIN_BOTTOM_FOREGROUND_RATIO': 0.3}):
        _, debug = run(image, **settings)
        assert debug['top_auto_steps'] == 0
    image[:25] = 0  # Lower upper density must never trigger correction.
    _, debug = run(image)
    assert debug['top_auto_steps'] == 0
    _, debug = run(np.zeros((100, 100), np.uint8))
    assert debug['top_auto_steps'] == 0 and debug['bottom_density'] == 0
    print('PASS: bounded upper density correction, percentage steps, fixed bottom and single morphology', flush=True)


def validate_preprocessing():
    validate_top_auto_adjust()
    pipeline = PreprocessingMixin()
    images = [road_image(), np.zeros((270, 512, 3), dtype=np.uint8), road_image(True)]
    for index, image in enumerate(images):
        binary = pipeline.preprocess_frame_bgr(image)
        assert_binary(binary)
        assert pipeline._last_threshold_debug['mode'] == 'regional_otsu_vertical'
        assert pipeline._last_threshold_debug['tophat'] is True
        for key in ('top_t', 'bottom_t'):
            assert np.isfinite(pipeline._last_threshold_debug[key])
            assert 0 <= pipeline._last_threshold_debug[key] <= 255
        assert 1 <= pipeline._last_threshold_debug['split_y'] < binary.shape[0]
        if index == 0:
            assert set(np.unique(binary)) == {0, 1}, 'Road fixture must contain foreground and background'
        if index == 1:
            assert not np.any(binary), 'Blank image should remain blank'
    gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    bev, matrix, inverse, src, dst = pipeline.get_birdseye_responsive_gray(
        gray, top_width_factor=tuning.BEV_SRC_TOP_WIDTH_FACTOR,
        bottom_width_factor=tuning.BEV_SRC_BOTTOM_WIDTH_FACTOR,
        bottom_offset_factor=tuning.BEV_SRC_BOTTOM_OFFSET_FACTOR,
        horizon_factor=tuning.BEV_SRC_HORIZON_FACTOR,
        empty_margins=tuning.BEV_DST_EMPTY_MARGINS, return_points=True)
    assert bev.shape == (270, 512) and bev.dtype == np.uint8
    assert np.array_equal(src, np.float32([[13, 270], [128, 191], [384, 191], [499, 270]]))
    transformed = cv2.perspectiveTransform(src[None, :, :], matrix)[0]
    assert np.allclose(transformed, dst, atol=1e-3)
    assert np.allclose(cv2.perspectiveTransform(dst[None, :, :], inverse)[0], src, atol=1e-3)
    class ThresholdObserver(PreprocessingMixin):
        def _apply_tophat_prethreshold(self, image):
            self.prethreshold = super()._apply_tophat_prethreshold(image)
            return self.prethreshold

        def _apply_threshold_morphology(self, binary):
            self.raw_binary = binary.copy()
            self.morphology_calls = getattr(self, 'morphology_calls', 0) + 1
            return super()._apply_threshold_morphology(binary)

    observer = ThresholdObserver()
    cleaned = observer._otsu_binary(bev)
    split_y = observer._last_threshold_debug['split_y']
    top_t, top_raw = cv2.threshold(
        observer.prethreshold[:split_y], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bottom_t, bottom_raw = cv2.threshold(
        observer.prethreshold[split_y:], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    debug = observer._last_threshold_debug
    assert debug['top_otsu_raw'] == top_t
    assert debug['bottom_otsu_raw'] == bottom_t
    assert debug['bottom_t'] == np.clip(bottom_t + tuning.OTSU_BOTTOM_OFFSET, 0, 255)
    expected = np.vstack((observer.prethreshold[:split_y] > debug['top_t'],
                          observer.prethreshold[split_y:] > debug['bottom_t'])).astype(np.uint8)
    assert np.array_equal(observer.raw_binary, expected)
    assert observer.morphology_calls == 1
    assert np.array_equal(cleaned, pipeline._apply_threshold_morphology(observer.raw_binary))
    assert np.array_equal(observer.prethreshold, pipeline._apply_tophat_prethreshold(bev))
    assert_binary(cleaned)
    # Threshold-only edge checks, independent of the optional historical comparison.
    from unittest.mock import patch
    for shape in ((400, 400), (271, 650), (2, 7)):
        for fraction in (-1.0, 0.5, 2.0):
            probe = ThresholdObserver()
            image = np.zeros(shape, dtype=np.uint8)
            with patch.object(tuning, 'OTSU_VERTICAL_SPLIT_FRAC', fraction):
                result = probe._otsu_binary(image)
            assert result.shape == shape and result.dtype == np.uint8
            assert not np.any(result)
            assert 1 <= probe._last_threshold_debug['split_y'] < shape[0]
            assert probe.morphology_calls == 1
    print('PASS: regional Otsu thresholds, recombination, split bounds and single morphology pass', flush=True)
    reference_check(images)
    print('PASS: grayscale / preprocessing pipeline and Bird\'s-Eye-View transform', flush=True)
    print('PASS: vertical regional Otsu, unchanged Top-Hat/morphology and uint8 0/1 binary output', flush=True)
    print('PASS: blank and altered-illumination frames handled safely', flush=True)


def validate_failure_recovery():
    queues = {name: Queue() for name in ('Config', 'Critical', 'General', 'Warning')}
    lane = threadLaneDetection(queues)
    lane.lane_enabled = True
    ok, jpeg = cv2.imencode('.jpg', road_image())
    assert ok
    payload = base64.b64encode(jpeg).decode('ascii')
    real_preprocess = lane.preprocess_frame_bgr
    def fail(frame):
        raise RuntimeError('injected preprocessing failure')
    def work():
        with lane._rx_lock:
            lane._latest_b64 = payload
            lane._rx_count += 1
        lane.thread_work()
    logs = io.StringIO()
    try:
        with redirect_stdout(logs):
            lane.preprocess_frame_bgr = fail
            for _ in range(3):
                work()
            lane.preprocess_frame_bgr = real_preprocess
            work()
            work()
            lane.preprocess_frame_bgr = fail
            work()
            work()
            lane.preprocess_frame_bgr = real_preprocess
            work()
            lane.lane_enabled = False
            lane.preprocess_frame_bgr = fail
            work()
        assert logs.getvalue().count('Frame preprocessing failed') == 2
        assert logs.getvalue().count('First frame preprocessing completed successfully') == 1
        assert queues['Critical'].empty(), 'Preprocessing generated actuation'
    finally:
        for subscriber in (lane.cameraSubscriber, lane.controlSubscriber):
            subscriber._pipeRecv.close()
            subscriber._pipeSend.close()
    print('PASS: preprocessing failures stay quiet, recover, and skip disabled frames', flush=True)


def validate():
    root = Path(__file__).parent
    # Phase 2 itself runs the unchanged Phase 1 validator.
    regression = subprocess.run([sys.executable, str(root / 'validate_phase2.py')],
                                capture_output=True, text=True, timeout=60)
    print(regression.stdout, end='')
    assert regression.returncode == 0, regression.stderr
    assert 'PASS: Phase 1 regression validation' in regression.stdout
    print('PASS: Phase 2 regression validation', flush=True)
    validate_preprocessing()
    validate_failure_recovery()
    # Reuse Phase 2's real-process transport test without editing that validator.
    integration = subprocess.run([sys.executable, str(root / 'validate_phase2.py'), '--transport-child'],
                                 capture_output=True, text=True, timeout=30)
    print(integration.stdout, end='')
    assert integration.returncode == 0, integration.stderr
    assert integration.stdout.count('First frame preprocessing completed successfully') == 1
    assert 'Frame preprocessing failed' not in integration.stdout
    assert 'PASS: safe transport child and clean shutdown' in integration.stdout
    print('PASS: Gateway -> LaneDetection -> preprocessing integration', flush=True)
    print('PASS: no valid movement command generated', flush=True)
    print('PASS: clean Phase 3.1 shutdown', flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    validate()
