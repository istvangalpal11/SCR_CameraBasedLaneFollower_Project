"""Optional notebook-only BEV source shift; no production tuning mutation."""
from unittest.mock import patch
import cv2
import numpy as np
from src.algorithms.LaneDetection.utils.preprocessing import PreprocessingMixin

_warned = set()


def custom_bev_points(gray_img, configuration):
    """Return validated custom src/dst, or None to use production unchanged."""
    if not configuration.get('USE_CUSTOM_TEST_BEV', False):
        return None
    names = ('TOP_WIDTH_FACTOR', 'BOTTOM_WIDTH_FACTOR', 'BOTTOM_OFFSET_FACTOR',
             'HORIZON_FACTOR', 'EMPTY_MARGINS', 'CENTER_OFFSET_PX')
    try:
        values = [configuration['TEST_BEV_' + name] for name in names]
        if any(isinstance(v, (bool, str)) for v in values):
            raise ValueError('geometry parameters must be finite numbers')
        top, bottom, offset, horizon, margins, shift = map(float, values)
        if not np.all(np.isfinite([top, bottom, offset, horizon, margins, shift])):
            raise ValueError('geometry parameters must be finite numbers')
        if not (0 < top <= 1 and 0 < bottom <= 1):
            raise ValueError('source width factors must be in (0, 1]')
        if not (0 <= horizon < 1 and 0 <= offset < 1):
            raise ValueError('horizon and bottom offset must be in [0, 1)')
        if not 0 <= margins < 0.5:
            raise ValueError('empty margins must be in [0, 0.5)')
        height, width = gray_img.shape[:2]
        if int(height * horizon) >= int(height * (1 - offset)):
            raise ValueError('source top must be above source bottom')
        # Reuse real production point construction, suppressing only its warp.
        with patch.object(cv2, 'warpPerspective', return_value=None):
            _, _, _, src, dst = PreprocessingMixin().get_birdseye_responsive_gray(
                gray_img, top_width_factor=top, bottom_width_factor=bottom,
                bottom_offset_factor=offset, horizon_factor=horizon,
                empty_margins=margins, return_points=True)
        src = src.copy()
        src[:, 0] += shift  # Translate all four corners by exactly the same amount.
        if src[:, 0].max() <= 0 or src[:, 0].min() >= width:
            raise ValueError('shifted source trapezoid is completely outside the image')
        if (not cv2.isContourConvex(src) or cv2.contourArea(src) <= 0
                or not cv2.isContourConvex(dst) or cv2.contourArea(dst) <= 0):
            raise ValueError('degenerate or crossed source/destination trapezoid')
        return src, dst
    except (ValueError, TypeError, KeyError, OverflowError, cv2.error) as exc:
        message = str(exc)
        warning_key = (message, tuple(repr(configuration.get('TEST_BEV_' + name)) for name in names))
        if warning_key not in _warned:
            print(f'WARNING: Invalid CUSTOM NOTEBOOK BEV: {message}. Using PRODUCTION geometry instead.')
            _warned.add(warning_key)
        return None
