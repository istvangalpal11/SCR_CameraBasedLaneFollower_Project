"""Historical BFMC/SCR percentile threshold implementation.
Archived after Otsu became the active production threshold. Not imported by runtime.
Combine PercentileThresholdLegacyMixin with PreprocessingMixin for historical study;
its threshold methods delegate Top-Hat and morphology to that supplied mixin.
Historical tuning below preserves the values at the time of archival.
"""
from types import SimpleNamespace
import cv2
import numpy as np

lane_tuning = SimpleNamespace(
    THRESH_PERCENTILE=90,
    THRESH_OFFSET=-4.0,
    THRESH_ROI_TOP_FRAC=0.3,
    THRESH_GAUSSIAN_BLUR_K=5,
    THRESH_TOPHAT_ENABLE=True,
    THRESH_TOPHAT_STATS_MIN_VALUE=3.0,
    THRESH_TOPHAT_STATS_MIN_SAMPLES=100,
    THRESH_TOPHAT_MIN_THRESHOLD=15.0,
)


class PercentileThresholdLegacyMixin:
    def _threshold_percentile(self, values, pct):
        vals = np.asarray(values)
        if vals.size == 0:
            return 0.0

        if bool(getattr(lane_tuning, "THRESH_TOPHAT_ENABLE", False)):
            stats_min = float(getattr(lane_tuning, "THRESH_TOPHAT_STATS_MIN_VALUE", 0.0))
            min_samples = max(1, int(getattr(lane_tuning, "THRESH_TOPHAT_STATS_MIN_SAMPLES", 100)))
            if stats_min > 0:
                filtered = vals[vals >= stats_min]
                if filtered.size >= min_samples:
                    vals = filtered

        if vals.size == 0:
            return 0.0

        return float(np.percentile(vals, pct))

    def _apply_tophat_threshold_floor(self, threshold_value):
        t = float(threshold_value)
        if bool(getattr(lane_tuning, "THRESH_TOPHAT_ENABLE", False)):
            t = max(t, float(getattr(lane_tuning, "THRESH_TOPHAT_MIN_THRESHOLD", 20.0)))
        return float(np.clip(t, 0, 255))

    def _dynamic_percentile_binary(self, gray_u8, pct=None, offset=None, roi_top=None):
        if gray_u8.dtype != np.uint8:
            gray_u8 = np.clip(gray_u8, 0, 255).astype(np.uint8)

        thresh_u8 = self._apply_tophat_prethreshold(gray_u8)

        pct = lane_tuning.THRESH_PERCENTILE if pct is None else pct
        offset = lane_tuning.THRESH_OFFSET if offset is None else offset
        roi_top = lane_tuning.THRESH_ROI_TOP_FRAC if roi_top is None else roi_top

        h, _ = thresh_u8.shape[:2]
        y0 = int(np.clip(h * float(roi_top), 0, max(h - 1, 0)))

        roi = thresh_u8[y0:, :]
        blur_k = max(3, int(getattr(lane_tuning, "THRESH_GAUSSIAN_BLUR_K", 5)))
        if blur_k % 2 == 0:
            blur_k += 1
        if bool(getattr(lane_tuning, "THRESH_TOPHAT_ENABLE", False)):
            roi_blur = roi
        else:
            roi_blur = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)

        t = self._threshold_percentile(roi_blur, pct) + offset
        t = self._apply_tophat_threshold_floor(t)
        binary = (thresh_u8 >= t).astype(np.uint8)
        binary = self._apply_threshold_morphology(binary)
        self._last_threshold_debug = {
            "mode": "global",
            "global_t": t,
            "tophat": bool(getattr(lane_tuning, "THRESH_TOPHAT_ENABLE", False)),
        }
        return binary
