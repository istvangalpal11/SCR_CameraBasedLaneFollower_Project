# preprocessing.py
import cv2, numpy as np

from src.algorithms.LaneDetection.utils import tuning as lane_tuning


class PreprocessingMixin:
    def get_birdseye_responsive_gray(self, gray_img,
                    top_width_factor=lane_tuning.BEV_SRC_TOP_WIDTH_FACTOR,
                    bottom_width_factor=lane_tuning.BEV_SRC_BOTTOM_WIDTH_FACTOR,
                    bottom_offset_factor=lane_tuning.BEV_SRC_BOTTOM_OFFSET_FACTOR,
                    horizon_factor=lane_tuning.BEV_SRC_HORIZON_FACTOR,
                    empty_margins=lane_tuning.BEV_DST_EMPTY_MARGINS,
                    return_points=False): # empty space on left/right
        if len(gray_img.shape) == 2:
            h, w = gray_img.shape
        else:
            h, w = gray_img.shape[:2]

        mid_x = w // 2
        top_w_half = int(w * top_width_factor / 2)
        bot_w_half = int(w * bottom_width_factor / 2)

        y_top = int(h * horizon_factor)
        y_bot = int(h * (1 - bottom_offset_factor))

        p1 = (mid_x - bot_w_half, y_bot)
        p2 = (mid_x - top_w_half, y_top)
        p3 = (mid_x + top_w_half, y_top)
        p4 = (mid_x + bot_w_half, y_bot)

        src = np.float32([p1, p2, p3, p4])

        dst_margin = int(w * empty_margins)
        dst = np.float32([
            [dst_margin, h],
            [dst_margin, 0],
            [w - dst_margin, 0],
            [w - dst_margin, h]
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        bev_gray = cv2.warpPerspective(gray_img, M, (w, h))

        if return_points:
            return bev_gray, M, Minv, src, dst

        return bev_gray, M, Minv

    def _apply_threshold_morphology(self, binary):
        binary = (np.asarray(binary) > 0).astype(np.uint8)

        if bool(getattr(lane_tuning, "THRESH_OPEN_ENABLE", False)):
            k_open_w = max(1, int(getattr(lane_tuning, "THRESH_OPEN_KERNEL_W", 3)))
            k_open_h = max(1, int(getattr(lane_tuning, "THRESH_OPEN_KERNEL_H", 3)))
            open_iter = max(1, int(getattr(lane_tuning, "THRESH_OPEN_ITER", 1)))
            k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open_w, k_open_h))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=open_iter)
            binary = (binary > 0).astype(np.uint8)

        k_close_w = max(1, int(getattr(lane_tuning, "THRESH_CLOSE_KERNEL_W", 3)))
        k_close_h = max(1, int(getattr(lane_tuning, "THRESH_CLOSE_KERNEL_H", 11)))
        close_iter = max(1, int(getattr(lane_tuning, "THRESH_CLOSE_ITER", 1)))
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close_w, k_close_h))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=close_iter)

        return (binary > 0).astype(np.uint8)

    def _apply_tophat_prethreshold(self, gray_u8):
        gray_u8 = np.asarray(gray_u8)
        if gray_u8.dtype != np.uint8:
            gray_u8 = np.clip(gray_u8, 0, 255).astype(np.uint8)

        if not bool(getattr(lane_tuning, "THRESH_TOPHAT_ENABLE", False)):
            return gray_u8

        blur_k = int(getattr(lane_tuning, "THRESH_TOPHAT_BLUR_K", 0) or 0)
        if blur_k <= 0:
            blur_k = int(getattr(lane_tuning, "THRESH_GAUSSIAN_BLUR_K", 5))
        blur_k = max(1, blur_k)
        if blur_k > 1:
            if blur_k % 2 == 0:
                blur_k += 1
            work = cv2.GaussianBlur(gray_u8, (blur_k, blur_k), 0)
        else:
            work = gray_u8

        kernel_w = max(1, int(getattr(lane_tuning, "THRESH_TOPHAT_KERNEL_W", 41)))
        kernel_h = max(1, int(getattr(lane_tuning, "THRESH_TOPHAT_KERNEL_H", 1)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))

        tophat = cv2.morphologyEx(work, cv2.MORPH_TOPHAT, kernel)
        return np.clip(tophat, 0, 255).astype(np.uint8)

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

    # dinamic threshold
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

    def apply_bev_black_trapezoid(
        self,
        bev_img,
        top_width_frac,
        bottom_width_frac,
        top_y_frac,
        bottom_y_frac,
        center_x_frac,
        return_points=False,
    ):
        """
        Fill a bottom-center trapezoid on the BEV image with black.
        Use this to suppress the car front footprint before thresholding.
        """
        if bev_img is None:
            return (None, None) if return_points else None

        if len(bev_img.shape) == 2:
            h, w = bev_img.shape
            fill_color = 0
        else:
            h, w = bev_img.shape[:2]
            fill_color = (0, 0, 0)

        cx = int(np.clip(w * float(center_x_frac), 0, w - 1))
        y_top = int(np.clip(h * float(top_y_frac), 0, h - 1))
        y_bot = int(np.clip(h * float(bottom_y_frac), 0, h - 1))
        if y_bot <= y_top:
            y_bot = min(h - 1, y_top + 1)

        top_half = max(1, int(w * max(0.0, float(top_width_frac)) * 0.5))
        bot_half = max(1, int(w * max(0.0, float(bottom_width_frac)) * 0.5))

        p1 = (int(np.clip(cx - bot_half, 0, w - 1)), y_bot)
        p2 = (int(np.clip(cx - top_half, 0, w - 1)), y_top)
        p3 = (int(np.clip(cx + top_half, 0, w - 1)), y_top)
        p4 = (int(np.clip(cx + bot_half, 0, w - 1)), y_bot)

        points = np.array([p1, p2, p3, p4], dtype=np.int32)

        out = bev_img.copy()
        cv2.fillConvexPoly(out, points, fill_color)

        if return_points:
            return out, points
        return out

    def preprocess_frame_bgr(self, frame_bgr):
        """BFMC initial/global path, ending before any lane-pixel search."""
        gray_input = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bevGray, _, _ = self.get_birdseye_responsive_gray(
            gray_input,
            top_width_factor=lane_tuning.BEV_SRC_TOP_WIDTH_FACTOR,
            bottom_width_factor=lane_tuning.BEV_SRC_BOTTOM_WIDTH_FACTOR,
            bottom_offset_factor=lane_tuning.BEV_SRC_BOTTOM_OFFSET_FACTOR,
            horizon_factor=lane_tuning.BEV_SRC_HORIZON_FACTOR,
            empty_margins=lane_tuning.BEV_DST_EMPTY_MARGINS,
        )
        if lane_tuning.BEV_BLACK_TRAPEZOID_ENABLE:
            bevGray = self.apply_bev_black_trapezoid(
                bevGray,
                top_width_frac=lane_tuning.BEV_BLACK_TRAPEZOID_TOP_WIDTH_FRAC,
                bottom_width_frac=lane_tuning.BEV_BLACK_TRAPEZOID_BOTTOM_WIDTH_FRAC,
                top_y_frac=lane_tuning.BEV_BLACK_TRAPEZOID_TOP_Y_FRAC,
                bottom_y_frac=lane_tuning.BEV_BLACK_TRAPEZOID_BOTTOM_Y_FRAC,
                center_x_frac=lane_tuning.BEV_BLACK_TRAPEZOID_CENTER_X_FRAC,
            )
        # No previous fits exist yet. Use BFMC's original global fallback.
        # The original runtime also leaves apply_roi_mask disabled.
        return self._dynamic_percentile_binary(bevGray)
