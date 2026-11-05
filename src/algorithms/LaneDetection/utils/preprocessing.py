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
                    return_points=False, output_size=None): # output_size is (width, height)
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

        if output_size is None:
            out_w, out_h = w, h
        else:
            if (not isinstance(output_size, (tuple, list)) or len(output_size) != 2
                    or any(isinstance(v, (bool, np.bool_))
                           or not isinstance(v, (int, np.integer)) or v <= 0
                           for v in output_size)):
                raise ValueError('output_size must be a pair of positive integer dimensions')
            out_w, out_h = map(int, output_size)
        dst_margin = int(out_w * empty_margins)
        dst = np.float32([
            [dst_margin, out_h],
            [dst_margin, 0],
            [out_w - dst_margin, 0],
            [out_w - dst_margin, out_h]
        ])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        bev_gray = cv2.warpPerspective(gray_img, M, (out_w, out_h))

        if return_points:
            return bev_gray, M, Minv, src, dst

        return bev_gray, M, Minv

    def _apply_threshold_morphology(self, binary):
        binary = (np.asarray(binary) > 0).astype(np.uint8)

        open_iter = int(getattr(lane_tuning, "THRESH_OPEN_ITER", 0))
        if open_iter > 0:
            k_open_w = max(1, int(getattr(lane_tuning, "THRESH_OPEN_KERNEL_W", 3)))
            k_open_h = max(1, int(getattr(lane_tuning, "THRESH_OPEN_KERNEL_H", 3)))
            k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open_w, k_open_h))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=open_iter)
            binary = (binary > 0).astype(np.uint8)

        close_iter = int(getattr(lane_tuning, "THRESH_CLOSE_ITER", 0))
        if close_iter > 0:
            k_close_w = max(1, int(getattr(lane_tuning, "THRESH_CLOSE_KERNEL_W", 3)))
            k_close_h = max(1, int(getattr(lane_tuning, "THRESH_CLOSE_KERNEL_H", 11)))
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

    def _otsu_binary(self, gray_u8):
        """Existing Top-Hat -> upper/lower Otsu -> one full-image morphology pass."""
        if gray_u8.dtype != np.uint8:
            gray_u8 = np.clip(gray_u8, 0, 255).astype(np.uint8)
        prethreshold_u8 = self._apply_tophat_prethreshold(gray_u8)
        height = prethreshold_u8.shape[0]
        if height < 2:
            raise ValueError('Vertical regional Otsu requires at least two image rows')
        split_frac = float(np.clip(lane_tuning.OTSU_VERTICAL_SPLIT_FRAC, 0.0, 1.0))
        if not np.isfinite(split_frac):
            raise ValueError('OTSU_VERTICAL_SPLIT_FRAC must be finite')
        split_y = max(1, min(height - 1, int(height * split_frac)))
        top_otsu_raw, _ = cv2.threshold(
            prethreshold_u8[:split_y, :], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bottom_otsu_raw, _ = cv2.threshold(
            prethreshold_u8[split_y:, :], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        top_offset = float(lane_tuning.OTSU_TOP_OFFSET)
        bottom_offset = float(lane_tuning.OTSU_BOTTOM_OFFSET)
        top_threshold = float(np.clip(top_otsu_raw + top_offset, 0, 255))
        bottom_threshold = float(np.clip(bottom_otsu_raw + bottom_offset, 0, 255))
        # Match OpenCV THRESH_BINARY's strict comparison, preserving zero-offset behavior.
        top_binary = prethreshold_u8[:split_y, :] > top_threshold
        bottom_binary = prethreshold_u8[split_y:, :] > bottom_threshold
        top_threshold_before_auto = top_threshold
        top_density_initial = float(np.count_nonzero(top_binary) / top_binary.size)
        top_density = top_density_initial
        bottom_density = float(np.count_nonzero(bottom_binary) / bottom_binary.size)
        max_ratio = float(lane_tuning.OTSU_TOP_MAX_TO_BOTTOM_RATIO)
        min_bottom_density = float(lane_tuning.OTSU_MIN_BOTTOM_FOREGROUND_RATIO)
        max_steps = max(0, int(lane_tuning.OTSU_TOP_ADJUST_MAX_STEPS))
        auto_steps = 0
        while (lane_tuning.OTSU_TOP_AUTO_ADJUST_ENABLE
               and bottom_density >= min_bottom_density
               and top_density > bottom_density * max_ratio
               and auto_steps < max_steps and top_threshold < 255):
            step = max(1, round(top_threshold * lane_tuning.OTSU_TOP_ADJUST_PERCENT))
            top_threshold = float(np.clip(top_threshold + step, 0, 255))
            top_binary = prethreshold_u8[:split_y, :] > top_threshold
            top_density = float(np.count_nonzero(top_binary) / top_binary.size)
            auto_steps += 1
        binary = np.vstack((top_binary, bottom_binary)).astype(np.uint8)
        binary = self._apply_threshold_morphology(binary)
        self._last_threshold_debug = {
            "mode": "regional_otsu_vertical",
            "top_otsu_raw": float(top_otsu_raw),
            "bottom_otsu_raw": float(bottom_otsu_raw),
            "top_offset": top_offset,
            "bottom_offset": bottom_offset,
            "top_t": float(top_threshold),
            "bottom_t": float(bottom_threshold),
            "top_threshold_before_auto": top_threshold_before_auto,
            "top_auto_steps": auto_steps,
            "top_auto_added_total": top_threshold - top_threshold_before_auto,
            "top_density_initial": top_density_initial,
            "top_density_final": top_density,
            "bottom_density": bottom_density,
            "max_top_to_bottom_ratio": max_ratio,
            "min_bottom_foreground_ratio": min_bottom_density,
            "split_y": split_y,
            "split_frac": split_frac,
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

    def _draw_bev_trapezoid_debug(self, image, src_points):
        """Draw actual warp source points on a display-only BGR copy."""
        overlay = (cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
                   if image.ndim == 2 else image.copy())
        polygon = np.rint(src_points).astype(np.int32)
        cv2.polylines(overlay, [polygon], True, (0, 255, 255), 2)
        for number, point in enumerate(polygon, 1):
            x, y = map(int, point)
            cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)
            label_x = max(2, min(x + 5, overlay.shape[1] - 30))
            label_y = max(15, min(y - 6, overlay.shape[0] - 5))
            cv2.putText(overlay, f'P{number}', (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        return overlay

    def preprocess_frame_bgr(self, frame_bgr):
        """BFMC preprocessing with upper/lower Otsu, ending before lane search."""
        gray_input = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bevGray, _, _, src_points, _ = self.get_birdseye_responsive_gray(
            gray_input,
            top_width_factor=lane_tuning.BEV_SRC_TOP_WIDTH_FACTOR,
            bottom_width_factor=lane_tuning.BEV_SRC_BOTTOM_WIDTH_FACTOR,
            bottom_offset_factor=lane_tuning.BEV_SRC_BOTTOM_OFFSET_FACTOR,
            horizon_factor=lane_tuning.BEV_SRC_HORIZON_FACTOR,
            empty_margins=lane_tuning.BEV_DST_EMPTY_MARGINS,
            return_points=True,
            output_size=(lane_tuning.BEV_OUTPUT_WIDTH, lane_tuning.BEV_OUTPUT_HEIGHT),
        )
        self._last_bev_trapezoid_debug = self._draw_bev_trapezoid_debug(gray_input, src_points)
        if lane_tuning.BEV_BLACK_TRAPEZOID_ENABLE:
            bevGray = self.apply_bev_black_trapezoid(
                bevGray,
                top_width_frac=lane_tuning.BEV_BLACK_TRAPEZOID_TOP_WIDTH_FRAC,
                bottom_width_frac=lane_tuning.BEV_BLACK_TRAPEZOID_BOTTOM_WIDTH_FRAC,
                top_y_frac=lane_tuning.BEV_BLACK_TRAPEZOID_TOP_Y_FRAC,
                bottom_y_frac=lane_tuning.BEV_BLACK_TRAPEZOID_BOTTOM_Y_FRAC,
                center_x_frac=lane_tuning.BEV_BLACK_TRAPEZOID_CENTER_X_FRAC,
            )
        # The original runtime also leaves apply_roi_mask disabled.
        return self._otsu_binary(bevGray)
