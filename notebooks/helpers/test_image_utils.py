"""Notebook-only screenshot preparation; never used by camera runtime."""
import cv2
import numpy as np


def normalize_notebook_image(image):
    """Notebook-only screenshot preparation, not production preprocessing."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected an OpenCV uint8 BGR image')
    original_height, original_width = image.shape[:2]
    if original_height < 1 or original_width < 1:
        raise ValueError('Empty image dimensions')
    scale = 270 / original_height
    resized_width = max(1, round(original_width * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, 270), interpolation=interpolation)
    if resized_width < 512:
        extra = 512 - resized_width
        left = extra // 2
        right = extra - left
        # One constant color, computed from the resized image (not the original).
        mean_bgr = tuple(int(v) for v in np.rint(resized.mean(axis=(0, 1))))
        normalized = cv2.copyMakeBorder(resized, 0, 0, left, right,
                                        cv2.BORDER_CONSTANT, value=mean_bgr)
    elif resized_width > 512:
        left = (resized_width - 512) // 2
        normalized = resized[:, left:left + 512].copy()
    else:
        normalized = resized
    if normalized.shape != (270, 512, 3) or normalized.dtype != np.uint8:
        raise ValueError('Normalization must produce 270x512 uint8 BGR')
    return normalized
