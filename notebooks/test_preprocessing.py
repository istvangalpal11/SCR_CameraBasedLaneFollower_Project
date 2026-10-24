"""Notebook-only threshold experiments; never imported by the car runtime."""
import cv2
import numpy as np


def otsu_threshold(prethreshold_u8):
    """Return (whole-image Otsu threshold, raw uint8 0/1 mask).

    Input must already have production grayscale/BEV/mask/Top-Hat preparation.
    This function does not perform morphology or read/change tuning values.
    """
    if (not isinstance(prethreshold_u8, np.ndarray)
            or prethreshold_u8.dtype != np.uint8
            or prethreshold_u8.ndim != 2 or prethreshold_u8.size == 0):
        raise ValueError('Otsu requires a nonempty 2D uint8 pre-threshold image')
    threshold, binary = cv2.threshold(
        prethreshold_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return float(threshold), (binary > 0).astype(np.uint8)
