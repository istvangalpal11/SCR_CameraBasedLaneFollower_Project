"""Observation hooks around the real production preprocessing mixin."""
from src.algorithms.LaneDetection.utils.preprocessing import PreprocessingMixin
from .bev_test import custom_bev_points
import cv2


class _GrayscaleObserved(Exception):
    pass


class PreprocessingTester(PreprocessingMixin):
    def __init__(self, notebook_namespace):
        # Resolve later notebook crop controls at call time, without editing those cells.
        self._notebook_namespace = notebook_namespace

    def get_birdseye_responsive_gray(self, gray_img, *args, notebook_prepared=False, **kwargs):
        self.observed_gray = gray_img.copy()
        if getattr(self, '_capture_gray_only', False):
            raise _GrayscaleObserved()
        if not notebook_prepared:
            gray_img = self._notebook_namespace["test_bottom_crop"](gray_img)
        self.observed_gray_for_bev = gray_img.copy()
        geometry = custom_bev_points(gray_img, self._notebook_namespace)
        if geometry is not None:
            src, dst = geometry
            matrix = cv2.getPerspectiveTransform(src, dst)
            inverse = cv2.getPerspectiveTransform(dst, src)
            height, width = gray_img.shape[:2]
            bev = cv2.warpPerspective(gray_img, matrix, (width, height))
            if kwargs.get('return_points', False):
                return bev, matrix, inverse, src, dst
            return bev, matrix, inverse
        return super().get_birdseye_responsive_gray(gray_img, *args, **kwargs)

    def _apply_tophat_prethreshold(self, gray_u8):
        prepared = super()._apply_tophat_prethreshold(gray_u8)
        self.observed_prethreshold = prepared.copy()
        return prepared

    def _apply_threshold_morphology(self, binary):
        self.observed_threshold = binary.copy()
        return super()._apply_threshold_morphology(binary)
