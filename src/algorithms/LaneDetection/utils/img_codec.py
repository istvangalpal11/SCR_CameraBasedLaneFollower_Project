"""Image codec adapted from the BFMC ImgCodecMixin, without logger coupling."""
import base64
import cv2
import numpy as np


class ImgCodecMixin:
    def _encode_frame_to_b64(self, frame_bgr):
        ok, jpg = cv2.imencode(
            '.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return None
        return base64.b64encode(jpg).decode('ascii')

    def _decode_b64_jpg_to_bgr(self, b64_str: str):
        """Return a decoded BGR array, or None for malformed base64/image data."""
        try:
            image_data = base64.b64decode(b64_str, validate=True)
            img_as_np = np.frombuffer(image_data, dtype=np.uint8)
            return cv2.imdecode(img_as_np, cv2.IMREAD_COLOR)
        except Exception:
            # The LaneDetection thread owns the non-spammy diagnostic.
            return None
