"""Optional exports of notebook display arrays, without Matplotlib decorations."""
from pathlib import Path
import re
import shutil
import cv2
import numpy as np


def save_pairs_if_enabled(enabled, pairs, output_directory, *, before_name='before', after_name='after'):
    """Reset only a designated section folder and save lossless before/after PNGs.

    pairs contains (source_image_name, before_array, after_array).
    uint8 0/1 masks are saved as visible 0/255; source arrays are never modified.
    False returns immediately without inspecting, creating or deleting paths.
    """
    if not enabled:
        return []
    export_root = Path(__file__).resolve().parents[1] / 'debug_exports'
    directory = Path(output_directory).resolve()
    allowed = {export_root / section for section in ('bev', 'tophat', 'threshold')}
    if directory not in allowed:
        raise ValueError('Export reset is restricted to notebooks/debug_exports/{bev,tophat,threshold}')
    def safe_name(value):
        name = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._')
        if not name:
            raise ValueError('Export names must contain a filename character')
        return name
    suffixes = (safe_name(before_name), safe_name(after_name))
    pending = []
    seen = set()
    for source_name, before, after in pairs:
        stem = safe_name(Path(str(source_name)).stem)
        for suffix, image in zip(suffixes, (before, after)):
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.size == 0 or image.ndim not in (2, 3)
                    or (image.ndim == 3 and image.shape[2] not in (3, 4))):
                raise ValueError('Exports require nonempty uint8 grayscale/BGR/BGRA arrays')
            filename = stem + '_' + suffix + '.png'
            if filename in seen:
                raise ValueError('Duplicate export filename: ' + filename)
            seen.add(filename)
            display_image = image * np.uint8(255) if image.max() <= 1 else image
            pending.append((directory / filename, display_image))
    # Validate inputs and resolved destination before the explicitly requested reset.
    if directory.exists():
        if not directory.is_dir():
            raise ValueError('Export destination exists but is not a directory')
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    for path, image in pending:
        if not cv2.imwrite(str(path), image):
            raise OSError('Could not save notebook export: ' + str(path))
        saved.append(path)
    print(f'Saved {len(saved)} images to {directory}')
    return saved
