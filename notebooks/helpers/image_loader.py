"""Saved notebook image discovery and loading."""
import re
import cv2
from .test_image_utils import normalize_notebook_image


def natural_key(path):
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", path.name))


def load_saved_images(root, directory, extensions):
    directory = root / directory
    allowed = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    valid_extensions = []
    for entry in extensions:
        extension = entry.strip().lower() if isinstance(entry, str) else ''
        if not extension.startswith('.') or extension not in allowed:
            print(f'WARNING: Invalid or unsupported image extension {entry!r}; ignored')
            continue
        if extension not in valid_extensions:
            valid_extensions.append(extension)
    candidates = [p for p in directory.iterdir() if p.is_file() and p.stem.isdigit()]
    paths = []
    for extension in valid_extensions:
        paths = sorted((p for p in candidates if p.suffix.lower() == extension), key=natural_key)
        if paths:
            print(f'Using {extension} files from {directory}')
            break
    if not paths:
        print(f'WARNING: No numerically named images found in {directory} for {valid_extensions}')
    items = []
    for path in paths:
        try:
            original = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if original is None:
                raise ValueError('OpenCV could not read the image')
            normalized = normalize_notebook_image(original)
        except (cv2.error, ValueError) as exc:
            print(f'WARNING: {path.name}: {exc}; skipped')
            continue
        items.append({'name': path.name, 'original_bgr': original, 'frame_bgr': normalized})
    return items
