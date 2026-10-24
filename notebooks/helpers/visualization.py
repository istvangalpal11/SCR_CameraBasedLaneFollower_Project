"""Notebook display and pagination, independent of preprocessing mathematics."""
import cv2
import matplotlib.pyplot as plt
from src.algorithms.LaneDetection.utils import tuning


def draw_image(ax, image, title):
    if image.ndim == 3:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(image, cmap="gray", vmin=0, vmax=1 if image.max() <= 1 else 255)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def show_pairs(pairs, stage, before_label="Input", after_label="Result", *, configuration):
    USE_GRID = configuration['USE_GRID']
    GRID_ROWS = configuration['GRID_ROWS']
    GRID_COLS = configuration['GRID_COLS']
    # pairs: (source name, before array, after array); one source per grid position.
    if not USE_GRID:
        for name, before, after in pairs:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            draw_image(axes[0], before, f"{name} | {stage} | {before_label}")
            draw_image(axes[1], after, f"{name} | {stage} | {after_label}")
            fig.tight_layout()
            plt.show()
        return
    if (type(GRID_ROWS) is not int or type(GRID_COLS) is not int
            or GRID_ROWS < 1 or GRID_COLS < 1):
        raise ValueError("GRID_ROWS and GRID_COLS must be positive integers")
    capacity = GRID_ROWS * GRID_COLS
    for offset in range(0, len(pairs), capacity):
        page = pairs[offset:offset + capacity]
        fig = plt.figure(figsize=(8 * GRID_COLS, 3.5 * GRID_ROWS), layout="constrained")
        outer = fig.add_gridspec(GRID_ROWS, GRID_COLS)
        for position in range(capacity):
            if position >= len(page):
                fig.add_subplot(outer[position]).axis("off")
                continue
            name, before, after = page[position]
            inner = outer[position].subgridspec(1, 2)
            draw_image(fig.add_subplot(inner[0, 0]), before, f"{name} | {stage}\n{before_label}")
            draw_image(fig.add_subplot(inner[0, 1]), after, f"{name} | {stage}\n{after_label}")
        fig.suptitle(f"{stage} â€” page {offset // capacity + 1}")
        plt.show()


def show_tuning(prefixes):
    for name in sorted(vars(tuning)):
        if name.startswith(prefixes):
            print(f"{name} = {getattr(tuning, name)}")
