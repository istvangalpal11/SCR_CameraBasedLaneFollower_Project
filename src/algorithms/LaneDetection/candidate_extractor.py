"""Global white-run observations from binary BEV images, without lane classification."""
import numpy as np


def extract_row_markings(binary_bev, row_step=5):
    """Scan rows 0, row_step, ... and return every contiguous run of ones.

    Input is a 2-D uint8 array containing only 0 and 1; dimensions are arbitrary.
    x_start and x_end are inclusive; x is their midpoint (possibly half a pixel).
    Runs are returned in row order, then left to right. Input is never modified.
    """
    if (not isinstance(binary_bev, np.ndarray) or binary_bev.ndim != 2
            or binary_bev.dtype != np.uint8):
        raise ValueError('binary_bev must be a 2-D uint8 array containing 0 and 1')
    if np.any((binary_bev != 0) & (binary_bev != 1)):
        raise ValueError('binary_bev must contain only 0 and 1')
    if (isinstance(row_step, (bool, np.bool_))
            or not isinstance(row_step, (int, np.integer)) or row_step <= 0):
        raise ValueError('row_step must be a positive integer')
    height, width = binary_bev.shape
    observations = []
    for y in range(0, height, row_step):
        # Signed transitions with zero padding retain runs at either image edge.
        transitions = np.diff(np.pad(binary_bev[y].astype(np.int8), (1, 1)))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1) - 1
        for start, end in zip(starts, ends):
            x_start, x_end = int(start), int(end)
            observations.append({
                'x': (x_start + x_end) / 2,
                'y': y,
                'width_px': x_end - x_start + 1,
                'x_start': x_start,
                'x_end': x_end,
                'touches_left_edge': x_start == 0,
                'touches_right_edge': x_end == width - 1,
            })
    return observations
