"""Per-frame continuity baseline for unlabeled BEV marking candidates.

All thresholds are provisional pixel-space gates, without metric calibration.
No image splitting, fitted road geometry, tracking state or hardware dependencies.
"""
from bisect import bisect_left, bisect_right
from itertools import groupby
from math import atan, degrees, isfinite
from numbers import Integral


def _local_slope(points):
    # Span up to three observations to reduce single-row center jitter.
    start, end = points[max(0, len(points) - 3)], points[-1]
    return (end['x'] - start['x']) / (end['y'] - start['y'])


def build_boundary_candidates(observations, row_step=5, *, max_link_gap_rows=16,
                              max_horizontal_jump_px=12, max_direction_change_deg=25,
                              min_candidate_points=4, max_width_ratio=3):
    """Link original Phase 1 observations, returning supported candidate dictionaries.

    Process increasing Y, globally across X. Each row is matched one-to-one to
    recent chains by prediction error, preferring established direction evidence
    over a singleton seed. Search is restricted to a local X interval and chains
    within max_link_gap_rows * row_step pixels of the current row.

    max_horizontal_jump_px bounds both displacement per row_step and prediction
    error in pixels. Direction uses the last up to three real points; changes
    exceeding max_direction_change_deg are rejected. A seed needs an adjacent
    observation before bridging gaps. Abrupt relative width changes greater than
    max_width_ratio are rejected, without an absolute marking-width cutoff.

    max_link_gap_rows is total endpoint separation in scan strides, not the count
    of missing rows. Gaps retain only endpoints and missing-row counts, never
    invented observations. Short chains remain unassigned in the caller's raw
    observations. Original dictionaries (including edge flags) are retained by
    reference and never modified. IDs restart on each call, with no frame history.

    Limitations: greedy local matching can split sharp turns or long gaps and can
    misassociate geometrically indistinguishable markings/crossings. This is not
    a confidence estimate or a fitted boundary model.
    """
    for name, value in (('row_step', row_step), ('max_link_gap_rows', max_link_gap_rows),
                        ('min_candidate_points', min_candidate_points)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not isfinite(max_horizontal_jump_px) or max_horizontal_jump_px <= 0:
        raise ValueError('max_horizontal_jump_px must be finite and positive')
    if not isfinite(max_direction_change_deg) or not 0 < max_direction_change_deg <= 180:
        raise ValueError('max_direction_change_deg must be in (0, 180]')
    if not isfinite(max_width_ratio) or max_width_ratio < 1:
        raise ValueError('max_width_ratio must be finite and at least 1')

    ordered = sorted(observations, key=lambda o: (o['y'], o['x'], o['x_start'], o['x_end']))
    chains, active = [], []
    max_gap_px = max_link_gap_rows * row_step
    for y, group in groupby(ordered, key=lambda o: o['y']):
        row = list(group)
        xs = [o['x'] for o in row]
        active = [i for i in active if y - chains[i][-1]['y'] <= max_gap_px]
        links = []
        for chain_index in active:
            points = chains[chain_index]
            last = points[-1]
            dy = y - last['y']
            has_direction = len(points) >= 2
            if not has_direction and dy > row_step:
                continue  # A lone point cannot justify a long dashed association.
            slope = _local_slope(points) if has_direction else 0.0
            predicted_x = last['x'] + slope * dy
            recent = points[-3:]
            reference_width = sum(p['width_px'] for p in recent) / len(recent)
            lo = bisect_left(xs, predicted_x - max_horizontal_jump_px)
            hi = bisect_right(xs, predicted_x + max_horizontal_jump_px)
            for index in range(lo, hi):
                point = row[index]
                dx = point['x'] - last['x']
                if abs(dx) > max_horizontal_jump_px * dy / row_step:
                    continue
                width_ratio = max(reference_width, point['width_px']) / min(reference_width, point['width_px'])
                if width_ratio > max_width_ratio:
                    continue
                angle_change = abs(degrees(atan(dx / dy) - atan(slope))) if has_direction else 0.0
                if angle_change > max_direction_change_deg:
                    continue
                links.append((not has_direction, abs(point['x'] - predicted_x),
                              angle_change, dy, chain_index, index))
        used_chains, used_points = set(), set()
        for _, _, _, _, chain_index, index in sorted(links):
            if chain_index not in used_chains and index not in used_points:
                chains[chain_index].append(row[index])
                used_chains.add(chain_index)
                used_points.add(index)
        for index, point in enumerate(row):
            if index not in used_points:
                active.append(len(chains))
                chains.append([point])

    candidates = []
    for points in chains:
        if len(points) < min_candidate_points:
            continue
        separations = [b['y'] - a['y'] for a, b in zip(points, points[1:])]
        gaps = [{'y_from': a['y'], 'y_to': b['y'], 'gap_px': b['y'] - a['y'],
                 'missing_scan_rows': max(0, (b['y'] - a['y'] - 1) // row_step)}
                for a, b in zip(points, points[1:]) if b['y'] - a['y'] > row_step]
        candidates.append({'candidate_id': len(candidates), 'observations': points,
                           'point_count': len(points), 'y_min': points[0]['y'],
                           'y_max': points[-1]['y'], 'max_gap_px': max(separations, default=0),
                           'gaps': gaps})
    return candidates
