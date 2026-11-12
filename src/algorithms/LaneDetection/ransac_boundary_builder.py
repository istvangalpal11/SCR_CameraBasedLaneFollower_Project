"""Experimental, bounded per-frame multi-model RANSAC in BEV pixel space.

Hypotheses are diagnostic quadratics, not metric lane geometry or tracked lanes.
Only NumPy is needed. The independent continuity builder is deliberately unused.
"""
from numbers import Integral
import numpy as np


def _fit(design, xs):
    coefficients, _, rank, _ = np.linalg.lstsq(design, xs, rcond=None)
    return coefficients if rank == 3 and np.isfinite(coefficients).all() else None


def _inliers(design, xs, ys, coefficients, threshold):
    residuals = np.abs(xs - design @ coefficients)
    eligible = np.flatnonzero(residuals <= threshold)
    # A row cannot boost support with multiple neighboring parallel markings.
    # Keep its closest point; exact ties use the deterministic input ordering.
    order = np.lexsort((eligible, residuals[eligible], ys[eligible]))
    eligible = eligible[order]
    if len(eligible):
        eligible = eligible[np.r_[True, np.diff(ys[eligible]) != 0]]
    return eligible


def build_ransac_candidates(observations, row_step=5, *, max_iterations=150,
                            inlier_threshold_px=3.0, min_inliers=6,
                            min_y_span_px=60.0, max_candidates=8, random_seed=42):
    """Return independent candidates with the Phase 2.1 base field semantics.

    Sort observations deterministically and sample three distinct rows. Fit
    x = a*t**2 + b*t + c where t = (y - y_offset) / y_scale. Count at most one
    inlier per row; require min_inliers distinct rows and min_y_span_px coverage.
    Refine each supported hypothesis with at most two least-squares refits.
    Rank by row support, then longitudinal span, then lower RMS residual.

    Peel the best candidate's assigned observations, then repeat with the same
    call-local seeded generator, capped by max_iterations per candidate search
    and max_candidates accepted models. No observation is assigned twice.
    All thresholds are provisional pixels, without physical lane-width gates.

    Base fields preserve original observation dictionaries/edge flags by reference.
    gaps and max_gap_px are observed endpoint separations, exactly as in the
    continuity builder, with no invented evidence. inlier_ratio uses the remaining
    observation count before accepting this candidate; it is not a probability.
    model_coefficients are [a, b, c] in normalized Y, not raw pixel Y.

    Limitations: greedy peeling is order-dependent; nearby/crossing/branching
    markings may mix, and a smooth polynomial through a long gap is not proof of
    paint. Short fragments may fail the span/support gates. Quadratics can be
    unsuitable for tight curves; no model is justified outside [y_min, y_max].
    """
    for name, value, minimum in (('row_step', row_step, 1),
                                  ('max_iterations', max_iterations, 1),
                                  ('min_inliers', min_inliers, 3),
                                  ('max_candidates', max_candidates, 1),
                                  ('random_seed', random_seed, 0)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    for name, value in (('inlier_threshold_px', inlier_threshold_px),
                        ('min_y_span_px', min_y_span_px)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    ordered = sorted(observations, key=lambda o: (o['y'], o['x'], o['x_start'], o['x_end']))
    if len(ordered) < min_inliers:
        return []
    xs = np.array([o['x'] for o in ordered], dtype=float)
    ys = np.array([o['y'] for o in ordered], dtype=float)
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        raise ValueError('Observation centers must be finite')
    y_offset = float((ys.min() + ys.max()) / 2)
    y_scale = float(max((ys.max() - ys.min()) / 2, row_step))
    normalized_y = (ys - y_offset) / y_scale
    design = np.column_stack((normalized_y**2, normalized_y, np.ones(len(ordered))))
    remaining = np.arange(len(ordered))
    rng = np.random.default_rng(random_seed)
    candidates = []
    for _ in range(max_candidates):
        if len(remaining) < min_inliers or len(np.unique(ys[remaining])) < min_inliers:
            break
        local_x, local_y, local_design = xs[remaining], ys[remaining], design[remaining]
        best = None
        for _ in range(max_iterations):
            sample = rng.choice(len(remaining), size=3, replace=False)
            if len(np.unique(local_y[sample])) < 3 or np.ptp(local_y[sample]) < min_y_span_px:
                continue
            coefficients = _fit(local_design[sample], local_x[sample])
            if coefficients is None:
                continue
            indices = _inliers(local_design, local_x, local_y, coefficients, inlier_threshold_px)
            for _ in range(2):
                if len(indices) < min_inliers or np.ptp(local_y[indices]) < min_y_span_px:
                    break
                refined = _fit(local_design[indices], local_x[indices])
                if refined is None:
                    break
                coefficients = refined
                indices = _inliers(local_design, local_x, local_y, coefficients, inlier_threshold_px)
            if len(indices) < min_inliers or np.ptp(local_y[indices]) < min_y_span_px:
                continue
            residuals = local_x[indices] - local_design[indices] @ coefficients
            rms = float(np.sqrt(np.mean(residuals**2)))
            score = (len(indices), float(np.ptp(local_y[indices])), -rms)
            if best is None or score > best[0]:
                best = score, indices, coefficients, rms
        if best is None:
            break
        _, indices, coefficients, rms = best
        selected = remaining[indices]
        points = [ordered[int(i)] for i in selected]  # Already ordered by increasing Y.
        separations = [b['y'] - a['y'] for a, b in zip(points, points[1:])]
        gaps = [{'y_from': a['y'], 'y_to': b['y'], 'gap_px': b['y'] - a['y'],
                 'missing_scan_rows': max(0, (b['y'] - a['y'] - 1) // row_step)}
                for a, b in zip(points, points[1:]) if b['y'] - a['y'] > row_step]
        candidates.append({'candidate_id': len(candidates), 'observations': points,
                           'point_count': len(points), 'y_min': points[0]['y'],
                           'y_max': points[-1]['y'], 'max_gap_px': max(separations, default=0),
                           'gaps': gaps, 'rms_residual_px': rms,
                           'inlier_ratio': len(points) / len(remaining),
                           'model_type': 'quadratic_normalized_y',
                           'model_coefficients': coefficients.tolist(),
                           'y_offset': y_offset, 'y_scale': y_scale})
        keep = np.ones(len(remaining), dtype=bool)
        keep[indices] = False
        remaining = remaining[keep]
    return candidates
