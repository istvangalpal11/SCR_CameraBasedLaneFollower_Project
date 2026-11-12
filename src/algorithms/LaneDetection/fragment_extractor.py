"""Short-range grouping of real Phase 1 observations into visible marking fragments."""
from statistics import median
from src.algorithms.LaneDetection.boundary_builder import build_boundary_candidates


def _endpoint_tangent(points, row_step):
    """Approximate dx/dy in increasing image Y; never claim precision for 1-2 points."""
    valid = len(points) >= 3 and points[-1]['y'] - points[0]['y'] >= 2 * row_step
    slope = ((points[-1]['x'] - points[0]['x']) / (points[-1]['y'] - points[0]['y'])
             if valid else None)
    return {'dx_dy': slope, 'valid': valid, 'support_points': len(points)}


def extract_marking_fragments(observations, row_step=5, *, max_gap_rows=2,
                              max_horizontal_jump_px=6, max_direction_change_deg=25,
                              max_width_ratio=3.0):
    """Reuse local continuity linking with independent, strict fragment settings.

    max_gap_rows * row_step is maximum observed endpoint separation (10 px at
    defaults), not the number of missing scan rows. No large-gap boundary linking,
    fragment merging, RANSAC, polynomial fitting, history or image-half labels.
    The existing continuity algorithm supplies position, local direction and
    relative width gates; none of its baseline defaults are changed.

    min_candidate_points=1 preserves every real observation, even singleton/bottom
    edge fragments. Output observations and endpoints reference original Phase 1
    dictionaries, without modifying them or synthesizing pixels. IDs are per-call.
    Tangents use at most three actual observations at each endpoint. valid means
    enough local support only, not calibrated accuracy; short fragments return
    dx_dy=None. Tangents are oriented toward increasing image Y at both ends.

    Parameters are provisional pixels/degrees. Sampling cannot prove physical
    connectivity between rows: very close dashes can still join within the chosen
    small gap tolerance; noise/occlusion/strong curvature may split real paint.
    Crossing stripes with indistinguishable center/width evidence remain ambiguous.
    """
    chains = build_boundary_candidates(
        observations, row_step=row_step, max_link_gap_rows=max_gap_rows,
        max_horizontal_jump_px=max_horizontal_jump_px,
        max_direction_change_deg=max_direction_change_deg,
        min_candidate_points=1, max_width_ratio=max_width_ratio)
    fragments = []
    for chain in chains:
        points = chain['observations']
        fragments.append({
            'fragment_id': len(fragments),
            'observations': points,
            'point_count': len(points),
            'start_observation': points[0],
            'end_observation': points[-1],
            'start_tangent': _endpoint_tangent(points[:3], row_step),
            'end_tangent': _endpoint_tangent(points[-3:], row_step),
            'y_min': chain['y_min'],
            'y_max': chain['y_max'],
            'typical_width_px': float(median(p['width_px'] for p in points)),
            'touches_left_edge': any(p.get('touches_left_edge', False) for p in points),
            'touches_right_edge': any(p.get('touches_right_edge', False) for p in points),
            'max_gap_px': chain['max_gap_px'],
            'gaps': chain['gaps'],
        })
    return fragments
