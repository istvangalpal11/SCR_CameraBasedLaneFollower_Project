"""Conservative deterministic fragment association, without fitting random models.

All gates are provisional pixel-space quantities. No hardware, metadata or history.
"""
from bisect import bisect_right
from math import atan, degrees, isfinite, tan, radians
from numbers import Integral
import numpy as np


def _slope(tangent):
    value = tangent.get('dx_dy')
    return value if tangent.get('valid', False) and value is not None and isfinite(value) else None


def _crosses(a, b):
    """Intersect two center-line segments in their overlapping image-Y interval."""
    lo, hi = max(a[0][1], b[0][1]), min(a[1][1], b[1][1])
    if lo > hi:
        return False
    def x_at(segment, y):
        (x0, y0), (x1, y1) = segment
        return x0 + (x1 - x0) * (y - y0) / (y1 - y0)
    d0, d1 = x_at(a, lo) - x_at(b, lo), x_at(a, hi) - x_at(b, hi)
    return d0 * d1 < 0 or abs(d0) < 1e-9 or abs(d1) < 1e-9


def _collect_pair_diagnostics(ordered, destination, *, max_gap_px, lateral_tolerance_px,
                              tangent_uncertainty_deg, max_direction_change_deg,
                              max_width_ratio, short_max_gap_px, min_strong_points):
    """Debug only: compare gates, including unreached ones, without affecting decisions.

    Record forward non-overlapping pairs in the normal Y window, plus the nearest
    start row beyond it to explain the search-window exclusion. No image/point
    copies. The caller may filter presentation; full records remain in destination.
    """
    records = {}
    for i, source in enumerate(ordered):
        later = [(j, target) for j, target in enumerate(ordered) if target['y_min'] > source['y_max']]
        beyond = next((target['y_min'] for _, target in later
                       if target['y_min'] - source['y_max'] > max_gap_px), None)
        for j, target in later:
            a, b = source['end_observation'], target['start_observation']
            gap, dx = b['y'] - a['y'], b['x'] - a['x']
            if gap > max_gap_px and target['y_min'] != beyond:
                continue
            sa, sb = _slope(source['end_tangent']), _slope(target['start_tangent'])
            slopes = [value for value in (sa, sb) if value is not None]
            strong = sa is not None and sb is not None and min(source['point_count'], target['point_count']) >= min_strong_points
            tolerance = lateral_tolerance_px + (gap * tan(radians(tangent_uncertainty_deg)) if strong else 0)
            width_ratio = max(source['typical_width_px'], target['typical_width_px']) / min(source['typical_width_px'], target['typical_width_px'])
            if slopes:
                angles = [degrees(atan(value)) for value in slopes]
                chord_angle = degrees(atan(dx / gap))
                angle_change = max([abs(chord_angle - angle) for angle in angles] + [max(angles) - min(angles)])
                error = abs(dx - sum(slopes) / len(slopes) * gap)
            else:
                angle_change = error = None
            checks = {}
            reason = None
            # Same short-circuit order as the actual algorithm (window included).
            for key, label, value, limit, passed, message in (
                ('source_support', 'Source points', source['point_count'], 2, source['point_count'] >= 2, 'Source fragment is a singleton'),
                ('gap', 'Longitudinal gap (px)', gap, max_gap_px, gap <= max_gap_px, 'Longitudinal gap exceeds search window'),
                ('target_support', 'Target points', target['point_count'], 2, target['point_count'] >= 2, 'Target fragment is a singleton'),
                ('tangent', 'At least one valid tangent', bool(slopes), True, bool(slopes), 'Neither endpoint has a valid tangent'),
                ('support_gap', 'Support-dependent gap (px)', gap, max_gap_px if strong else short_max_gap_px,
                 strong or gap <= short_max_gap_px, 'Weak fragment support limits the allowed gap'),
                ('width', 'Width ratio', width_ratio, max_width_ratio, width_ratio <= max_width_ratio, 'Width ratio exceeds threshold'),
                ('direction', 'Direction difference (deg)', angle_change, max_direction_change_deg,
                 angle_change <= max_direction_change_deg if angle_change is not None else None, 'Direction difference exceeds threshold'),
                ('lateral', 'Lateral prediction error (px)', error, tolerance,
                 error <= tolerance if error is not None else None, 'Lateral prediction error exceeds tolerance')):
                reached = reason is None
                checks[key] = dict(label=label, value=value, limit=limit, passed=passed, reached=reached)
                if reached and (passed is None or not passed):
                    reason = message
            record = dict(source_fragment_id=source['fragment_id'], target_fragment_id=target['fragment_id'],
                          gap_px=gap, lateral_error_px=error, lateral_tolerance_px=tolerance,
                          direction_difference_deg=angle_change, width_ratio=width_ratio,
                          strong_support=strong, checks=checks, status='rejected' if reason else 'pending',
                          reason=reason, display_distance=abs(dx) if error is None else error)
            destination.append(record)
            records[i, j] = record
    return records


def _validate_chain(chain, fragments, max_residual, max_angle, diagnostic=None):
    """Validate observations and endpoint tangents; never generate fitted evidence."""
    points = [p for i in chain for p in fragments[i]['observations']]
    ys = np.array([p['y'] for p in points], dtype=float)
    xs = np.array([p['x'] for p in points], dtype=float)
    offset, scale = (ys.min() + ys.max()) / 2, (ys.max() - ys.min()) / 2
    if scale <= 0:
        if diagnostic is not None:
            diagnostic['chain_fit_reason'] = 'Degenerate longitudinal span'
        return False, None
    t = (ys - offset) / scale
    design = np.column_stack((t*t, t, np.ones(len(t))))
    coeff, _, rank, _ = np.linalg.lstsq(design, xs, rcond=None)
    if rank < 3 or not np.isfinite(coeff).all():
        if diagnostic is not None:
            diagnostic['chain_fit_reason'] = 'Rank-deficient or non-finite validation fit'
        return False, None
    residual = float(np.max(np.abs(xs - design @ coeff)))
    if diagnostic is not None:
        diagnostic['validation_max_residual_px'] = residual
    if residual > max_residual:
        if diagnostic is not None:
            diagnostic['chain_fit_reason'] = 'Chain fit residual exceeds threshold'
        return False, residual
    for i in chain:
        f = fragments[i]
        for endpoint, tangent in ((f['start_observation'], f['start_tangent']),
                                  (f['end_observation'], f['end_tangent'])):
            slope = _slope(tangent)
            if slope is not None:
                at = (endpoint['y'] - offset) / scale
                fitted = (2 * coeff[0] * at + coeff[1]) / scale
                if abs(degrees(atan(fitted) - atan(slope))) > max_angle:
                    if diagnostic is not None:
                        diagnostic['chain_fit_reason'] = 'Validation-fit tangent disagrees with fragment endpoint'
                        diagnostic['chain_tangent_fragment_id'] = f['fragment_id']
                        diagnostic['chain_tangent_error_deg'] = abs(degrees(atan(fitted) - atan(slope)))
                    return False, residual
    return True, residual


def link_marking_fragments(fragments, row_step=5, *, max_gap_px=140.0,
                           lateral_tolerance_px=6.0, tangent_uncertainty_deg=8.0,
                           max_direction_change_deg=25.0, max_width_ratio=2.0,
                           short_max_gap_px=15.0, min_strong_points=4,
                           chain_max_residual_px=5.0, ambiguity_margin=0.05, diagnostics=None):
    """Return unlabeled, disjoint chains containing every original fragment/point.

    Search only later, non-overlapping fragments within max_gap_px. Reliable pairs
    use endpoint direction predictions with tolerance base + gap*tan(uncertainty).
    The mean outgoing/incoming slope allows gradual turning; the first endpoint's
    raw forward-prediction residual remains diagnostic. Chord and tangent angles
    must also agree. Width is relative, without metric lane-width assumptions.

    Long links require two valid tangents and min_strong_points at both ends.
    Weaker pairs need at least one real valid tangent, two observations per
    fragment, short_max_gap_px separation, and the base lateral tolerance only.
    Singletons and pairs with no valid direction remain independent candidates.
    Invalid tangents are never filled in or inferred from a long gap.

    Cost equally averages normalized position error, angle change, relative width
    change, gap and inverse support. Reject all alternatives at an endpoint if its
    two lowest costs differ by <= ambiguity_margin. Globally sort the remaining
    links; each fragment can have only one predecessor/successor. Reject segment
    crossings with accepted links or other observed fragment polylines. A normalized
    quadratic validates each proposed merged chain's residual and real tangents;
    reject connections, never observations. No RANSAC or fitted model is returned.

    Optional diagnostics is a caller-owned list receiving pair records separately
    from candidates. None (default) performs no diagnostic collection/calculations.
    Records distinguish reached checks from diagnostic-only comparisons after an
    early rejection. Nearest beyond-window rows are included for gap explanations.
    Supply a fresh list per frame; no diagnostic history is stored in this module.

    Base output/gap semantics match boundary_builder.py. Links retain per-connection
    cost and residual diagnostics; IDs are per-frame only. Original observation
    references and fragment IDs are preserved. Input order does not affect output.

    Limitations: conservative greedy/ambiguity rules can leave real boundaries split.
    Short fragments may remain disconnected. Center-line crossing checks are a
    geometric guard, not intersection classification or proof of paint in gaps.
    A quadratic may reject valid tight/S curves; compatible-looking artifacts may
    still associate. All thresholds require visual review, not physical calibration.
    """
    for name, value, minimum in (('row_step', row_step, 1), ('min_strong_points', min_strong_points, 3)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    for name, value in (('max_gap_px', max_gap_px), ('lateral_tolerance_px', lateral_tolerance_px),
                        ('short_max_gap_px', short_max_gap_px), ('chain_max_residual_px', chain_max_residual_px)):
        if not isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not isfinite(max_width_ratio) or max_width_ratio <= 1:
        raise ValueError('max_width_ratio must be finite and greater than 1')
    if not isfinite(tangent_uncertainty_deg) or not 0 <= tangent_uncertainty_deg < 90:
        raise ValueError('tangent_uncertainty_deg must be in [0, 90)')
    if not isfinite(max_direction_change_deg) or not 0 < max_direction_change_deg < 90:
        raise ValueError('max_direction_change_deg must be in (0, 90)')
    if not isfinite(ambiguity_margin) or ambiguity_margin < 0:
        raise ValueError('ambiguity_margin must be finite and nonnegative')
    ordered = sorted(fragments, key=lambda f: (f['y_min'], f['start_observation']['x'],
                                              f['y_max'], f['end_observation']['x'], f['fragment_id']))
    if len({f['fragment_id'] for f in ordered}) != len(ordered):
        raise ValueError('Fragment IDs must be unique within a frame')
    point_ids = [id(p) for f in ordered for p in f['observations']]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError('Input fragments must not share observations')
    debug_records = None
    if diagnostics is not None:
        debug_records = _collect_pair_diagnostics(
            ordered, diagnostics, max_gap_px=max_gap_px, lateral_tolerance_px=lateral_tolerance_px,
            tangent_uncertainty_deg=tangent_uncertainty_deg,
            max_direction_change_deg=max_direction_change_deg, max_width_ratio=max_width_ratio,
            short_max_gap_px=short_max_gap_px, min_strong_points=min_strong_points)
    starts = [f['y_min'] for f in ordered]
    pairs = []
    for i, source in enumerate(ordered):
        if source['point_count'] < 2:
            continue
        a = source['end_observation']
        sa = _slope(source['end_tangent'])
        lo, hi = bisect_right(starts, source['y_max']), bisect_right(starts, source['y_max'] + max_gap_px)
        for j in range(lo, hi):
            target = ordered[j]
            if target['point_count'] < 2:
                continue
            b, sb = target['start_observation'], _slope(target['start_tangent'])
            if sa is None and sb is None:
                continue
            gap, dx = b['y'] - a['y'], b['x'] - a['x']
            strong = sa is not None and sb is not None and min(source['point_count'], target['point_count']) >= min_strong_points
            if not strong and gap > short_max_gap_px:
                continue
            width_ratio = max(source['typical_width_px'], target['typical_width_px']) / min(source['typical_width_px'], target['typical_width_px'])
            if width_ratio > max_width_ratio:
                continue
            chord_angle = degrees(atan(dx / gap))
            slopes = [s for s in (sa, sb) if s is not None]
            angles = [degrees(atan(s)) for s in slopes]
            angle_change = max([abs(chord_angle - angle) for angle in angles] + [max(angles) - min(angles)])
            if angle_change > max_direction_change_deg:
                continue
            predicted_dx = sum(slopes) / len(slopes) * gap
            error = abs(dx - predicted_dx)
            tolerance = lateral_tolerance_px + (gap * tan(radians(tangent_uncertainty_deg)) if strong else 0)
            if error > tolerance:
                continue
            cost = (error / tolerance + angle_change / max_direction_change_deg
                    + (width_ratio - 1) / (max_width_ratio - 1) + gap / max_gap_px
                    + 1 / min(source['point_count'], target['point_count'])) / 5
            pairs.append({'source': i, 'target': j, 'cost': cost, 'gap_px': gap,
                          'prediction_residual_px': error, 'lateral_tolerance_px': tolerance,
                          'forward_residual_px': abs(dx - sa * gap) if sa is not None else None,
                          'direction_change_deg': angle_change, 'width_ratio': width_ratio})
            if debug_records is not None:
                debug_records[i, j]['cost'] = cost
    # Comparable alternatives stay unlinked rather than being resolved by ID order.
    ambiguous_out, ambiguous_in = set(), set()
    for key, ambiguous in (('source', ambiguous_out), ('target', ambiguous_in)):
        groups = {}
        for pair in pairs:
            groups.setdefault(pair[key], []).append(pair['cost'])
        for index, costs in groups.items():
            costs.sort()
            if len(costs) > 1 and costs[1] - costs[0] <= ambiguity_margin:
                ambiguous.add(index)
    segments = [(i, ((a['x'], a['y']), (b['x'], b['y'])))
                for i, f in enumerate(ordered) for a, b in zip(f['observations'], f['observations'][1:])]
    predecessor, successor, accepted = {}, {}, []
    for pair in sorted(pairs, key=lambda p: (p['cost'], p['source'], p['target'])):
        i, j = pair['source'], pair['target']
        if debug_records is not None:
            record = debug_records[i, j]
            unambiguous = i not in ambiguous_out and j not in ambiguous_in
            record['checks']['ambiguity'] = dict(label='Unambiguous association', passed=unambiguous, reached=True)
            record['checks']['capacity'] = dict(label='Free predecessor/successor slots',
                                               passed=i not in successor and j not in predecessor,
                                               reached=unambiguous)
        if i in ambiguous_out or j in ambiguous_in or i in successor or j in predecessor:
            if debug_records is not None:
                if i in ambiguous_out or j in ambiguous_in:
                    record['reason'] = 'Ambiguity: comparable outgoing/incoming pair costs'
                    record['ambiguity_margin'] = ambiguity_margin
                    record['competing_pairs'] = [
                        dict(source_fragment_id=ordered[p['source']]['fragment_id'],
                             target_fragment_id=ordered[p['target']]['fragment_id'], cost=p['cost'])
                        for p in pairs if (i in ambiguous_out and p['source'] == i)
                        or (j in ambiguous_in and p['target'] == j)]
                else:
                    record['reason'] = 'Greedy assignment: predecessor/successor slot already occupied'
                record['status'] = 'rejected'
            continue
        a, b = ordered[i]['end_observation'], ordered[j]['start_observation']
        segment = ((a['x'], a['y']), (b['x'], b['y']))
        if any(owner not in (i, j) and _crosses(segment, other) for owner, other in segments):
            if debug_records is not None:
                record['checks']['crossing_observed'] = dict(label='No crossing with observed fragments', passed=False, reached=True)
                record['status'], record['reason'] = 'rejected', 'Crossing: inferred connection intersects an observed fragment'
            continue
        if debug_records is not None:
            record['checks']['crossing_observed'] = dict(label='No crossing with observed fragments', passed=True, reached=True)
        if any(not ({i, j} & {link['source'], link['target']}) and _crosses(segment, other)
               for link, other in accepted):
            if debug_records is not None:
                record['checks']['crossing_accepted'] = dict(label='No crossing with accepted links', passed=False, reached=True)
                record['status'], record['reason'] = 'rejected', 'Crossing: inferred connection intersects an accepted link'
            continue
        if debug_records is not None:
            record['checks']['crossing_accepted'] = dict(label='No crossing with accepted links', passed=True, reached=True)
        left, node = [i], i
        while node in predecessor:
            node = predecessor[node]
            left.append(node)
        right, node = [j], j
        while node in successor:
            node = successor[node]
            right.append(node)
        chain = left[::-1] + right
        if debug_records is None:
            valid, residual = _validate_chain(chain, ordered, chain_max_residual_px, max_direction_change_deg)
        else:
            record['proposed_fragment_ids'] = [ordered[index]['fragment_id'] for index in chain]
            valid, residual = _validate_chain(chain, ordered, chain_max_residual_px, max_direction_change_deg, diagnostic=record)
            record['checks']['chain_fit'] = dict(label='Combined chain fit / endpoint tangents',
                                               value=residual, limit=chain_max_residual_px, passed=valid, reached=True)
        if not valid:
            if debug_records is not None:
                record['status'] = 'rejected'
                record['reason'] = 'Chain-fit validation: ' + record.get('chain_fit_reason', 'Validation failed')
            continue
        if debug_records is not None:
            record['status'], record['reason'] = 'accepted', 'Accepted connection'
        predecessor[j], successor[i] = i, j
        pair['validation_max_residual_px'] = residual
        accepted.append((pair, segment))
    by_source = {pair['source']: pair for pair, _ in accepted}
    candidates = []
    for head in range(len(ordered)):
        if head in predecessor:
            continue
        chain, node = [head], head
        while node in successor:
            node = successor[node]
            chain.append(node)
        points = [p for i in chain for p in ordered[i]['observations']]
        separations = [b['y'] - a['y'] for a, b in zip(points, points[1:])]
        gaps = [{'y_from': a['y'], 'y_to': b['y'], 'gap_px': b['y'] - a['y'],
                 'missing_scan_rows': max(0, (b['y'] - a['y'] - 1) // row_step)}
                for a, b in zip(points, points[1:]) if b['y'] - a['y'] > row_step]
        links = []
        for i in chain[:-1]:
            pair = by_source[i]
            links.append(dict(source_fragment_id=ordered[i]['fragment_id'],
                              target_fragment_id=ordered[pair['target']]['fragment_id'],
                              **{k: v for k, v in pair.items() if k not in ('source', 'target')}))
        candidates.append({'candidate_id': len(candidates), 'observations': points,
                           'point_count': len(points), 'y_min': points[0]['y'], 'y_max': points[-1]['y'],
                           'gaps': gaps, 'max_gap_px': max(separations, default=0),
                           'fragment_ids': [ordered[i]['fragment_id'] for i in chain], 'links': links})
    return candidates
