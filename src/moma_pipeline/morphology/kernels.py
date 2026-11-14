"""Numeric DT morphology kernels, independent of geometry assembly."""

from numba import njit


@njit(nogil=True, cache=True)
def _point_on_segment_xy(px, py, ax, ay, bx, by, eps=1e-8):
    return (min(ax, bx) - eps <= px <= max(ax, bx) + eps and
            min(ay, by) - eps <= py <= max(ay, by) + eps)


@njit(nogil=True, cache=True)
def _segments_intersect_xy(ax, ay, bx, by, cx, cy, dx, dy, eps=1e-8):
    """Match the core scalar intersection predicate, including touching ribs."""
    if (max(ax, bx) + eps < min(cx, dx) or
            max(cx, dx) + eps < min(ax, bx) or
            max(ay, by) + eps < min(cy, dy) or
            max(cy, dy) + eps < min(ay, by)):
        return False

    o1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    o2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
    o3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
    o4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    if abs(o1) <= eps and _point_on_segment_xy(cx, cy, ax, ay, bx, by, eps):
        return True
    if abs(o2) <= eps and _point_on_segment_xy(dx, dy, ax, ay, bx, by, eps):
        return True
    if abs(o3) <= eps and _point_on_segment_xy(ax, ay, cx, cy, dx, dy, eps):
        return True
    if abs(o4) <= eps and _point_on_segment_xy(bx, by, cx, cy, dx, dy, eps):
        return True
    return False


@njit(nogil=True, cache=True)
def _row_crosses_cap(row, cap):
    return _segments_intersect_xy(
        row[0], row[1], row[2], row[3],
        cap[0], cap[1], cap[2], cap[3])


@njit(nogil=True, cache=True)
def terminal_body_indices(mesh, cap0, cap1, first, last):
    """Trim cap-crossing terminal rows in the inclusive interval [first, last].

    ``mesh`` has four columns (x0, y0, x1, y1), and each cap has four
    coordinates in the same order. Nonempty input intervals must index valid
    mesh rows. Empty intervals return unchanged with reason 1. Inputs are not
    modified; geometry assembly remains with the caller.

    Return ``(first, last, reason)`` with reason 0 for a clear retained body,
    1 for all terminal ribs redundant, or 2 for a nonterminal rib crossing a
    cap. Preserve the core order: trim the prefix, trim the suffix, then scan
    the retained interval, short-circuiting cap0 before cap1.
    """
    while first <= last and (_row_crosses_cap(mesh[first], cap0) or
                             _row_crosses_cap(mesh[first], cap1)):
        first += 1
    while last >= first and (_row_crosses_cap(mesh[last], cap0) or
                             _row_crosses_cap(mesh[last], cap1)):
        last -= 1
    if first > last:
        return first, last, 1
    for i in range(first, last + 1):
        if _row_crosses_cap(mesh[i], cap0) or _row_crosses_cap(mesh[i], cap1):
            return first, last, 2
    return first, last, 0
