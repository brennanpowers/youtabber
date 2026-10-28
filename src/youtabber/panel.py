"""Bound a part's panel to the paper its tab is drawn on, where a clear bound is found, so footage beside, above or
below a tab band is kept out of the frame comparison, the playhead search and the measures."""
from collections import Counter

import numpy as np

# A column or row whose picture changes by more than this many gray levels between scanned frames half a second
# apart, averaged over the staffs' rows (or the bound's columns), in most scanned frames, is footage
MOVING_LEVELS = 4.0
# A column is staff when half or more of its tab lines are drawn there in at least this share of the scanned
# frames; paper beside the staff and footage show lines in almost none
LINE_SHARE = 0.1
# Moving columns without lines narrower than this many line gaps don't separate staff columns (noise, a cursor)
NOISE_GAPS = 1.0
# The bound reaches this many line gaps past the staff lines' ends, for the first measure's number and bracket on
# the left and a closing barline on the right, but never into moving columns
LEFT_GAPS, RIGHT_GAPS = 2.5, 1.5
# Staff lines on at least this many line gaps of moving columns are a see-through band over footage: no bound is used
SEE_THROUGH_GAPS = 4.0
# Staffs seen in less than this share of a layout's scanned frames don't set the rows the bound keeps
RARE_SHARE = 0.1
# Staffs whose line gap is within this share of the segment's set the frame's layout
LAYOUT_GAP_SHARE = 0.35
# A layout shown in fewer scanned frames than this gets no box of its own
MIN_LAYOUT_FRAMES = 8
# ...and takes the box of the nearest frame with one, if that frame is at most this many seconds away
BORROW_SECONDS = 2.0
# The staff must span at least this many line gaps for a bound to be used
MIN_STAFF_GAPS = 8.0


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each run of True, end exclusive."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def layout_of(ss: list, gap: float) -> tuple[int, ...]:
    """Where a frame's tab staffs sit, in half line gaps: frames of one layout show their staffs in the same rows."""
    return tuple(sorted(round(2 * s.tab.top / gap) for s in ss if abs(s.tab.gap - gap) <= LAYOUT_GAP_SHARE * gap))


def line_count(tab) -> int:
    """How many lines a tab staff holds from its top line to its bottom one, at its line gap."""
    return round((tab.bottom - tab.top) / tab.gap) + 1


def panel_bounds(found: dict, seg, cols: tuple[int, int] | None = None):
    """A function of time giving the box (x0, y0, x1, y1, crop coordinates) that the part's panel stays inside at
    that time, or None where there is no clear bound; None instead of a function when there is none at any time.
    Each layout, the frames whose staffs sit in the same rows, gets its own box, so a title page with its staff
    lower or wider than the pages after it keeps its own columns and rows. `found` holds the part's systems at each
    scan time, and `cols` gives the source columns of the crop when it is narrower than the segment. The function's
    `layouts` lists each layout's first and last time, frame count and box."""
    times = [t for t in sorted(found) if seg.t0 <= t <= seg.t1 and found[t]]
    if not times or any(getattr(s, "lines_at", None) is None for t in times for s in found[t]):
        return None
    groups: dict[tuple, list[float]] = {}
    for t in times:
        groups.setdefault(layout_of(found[t], seg.gap), []).append(t)
    counts = Counter(line_count(s.tab) for t in times for s in found[t]
                     if abs(s.tab.gap - seg.gap) <= LAYOUT_GAP_SHARE * seg.gap)
    lines = counts.most_common(1)[0][0] if counts else 0
    boxes = {k: layout_bound([found[t] for t in ts], seg, cols, lines) for k, ts in groups.items()
             if k and len(ts) >= MIN_LAYOUT_FRAMES}
    if not any(boxes.values()):
        return None
    key_at = {t: layout_of(found[t], seg.gap) for t in times}
    boxed = [t for t in times if boxes.get(key_at[t])]

    def bound_at(t: float):
        near = min(times, key=lambda k: abs(k - t))
        box = boxes.get(key_at[near])
        if box is None:
            # A layout shown too briefly for its own box (a page fading in or out, the last page) takes the box of
            # the nearest frame that has one
            other = min(boxed, key=lambda k: abs(k - t))
            if abs(other - t) <= BORROW_SECONDS:
                box = boxes[key_at[other]]
        return box
    bound_at.layouts = [(min(ts), max(ts), len(ts), boxes.get(k)) for k, ts in groups.items()]
    return bound_at


def layout_bound(systems: list[list], seg, cols: tuple[int, int] | None = None,
                 lines: int = 0) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) in crop coordinates for one layout's frames, each a list of systems, or None when they give
    no clear bound. Columns run from the first to the last where the staff lines are drawn, inside the still columns
    around them; rows are the still rows around the staffs. `cols` is as in `panel_bounds`; `lines` is how many
    lines the segment's staffs usually show."""
    w = len(systems[0][0].lines_at)
    share = np.mean([np.any([s.lines_at for s in ss], axis=0) for ss in systems], axis=0)
    moved = [(ss[0].motion, ss) for ss in systems if getattr(ss[0], "motion", None) is not None]
    if not moved:
        return None
    th, tw = moved[0][0].shape
    cell = w / tw
    # Each frame's change over its staffs' rows, per column, then the median over the frames
    profiles = []
    for m, ss in moved:
        rows = np.zeros(th, bool)
        for s in ss:
            rows[int(s.tab.top / cell):int(s.tab.bottom / cell) + 1] = True
        profiles.append(m[rows].mean(axis=0))
    motion = np.median(profiles, axis=0)[np.minimum((np.arange(w) / cell).astype(int), tw - 1)]
    staff = share >= LINE_SHARE
    moving = motion > MOVING_LEVELS
    gap = seg.gap
    if (staff & moving).sum() >= SEE_THROUGH_GAPS * gap:
        return None
    # Columns that move and show no lines are footage. Staff columns separated by such a stretch belong to
    # different pictures; the staff is the stretch holding the most staff columns. Columns with lines
    # stay staff even where a cursor or highlight moves over them
    footage = moving & ~staff
    cuts = [(a, b) for a, b in runs(footage) if b - a >= NOISE_GAPS * gap]
    edges = [0] + [x for r in cuts for x in r] + [w]
    pieces = list(zip(edges[::2], edges[1::2]))
    a, b = max(pieces, key=lambda r: int(staff[r[0]:r[1]].sum()))
    on = np.flatnonzero(staff[a:b])
    if len(on) == 0:
        return None
    c0, c1 = a + int(on[0]), a + int(on[-1]) + 1
    if c1 - c0 < MIN_STAFF_GAPS * gap:
        return None
    # The reach past the lines' ends stops at the first moving column
    left = np.flatnonzero(moving[a:c0])
    right = np.flatnonzero(moving[c1:b])
    a = a + int(left[-1]) + 1 if len(left) else a
    b = c1 + int(right[0]) if len(right) else b
    x0, x1 = max(a, int(c0 - LEFT_GAPS * gap)), min(b, int(c1 + RIGHT_GAPS * gap))
    # Rows: each frame's change over the bound's columns, the median over the frames, and the still rows around
    # every staff the layout showed
    k0, k1 = int(x0 / cell), max(int(x0 / cell) + 1, int(x1 / cell))
    row_motion = np.median([m[:, k0:k1].mean(axis=1) for m, _ in moved], axis=0) > MOVING_LEVELS
    # A staff the scan saw in only a few frames (lines in the footage) doesn't hold the rows open
    top = float(np.percentile([min(s.top for s in ss) for ss in systems], 100 * RARE_SHARE))
    bottom = float(np.percentile([max(s.tab.bottom for s in ss) for ss in systems], 100 * (1 - RARE_SHARE)))
    r0, r1 = int(top / cell), min(th, int(bottom / cell) + 1)
    # Dense digits on an edge string can hide its line from the scan, and digits changing there would look like
    # footage: a staff showing fewer lines than usual keeps open the rows its missing lines could take, on both sides
    lost = float(np.percentile([max(lines - line_count(s.tab), 0) for ss in systems for s in ss
                                if abs(s.tab.gap - gap) <= LAYOUT_GAP_SHARE * gap] or [0], 100 * (1 - RARE_SHARE)))
    o0, o1 = max(0, int((top - lost * gap) / cell)), min(th, int((bottom + lost * gap) / cell) + 1)
    y0, y1 = seg.y0, seg.y1
    if not row_motion[r0:r1].any():
        up = np.flatnonzero(row_motion[:o0])
        down = np.flatnonzero(row_motion[o1:])
        if len(up):
            y0 = max(y0, int((up[-1] + 1) * cell))
        if len(down):
            y1 = min(y1, int((o1 + down[0]) * cell))
    p0, p1 = (seg.x0, seg.x1) if cols is None else cols
    # A bound reaching less than a staff's width into the panel's columns would print an empty panel
    if min(p1, x1) - max(p0, x0) < MIN_STAFF_GAPS * gap:
        return None
    return (max(0, x0 - p0), max(0, y0 - seg.y0), min(p1, x1) - p0, max(0, y1 - seg.y0))
