"""Scan the whole video at a low frame rate for tab staffs and group the frames that hold them into segments."""
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from youtabber.observe import PRINT_ABOVE_GAPS
from youtabber.staff import LINE_CONTRAST, Staff, find_staffs, line_rows, thin_response
from youtabber.video import frames

SCAN_FPS = 2
# Between staff lines, at least this share of a row must be plain paper (digits cover the rest)
PAPER_SHARE = 0.6
# Paper pixels are within this many gray levels of the row's median
PAPER_SPREAD = 22
# Staffs in one segment keep their line spacing within this share
GAP_DRIFT = 0.35
# Frames without tab shorter than this don't end a segment (page turns, fades)
BRIDGE_SECONDS = 1.6
MIN_SEGMENT_SECONDS = 2.0
# A stretch too short to be a segment carries on over frames without tab only into staffs spaced within this share of
# its own; a passing card is drawn at another spacing than the tab after it
PASSING_GAP_SHARE = 0.05
# A strip of chord diagrams is a "staff" whose lines all go missing together, between one diagram and the next, over
# stretches at least this many line gaps wide; dashed lines and digits knocked out of the lines leave shorter gaps
DIAGRAM_BREAK_GAPS = 2.0
# ...or whose lines run unbroken (shorter breaks closed up) in two or more pieces, none longer than DIAGRAM_PIECE_GAPS
# line gaps and the median at least DIAGRAM_PIECE_MIN: a row of diagrams set close together. A break shorter than
# DIAGRAM_SHORT_BREAK gaps is a digit or chord on the lines, and pieces whose median is under DIAGRAM_PIECE_MIN gaps
# are the dashes of a dashed line
DIAGRAM_PIECE_MIN, DIAGRAM_PIECE_GAPS, DIAGRAM_SHORT_BREAK = 1.5, 5.0, 0.7
# A diagram's lines end on its outer strings, which cross the whole grid; a tab staff's pieces end at digits. The
# pieces count as diagrams only when STROKE_ENDS of their inner ends have, within max(STROKE_PX, STROKE_GAPS line
# gaps), a column that is ink on STROKE_COVER of the rows from the top line to the bottom one
STROKE_ENDS, STROKE_COVER, STROKE_PX, STROKE_GAPS = 0.75, 0.85, 3, 0.1
# A notation staff sits at most this many tab line gaps above its tab staff...
NOTATION_GAPS = 9
# ...or up to this many when nothing but small marks (H, P, fingerings) lies between them
NOTATION_FAR_GAPS = 12
# Between a far notation staff and its tab, no line or row of ink may run across more than this share of the staffs
OPEN_ROW_SHARE = 0.5
# A system spaced closer than this share of the frame's widest is a title bar or text lines, which don't set the theme
THEME_GAP_SHARE = 0.4
# A tab staff found again half a second later sits within this many pixels of where it was
STEADY_PX = 2.5
# A staff that moves up by more than STEADY_PX and at most this many line gaps between scanned frames is scrolling
SCROLL_GAPS = 4.0
# A page counts as scrolling when it moves for at least this long
SCROLL_SECONDS = 10.0
# A staff shorter than a fifth of the frame still counts when it starts within this many line gaps of where a
# full-length staff with the same spacing starts: the last system of a page, holding only its final measures
SHORT_ALIGN_GAPS = 3.0
# ...and is at least this many line gaps long, room for two measures
SHORT_MIN_GAPS = 10.0
# Each scanned frame keeps how much it changed from the frame before, averaged over cells this many pixels square,
# so the panel bound can tell still paper from footage
MOTION_CELL = 8
# A split screen shows two panels side by side with a divider between them: a column at most DIVIDER_GAPS line gaps
# wide, within DIVIDER_MIDDLE of the staff's width, unlike paper on DIVIDER_COVER of the rows from 2 gaps above the
# system to 2 below it, that also stands out from the columns beside it on DIVIDER_FOOTAGE of the frame's rows above
# or below those. A barline stops at its staff, so it fails that last test
DIVIDER_GAPS, DIVIDER_COVER, DIVIDER_FOOTAGE = 3.0, 0.95, 0.5
DIVIDER_MIDDLE = (0.2, 0.8)
# A row stands out when the divider differs by more than DIVIDER_CONTRAST gray levels from every pixel beside it, in
# the columns DIVIDER_SIDE_PX (nearest, farthest) pixels from its edges. A divider is taken only with at least
# DIVIDER_SIDE_PX[1] + 1 columns of the frame left of it and DIVIDER_SIDE_PX[1] right of it, so the side columns lie
# inside the frame
DIVIDER_CONTRAST = 25
DIVIDER_SIDE_PX = (4, 6)


@dataclass
class System:
    tab: Staff
    notation: Staff | None
    panel: bool = field(default=False, repr=False)   # cut from a wider system at a split screen's divider
    cut_from: "System | None" = field(default=None, repr=False)   # the wider system a panel was cut from

    @property
    def top(self) -> float:
        return (self.notation or self.tab).top


@dataclass
class Segment:
    t0: float
    t1: float
    gap: float
    polarity: str
    times: list[float] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)
    paper: np.ndarray | None = None
    y0: int = 0
    y1: int = 0
    x0: int = 0
    x1: int = 0
    # A stretch too short to keep just before this segment, left out because its staff has as many lines but is
    # spaced over PASSING_GAP_SHARE apart from this one's: a brief page that may hold tab
    passed: tuple[float, float] | None = None


def paper_between_lines(gray: np.ndarray, st: Staff) -> bool:
    """True when the rows midway between the staff lines are mostly one flat color, as paper is."""
    good = 0
    for a, b in zip(st.lines, st.lines[1:]):
        row = gray[int(round((a + b) / 2)), st.x0:st.x1].astype(int)
        if row.size == 0:
            return False
        med = np.median(row)
        good += np.mean(np.abs(row - med) <= PAPER_SPREAD) >= PAPER_SHARE
    return good >= len(st.lines) - 1 - 1  # one row may be crowded with digits


def inside(s: System, region: tuple[int, int, int, int]) -> bool:
    """True when the tab staff's middle lies inside region (x, y, w, h)."""
    rx, ry, rw, rh = region
    cx, cy = (s.tab.x0 + s.tab.x1) / 2, (s.tab.top + s.tab.bottom) / 2
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


def short_staffs(gray: np.ndarray, long: list[Staff]) -> list[Staff]:
    """Staffs too short for the usual length rule that line up under or over one of the long staffs: same polarity
    and spacing, the same left end, and clear of every long staff."""
    w = gray.shape[1]
    # The long staffs are painted over first, so their strong lines can't make a short staff's outer lines look weak
    rest = gray.copy()
    for o in long:
        rest[max(0, int(o.top - o.gap)):int(o.bottom + o.gap) + 1] = int(np.median(gray))
    out = []
    for s in find_staffs(rest, max(w // 40, 20)):
        if s.x1 - s.x0 >= w / 5 or s.x1 - s.x0 < SHORT_MIN_GAPS * s.gap:
            continue
        if any(o.top <= s.bottom + o.gap and s.top <= o.bottom + o.gap for o in long):
            continue
        if any(o.polarity == s.polarity and abs(o.gap - s.gap) <= 0.1 * o.gap
               and abs(o.x0 - s.x0) <= SHORT_ALIGN_GAPS * o.gap for o in long):
            out.append(s)
    return out


def open_between(gray: np.ndarray, upper: Staff, lower: Staff) -> bool:
    """True when nothing between two staffs runs across them: no line (another staff's, a rule) and no solid band
    (a footage edge). Words, slurs and P.M. or H marks are allowed. Notes and stems hang up to two gaps below the
    upper staff, and marks sit a gap above the lower one."""
    a, b = int(upper.bottom + 2 * upper.gap), int(lower.top - lower.gap)
    x0, x1 = max(upper.x0, lower.x0), min(upper.x1, lower.x1)
    if b <= a or x1 <= x0:
        return False
    band = gray[a:b, x0:x1]
    if line_rows(band, upper.polarity, int(OPEN_ROW_SHARE * (x1 - x0))):
        return False
    ink = np.abs(band.astype(np.int16) - np.median(band)) > 60
    return bool(ink.mean(axis=1).max() < OPEN_ROW_SHARE)


def broken_up(gray: np.ndarray, st: Staff) -> bool:
    """True when most of a staff's lines go missing together over two or more stretches DIAGRAM_BREAK_GAPS wide, or
    when they run on in short pieces, one per diagram, that end on the diagrams' outer strings: a strip of chord
    diagrams."""
    a = max(0, int(st.top) - 3)
    resp = thin_response(gray[a:int(st.bottom) + 4, st.x0:st.x1], st.polarity)
    present = sum(resp[max(0, int(round(y)) - a - 1):int(round(y)) - a + 2].any(axis=0) for y in st.lines)
    missing = np.concatenate([[False], present < len(st.lines) / 2, [False]]).astype(np.int8)
    starts, ends = np.flatnonzero(np.diff(missing) == 1), np.flatnonzero(np.diff(missing) == -1)
    if int(((ends - starts) >= DIAGRAM_BREAK_GAPS * st.gap).sum()) >= 2:
        return True
    breaks = [(s, e) for s, e in zip(starts, ends) if e - s >= DIAGRAM_SHORT_BREAK * st.gap]
    edges = [0] + [x for b in breaks for x in b] + [len(present)]
    spans = [(p, q) for p, q in zip(edges[::2], edges[1::2]) if q > p]
    pieces = [q - p for p, q in spans]
    if not (len(pieces) >= 2 and max(pieces) <= DIAGRAM_PIECE_GAPS * st.gap
            and float(np.median(pieces)) >= DIAGRAM_PIECE_MIN * st.gap):
        return False
    inner = [q for _, q in spans[:-1]] + [p for p, _ in spans[1:]]
    return sum(stroke_near(gray, st, st.x0 + x) for x in inner) >= STROKE_ENDS * len(inner)


def stroke_near(gray: np.ndarray, st: Staff, x: int) -> bool:
    """True when a column near x is ink on nearly every row from the staff's top line to its bottom line: a
    vertical line crossing the whole staff, as a chord diagram's outer string does."""
    paper = np.median([np.median(gray[int(round((a + b) / 2)), st.x0:st.x1]) for a, b in zip(st.lines, st.lines[1:])])
    slack = max(STROKE_PX, int(STROKE_GAPS * st.gap))
    cols = gray[int(round(st.top)):int(round(st.bottom)) + 1, max(0, x - slack):x + slack + 1].astype(np.int16)
    ink = cols < paper - LINE_CONTRAST if st.polarity == "dark" else cols > paper + LINE_CONTRAST
    return bool(ink.size and ink.mean(axis=0).max() >= STROKE_COVER)


def line_columns(gray: np.ndarray, st: Staff) -> np.ndarray:
    """Columns where at least half of the staff's lines are drawn in this frame."""
    a = max(0, int(st.top) - 4)
    resp = thin_response(gray[a:int(st.bottom) + 5], st.polarity)
    hits = sum(resp[max(0, int(round(y)) - a - 1):int(round(y)) - a + 2].any(axis=0) for y in st.lines)
    return hits >= 0.5 * len(st.lines)


def divider_at(gray: np.ndarray, st: Staff, top: float, rows: tuple[float, float] | None = None) -> int | None:
    """The x of a split screen's divider through a staff whose system starts at row `top`, or None. The divider
    must run on through the frame above or below the system, and also above or below `rows` (top, bottom) when
    given: the rows of the whole stack of staffs over the staff's columns."""
    h, w = gray.shape
    gap = st.gap
    a, b = int(max(0, top - 2 * gap)), int(min(h, st.bottom + 2 * gap))
    above, below = (a, b) if rows is None else (int(max(0, rows[0] - 2 * gap)), int(min(h, rows[1] + 2 * gap)))
    band = gray[a:b, st.x0:st.x1].astype(np.int16)
    if band.size == 0:
        return None
    far = np.abs(band - np.median(band)) > 60
    cover = far.mean(axis=0)
    n = len(cover)
    cols = np.flatnonzero(cover >= DIVIDER_COVER)
    cols = cols[(cols >= DIVIDER_MIDDLE[0] * n) & (cols < DIVIDER_MIDDLE[1] * n)]
    if not len(cols):
        return None
    for r in np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1):
        if r[-1] - r[0] + 1 > DIVIDER_GAPS * gap:
            continue
        x0, x1 = int(r[0]) + st.x0, int(r[-1]) + st.x0
        if x0 < DIVIDER_SIDE_PX[1] + 1 or x1 + DIVIDER_SIDE_PX[1] + 1 > w:
            continue

        def runs_on(lo: int, hi: int) -> bool:
            for out in (gray[:lo], gray[hi:]):
                if out.shape[0] < b - a:
                    continue
                line = out[:, x0:x1 + 1].astype(np.float32).mean(axis=1)
                d0, d1 = DIVIDER_SIDE_PX
                side = np.concatenate([out[:, x0 - d1:x0 - d0 + 1], out[:, x1 + d0:x1 + d1 + 1]],
                                      axis=1).astype(np.float32)
                if (np.abs(side - line[:, None]).min(axis=1) > DIVIDER_CONTRAST).mean() >= DIVIDER_FOOTAGE:
                    return True
            return False

        if runs_on(a, b) and runs_on(above, below):
            return (x0 + x1) // 2
    return None


def split_panels(gray: np.ndarray, systems: list["System"]) -> list["System"]:
    """Each system that runs across a split screen's divider, cut into its left and right panels."""
    out = []
    for s in systems:
        # Staffs stacked over the same columns can share barlines drawn through all of them, so the divider must
        # run on past the whole stack
        stack = [o for o in systems if o.tab.x0 < s.tab.x1 and s.tab.x0 < o.tab.x1]
        rows = (min(o.top for o in stack), max(o.tab.bottom for o in stack))
        # The strings of a strip of chord diagrams cross its lines like dividers
        x = None if broken_up(gray, s.tab) else divider_at(gray, s.tab, s.top, rows)
        if x is None:
            out.append(s)
            continue
        gap = s.tab.gap
        for lo, hi in ((s.tab.x0, int(x - gap / 2)), (int(x + gap / 2), s.tab.x1)):
            tab = Staff(s.tab.lines, max(lo, s.tab.x0), min(hi, s.tab.x1), s.tab.gap, s.tab.polarity, s.tab.dashed)
            n = s.notation
            if n is not None:
                n = Staff(n.lines, max(lo, n.x0), min(hi, n.x1), n.gap, n.polarity, n.dashed)
            out.append(System(tab, n, True, s))
    return out


def systems_in(gray: np.ndarray, whole_lines: bool = False) -> list[System]:
    """Tab staffs in a frame, each with the notation staff drawn above it if there is one.

    With `whole_lines`, staffs whose lines break up across their width (a strip of chord diagrams) are left out
    before pairing. The scan keeps them: the broken staff that paper shows between light lines on a dark theme
    marks frames the scan must not join to its neighbors."""
    h, w = gray.shape
    staffs = [s for s in find_staffs(gray) if s.x1 - s.x0 >= w / 5 and s.gap >= 4]
    if staffs:
        staffs = sorted(staffs + short_staffs(gray, staffs), key=lambda s: s.top)
    # A light "staff" between dark lines is the paper showing between them, and the other way round;
    # a staff inside a wider-spaced staff is beams or digit rows
    keep = []
    for s in staffs:
        clash = [o for o in staffs if o is not s and o.top <= s.bottom + 1 and s.top <= o.bottom + 1]
        if any(o.gap > s.gap * 1.2 for o in clash):
            continue
        if any(o.polarity != s.polarity and o.polarity == "dark" and abs(o.gap - s.gap) < 0.3 * s.gap
               for o in clash):
            continue
        keep.append(s)
    staffs = [s for s in keep if paper_between_lines(gray, s) and not (whole_lines and broken_up(gray, s))]
    out: list[System] = []
    used = set()
    # From the bottom up: each tab staff takes the staff just above it as its notation staff, so a stray group of
    # close lines above a notation staff can't pull the notation staff away from its tab
    for i in range(len(staffs) - 1, -1, -1):
        if i in used:
            continue
        s = staffs[i]
        above = staffs[i - 1] if i > 0 and i - 1 not in used else None
        # Closer-spaced lines above are notation (a ledger line can add a sixth), unless they outnumber the lines
        # below: then the lower group is strings or frets under a tab staff, not a tab staff. Two 5-line staffs
        # spaced alike are two systems of 5-string tab, and notation set far above its tab is spaced closer than it.
        twin = (above is not None and len(above.lines) == len(s.lines) == 5
                and abs(above.gap - s.gap) <= 0.1 * s.gap)
        if (above is not None and above.polarity == s.polarity and 0 < s.top - above.bottom
                and (s.top - above.bottom < NOTATION_GAPS * s.gap
                     or s.top - above.bottom < NOTATION_FAR_GAPS * s.gap and above.gap < 0.85 * s.gap
                     and open_between(gray, above, s))
                and (len(above.lines) == 5 and above.gap <= 1.1 * s.gap and not twin
                     or above.gap < 0.85 * s.gap and len(above.lines) <= len(s.lines))):
            out.append(System(s, above))
            used.update({i, i - 1})
        else:
            out.append(System(s, None))
            used.add(i)
    out.reverse()
    return out


def vertical_scrolls(found: dict[float, list[System]]) -> list[tuple[float, float]]:
    """Time spans where a page of staffs moves steadily up the screen, frame after frame, instead of jumping."""
    times = sorted(found)
    moving = []
    for a, b in zip(times, times[1:]):
        for s in found[a]:
            g = s.tab.gap
            if any(STEADY_PX < s.tab.top - o.tab.top <= SCROLL_GAPS * g and abs(o.tab.gap - g) <= 0.05 * g
                   and abs(o.tab.x0 - s.tab.x0) <= 4 * g for o in found[b]):
                moving.append(a)
                break
    spans: list[list[float]] = []
    for t in moving:
        if spans and t - spans[-1][1] <= BRIDGE_SECONDS:
            spans[-1][1] = t + 1 / SCAN_FPS
        else:
            spans.append([t, t + 1 / SCAN_FPS])
    return [(a, b) for a, b in spans if b - a >= SCROLL_SECONDS]


def lines_of(syss: list[System]) -> int:
    """The number of lines of the widest-spaced tab staff among a frame's systems."""
    return len(max(syss, key=lambda s: s.tab.gap).tab.lines)


def scan(path: Path, width: int, height: int, region: tuple[int, int, int, int] | None = None,
         progress: Callable[[float], None] | None = None
         ) -> tuple[list[Segment], dict[float, list[System]], list[tuple[float, float]]]:
    """Segments of steady tab, the systems seen in each scanned frame, and spans where the page scrolls up
    continuously (read by nothing, so they are reported). `progress(t)` is called with each scanned frame's time."""
    found: dict[float, list[System]] = {}
    papers: list[tuple[float, np.ndarray]] = []
    before = None
    for t, f in frames(path, SCAN_FPS):
        if progress is not None:
            progress(t)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        whole = systems_in(g)
        found[t] = [s for s in split_panels(g, whole) if region is None or inside(s, region)]
        # How much each part of the frame changed since the frame before, and where each staff's lines are drawn
        h, w = g.shape
        motion = None if before is None else cv2.resize(cv2.absdiff(g, before), (w // MOTION_CELL, h // MOTION_CELL),
                                                        interpolation=cv2.INTER_AREA)
        before = g
        for s in found[t]:
            s.motion = motion
            s.lines_at = line_columns(g, s.tab)
            # A panel's lines run on into the panel beside it; only its own columns count for its bound
            if s.panel:
                s.lines_at[:s.tab.x0] = False
                s.lines_at[s.tab.x1:] = False
        # Paper is sampled between the lines of every staff in the frame, including staffs a segment later drops
        # as footage (guitar strings), so a segment's paper color can come partly from the footage. The staffs are
        # sampled whole, before any divider cut, so a staff cut into panels counts as often as one that isn't. A
        # staff whose middle is outside the region is sampled when one of its panels is kept
        kept = {id(s.cut_from) for s in found[t] if s.cut_from is not None}
        for s in (s for s in whole if region is None or inside(s, region) or id(s) in kept):
            st = s.tab
            for a, b in zip(st.lines, st.lines[1:]):
                row = f[int(round((a + b) / 2)), st.x0:st.x1]
                papers.append((t, np.median(row, axis=0)))
    scrolls = vertical_scrolls(found)
    # Tab holds still between page turns; guitar strings and other lines in the footage wobble.
    # Keep a system only if the frame before or after shows it in the same place.
    times = sorted(found)
    steady: dict[float, list[System]] = {}
    for k, t in enumerate(times):
        near = [found[times[j]] for j in (k - 1, k + 1) if 0 <= j < len(times)]
        steady[t] = [s for s in found[t]
                     if any(abs(o.tab.top - s.tab.top) <= STEADY_PX and abs(o.tab.gap - s.tab.gap) <= 0.05 * s.tab.gap
                            and abs(o.tab.x0 - s.tab.x0) <= 4 * s.tab.gap for other in near for o in other)]
    # Split the frames into segments; if a segment too short to keep hid staffs of the other polarity, split again
    # with those frames showing every staff
    no_bridge: set[float] = set()
    while True:
        found = dict(steady)
        segments: list[Segment] = []
        cur: Segment | None = None
        last_seen = -1e9
        dropped: set[float] = set()   # times of frames whose staffs of the other polarity were left out
        for t in sorted(found):
            syss = found[t]
            if not syss:
                continue
            gap = max(s.tab.gap for s in syss)
            # The topmost system sets the theme, unless it is spaced much closer than the widest: a title bar or
            # banner above the tab can pass for a close staff of the other polarity
            pol = next(s for s in syss if s.tab.gap >= THEME_GAP_SHARE * gap).tab.polarity
            # Lines in the footage (guitar strings) come and go above the tab as a staff of the other polarity;
            # while the frame still shows a staff of the segment's polarity and spacing, the segment goes on
            # without them. Whichever polarity starts the segment keeps it, so footage that starts one hides the
            # real tab after it
            if cur is not None and pol != cur.polarity and t - last_seen <= BRIDGE_SECONDS and t not in no_bridge:
                same = [s for s in syss if s.tab.polarity == cur.polarity
                        and abs(s.tab.gap - cur.gap) <= GAP_DRIFT * cur.gap]
                if same:
                    pol, gap = cur.polarity, max(s.tab.gap for s in same)
                    dropped.add(t)
                    found[t] = [s for s in syss if s.tab.polarity == pol]
            passing = (cur is not None and t - last_seen > 1 / SCAN_FPS + 1e-6
                       and cur.t1 - cur.t0 + 1 / SCAN_FPS < MIN_SEGMENT_SECONDS
                       and abs(gap - np.median(cur.gaps)) > PASSING_GAP_SHARE * np.median(cur.gaps))
            carries = (cur is not None and t - last_seen <= BRIDGE_SECONDS and pol == cur.polarity
                       and abs(gap - cur.gap) <= GAP_DRIFT * cur.gap)
            if carries and not passing:
                cur.t1 = t
                cur.times.append(t)
                cur.gaps.append(gap)
            else:
                last = cur
                if cur is not None:
                    segments.append(cur)
                cur = Segment(t, t, gap, pol, [t], [gap])
                if carries and lines_of(found[t]) == lines_of(found[last.times[-1]]):
                    cur.passed = (last.t0, last.t1 + 1 / SCAN_FPS)
            last_seen = t
        if cur is not None:
            segments.append(cur)
        short = [s for s in segments if s.t1 - s.t0 + 1 / SCAN_FPS < MIN_SEGMENT_SECONDS]
        hid = {t for s in short for t in s.times} & dropped
        if not hid:
            break
        no_bridge |= hid
    segments = [s for s in segments if s.t1 - s.t0 + 1 / SCAN_FPS >= MIN_SEGMENT_SECONDS]
    for seg in segments:
        cols = [c for t, c in papers if seg.t0 <= t <= seg.t1]
        seg.paper = np.median(np.array(cols), axis=0) if cols else np.array([255.0, 255.0, 255.0])
    for seg in segments:
        seg.gap = float(np.median(seg.gaps))
        tops, bots, lefts, rights = [], [], [], []
        # Dark ink on paper prints marks as high as chord names and section labels sit, so the crop reaches them;
        # a dark theme's print area stops at its panel
        above = PRINT_ABOVE_GAPS if seg.polarity == "dark" else 5
        for t in seg.times:
            for s in found[t]:
                tops.append(s.top - s.tab.gap * above)
                bots.append(s.tab.bottom + s.tab.gap * 4.5)
                lefts.append(s.tab.x0)
                rights.append(s.tab.x1)
        seg.y0 = int(max(0, min(tops)))
        seg.y1 = int(min(height, max(bots)))
        seg.x0 = int(max(0, min(lefts) - seg.gap * 3))
        seg.x1 = int(min(width, max(rights) + seg.gap * 3))
        if region is not None:
            # The region given on the command line bounds what is read
            rx, ry, rw, rh = region
            seg.x0, seg.y0 = max(seg.x0, rx), max(seg.y0, ry)
            seg.x1, seg.y1 = min(seg.x1, rx + rw), min(seg.y1, ry + rh)
        seg.x0 -= seg.x0 % 2
        seg.y0 -= seg.y0 % 2
        seg.x1 -= (seg.x1 - seg.x0) % 2
        seg.y1 -= (seg.y1 - seg.y0) % 2
        seg.t1 += 1 / SCAN_FPS
    return segments, found, scrolls
