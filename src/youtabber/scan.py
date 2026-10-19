"""Scan the whole video at a low frame rate for tab staffs and group the frames that hold them into segments."""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from youtabber.observe import PRINT_ABOVE_GAPS
from youtabber.staff import Staff, find_staffs, line_rows, thin_response
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
# A strip of chord diagrams is a "staff" whose lines all go missing together, between one diagram and the next, over
# stretches at least this many line gaps wide; dashed lines and digits knocked out of the lines leave shorter gaps
DIAGRAM_BREAK_GAPS = 2.0
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


@dataclass
class System:
    tab: Staff
    notation: Staff | None

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
    """True when most of a staff's lines go missing together over two or more stretches DIAGRAM_BREAK_GAPS wide."""
    a = max(0, int(st.top) - 3)
    resp = thin_response(gray[a:int(st.bottom) + 4, st.x0:st.x1], st.polarity)
    present = sum(resp[max(0, int(round(y)) - a - 1):int(round(y)) - a + 2].any(axis=0) for y in st.lines)
    missing = np.concatenate([[False], present < len(st.lines) / 2, [False]]).astype(np.int8)
    starts, ends = np.flatnonzero(np.diff(missing) == 1), np.flatnonzero(np.diff(missing) == -1)
    return int(((ends - starts) >= DIAGRAM_BREAK_GAPS * st.gap).sum()) >= 2


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


def scan(path: Path, width: int, height: int, region: tuple[int, int, int, int] | None = None
         ) -> tuple[list[Segment], dict[float, list[System]], list[tuple[float, float]]]:
    """Segments of steady tab, the systems seen in each scanned frame, and spans where the page scrolls up
    continuously (read by nothing, so they are reported)."""
    found: dict[float, list[System]] = {}
    papers: list[np.ndarray] = []
    for t, f in frames(path, SCAN_FPS):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        found[t] = [s for s in systems_in(g) if region is None or inside(s, region)]
        for s in found[t]:
            st = s.tab
            for a, b in zip(st.lines, st.lines[1:]):
                row = f[int(round((a + b) / 2)), st.x0:st.x1]
                papers.append((t, np.median(row, axis=0)))
    # Tab holds still between page turns; guitar strings and other lines in the footage wobble.
    # Keep a system only if the frame before or after shows it in the same place.
    scrolls = vertical_scrolls(found)
    times = sorted(found)
    steady: dict[float, list[System]] = {}
    for k, t in enumerate(times):
        near = [found[times[j]] for j in (k - 1, k + 1) if 0 <= j < len(times)]
        steady[t] = [s for s in found[t]
                     if any(abs(o.tab.top - s.tab.top) <= STEADY_PX and abs(o.tab.gap - s.tab.gap) <= 0.05 * s.tab.gap
                            and abs(o.tab.x0 - s.tab.x0) <= 4 * s.tab.gap for other in near for o in other)]
    found = steady
    segments: list[Segment] = []
    cur: Segment | None = None
    last_seen = -1e9
    for t in sorted(found):
        syss = found[t]
        if not syss:
            continue
        gap = max(s.tab.gap for s in syss)
        # The topmost system sets the theme, unless it is spaced much closer than the widest: a title bar or banner
        # above the tab can pass for a close staff of the other polarity
        pol = next(s for s in syss if s.tab.gap >= THEME_GAP_SHARE * gap).tab.polarity
        if (cur is not None and t - last_seen <= BRIDGE_SECONDS and pol == cur.polarity
                and abs(gap - cur.gap) <= GAP_DRIFT * cur.gap):
            cur.t1 = t
            cur.times.append(t)
            cur.gaps.append(gap)
        else:
            if cur is not None:
                segments.append(cur)
            cur = Segment(t, t, gap, pol, [t], [gap])
        last_seen = t
    if cur is not None:
        segments.append(cur)
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
