"""Scan the whole video at a low frame rate for tab staffs and group the frames that hold them into segments."""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from youtabber.staff import Staff, find_staffs
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
# A notation staff sits at most this many tab line gaps above its tab staff
NOTATION_GAPS = 9
# A tab staff found again half a second later sits within this many pixels of where it was
STEADY_PX = 2.5


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


def systems_in(gray: np.ndarray) -> list[System]:
    """Tab staffs in a frame, each with the notation staff drawn above it if there is one."""
    h, w = gray.shape
    staffs = [s for s in find_staffs(gray) if s.x1 - s.x0 >= w / 5 and s.gap >= 4]
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
    staffs = [s for s in keep if paper_between_lines(gray, s)]
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
        # below: then the lower group is strings or frets under a tab staff, not a tab staff
        if (above is not None and above.polarity == s.polarity and 0 < s.top - above.bottom < NOTATION_GAPS * s.gap
                and (len(above.lines) == 5 and above.gap <= 1.1 * s.gap
                     or above.gap < 0.85 * s.gap and len(above.lines) <= len(s.lines))):
            out.append(System(s, above))
            used.update({i, i - 1})
        else:
            out.append(System(s, None))
            used.add(i)
    out.reverse()
    return out


def scan(path: Path, width: int, height: int, region: tuple[int, int, int, int] | None = None
         ) -> tuple[list[Segment], dict[float, list[System]]]:
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
        pol = syss[0].tab.polarity
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
        for t in seg.times:
            for s in found[t]:
                tops.append(s.top - s.tab.gap * 5)
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
    return segments, found
