"""Find the parts a segment shows at once: bands of tab stacked with footage between them, each read on its own."""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from youtabber.observe import ABOVE_GAPS, BELOW_GAPS, PANEL_SHARE, PAPER_TOL
from youtabber.video import frames

# Staffs whose rows come within this many line gaps of each other over the segment belong to one lane
LANE_SLACK_GAPS = 1.0
# Two lanes are one page when at least this share of the rows between them is paper in the frames tested
PAPER_BETWEEN = 0.5
# Frames tested for paper between two lanes, spread over the time both show
PAPER_FRAMES = 5
# A lane is a part only when the scan sees a staff in it for at least this long; shorter ones are a title staff or
# lines in the footage caught for a moment
PART_MIN_SECONDS = 4.0
# Views of different parts that start within this long of each other start on one screen. View starts come in
# 1/6 s steps, so this allows one step between them
SAME_SCREEN_SECONDS = 0.3


@dataclass
class Part:
    id: int
    seg: int
    rows: tuple[int, int]                # source rows its staffs occupy
    kind: str                            # "page" (the segment's only part) or "band" (stacked)
    found: dict = field(default_factory=dict)   # scan time -> the part's systems at that time
    seen: float = 0.0                    # seconds the scan saw it
    stream: int = 0                      # the assembly it joins: parts of one stream are numbered and joined together
    cols: tuple[int, int] | None = None  # source columns of a part narrower than the segment; None when it spans it


def lanes(found: dict, seg) -> list[list[float]]:
    """Row ranges [top, bottom] that the segment's staffs occupy, merged where they touch."""
    spans = sorted((s.top, s.tab.bottom) for t, ss in found.items() if seg.t0 <= t <= seg.t1 for s in ss)
    out: list[list[float]] = []
    for a, b in spans:
        if out and a <= out[-1][1] + LANE_SLACK_GAPS * seg.gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def in_lane(found: dict, seg, lane) -> dict:
    """Scan time -> the segment's systems that lie inside the lane's rows."""
    a, b = lane
    return {t: [s for s in ss if a <= s.top and s.tab.bottom <= b] for t, ss in found.items() if seg.t0 <= t <= seg.t1}


def paper_rows_share(strip: np.ndarray, paper: np.ndarray) -> float:
    """Share of the strip's rows that are paper colored on at least PANEL_SHARE of their width."""
    p = paper.astype(np.int16)
    lo = tuple(int(v) for v in np.clip(p - (PAPER_TOL - 1), 0, 255))
    hi = tuple(int(v) for v in np.clip(p + (PAPER_TOL - 1), 0, 255))
    rows = (cv2.inRange(strip, lo, hi) > 0).mean(axis=1) >= PANEL_SHARE
    return float(rows.mean())


def paper_between(path: Path, seg, upper: dict, lower: dict, ua, lb) -> float:
    """Share of the rows between two lanes that are paper, over frames where both lanes show a staff."""
    both = sorted(t for t in upper if upper[t] and lower.get(t))
    if not both:
        return 1.0
    top = int(ua + BELOW_GAPS * seg.gap)
    bottom = int(lb - ABOVE_GAPS * seg.gap)
    if bottom - top < 2 * seg.gap:
        return 1.0
    picks = [both[int(k * (len(both) - 1) / max(PAPER_FRAMES - 1, 1))] for k in range(PAPER_FRAMES)]
    shares = []
    for t in sorted(set(picks)):
        f = next((f for _, f in frames(path, 1, start=t, length=0.5)), None)
        if f is None:
            continue
        shares.append(paper_rows_share(f[top:bottom, seg.x0:seg.x1], seg.paper))
    return float(np.median(shares)) if shares else 1.0


def find_parts(path: Path, seg, si: int, found: dict, scan_step: float) -> list[Part]:
    """The segment's parts, top to bottom. A segment whose staffs share one page is one part."""
    ls = lanes(found, seg)
    groups = [[ls[0]]] if ls else []
    for lane in ls[1:]:
        last = groups[-1][-1]
        share = paper_between(path, seg, in_lane(found, seg, last), in_lane(found, seg, lane), last[1], lane[0])
        if share >= PAPER_BETWEEN:
            groups[-1].append(lane)
        else:
            groups.append([lane])
    rows = [(g[0][0], g[-1][1]) for g in groups]
    parts = []
    for r in rows:
        sub = in_lane(found, seg, r)
        seen = sum(1 for ss in sub.values() if ss) * scan_step
        if seen < PART_MIN_SECONDS:
            continue
        parts.append(Part(len(parts), si, (int(r[0]), int(r[1])), "band", sub, seen))
    if len(parts) <= 1:
        return [Part(0, si, (seg.y0, seg.y1), "page", {}, seg.t1 - seg.t0)]
    return parts


def stream_of(part: Part, seg_parts: list[Part]) -> int:
    """The stream a part joins: 0 for the segment's longest-seen part (the first one on a tie), else its own.
    Part ids are unique in the video, so streams of different parts never collide."""
    main = max(seg_parts, key=lambda p: p.seen)
    return 0 if part is main else 1 + part.id


def merge_order(keys: list[tuple[float, float, int]]) -> list[int]:
    """Print order for measures keyed (first seen, part's place on screen, order within its part): by when they were
    first seen, and those first seen on one screen top before bottom."""
    order = sorted(range(len(keys)), key=lambda i: keys[i][0])
    out, k = [], 0
    while k < len(order):
        t = keys[order[k]][0]
        j = k
        while j < len(order) and keys[order[j]][0] - t <= SAME_SCREEN_SECONDS:
            j += 1
        out += sorted(order[k:j], key=lambda i: (keys[i][1], keys[i][0], keys[i][2]))
        k = j
    return out
