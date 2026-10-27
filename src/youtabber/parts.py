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
    continues: "Part | None" = None      # the part of the segment whose stream this one joins
    carries: bool = False                # a split screen's panel that goes on with the stream of the segments around it
    cut_from: "Part | None" = None       # the part a split screen's panels and full-width frames were cut from


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


def side_by_side(sub: dict) -> list[tuple[float, float]]:
    """(time, divider x) for each scan time where two systems share rows but not columns: the panels of a split
    screen, which the scan cuts at the divider."""
    out = []
    for t, ss in sub.items():
        for a in ss:
            b = next((b for b in ss if b is not a and b.tab.x0 > a.tab.x1
                      and a.top <= b.tab.bottom and b.top <= a.tab.bottom), None)
            if b is not None:
                out.append((t, (a.tab.x1 + b.tab.x0) / 2))
                break
    return out


def panels(part: Part, seg, scan_step: float) -> list[Part]:
    """The part cut where a split screen shows two panels side by side for at least PART_MIN_SECONDS, else the part
    alone. The cut gives the frames showing the tab across the full width (when seen that long), then the left and
    right panels of the split frames, each read in its own columns. The left panel goes on with the full width's
    stream unless `pipeline.carry_on` finds the measure numbers go on from the right one."""
    hits = side_by_side(part.found)
    if len(hits) * scan_step < PART_MIN_SECONDS:
        return [part]
    # An even column, as `video.frames` rounds crop offsets down to even, so each panel's frames start at its cols
    div = int(np.median([x for _, x in hits]))
    div -= div % 2
    times = {t for t, _ in hits}
    whole = {t: (ss if t not in times else []) for t, ss in part.found.items()}
    left = {t: ([s for s in ss if s.tab.x1 <= div] if t in times else []) for t, ss in part.found.items()}
    right = {t: ([s for s in ss if s.tab.x0 >= div] if t in times else []) for t, ss in part.found.items()}

    def seen(sub):
        return sum(1 for ss in sub.values() if ss) * scan_step

    out = []
    full = None
    if seen(whole) >= PART_MIN_SECONDS:
        full = Part(0, part.seg, part.rows, "band", whole, seen(whole), cut_from=part)
        out.append(full)
    out.append(Part(0, part.seg, part.rows, "band", left, seen(left), cols=(seg.x0, div), continues=full,
                    carries=full is None, cut_from=part))
    out.append(Part(0, part.seg, part.rows, "band", right, seen(right), cols=(div, seg.x1), cut_from=part))
    return out


def find_parts(path: Path, seg, si: int, found: dict, scan_step: float) -> list[Part]:
    """The segment's parts, top to bottom, each split screen's panels left to right. A segment whose staffs share
    one page is one part."""
    parts = [p for part in stacked(path, seg, si, found, scan_step)
             for p in panels(part, seg, scan_step)]
    for k, p in enumerate(parts):
        p.id = k
    return parts


def stacked(path: Path, seg, si: int, found: dict, scan_step: float) -> list[Part]:
    """The segment's stacked parts, top to bottom. A segment whose staffs share one page is one part."""
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
        page = {t: ss for t, ss in found.items() if seg.t0 <= t <= seg.t1}
        return [Part(0, si, (seg.y0, seg.y1), "page", page, seg.t1 - seg.t0)]
    return parts


def stream_of(part: Part, seg_parts: list[Part]) -> int:
    """The stream a part joins: a part that continues another joins that one's stream. Otherwise 0 for the segment's
    panel that carries the stream on, else for its longest-seen part (the first one on a tie), counting the seconds
    of the parts that continue it; any other part gets its own. Part ids are unique in the video, as `read` numbers
    every part by its place in the video's list before calling this, so streams of different parts never collide."""
    if part.continues is not None:
        return stream_of(part.continues, seg_parts)
    heads = [p for p in seg_parts if p.continues is None]
    main = max(heads, key=lambda p: (p.carries, p.seen + sum(q.seen for q in seg_parts if q.continues is p)))
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
