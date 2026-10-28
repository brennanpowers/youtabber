"""Measure how the band moved between two still views, from the frames between them.

A band that slides (a jump scroll drawn as a short animation, or one instant jump) moves the music by a measurable
shift; a page flip or cross-fade replaces it; a band that only had a cursor move over it didn't move at all. The
frames between two views are read at a high rate so each step of a slide is small enough to measure.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Hashable

import cv2
import numpy as np

from youtabber.observe import colored, to_gray
from youtabber.video import frames

# Frames per second read between two views; a slide drawn over a tenth of a second still shows three steps
TRACK_FPS = 30
# Seconds of frames read either side of the gap between two views
TRACK_PAD = 0.1
# Frames this far apart are not tracked; something else filled the time between the views
TRACK_MAX_SECONDS = 3.0
# A step moves when phase correlation finds a shift of at least this many pixels with at least this response
MOVE_PX, MOVE_RESPONSE = 1.0, 0.2
# A step that shifts the band up or down by more than this many pixels is no sideways slide
MAX_DY_PX = 2.0


@dataclass
class Move:
    kind: str                          # "still", "slide", "cut" or "glide"; "unknown" when the frames couldn't be read
    shifts: list[int] = field(default_factory=list)   # candidate content shifts for a slide, best guess first
    drawn: list[bool] = field(default_factory=list)   # per shift: reached in two or more steps the same way
    steps: int = 0
    moves: list[float] = field(default_factory=list)  # each step's measured shift


def content(band: np.ndarray, polarity: str, thr: float, gap: float) -> np.ndarray:
    """Soft ink of a band without staff lines and without colored marks (cursors, highlights)."""
    g = to_gray(band, polarity)
    ink = np.clip((thr + 40 - g.astype(np.float32)) / 80.0, 0, 1)
    ink[colored(band)] = 0
    lines = cv2.morphologyEx((g < thr).astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(3 * gap)), 1)))
    ink[cv2.dilate(lines, np.ones((3, 1), np.uint8)).astype(bool)] = 0
    return ink


def transition(path: Path, crop: tuple[int, int, int, int], rows: tuple[int, int], cols: tuple[int, int],
               polarity: str, thr: float, gap: float, t0: float, t1: float) -> Move:
    """How the band in crop rows `rows` and columns `cols` moved from time t0 (the last frame of one view) to t1
    (the first frame of the next)."""
    if t1 - t0 > TRACK_MAX_SECONDS or rows[1] - rows[0] < 4 or cols[1] - cols[0] < 4 * gap:
        return Move("unknown")
    y0, y1 = rows
    x0, x1 = cols
    prev = None
    moves: list[float] = []
    cut = False
    n = 0
    start = max(0.0, t0 - TRACK_PAD)
    for _, f in frames(path, TRACK_FPS, crop=crop, start=start, length=t1 - start + TRACK_PAD):
        ink = content(f[y0:y1, x0:x1], polarity, thr, gap)
        if prev is not None:
            n += 1
            (dx, dy), resp = cv2.phaseCorrelate(prev, ink)
            if abs(dx) >= MOVE_PX or abs(dy) >= MOVE_PX:
                if resp >= MOVE_RESPONSE and abs(dy) <= MAX_DY_PX:
                    moves.append(dx)
                else:
                    cut = True
        prev = ink
    if n == 0:
        return Move("unknown")
    if cut:
        return Move("cut", steps=n, moves=moves)
    if not moves:
        return Move("still", steps=n)
    # Phase correlation wraps round the band's width: a step longer than half of it reads as a step the other way
    w = x1 - x0
    paths = [[]]
    for dx in moves:
        options = [dx] + ([dx - np.sign(dx) * w] if abs(dx) > 0.15 * w else [])
        paths = [q + [d] for q in paths for d in options]
        if len(paths) > 16:
            break
    # Content shifts in the assembler's sense: content of the later view sits that far left of where it was
    found: dict[int, bool] = {}
    for q in paths:
        d = int(round(-sum(q)))
        big = [x for x in q if abs(x) >= gap]
        found[d] = found.get(d, False) or (len(big) >= 2 and len({x > 0 for x in q if abs(x) >= MOVE_PX}) == 1)
    # A slide drawn in several steps the same way comes first: steps that turn back and forth are more likely
    # steps misread the wrong way round
    best = sorted(found, key=lambda d: (not found[d], abs(d + sum(moves))))
    return Move("slide", best, [found[d] for d in best], n, moves)


def pairs(views: list, key: Callable[[object], Hashable]) -> list[tuple]:
    """Each view paired with the view before it that has the same key, in order."""
    last: dict = {}
    out = []
    for v in views:
        k = key(v)
        if k in last:
            out.append((last[k], v))
        last[k] = v
    return out
