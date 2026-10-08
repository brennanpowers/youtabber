from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from youtabber import ink
from youtabber.views import View

# A real overlap must be at least this share of the view width
MIN_OVERLAP = 0.1
# A real scroll moves at least this share of the view width. A closer match is the same image shown
# twice, such as a repeated line, and both copies are kept.
MIN_SCROLL = 0.1
# Share of overlapping ink allowed to disagree at a real overlap
MAX_MISMATCH = 0.04
# Overlaps with less ink than this are too blank to prove anything
MIN_OVERLAP_INK = 300


@dataclass
class Placed:
    view: View
    x: int  # left edge of the view in strip coordinates
    new_from: int  # first column of the view that the strip didn't already have
    ambiguous: bool = False  # more than one overlap matched, so one had to be picked
    unjoined: bool = False  # the video scrolls, but this view matched nothing and was added whole


def candidate_shifts(prev: np.ndarray, cur: np.ndarray) -> list[int]:
    """Every shift where the left of `cur` matches the right of `prev`, best first.

    More than one candidate means the overlap holds repeated measures.
    """
    w = prev.shape[1]
    # Staff lines match at every shift, so compare everything else, light gray measure numbers included
    a, b = ink.without_staff_lines(prev), ink.without_staff_lines(cur)
    near_a, near_b = ink.near(a), ink.near(b)
    scores = np.full(w, np.inf)
    for shift in range(int(w * MIN_SCROLL), int(w * (1 - MIN_OVERLAP))):
        pa, pb = a[:, shift:], b[:, :w - shift]
        total = pa.sum() + pb.sum()
        if total < MIN_OVERLAP_INK:
            continue
        missing = (pa & ~near_b[:, :w - shift]).sum() + (pb & ~near_a[:, shift:]).sum()
        scores[shift] = missing / total
    # Keep only the best shift within each cluster of neighbors, which differ by a pixel or two
    found = []
    for shift in np.argsort(scores):
        if scores[shift] > MAX_MISMATCH:
            break
        if all(abs(int(shift) - f) > 20 for f in found):
            found.append(int(shift))
    return found


def align(views: list[View]) -> list[View]:
    """Views moved up or down so their tab staffs sit at the same height.

    Some videos draw each page with the staffs a few pixels higher or lower. Views are padded rather
    than cropped, and a view whose tab staff has an unusual number of lines is left where it is.
    """
    tabs = [ink.tab_staff(ink.staff_lines(v.image)) for v in views]
    usual = Counter(len(t) for t in tabs if t).most_common(1)
    if not usual:
        return views
    tops = [t[0] for t in tabs if len(t) == usual[0][0]]
    target = int(np.median(tops))
    moves = [target - t[0] if len(t) == usual[0][0] else 0 for t in tabs]
    if not any(moves):
        return views
    h, w = views[0].image.shape
    up, down = -min(moves + [0]), max(moves + [0])
    out = []
    for view, move in zip(views, moves):
        image = np.full((h + up + down, w), 255, np.uint8)
        image[up + move:up + move + h] = view.image
        out.append(replace(view, image=image))
    return out


def stitch(views: list[View]) -> tuple[np.ndarray, list[Placed]]:
    """Join views into one long strip, keeping each overlapping part once."""
    views = align(views)
    h, w = views[0].image.shape
    candidates = [candidate_shifts(p.image, c.image) for p, c in zip(views, views[1:])]
    # Players scroll by the same amount each time, so clear-cut pairs settle the repeated-measure ones
    clear = [c[0] for c in candidates if len(c) == 1]
    typical = float(np.median(clear)) if clear else None

    placed = [Placed(views[0], 0, 0)]
    x = 0
    for view, found in zip(views[1:], candidates):
        if not found:
            x += w
            placed.append(Placed(view, x, 0, unjoined=typical is not None))
            continue
        if len(found) == 1:
            shift = found[0]
        elif typical is not None:
            shift = min(found, key=lambda s: abs(s - typical))
        else:
            # With no clear-cut pair to learn the scroll distance from, take the largest overlap
            shift = min(found)
        x += shift
        placed.append(Placed(view, x, w - shift, ambiguous=len(found) > 1))
    strip = np.full((h, x + w), 255, np.uint8)
    for p in placed:
        strip[:, p.x + p.new_from:p.x + w] = p.view.image[:, p.new_from:]
    return strip, placed
