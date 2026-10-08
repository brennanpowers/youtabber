from dataclasses import dataclass

import cv2
import numpy as np

from youtabber.views import View

# A real overlap must be at least this share of the view width
MIN_OVERLAP = 0.1
# Share of overlapping ink allowed to disagree at a real overlap
MAX_MISMATCH = 0.04
# Overlaps with less ink than this are too blank to prove anything
MIN_OVERLAP_INK = 300
_NEAR = np.ones((3, 3), np.uint8)


@dataclass
class Placed:
    view: View
    x: int  # left edge of the view in strip coordinates
    new_from: int  # first column of the view that the strip didn't already have
    ambiguous: bool = False  # more than one overlap matched, so the usual scroll distance decided


def _ink(image: np.ndarray) -> np.ndarray:
    """Ink to compare between views: everything darker than paper, minus the staff lines.

    Staff lines look the same at every shift, so they add nothing but compression flicker.
    The loose threshold keeps light gray measure numbers, which help tell repeated measures apart.
    """
    ink = (image < 200).astype(np.uint8)
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (51, 1)))
    return (ink & ~lines).astype(bool)


def candidate_shifts(prev: np.ndarray, cur: np.ndarray) -> list[int]:
    """Every shift where the left of `cur` matches the right of `prev`, best first.

    More than one candidate means the overlap holds repeated measures.
    """
    w = prev.shape[1]
    a, b = _ink(prev), _ink(cur)
    near_a = cv2.dilate(a.astype(np.uint8), _NEAR).astype(bool)
    near_b = cv2.dilate(b.astype(np.uint8), _NEAR).astype(bool)
    scores = np.full(w, np.inf)
    for shift in range(1, int(w * (1 - MIN_OVERLAP))):
        pa, pb = a[:, shift:], b[:, :w - shift]
        ink = pa.sum() + pb.sum()
        if ink < MIN_OVERLAP_INK:
            continue
        missing = (pa & ~near_b[:, :w - shift]).sum() + (pb & ~near_a[:, shift:]).sum()
        scores[shift] = missing / ink
    # Keep only the best shift within each cluster of neighbors, which differ by a pixel or two
    found = []
    for shift in np.argsort(scores):
        if scores[shift] > MAX_MISMATCH:
            break
        if all(abs(int(shift) - f) > 20 for f in found):
            found.append(int(shift))
    return found


def stitch(views: list[View]) -> tuple[np.ndarray, list[Placed]]:
    """Join views into one long strip, keeping each overlapping part once."""
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
            placed.append(Placed(view, x, 0))
            continue
        if len(found) == 1:
            shift = found[0]
        elif typical is not None:
            shift = min(found, key=lambda s: abs(s - typical))
        else:
            shift = min(found)
        x += shift
        placed.append(Placed(view, x, w - shift, ambiguous=len(found) > 1))
    strip = np.full((h, x + w), 255, np.uint8)
    for p in placed:
        strip[:, p.x + p.new_from:p.x + w] = p.view.image[:, p.new_from:]
    return strip, placed
