from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from youtabber import ink
from youtabber.region import Region
from youtabber.video import frames

FPS = 4
# Fraction of ink that must differ from the view's first frame to start a new view
CHANGE_THRESHOLD = 0.15
MIN_VIEW_SECONDS = 1.0
# Bounds memory for long static views; the median needs only a sample of frames
MAX_FRAMES_PER_VIEW = 60


@dataclass
class View:
    start: float
    end: float
    image: np.ndarray  # grayscale median of the view's frames, cursor removed


def brightness(frame: np.ndarray) -> np.ndarray:
    """HSV value channel: the brightest of B, G and R for each pixel.

    Black ink stays dark even under a colored highlight, while colored cursors and
    highlights come out nearly as bright as the paper.
    """
    return frame.max(axis=2)


def has_notes(image: np.ndarray) -> bool:
    """True when the view holds ink beyond its staff lines, ruling out blank staves and end screens."""
    dark = image < ink.DARK
    if dark.mean() > 0.5:
        return False
    lines = ink.horizontal_lines(dark, image.shape[1] // 8)
    return (dark & ~lines).mean() > 0.002


def find_views(path: Path, region: Region) -> Iterator[View]:
    crop = (region.x, region.y, region.w, region.h)
    batch: list[np.ndarray] = []
    ref_ink: np.ndarray | None = None
    start = last = 0.0

    def finish() -> View | None:
        if last - start + 1 / FPS < MIN_VIEW_SECONDS:
            return None
        image = np.median(np.stack([brightness(f) for f in batch]), axis=0).astype(np.uint8)
        return View(start, last + 1 / FPS, image) if has_notes(image) else None

    for t, frame in frames(path, fps=FPS, crop=crop):
        dark = brightness(frame) < ink.DARK
        if ref_ink is None or ink.difference(ref_ink, dark) > CHANGE_THRESHOLD:
            if batch and (view := finish()):
                yield view
            batch, ref_ink, start = [], dark, t
        batch.append(frame)
        if len(batch) > MAX_FRAMES_PER_VIEW:
            batch = batch[::2]
        last = t
    if batch and (view := finish()):
        yield view
