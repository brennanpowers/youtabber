"""Turn one measure, as the video showed it, into a clean black-on-white image for printing."""
from dataclasses import dataclass

import numpy as np

from youtabber.assemble import Obs
from youtabber.observe import to_gray
from youtabber.parse import Cell
from youtabber.scan import Segment

# Paper this close to white prints white
WHITE_FROM = 240


@dataclass
class PrintCell:
    image: np.ndarray        # gray, black ink on white paper
    gap: float               # tab line spacing in image pixels
    top: int                 # image row of the tab staff's top line
    left: int                # image columns left of the measure's left barline (marks that hang over it)
    advance: int             # columns from this measure's left barline to the next measure's


def whiten(gray: np.ndarray) -> np.ndarray:
    """Scale brightness so the paper prints white, and anything close to it too."""
    paper = float(np.median(gray))
    out = np.clip(gray.astype(np.float32) * (255 / max(paper, 1.0)), 0, 255)
    out[out > WHITE_FROM] = 255
    return out.astype(np.uint8)


def print_cell(o: Obs, c: Cell, seg: Segment) -> PrintCell:
    """Measure c of view o as printed: the cell's columns over its system's print band, gray with dark ink,
    paper white."""
    # The cell's ink box starts at its left barline, or a little before it for a system's first measure
    x_left = c.x1 + 1 - c.ink.shape[1]
    box = o.image[c.band[0]:c.band[1], x_left:c.x1 + 1]
    top = c.y0 + c.ref_y - c.band[0]
    return PrintCell(whiten(to_gray(box, seg.polarity)), c.gap, top, c.x0 - x_left, c.x1 - c.x0)
