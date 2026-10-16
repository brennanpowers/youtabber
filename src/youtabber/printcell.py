"""Turn one measure, as the video showed it, into a clean black-on-white image for printing."""
import bisect
from dataclasses import dataclass

import cv2
import numpy as np

from youtabber.assemble import Obs
from youtabber.observe import chroma, set_paper, to_gray
from youtabber.parse import Cell
from youtabber.scan import Segment

# Paper this close to white prints white
WHITE_FROM = 240
# A mark crossing the next barline this many line gaps or less from its left edge is a measure number centered
# on that barline, and belongs to the measure starting there
NUMBER_REACH = 1.5
# A cell left out of the measure list joins the next measure only up to this many line gaps wide: a clef, key or
# time signature is short, a piece of music cut off by stray barlines is often wider
JOIN_GAPS = 4.5
# Anything this much darker than white, once the paper is whitened, is a mark
MARK_LEVEL = 235
# Tinted areas at least this many square line gaps big, or taller than three gaps, are cursor boxes, highlight
# bars or playhead lines; smaller colored marks, such as red measure numbers, are ink
TINT_AREA_GAPS = 2.0
# Color spread, after dividing out the paper's color, above which a pixel is tinted. Gray staff lines on yellow
# paper come out of video compression at about 31, so the limit sits above that.
PRINT_COLOR_SAT = 40
# Share of the paper level a tinted pixel may be darker than its area's typical gray and still count as tint
LIFT_SLACK = 0.1


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


def owner(left: int, right: int, starts: list[int], gap: float) -> int:
    """Index of the measure a mark above or below the staffs belongs to. `starts` are the measures' left edges,
    left to right. A mark belongs to the measure its left edge is in, except one that crosses the next barline
    close to its left edge."""
    k = max(bisect.bisect_right(starts, left) - 1, 0)
    if k + 1 < len(starts) and left < starts[k + 1] < right and starts[k + 1] - left <= NUMBER_REACH * gap:
        return k + 1
    return k


def mark_groups(marks: np.ndarray, x_off: int, starts: list[int], gap: float) -> list[tuple[np.ndarray, int, int, int]]:
    """The words and numbers in marks, a 0/1 image of marks above or below the staffs whose column 0 is image
    column x_off. For each: its own pixels, its left and right image columns, and the measure it belongs to."""
    # Thin long strokes, such as the edge of a staff line, print in the measure's own columns like the staff
    # and never join a word
    n, labels, stats, _ = cv2.connectedComponentsWithStats(marks, connectivity=8)
    thin = (stats[:, cv2.CC_STAT_HEIGHT] < 0.2 * gap) & (stats[:, cv2.CC_STAT_WIDTH] > gap)
    thin[0] = False
    marks = np.where(thin[labels], 0, marks).astype(np.uint8)
    # The digits of a number and the letters of a word are a few pixels apart: smear the marks sideways so
    # each word is one group, then keep only the marks' own pixels in each group
    grouped = cv2.dilate(marks, np.ones((1, max(3, int(0.3 * gap) | 1)), np.uint8))
    n, labels = cv2.connectedComponents(grouped, connectivity=8)
    labels[marks == 0] = 0
    groups = []
    for j in range(1, n):
        mask = labels == j
        cols = np.flatnonzero(mask.any(axis=0))
        left, right = int(cols[0]) + x_off, int(cols[-1]) + x_off + 1
        groups.append((mask, left, right, owner(left, right, starts, gap)))
    return groups


def joined_cells(cells: list[Cell], kept: list[Cell]) -> list[tuple[Cell, int, int]]:
    """A system's cells as printed, left to right: each cell, the column its printed measure starts at, and the
    first column it prints. A short complete cell left out of the measure list, with no other left-out cell next to
    it (a clef or time signature set off by a barline of its own), prints with the measure after it, so its measure
    number and signature print whole."""
    ids = {id(x) for x in kept}
    dropped = [id(x) not in ids for x in cells]
    out, carry = [], None
    for k, x in enumerate(cells):
        alone = not (k > 0 and dropped[k - 1]) and k + 1 < len(cells) and not dropped[k + 1]
        if dropped[k] and x.complete and alone and x.x1 - x.x0 <= JOIN_GAPS * x.gap:
            carry = (x.x0, x.print_x0)
            continue
        out.append((x, *(carry or (x.x0, x.print_x0))))
        carry = None
    return out


def clean(band: np.ndarray, seg: Segment, gap: float, cols: tuple[int, int] | None = None) -> np.ndarray:
    """Gray image of a band with dark ink on white paper, with cursor and highlight tint lifted to paper. Only
    columns cols (the paper panel) are kept; the rest print white."""
    set_paper(seg.paper)
    a, b = cols or (0, band.shape[1])
    out = np.full(band.shape[:2], 255, np.uint8)
    # OpenCV crashes on an empty image
    if b <= a:
        return out
    part = band[:, a:b]
    gray = to_gray(part, seg.polarity).astype(np.float32)
    paper = float(np.median(gray))
    tinted = (chroma(part) > PRINT_COLOR_SAT).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(tinted, connectivity=8)
    big = [j for j in range(1, n) if stats[j][4] >= TINT_AREA_GAPS * gap * gap or stats[j][3] >= 3 * gap]
    # On dark themes the ink itself is colored, so nothing is lifted
    if seg.polarity != "light":
        for j in big:
            body = labels == j
            # Only the light body of a box becomes paper. Ink keeps its darkness: lines and digits under a box are
            # darker than the box, and a tinted barline is darker than half the paper.
            level = float(np.median(gray[body])) - LIFT_SLACK * paper
            gray[body & (gray >= level) & (gray > 0.5 * paper)] = paper
    out[:, a:b] = whiten(gray.astype(np.uint8))
    return out


def print_cell(o: Obs, c: Cell, seg: Segment) -> PrintCell:
    """Measure c of view o as printed: its system's band, from its left barline to the next one, with the marks
    above and below the staffs that belong to it and none that belong to its neighbors."""
    ps = o.systems[c.system]
    parts = joined_cells(ps.cells, o.cells)
    i = next(k for k, (x, _, _) in enumerate(parts) if x is c)
    _, left_bar, first_col = parts[i]
    starts = [p[1] for p in parts]
    b0, b1 = c.band
    # Only the paper panel counts: footage left or right of it would skew the paper level
    px0, _, px1, _ = o.print_rect
    gray = clean(o.image[b0:b1], seg, c.gap, (px0, px1))
    st = ps.system.tab
    sys_top = int(round((ps.system.notation or st).top)) - b0
    sys_bottom = int(round(st.bottom)) - b0
    # Mark rows reach close to the outer staff lines, so a number sitting on the staff is labeled whole
    margin = max(2, int(0.15 * c.gap))
    # The measure's own columns: its left edge through its right barline. A system's first measure keeps the
    # margin before it, and a measure joined with a clef or time signature cell starts at that cell
    x0 = first_col
    x1 = next((q for p, q in ps.bars if p == c.x1), c.x1) + 1
    # Marks are looked at between the neighbors' far edges, so a mark's left edge is never cut off
    w0 = starts[i - 1] if i > 0 else max(0, x0 - int(3 * c.gap))
    w1 = starts[i + 2] if i + 2 < len(starts) else gray.shape[1]
    keep = np.zeros(gray.shape, bool)
    keep[:, x0:x1] = True
    xa, xb = x0, x1
    for rows in (slice(0, max(0, sys_top - margin)), slice(min(gray.shape[0], sys_bottom + margin), gray.shape[0])):
        marks = (gray[rows, w0:w1] < MARK_LEVEL).astype(np.uint8)
        # OpenCV crashes on an empty image: a band with no room above or below the staffs has no marks there
        if marks.size == 0:
            continue
        part = keep[rows, w0:w1]
        for mask, left, right, k in mark_groups(marks, w0, starts, c.gap):
            part[mask] = k == i
            if k == i:
                xa, xb = min(xa, left), max(xb, right)
    out = np.where(keep, gray, 255).astype(np.uint8)[:, xa:xb]
    # Overhang is measured from the left barline (a system's first measure: its staff start)
    return PrintCell(out, c.gap, c.y0 + c.ref_y - b0, left_bar - xa, c.x1 - left_bar)
