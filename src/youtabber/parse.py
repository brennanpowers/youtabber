"""Cut a view's plate into systems and measure cells."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from youtabber.observe import colored, gray_pixels, ink_map, to_gray
from youtabber.scan import System, systems_in

# A barline covers at least this share of the rows between the tab staff's top and bottom lines
BAR_COVER = 0.92
# ...and is at most this many line gaps wide (thick and double barlines merge into one)
BAR_WIDTH = 0.6
# Barline columns closer than this many gaps belong to one barline
BAR_MERGE = 0.7
# A measure is at least this many gaps wide
MIN_CELL = 1.5
# A staff that starts or ends this many line gaps inside the panel starts or ends there, not at the panel edge
INSIDE_GAPS = 1.0
# Line groups spaced closer than this share of the view's widest-spaced tab staff are not tab staffs
CLOSE_STAFF_SHARE = 0.6


@dataclass
class Cell:
    view: int
    system: int
    index: int                 # left to right within the system
    x0: int                    # left barline (or staff start for a lead cell), crop coordinates
    x1: int
    kind: str                  # "full" between two barlines, "lead" before the first, "tail" after the last
    ink: np.ndarray            # solid ink without staff lines, cell box
    faint: np.ndarray          # faint gray marks away from solid ink, cell box
    y0: int                    # top of the cell box
    number_crop: np.ndarray | None = None   # gray crop of the measure number, if one is printed there
    gap: float = 0.0                        # tab line spacing
    ref_y: int = 0                          # row of the tab staff's top line within the cell box
    mid_y: float = 0.0                      # middle of the tab staff, crop coordinates
    tint: np.ndarray | None = None          # colored (cursor-tinted) pixels in the cell box
    number: int | None = None
    number_text: str | None = None
    slot: int | None = None
    glided: bool = False                    # read from a glide mosaic
    # Between two barlines, or a system's own first or last measure; False when the panel edge cuts it
    complete: bool = True


@dataclass
class ParsedSystem:
    system: System
    bars: list[tuple[int, int]]
    y0: int
    y1: int
    cells: list[Cell] = field(default_factory=list)
    repeat_end: bool = False   # an end-repeat sign (dots before a barline) somewhere in the system


def barlines(dark: np.ndarray, tab_top: float, tab_bottom: float, gap: float, x0: int, x1: int,
             notation: tuple[float, float] | None) -> list[tuple[int, int]]:
    """(first, last) columns of each barline across the tab staff.

    With a notation staff, a barline also crosses the middle of that staff (its outer lines may be ledger or
    beam rows found as lines).
    """
    a, b = int(round(tab_top)), int(round(tab_bottom))
    band = dark[a:b + 1, x0:x1]
    if band.size == 0:
        return []
    full = band.mean(axis=0) >= BAR_COVER
    if notation is None:
        # In tab with rhythm stems below it, a barline stops at the bottom line; a stem that starts on the
        # top string runs on down to its beam
        below = dark[int(round(b + 0.5 * gap)):int(round(b + 2.0 * gap)), x0:x1]
        if below.size:
            full &= below.mean(axis=0) < 0.5
    else:
        # Barlines cross the notation staff too (drawn through the space between the staffs or not)
        n0, n1 = int(round(notation[0])), int(round(notation[1]))
        cover = dark[n0:n1 + 1, x0:x1]
        # A pixel either side: the two staffs' barlines may be a pixel apart after scaling
        cover = np.maximum(cover, np.roll(cover, 1, axis=1))
        cover = np.maximum(cover, np.roll(cover, -1, axis=1))
        full &= cover.mean(axis=0) >= BAR_COVER
    bars: list[list[int]] = []
    for x in np.flatnonzero(full):
        x = int(x) + x0
        if bars and x - bars[-1][1] <= BAR_MERGE * gap:
            bars[-1][1] = x
        else:
            bars.append([x, x])
    return [(p, q) for p, q in bars if q - p <= BAR_WIDTH * gap * 2.5]


def staff_extent(struct: np.ndarray, lines: list[float], x0: int, x1: int, gap: float) -> tuple[int, int]:
    """Exact left and right ends of a staff: walk outward while most lines still have ink in the column,
    stepping over gaps up to one line gap wide (digits sit on the lines)."""
    h, w = struct.shape
    rows = [int(round(y)) for y in lines]

    def present(x: int) -> bool:
        hits = sum(struct[max(0, r - 1):r + 2, x].any() for r in rows)
        return hits >= 0.6 * len(rows)

    def walk(x: int, step: int) -> int:
        last, miss = x, 0
        while 0 <= x + step < w:
            x += step
            if present(x):
                last, miss = x, 0
            else:
                miss += 1
                if miss > gap:
                    break
        return last

    return walk(x0, -1), walk(x1 - 1, 1) + 1


def number_crop(g: np.ndarray, marks: np.ndarray, top: float, top_gap: float, bar_x: int) -> np.ndarray | None:
    """Gray crop of the small mark group above the top staff line nearest the barline at bar_x, if any.

    Measure numbers sit just above the staff at the start of the measure (or just left of it at a
    system start). `marks` is every visible mark, faint ones included.
    """
    h, w = g.shape
    y0, y1 = int(max(0, top - 2.6 * top_gap)), int(max(0, top + 0.4 * top_gap))
    x0, x1 = int(max(0, bar_x - 3.0 * top_gap)), int(min(w, bar_x + 2.5 * top_gap))
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None
    sub = marks[y0:y1, x0:x1].astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    boxes = [stats[i] for i in range(1, n)
             if max(5, 0.2 * top_gap) <= stats[i][3] <= 1.6 * top_gap and stats[i][2] <= 1.4 * top_gap and stats[i][4] >= 6]
    if not boxes:
        return None
    # Digits of one number sit side by side with small gaps; take the group nearest the barline
    boxes.sort(key=lambda b: b[0])
    groups: list[list] = []
    for b in boxes:
        for gr in groups:
            last = gr[-1]
            if (b[0] - (last[0] + last[2]) <= 0.6 * max(b[3], last[3])
                    and abs((b[1] + b[3] / 2) - (last[1] + last[3] / 2)) < 0.3 * max(b[3], last[3])):
                gr.append(b)
                break
        else:
            groups.append([b])
    rel = bar_x - x0

    def dist(gr):
        left = gr[0][0]
        right = gr[-1][0] + gr[-1][2]
        return 0 if left - 0.6 * top_gap <= rel <= right + 0.6 * top_gap else min(abs(left - rel), abs(right - rel))

    best = min(groups, key=dist)
    if dist(best) > 1.0 * top_gap or len(best) > 3:
        return None
    gx0 = min(b[0] for b in best) + x0
    gx1 = max(b[0] + b[2] for b in best) + x0
    gy0 = min(b[1] for b in best) + y0
    gy1 = max(b[1] + b[3] for b in best) + y0
    pad = 2
    return g[max(0, gy0 - pad):gy1 + pad, max(0, gx0 - pad):gx1 + pad].copy()


def end_repeat(dark: np.ndarray, st, bars: list[tuple[int, int]]) -> bool:
    """True when two or more dots sit just left of some barline inside the staff: an end-repeat sign."""
    a, b = int(st.top) + 1, int(st.bottom)
    for p, _ in bars:
        x0, x1 = int(p - 1.3 * st.gap), int(p - 0.1 * st.gap)
        if x0 < 0:
            continue
        sub = dark[a:b, x0:x1].astype(np.uint8)
        # Staff lines would join the dots; keep only marks taller than a line
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        dots = [s for s in stats[1:] if 0.2 * st.gap <= s[2] <= 0.7 * st.gap and 0.2 * st.gap <= s[3] <= 0.7 * st.gap
                and s[4] >= 0.5 * s[2] * s[3]]
        if len(dots) >= 2:
            return True
    return False


# A playhead line held still over a view is lighter than the staff lines and than the staff's real barlines.
# A candidate is dropped only when it is lighter than both: past the staff lines' level by this share of the
# line-to-paper contrast...
BAR_LINE_SLACK = 0.15
# ...and past the staff's typical barline by this share of the barline-to-paper contrast
BAR_TYPICAL_SLACK = 0.5
# The typical barline is known only from this many candidates or more
BAR_TYPICAL_MIN = 3


def line_darkness(gray: np.ndarray, st) -> tuple[float, float]:
    """(gray level of a staff's lines, gray level of the paper between them)."""
    xs = slice(int(st.x0), int(st.x1))
    lines = [float(np.median(gray[max(0, int(round(y)) - 1):int(round(y)) + 2, xs].min(axis=0))) for y in st.lines]
    papers = [float(np.median(gray[int(round((a + b) / 2)), xs])) for a, b in zip(st.lines, st.lines[1:])]
    return float(np.median(lines)), float(np.median(papers))


def drawn_bars(gray: np.ndarray, st, bars: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The barline candidates left once lines lighter than both the staff lines and the typical barline are dropped."""
    if len(bars) < BAR_TYPICAL_MIN:
        return bars
    a, b = int(round(st.top)), int(round(st.bottom))
    levels = [float(np.median(gray[a:b + 1, p:q + 1].min(axis=1))) for p, q in bars]
    line, paper = line_darkness(gray, st)
    typical = float(np.median(levels))
    lighter_than_lines = line + BAR_LINE_SLACK * (paper - line)
    lighter_than_bars = typical + BAR_TYPICAL_SLACK * (paper - typical)
    return [bar for bar, level in zip(bars, levels) if level <= max(lighter_than_lines, lighter_than_bars)]


def opening_line(struct: np.ndarray, st) -> bool:
    """True when a solid line crosses the whole tab staff within one line gap of the staff's left end.

    Only solid ink counts: the soft gray edge of a panel or of the frame is faint."""
    a, b = int(round(st.top)), int(round(st.bottom))
    x0, x1 = int(st.x0), int(st.x0 + INSIDE_GAPS * st.gap) + 1
    band = struct[a:b + 1, x0:x1]
    return band.size > 0 and bool((band.mean(axis=0) >= BAR_COVER).any())


def parse_view(vi: int, plate: np.ndarray, rect: tuple[int, int, int, int], polarity: str, thr: float, fthr: float,
               gap: float) -> list[ParsedSystem]:
    g = to_gray(plate, polarity)
    rx0, ry0, rx1, ry1 = rect
    sub = g.copy()
    # Only the panel counts; footage outside it is painted paper
    paper = 255
    sub[:ry0] = paper
    sub[ry1:] = paper
    sub[:, :rx0] = paper
    sub[:, rx1:] = paper
    # A "tab staff" spaced much closer than the view's widest-spaced one is beams or text lines, not a staff
    syss = systems_in(sub)
    widest = max((s.tab.gap for s in syss), default=0.0)
    syss = [s for s in syss if s.tab.gap >= CLOSE_STAFF_SHARE * widest]
    colored_plate = colored(plate)
    # Where a cursor or highlight tinted the page, ink can't be compared between sightings
    tint = cv2.dilate(colored_plate.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    marks = ink_map(sub, fthr, gap) & ~colored_plate
    dark = sub < thr
    faint_struct = sub < fthr
    ink = ink_map(sub, thr, gap, plate)
    faint = ink_map(sub, fthr, gap) & ~ink & gray_pixels(plate)
    faint &= ~cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    out: list[ParsedSystem] = []
    for si, s in enumerate(syss):
        st = s.tab
        top_gap = (s.notation or s.tab).gap
        y0 = int(max(ry0, s.top - 3.0 * top_gap))
        y1 = int(min(ry1, st.bottom + 3.0 * st.gap))
        if si > 0:
            prev = out[-1]
            mid = int((syss[si - 1].tab.bottom + s.top) / 2)
            y0 = max(y0, mid)
            prev.y1 = min(prev.y1, mid)
        st.x0, st.x1 = staff_extent(faint_struct, st.lines, max(st.x0, rx0), min(st.x1, rx1), st.gap)
        bars = barlines(faint_struct, st.top, st.bottom, st.gap, max(rx0, st.x0 - 2), min(rx1, st.x1 + 3),
                        (s.notation.lines[1], s.notation.lines[-2]) if s.notation else None)
        bars = drawn_bars(sub, st, bars)
        out.append(ParsedSystem(s, bars, y0, y1, repeat_end=end_repeat(dark, st, bars)))
    for si, ps in enumerate(out):
        st = ps.system.tab
        edges: list[tuple[int, int, str]] = []
        xs = [b for b in ps.bars]
        if xs and xs[0][0] - st.x0 > MIN_CELL * st.gap:
            edges.append((st.x0, xs[0][0], "lead"))
        for (a, _), (b, _) in zip(xs, xs[1:]):
            if b - a >= MIN_CELL * st.gap:
                edges.append((a, b, "full"))
        if xs and st.x1 - xs[-1][1] > MIN_CELL * st.gap:
            edges.append((xs[-1][1], st.x1, "tail"))
        if not xs:
            edges.append((st.x0, st.x1, "lead"))
        # A lead cell is the system's own first measure when its staff starts inside the panel or has an
        # opening line; otherwise the panel edge cut a measure in two
        starts_inside = st.x0 >= rx0 + INSIDE_GAPS * st.gap or opening_line(dark, st)
        ends_inside = st.x1 <= rx1 - INSIDE_GAPS * st.gap
        for ci, (a, b, kind) in enumerate(edges):
            # The first measure of a system takes in the margin to its left, where some players print the
            # measure number; two pages that differ only there are different pages
            left = int(max(rx0, a - 1.5 * st.gap)) if ci == 0 and a <= st.x0 + 2 else a
            box_ink = ink[ps.y0:ps.y1, left:b + 1].copy()
            box_tint = tint[ps.y0:ps.y1, left:b + 1].copy()
            # Faint marks count only above the system, where light-gray measure numbers are printed; inside
            # the staffs they are mostly cursor residue
            box_faint = faint[ps.y0:ps.y1, left:b + 1].copy()
            box_faint[max(0, int(ps.system.top) - ps.y0):] = False
            # Content of a lead or tail cell that is only staff lines is no measure
            if kind != "full" and box_ink.sum() < 0.1 * st.gap * st.gap:
                continue
            top = (ps.system.notation or st).top
            top_gap = (ps.system.notation or st).gap
            crop = number_crop(sub, marks, top, top_gap, a)
            ps.cells.append(Cell(vi, si, len(ps.cells), a, b, kind, box_ink, box_faint, ps.y0, crop,
                                 gap=st.gap, ref_y=int(round(st.top)) - ps.y0, mid_y=(st.top + st.bottom) / 2,
                                 tint=box_tint,
                                 complete=kind == "full" or (kind == "lead" and starts_inside)
                                 or (kind == "tail" and ends_inside)))
    return out
