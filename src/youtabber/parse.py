"""Cut a view's plate into systems and measure cells."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from youtabber.observe import PRINT_ABOVE_GAPS, PRINT_BELOW_GAPS, PRINT_MARGIN_GAPS, colored, gray_pixels, ink_map, to_gray
from youtabber.scan import System, line_columns, systems_in
from youtabber.staff import without_border

# A barline covers at least this share of the rows between the tab staff's top and bottom lines
BAR_COVER = 0.92
# ...and is at most this many line gaps wide (thick and double barlines merge into one)
BAR_WIDTH = 0.6
# Barline columns closer than this many gaps belong to one barline
BAR_MERGE = 0.7
# The columns this many line gaps either side of a barline are checked for ink...
BAR_SIDE_GAPS = 0.12
# ...and a stroke with more than this share of the tab staff's rows inked there on both sides is not a barline
BAR_SIDE_INK = 0.45
# A pixel beside a line counts as ink there only when it is darker than the faint-ink threshold and than the line's
# own gray level plus this share of the way to the paper: footage seen through a panel is much lighter than the line
SIDE_LINE_SHARE = 0.45
# A line lighter than the ink threshold is judged against its own gray level only when it is darker than the staff's
# paper by at least this share of the way from the paper to the ink threshold: a lighter stripe is footage, not a line
SIDE_LINE_DEPTH = 0.25
# A measure is at least this many gaps wide
MIN_CELL = 1.5
# A staff that starts or ends this many line gaps inside the panel starts or ends there, not at the panel edge
INSIDE_GAPS = 1.0
# A lead or tail cell is a measure only where the staff lines are drawn across at least this share of the columns
# they cover in the system's whole measures
EDGE_LINES_SHARE = 0.5
# A colored mark at most this many tab line gaps tall and wide may be a digit of a colored measure number
DIGIT_GAPS = 1.6
# Line groups spaced closer than this share of the view's widest-spaced tab staff are not tab staffs
CLOSE_STAFF_SHARE = 0.6


# A notation staff's line spacing is within this share of the other notation staffs in the same view
NOTATION_MATCH = 0.15
# ...and below this share of the view's widest tab staff
NOTATION_SHARE = 0.85


def lone_notation(s: System, notation_gaps: list[float], widest: float) -> bool:
    """True when a "tab staff" with no notation of its own has at most 5 lines, spaced like the view's notation and
    closer together than the view's widest tab staff."""
    if s.notation is not None or len(s.tab.lines) > 5 or not notation_gaps or s.tab.gap >= NOTATION_SHARE * widest:
        return False
    return any(abs(s.tab.gap - g) <= NOTATION_MATCH * g for g in notation_gaps)


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
    read: int | None = None                 # the number OCR read from number_crop, trusted or not
    slot: int | None = None
    glided: bool = False                    # read from a glide mosaic
    # Between two barlines, or a system's own first or last measure; False when the panel edge cuts it
    complete: bool = True
    band: tuple[int, int] = (0, 0)          # rows to print, crop coordinates
    print_x0: int = 0                       # first column to print, crop coordinates
    notation_only: bool = False             # on a notation staff whose tab staff is out of the panel
    header: bool = False                    # a system's clef, key and time signature standing alone before a barline
    # Shaped like a header but with marks past the clef: a pickup bar, or a header with a time signature on the tab
    maybe_header: bool = False
    top_y: int = 0                          # row of the system's top staff line (notation or tab) in the cell box
    colored_bar: bool = False               # a barline at either end is a strongly colored stroke
    faint_bar: bool = False                 # a faint line inside may be a barline that was left out


@dataclass
class ParsedSystem:
    system: System
    bars: list[tuple[int, int]]
    y0: int
    y1: int
    cells: list[Cell] = field(default_factory=list)
    repeat_end: bool = False   # an end-repeat sign (dots before a barline) somewhere in the system
    repeat_bars: list = field(default_factory=list)   # left edges of the barlines that end a repeat
    colored_bars: set = field(default_factory=set)    # left edges of the barlines drawn as colored strokes
    unsure_bars: set = field(default_factory=set)     # columns of faint lines left out that may be barlines
    band: tuple[int, int] = (0, 0)   # rows to print, crop coordinates


# A barline joins two staffs into one system when a column of it is ink on this share of the rows from the upper
# tab staff's top line to the lower one's bottom line...
JOINED_COVER = 0.9
# ...for at least JOINED_MIN_BARS barlines and JOINED_SHARE of the lower staff's barlines
JOINED_MIN_BARS, JOINED_SHARE = 2, 0.6
# Barlines of the two staffs this close are one column
JOINED_SLACK_PX = 3

# A staff whose bottom line lies within this many line gaps of the plate's bottom edge, with fewer lines than the
# view's other tab staffs, is cut by that edge
CUT_STAFF_GAPS = 1.5

# In a glide, a barline drawn behind the note tiles covers less of the staff: a thin line covering this share
# counts too, when the columns on each side of it are inked in less than THIN_BESIDE of the rows the line is inked
# in (the edge of a column of digits is inked on one side)
HIDDEN_BAR_COVER, THIN_BESIDE = 0.85, 0.5


def side_ink(gray: np.ndarray, a: int, b: int, x: int, level: float, paper: float, fthr: float) -> float:
    """Share of rows a..b where column x is ink beside a line of gray `level` on paper of gray `paper`."""
    return float((gray[a:b + 1, x] < min(fthr, level + SIDE_LINE_SHARE * (paper - level))).mean())


def digit_sized(mask: np.ndarray, gap: float) -> np.ndarray:
    """The parts of a mask that are separate marks no bigger than a digit printed over a staff."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    small = np.zeros(n, bool)
    small[1:] = (stats[1:, 3] <= DIGIT_GAPS * gap) & (stats[1:, 2] <= DIGIT_GAPS * gap)
    return small[labels]


def barlines(gray: np.ndarray, thr: float, fthr: float, tab_top: float, tab_bottom: float, gap: float, x0: int, x1: int,
             notation: tuple[float, float] | None, hidden: bool = False) -> list[tuple[int, int]]:
    """(first, last) columns of each barline across the tab staff, found on the gray plate's faint-ink mask (`fthr`).

    With a notation staff, a barline also crosses the middle of that staff (its outer lines may be ledger or
    beam rows found as lines). With `hidden`, thin lines partly covered by notes count as barlines too.
    """
    dark = gray < fthr
    a, b = int(round(tab_top)), int(round(tab_bottom))
    band = dark[a:b + 1, x0:x1]
    if band.size == 0:
        return []
    cover = band.mean(axis=0)
    full = cover >= BAR_COVER
    if hidden:
        inked = np.maximum(band.sum(axis=0), 1)
        left, right = ((sum((band & np.roll(band, d, axis=1)).sum(axis=0) for d in side) / (2 * inked))
                       for side in ((2, 3), (-2, -3)))
        full |= (cover >= HIDDEN_BAR_COVER) & (np.maximum(left, right) < THIN_BESIDE)
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
    bars = [(p, q) for p, q in bars if q - p <= BAR_WIDTH * gap * 2.5]
    # In a glide the note tiles crowd the barlines, so ink beside a line proves nothing there
    if hidden:
        return bars
    # A barline is a straight stroke with paper beside it; an arpeggio squiggle or a letter stroke has ink on
    # both sides
    d = max(2, int(round(BAR_SIDE_GAPS * gap)))
    paper = float(np.median(gray[a:b + 1, x0:x1]))

    def squiggle(p: int, q: int) -> bool:
        if not (p - d >= 0 and q + d < dark.shape[1] and dark[a:b + 1, p - d].mean() > BAR_SIDE_INK
                and dark[a:b + 1, q + d].mean() > BAR_SIDE_INK):
            return False
        # The soft edges of a dark stroke are lighter than the stroke itself, so only a line drawn lighter than the
        # ink threshold is judged against its own gray level. A light stripe of footage hardly darker than the paper
        # around it is no line, whatever lies beside it
        level = float(np.median(gray[a:b + 1, p:q + 1].min(axis=1)))
        if level < thr or paper - level < SIDE_LINE_DEPTH * (paper - thr):
            return True
        return (side_ink(gray, a, b, p - d, level, paper, fthr) > BAR_SIDE_INK
                and side_ink(gray, a, b, q + d, level, paper, fthr) > BAR_SIDE_INK)
    return [(p, q) for p, q in bars if not squiggle(p, q)]


# A barline whose dark pixels across the tab staff are more than this share strongly colored may be the side of a
# colored annotation box drawn over the music, which looks the same whether or not a real barline lies under it...
BAR_COLORED_SHARE = 0.5
# ...when it is BOX_SIDE_GAPS line gaps wide: thinner ones are a line cursor resting on a barline, wider ones the edge
# of a highlight
BOX_SIDE_GAPS = (0.3, 0.6)


def colored_bars(dark: np.ndarray, colored_plate: np.ndarray, st, bars: list[tuple[int, int]]) -> set[int]:
    """Left edges of the barlines that are colored strokes as wide as an annotation box's side; printed barlines are
    gray."""
    a, b = int(round(st.top)), int(round(st.bottom))
    out = set()
    for p, q in bars:
        if not BOX_SIDE_GAPS[0] * st.gap <= q - p + 1 <= BOX_SIDE_GAPS[1] * st.gap:
            continue
        ink = dark[a:b + 1, p:q + 1]
        if ink.any() and (colored_plate[a:b + 1, p:q + 1] & ink).sum() > BAR_COLORED_SHARE * ink.sum():
            out.add(p)
    return out


# A staff running to the panel's right edge has its barlines looked for up to this many line gaps past that edge
EDGE_BAR_GAPS = 2.0
# ...and a barline found there counts only when at least this many gaps wide, as a double, final or end-repeat one is...
EDGE_BAR_MIN = 0.4
# ...and in every column just past it, less than this share of the staff is ink
EDGE_PAPER_INK = 0.5


def paper_after(gray: np.ndarray, st, bar: tuple[int, int], fthr: float) -> bool:
    """True when paper, not footage, follows the barline across the tab staff."""
    a, b = int(round(st.top)), int(round(st.bottom))
    x0 = bar[1] + 1
    x1 = x0 + max(2, int(0.5 * st.gap))
    if x1 > gray.shape[1]:
        return False
    return bool(((gray[a:b + 1, x0:x1] < fthr).mean(axis=0) < EDGE_PAPER_INK).all())


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


# A digit of a measure number is at most this many line gaps of its staff tall; some scores print numbers larger than
# the staff's spaces
NUMBER_DIGIT_GAPS = 2.0
# A mark at most this many line gaps wide that runs down past the top staff line is the top end of a barline or a
# stem, not a digit
STUB_GAPS = 0.3
# A barline's width is read this many line gaps below the top staff line, clear of the line itself
BAR_PROBE_GAPS = 0.3
# A digit of a measure number covers at least this many pixels; a smaller mark is a speck
MIN_DIGIT_AREA = 6
# A mark not taken for a digit, level with a number and this many digit heights or less from its side, may hold the
# number's first or last digit merged with a stem or a line
MERGED_DIGIT_GAP = 0.6


def number_crop(g: np.ndarray, marks: np.ndarray, top: float, top_gap: float, bar_x: int,
                panel_top: int = 0) -> np.ndarray | None:
    """Gray crop of the small mark group above the top staff line nearest the barline at bar_x, if any.

    Measure numbers sit just above the staff at the start of the measure (or just left of it at a
    system start). `marks` is every visible mark, faint ones included. Above row `panel_top` the plate is painted
    paper, so a group starting at the panel's top row is a number cut by the panel's top edge.
    """
    h, w = g.shape
    y0, y1 = int(max(0, top - 2.6 * top_gap)), int(max(0, top + 0.4 * top_gap))
    x0, x1 = int(max(0, bar_x - 3.0 * top_gap)), int(min(w, bar_x + 2.5 * top_gap))
    if y1 - y0 < 4 or x1 - x0 < 4:
        return None
    sub = marks[y0:y1, x0:x1].astype(np.uint8)
    # Clear the barline's columns from just above the top line down, so a digit touching the barline is read without it
    probe = min(sub.shape[0] - 1, int(top + BAR_PROBE_GAPS * top_gap) - y0)
    c0 = c1 = min(max(bar_x - x0, 0), sub.shape[1] - 1)
    while c0 > 0 and sub[probe, c0 - 1]:
        c0 -= 1
    while c1 + 1 < sub.shape[1] and sub[probe, c1 + 1]:
        c1 += 1
    sub[max(0, int(top) - 1 - y0):, max(0, c0 - 1):c1 + 2] = 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    boxes = [stats[i] for i in range(1, n)
             if max(5, 0.2 * top_gap) <= stats[i][3] <= NUMBER_DIGIT_GAPS * top_gap and stats[i][2] <= 1.4 * top_gap
             and stats[i][4] >= MIN_DIGIT_AREA
             and not (stats[i][1] + stats[i][3] >= y1 - y0 and stats[i][2] <= STUB_GAPS * top_gap)]
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

    # A measure number has at most three digits; a longer group is a word such as "let ring" printed nearby. A group
    # starting at the panel's top row is cut off
    groups = [gr for gr in groups if len(gr) <= 3 and min(b[1] for b in gr) + y0 > panel_top]
    if not groups:
        return None
    best = min(groups, key=dist)
    if dist(best) > 1.0 * top_gap or merged_beside(stats[1:], boxes, best):
        return None
    gx0 = min(b[0] for b in best) + x0
    gx1 = max(b[0] + b[2] for b in best) + x0
    gy0 = min(b[1] for b in best) + y0
    gy1 = max(b[1] + b[3] for b in best) + y0
    pad = 2
    return g[max(0, gy0 - pad):gy1 + pad, max(0, gx0 - pad):gx1 + pad].copy()


def merged_beside(stats: np.ndarray, boxes: list, group: list) -> bool:
    """True when a mark not taken for a digit stands level with the group, within MERGED_DIGIT_GAP digit heights of its
    side: reading the group alone could drop a digit, such as the 2 of 12 joined to a stem."""
    left, right = group[0][0], group[-1][0] + group[-1][2]
    top, bottom = min(b[1] for b in group), max(b[1] + b[3] for b in group)
    reach = MERGED_DIGIT_GAP * max(b[3] for b in group)
    digits = {tuple(b) for b in boxes}
    for x, y, bw, bh, area in stats:
        if (x, y, bw, bh, area) in digits or area < MIN_DIGIT_AREA or not (y < bottom and y + bh > top):
            continue
        if 0 <= x - right <= reach or 0 <= left - (x + bw) <= reach:
            return True
    return False


# An end-repeat sign's dots are DOT_SIZE line gaps across and stand one above the other: at most DOT_ALIGN line gaps
# apart in x and at least DOT_APART apart in y. Some scores put both dots in one line space, about 0.7 gaps apart;
# fret digits beside a barline stand at one height
DOT_SIZE, DOT_ALIGN, DOT_APART = (0.15, 0.7), 0.3, 0.5
# The dots sit evenly above and below the staff's middle: their midpoint is at most DOT_CENTER line gaps from it.
# Pieces of a time signature's digit, split by a staff line, are off center
DOT_CENTER = 0.3
# Both dots are one glyph printed twice, so their heights differ by at most this factor; a fret digit cut by a
# staff line leaves two pieces of unequal height
DOT_MATCH = 1.5
# A dot is round: at most this many times longer than wide. Stacked fret digits, string names and pieces of tie
# arcs beside a barline are long thin marks
DOT_ELONGATION = 1.75
# A barline at most this many line gaps right of the barline before it ends no measure, so dots before it are no
# end-repeat sign: the time signature's digits after a system's opening barline
REPEAT_ROOM_GAPS = 3.0


def end_repeats(dark: np.ndarray, st, bars: list[tuple[int, int]]) -> list[int]:
    """Left edges of the barlines with two dots one above the other just left of them inside the staff: end-repeat
    signs. Marks side by side at one height are fret digits, not a sign."""
    a, b = int(st.top) + 1, int(st.bottom)
    out = []
    for k, (p, _) in enumerate(bars):
        x0, x1 = int(p - 1.3 * st.gap), int(p - 0.1 * st.gap)
        if x0 < 0 or (k > 0 and p - bars[k - 1][1] <= REPEAT_ROOM_GAPS * st.gap):
            continue
        sub = dark[a:b, x0:x1].astype(np.uint8)
        # Staff lines would join the dots; keep only marks taller than a line
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((3, 1), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        lo, hi = DOT_SIZE[0] * st.gap, DOT_SIZE[1] * st.gap
        dots = [s for s in stats[1:] if lo <= min(s[2], s[3]) and max(s[2], s[3]) <= hi and s[4] >= 0.5 * s[2] * s[3]
                and max(s[2], s[3]) <= DOT_ELONGATION * min(s[2], s[3])]
        mid = (st.top + st.bottom) / 2 - a
        if any(abs(u[0] - v[0]) <= DOT_ALIGN * st.gap and abs(u[1] - v[1]) >= DOT_APART * st.gap
               and abs((u[1] + u[3] / 2 + v[1] + v[3] / 2) / 2 - mid) <= DOT_CENTER * st.gap
               and max(u[3], v[3]) <= DOT_MATCH * min(u[3], v[3])
               for u in dots for v in dots):
            out.append(p)
    return out


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


# A barline drawn light, thin or across two pixel columns can miss BAR_COVER on the faint-ink mask. Columns the rule
# passed over are looked at again on gray levels: a row of the staff counts when the darker of two neighbouring
# columns is darker than the paper on both sides by FAINT_HIT of the staff lines' contrast...
FAINT_HIT = 0.25
# ...and a column is a candidate when at least FAINT_COVER of the rows between the staff lines count
FAINT_COVER = 0.5
# Paper either side of a candidate is looked for this many line gaps away (2 pixels at least)
FAINT_SIDE_GAPS = 0.15


def faint_candidates(gray: np.ndarray, st, bars: list[tuple[int, int]], notation: tuple[float, float] | None,
                     fthr: float, colored_plate: np.ndarray,
                     dropped: list[tuple[int, int]] = ()) -> list[tuple[int, dict]]:
    """(column, measurements) of each column of a tab staff that stands out from the paper on both sides on most
    rows between its lines but is no barline found yet, nor a line dropped as lighter than the page's barlines. Each
    measurement is a share of rows or a darkness relative to the staff lines, so a light page is judged against its
    own ink. faint_rule says which measurements make a barline."""
    gap = st.gap
    a, b = int(round(st.top)), int(round(st.bottom))
    x0, x1 = max(int(st.x0), 3), min(int(st.x1), gray.shape[1] - 4)
    if x1 - x0 < 3 or b <= a:
        return []
    line, paper = line_darkness(gray, st)
    contrast = max(paper - line, 1.0)
    d = max(2, int(round(FAINT_SIDE_GAPS * gap)))
    # Rows on a staff line are dark in every column, so only the rows between the lines are judged
    keep = np.ones(b - a + 1, bool)
    for y in st.lines:
        r = int(round(y)) - a
        keep[max(0, r - 1):r + 2] = False
    if keep.sum() < 3:
        return []

    def strength(y0: int, y1: int, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per row and column: the darker of columns x, x+1, and how much darker that is than the paper beside."""
        rows = gray[max(0, y0):max(0, y1)].astype(np.float32)
        two = np.minimum(rows[:, xs], rows[:, np.minimum(xs + 1, gray.shape[1] - 1)])
        sides = np.minimum(rows[:, np.maximum(xs - d, 0)], rows[:, np.minimum(xs + 1 + d, gray.shape[1] - 1)])
        return two, (sides - two) / contrast

    xs = np.arange(x0, x1)
    two, c = strength(a, b + 1, xs)
    hit = (c >= FAINT_HIT)[keep]
    cover = hit.mean(axis=0)
    chosen: list[int] = []
    for i in np.argsort(-cover, kind="stable"):
        if cover[i] < FAINT_COVER:
            break
        x = x0 + int(i)
        # One candidate per half gap, none within 0.7 gaps of a found or dropped barline, and none within MIN_CELL gaps
        # of the staff's ends
        if (any(abs(x - y) <= 0.5 * gap for y in chosen)
                or any(p - 0.7 * gap <= x <= q + 0.7 * gap for p, q in [*bars, *dropped])
                or x - st.x0 < MIN_CELL * gap or st.x1 - x < MIN_CELL * gap):
            continue
        chosen.append(x)
    faint_mask = gray[a:b + 1] < fthr
    out = []
    for x in chosen:
        i = x - x0
        col = hit[:, i]
        runs = np.diff(np.flatnonzero(np.concatenate(([True], ~col, [True])))) - 1
        level = float(np.median(two[keep][:, i]))

        def share(y0: float, y1: float) -> float:
            _, s = strength(int(round(y0)), int(round(y1)), np.array([x]))
            return float((s >= FAINT_HIT).mean()) if s.size else 0.0

        f = {
             # Longest unbroken stretch of counted rows: a line is unbroken, a column of digits is not
             "run": float(runs.max()) / len(col),
             # Share of the staff's rows on the faint-ink mask, in the fuller of its two columns
             "fcover": float(max(faint_mask[:, x].mean(), faint_mask[:, x + 1].mean())),
             # Darkness against the paper, in units of the staff lines' darkness
             "dark": (paper - level) / contrast,
             # Share of ink on the columns just beside it: a cursor or shading is wider than a line
             "side": max(side_ink(gray, a, b, x - d, level, paper, fthr),
                         side_ink(gray, a, b, min(x + 1 + d, gray.shape[1] - 1), level, paper, fthr)),
             # Rhythm stems run on below the staff, a playhead line above it
             "below": share(b + 0.5 * gap, b + 2.0 * gap),
             "above": share(a - 2.0 * gap, a - 0.5 * gap),
             # A barline crosses the notation staff too; -1 when the system has none
             "notation": share(notation[0], notation[1]) if notation is not None else -1.0,
             "colored": float(colored_plate[a:b + 1, x:x + 2].any(axis=1).mean()),
             # A barline reaches both outer lines; a stem drawn inside the staff stops short of one
             "ends": min(share(st.lines[0] + 2, st.lines[1] - 1), share(st.lines[-2] + 2, st.lines[-1] - 1))}
        out.append((x, f))
    return sorted(out, key=lambda xf: xf[0])


# A faint line is a barline when its counted rows run unbroken over FAINT_RUN of the staff and fill FAINT_ENDS of both
# the top and the bottom line space. One reaching only FAINT_UNSURE_RUN and FAINT_UNSURE_ENDS is left out with a notice
FAINT_RUN, FAINT_ENDS = 0.6, 0.7
FAINT_UNSURE_RUN, FAINT_UNSURE_ENDS = 0.5, 0.6
# Any line is turned down when the columns beside it hold FAINT_SIDE or more ink (side_ink), when it runs on below or
# above the staff for FAINT_BELOW or FAINT_ABOVE of that stretch, or when its color covers FAINT_COLORED of the staff
FAINT_SIDE, FAINT_BELOW, FAINT_ABOVE, FAINT_COLORED = 0.3, 0.5, 0.5, 0.5
# ...or when it crosses less than FAINT_NOTATION of the notation staff above the tab
FAINT_NOTATION = 0.9
# ...or when it is FAINT_DARK times as dark as the staff lines or darker
FAINT_DARK = 1.6


def faint_rule(f: dict) -> str:
    """The verdict on a faint candidate: "bar" for a barline, "unsure" for a line that may be one, "" otherwise.

    A line dark enough for BAR_COVER was turned down by the cover rule's other tests, so it is left alone, and so is a
    stroke much darker than the staff lines that still misses BAR_COVER: it is broken, a stem or digits, not faint."""
    if not (f["fcover"] < BAR_COVER and f["dark"] < FAINT_DARK and f["side"] < FAINT_SIDE
            and f["below"] < FAINT_BELOW and f["above"] < FAINT_ABOVE and f["colored"] < FAINT_COLORED
            and (f["notation"] < 0 or f["notation"] >= FAINT_NOTATION)):
        return ""
    if f["run"] >= FAINT_RUN and f["ends"] >= FAINT_ENDS:
        return "bar"
    return "unsure" if f["run"] >= FAINT_UNSURE_RUN and f["ends"] >= FAINT_UNSURE_ENDS else ""


def faint_bars(gray: np.ndarray, st, bars: list[tuple[int, int]], notation: tuple[float, float] | None, fthr: float,
               colored_plate: np.ndarray, dropped: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], set[int]]:
    """Faint barlines the cover rule missed, and the columns of faint lines that may be barlines."""
    judged = [(x, faint_rule(f)) for x, f in faint_candidates(gray, st, bars, notation, fthr, colored_plate, dropped)]
    return [(x, x + 1) for x, v in judged if v == "bar"], {x for x, v in judged if v == "unsure"}


def opening_line(struct: np.ndarray, st) -> bool:
    """True when a solid line crosses the whole tab staff within one line gap of the staff's left end.

    Only solid ink counts: the soft gray edge of a panel or of the frame is faint."""
    a, b = int(round(st.top)), int(round(st.bottom))
    x0, x1 = int(st.x0), int(st.x0 + INSIDE_GAPS * st.gap) + 1
    band = struct[a:b + 1, x0:x1]
    return band.size > 0 and bool((band.mean(axis=0) >= BAR_COVER).any())


# A system's first cell is its clef, key and time signature alone when the barline closing it is at least
# HEADER_BAR_GAPS line gaps wide (a start-repeat or heavy barline; a plain one is about 0.1, a thin double one 0.4)
# and the cell is narrower than HEADER_SHARE of the system's median measure and than HEADER_MAX_GAPS line gaps
HEADER_BAR_GAPS, HEADER_SHARE, HEADER_MAX_GAPS = 0.55, 0.85, 11.0


def header_cell(edges: list[tuple[int, int, str]], bars: list[tuple[int, int]], repeats: list[int], x0: int,
                gap: float) -> bool:
    """True when a system's first cell, opening at the staff start, holds only the clef, key and time signature:
    it closes on a wide barline that is no end-repeat sign and is narrow beside the system's measures."""
    fulls = [b - a for a, b, kind in edges if kind == "full"]
    if len(fulls) < 2 or len(bars) < 2 or edges[0][2] != "full" or edges[0][0] > x0 + 2:
        return False
    p, q = bars[1]
    width = edges[0][1] - edges[0][0]
    return (q - p >= HEADER_BAR_GAPS * gap and p not in repeats
            and width < min(HEADER_SHARE * float(np.median(fulls)), HEADER_MAX_GAPS * gap))


def staff_ink(c: Cell, skip: int) -> tuple[float, float]:
    """Solid ink on a cell's tab staff and on the notation staff above it, in square line gaps, over the columns
    from `skip` past its left edge to its right barline."""
    # The ink box runs from a margin left of the left edge to the right barline's first column
    start = max(0, c.x0 - (c.x1 + 1 - c.ink.shape[1]) + skip)
    end = c.ink.shape[1] - 2
    bottom = int(round(2 * c.mid_y)) - (c.y0 + c.ref_y) - c.y0
    # The tab staff starts half a line gap above its top line, so a fret digit sitting on that line counts whole
    tab_top = max(0, c.ref_y - int(round(c.gap / 2)))
    tab = c.ink[tab_top:bottom + 1, start:end].sum()
    # A system cut by the panel top starts above the cell box; its notation rows then start at the box top
    notation = c.ink[max(0, c.top_y):tab_top, start:end].sum()
    return tab / (c.gap * c.gap), notation / (c.gap * c.gap)


# Past the first HEADER_SKIP_GAPS line gaps of a header cell, where the clef stands, its tab staff holds less than
# HEADER_TAB_INK square line gaps of ink; a narrow first measure with fret digits there is a pickup
HEADER_SKIP_GAPS, HEADER_TAB_INK = 4.0, 0.08


def bare_header(c: Cell) -> bool:
    """True when a cell has no fret digits on its tab staff right of where a clef stands."""
    return staff_ink(c, int(HEADER_SKIP_GAPS * c.gap))[0] < HEADER_TAB_INK


def mark_header(c: Cell) -> None:
    """Flags a system's first cell shaped like a header: a header when its tab staff is bare past the clef,
    otherwise a possible header that is kept as a measure."""
    c.header = bare_header(c)
    c.maybe_header = not c.header


# A run of blank rows this many line gaps tall ends the marks that belong to a system; chord names can sit
# 2.5 gaps of blank paper above the measure numbers
BLANK_GAPS = 3.0


def print_band(marks: np.ndarray, top: float, bottom: float, gap: float, lo: int, hi: int) -> tuple[int, int]:
    """Rows to print for a system: from its top line up to the last row of marks before a blank stretch, and
    the same below its bottom line, kept within lo..hi and within the print reach."""
    rows = marks.any(axis=1)

    def reach(start: int, step: int, limit: int) -> int:
        last, blank, y = start, 0, start
        while (y > limit) if step < 0 else (y < limit):
            y += step
            if rows[y]:
                last, blank = y, 0
            else:
                blank += 1
                if blank > BLANK_GAPS * gap:
                    break
        return last

    y0 = reach(int(top), -1, max(lo, int(top - PRINT_ABOVE_GAPS * gap)))
    y1 = reach(int(bottom), 1, min(hi - 1, int(bottom + PRINT_BELOW_GAPS * gap)))
    pad = max(2, int(0.3 * gap))
    # Digits sit centered on the outer lines, so the band always keeps a full margin of paper past them
    y0 = min(y0 - pad, int(np.floor(top - PRINT_MARGIN_GAPS * gap)))
    y1 = max(y1 + 1 + pad, int(np.ceil(bottom + PRINT_MARGIN_GAPS * gap)) + 1)
    return max(lo, y0), min(hi, y1)


def bar_through(struct: np.ndarray, x0: int, x1: int, top: float, bottom: float) -> bool:
    """True when some column of a barline is ink on nearly every row from top to bottom."""
    cols = struct[int(round(top)):int(round(bottom)) + 1, max(0, x0 - 1):x1 + 2]
    return bool(cols.size and cols.mean(axis=0).max() >= JOINED_COVER)


def joined_staffs(out: list, lone: list, struct: np.ndarray) -> tuple[list, list]:
    """Systems whose barlines run on down through the space below into the next system are one system of two staffs
    (two parts played together, or a second staff added under the first): one measure per barline column, printed
    across both. The upper staff's own barlines may be lost where the line runs on below it, so the columns come
    from both systems' barlines."""
    k = 0
    while k + 1 < len(out):
        u, l = out[k], out[k + 1]
        cands = sorted(set(u.bars) | set(l.bars))
        through = [b for b in cands if bar_through(struct, b[0], b[1], u.system.tab.top, l.system.tab.bottom)]
        # Columns found on both staffs count once
        bars: list = []
        for b in through:
            if bars and b[0] - bars[-1][1] <= JOINED_SLACK_PX:
                continue
            bars.append(b)
        if len(bars) >= JOINED_MIN_BARS and len(bars) >= JOINED_SHARE * max(len(l.bars), 1):
            out[k] = ParsedSystem(u.system, bars, u.y0, l.y1, repeat_end=u.repeat_end or l.repeat_end,
                                  band=(u.band[0], l.band[1]), repeat_bars=u.repeat_bars + l.repeat_bars,
                                  colored_bars=u.colored_bars | l.colored_bars,
                                  unsure_bars=u.unsure_bars | l.unsure_bars)
            del out[k + 1]
            lone = lone[:k + 1] + lone[k + 2:]
            continue
        k += 1
    return out, lone


def parse_view(vi: int, plate: np.ndarray, rect: tuple[int, int, int, int], polarity: str, thr: float, fthr: float,
               gap: float, print_rect: tuple[int, int, int, int] | None = None,
               hidden_bars: bool = False, whole_lines: bool = True) -> list[ParsedSystem]:
    g = to_gray(plate, polarity)
    rx0, ry0, rx1, ry1 = rect
    sub = g.copy()
    # Only the panel counts; footage outside it is painted paper
    paper = 255
    sub[:ry0] = paper
    sub[ry1:] = paper
    sub[:, :rx0] = paper
    sub[:, rx1:] = paper
    # A staff with no rows inside the print area is not on the paper, such as the strings of a fretboard drawn
    # below it
    px0, py0, px1, py1 = print_rect or rect
    syss = [s for s in systems_in(sub, whole_lines=whole_lines) if s.top < py1 and s.tab.bottom > py0]
    # Past the panel the plate is painted paper, so a border at the panel's edge passes for a staff line there; the
    # plate as it is shows what lies past it
    for s in syss:
        s.tab = without_border(g, s.tab)
    # A "tab staff" spaced much closer than the view's widest-spaced one is beams or text lines, not a staff
    widest = max((s.tab.gap for s in syss), default=0.0)
    syss = [s for s in syss if s.tab.gap >= CLOSE_STAFF_SHARE * widest]
    # A lone staff of 5 lines or fewer, spaced like this view's notation staffs and closer than its tab staffs, is
    # a notation staff whose tab is below the panel
    notation_gaps = [s.notation.gap for s in syss if s.notation is not None]
    lone = [lone_notation(s, notation_gaps, widest) for s in syss]
    colored_plate = colored(plate)
    # Where a cursor or highlight tinted the page, ink can't be compared between sightings
    tint = cv2.dilate(colored_plate.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    # Measure numbers may be printed in color, so digit-sized colored marks count; larger colored areas are cursors
    marks = ink_map(sub, fthr, gap) & ~(colored_plate & ~digit_sized(colored_plate, gap))
    dark = sub < thr
    faint_struct = sub < fthr
    ink = ink_map(sub, thr, gap, plate)
    faint = ink_map(sub, fthr, gap) & ~ink & gray_pixels(plate)
    faint &= ~cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    # Marks that may be printed with a system: everything visible on the unpainted plate inside the print area,
    # colored marks left out
    print_marks = np.zeros_like(ink)
    print_marks[py0:py1, px0:px1] = ink_map(to_gray(plate, polarity), fthr, gap, plate)[py0:py1, px0:px1]
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
        notation = (s.notation.lines[1], s.notation.lines[-2]) if s.notation else None
        bars = barlines(sub, thr, fthr, st.top, st.bottom, st.gap, max(rx0, st.x0 - 2), min(rx1, st.x1 + 3), notation,
                        hidden_bars)
        bar_sub = sub
        # The panel's edge search stops at a double or thick closing barline, so a staff that runs to the panel's
        # right edge may end on a barline just past it: those columns are read from the plate as it is
        if st.x1 >= rx1 - 1:
            x_end = min(g.shape[1], rx1 + int(EDGE_BAR_GAPS * st.gap))
            bar_sub = sub.copy()
            bar_sub[ry0:ry1, rx1:x_end] = g[ry0:ry1, rx1:x_end]
            found = barlines(bar_sub, thr, fthr, st.top, st.bottom, st.gap, max(rx0, st.x0 - 2), x_end, notation)
            closing = [bar for bar in found if bar[1] >= rx1 and bar[1] - bar[0] >= EDGE_BAR_MIN * st.gap
                       and paper_after(g, st, bar, fthr)]
            if closing:
                bars = [bar for bar in bars if bar[1] < closing[0][0]] + closing[:1]
        kept = drawn_bars(bar_sub, st, bars)
        dropped = [bar for bar in bars if bar not in kept]
        bars = kept
        unsure: set[int] = set()
        # Glide mosaics read their barlines with their own looser rule
        if not hidden_bars:
            light, unsure = faint_bars(bar_sub, st, bars, notation, fthr, colored_plate, dropped)
            bars = sorted(bars + light)
        repeats = end_repeats(dark, st, bars)
        # Glide mosaics are left out: a colored playhead caught over a barline there says nothing of the page
        tinted = set() if hidden_bars else colored_bars(faint_struct, colored_plate, st, bars)
        out.append(ParsedSystem(s, bars, y0, y1, repeat_end=bool(repeats), repeat_bars=repeats, colored_bars=tinted,
                                unsure_bars=unsure))
    # Print bands: each system's marks, up to halfway to the systems above and below
    for si, ps in enumerate(out):
        s, st = ps.system, ps.system.tab
        lo = py0 if si == 0 else max(py0, int((syss[si - 1].tab.bottom + s.top) / 2))
        hi = py1 if si == len(out) - 1 else min(py1, int((st.bottom + syss[si + 1].top) / 2))
        ps.band = print_band(print_marks[:, st.x0:st.x1], s.top, st.bottom, st.gap, lo, hi)
    out, lone = joined_staffs(out, lone, faint_struct)
    most_lines = max((len(ps.system.tab.lines) for ps in out), default=0)
    for si, ps in enumerate(out):
        st = ps.system.tab
        # The plate reaches a few line gaps below its lowest staff unless the frame ends first, so a staff this
        # close to its bottom is cut by the frame's edge: it shows only its upper lines and passes for a staff of
        # fewer strings. Its measures print from another screen that shows them whole
        cut_off = len(st.lines) < most_lines and st.bottom + CUT_STAFF_GAPS * st.gap >= g.shape[0]
        edges: list[tuple[int, int, str]] = []
        xs = [b for b in ps.bars]
        # How fully the lines show inside the system's whole measures: faint lines on a small frame show in fewer
        # columns, so an edge cell is held to that. A system with no whole measure keeps its edge cells
        lines_x = line_columns(g, st)
        whole = [lines_x[a:b].mean() for (a, _), (b, _) in zip(xs, xs[1:]) if b - a >= MIN_CELL * st.gap]
        drawn = float(np.median(whole)) if whole else 0.0
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
        # Glides read barlines more loosely, so their first cell isn't judged
        header = not hidden_bars and header_cell(edges, xs, ps.repeat_bars, st.x0, st.gap)
        ends_inside = st.x1 <= rx1 - INSIDE_GAPS * st.gap
        for ci, (a, b, kind) in enumerate(edges):
            # The first measure of a system takes in the margin to its left, where some players print the
            # measure number; two pages that differ only there are different pages
            first = ci == 0 and a <= st.x0 + 2
            left = int(max(rx0, a - 1.5 * st.gap)) if first else a
            # The printed first measure takes in the same left margin, kept inside the print area
            print_x0 = int(max(px0, a - 1.5 * st.gap)) if first else a
            box_ink = ink[ps.y0:ps.y1, left:b + 1].copy()
            box_tint = tint[ps.y0:ps.y1, left:b + 1].copy()
            # Faint marks count only above the system, where light-gray measure numbers are printed; inside
            # the staffs they are mostly cursor residue
            box_faint = faint[ps.y0:ps.y1, left:b + 1].copy()
            box_faint[max(0, int(ps.system.top) - ps.y0):] = False
            # Content of a lead or tail cell that is only staff lines is no measure
            if kind != "full" and box_ink.sum() < 0.1 * st.gap * st.gap:
                continue
            # Nor is one where most of the staff lines stop: footage beside the paper that the staff was followed into
            if kind != "full" and lines_x[a:b].mean() < EDGE_LINES_SHARE * drawn:
                continue
            top = (ps.system.notation or st).top
            top_gap = (ps.system.notation or st).gap
            crop = number_crop(sub, marks, top, top_gap, a, ry0)
            complete = kind == "full" or (kind == "lead" and starts_inside) or (kind == "tail" and ends_inside)
            cell = Cell(vi, si, len(ps.cells), a, b, kind, box_ink, box_faint, ps.y0, crop,
                        gap=st.gap, ref_y=int(round(st.top)) - ps.y0, mid_y=(st.top + st.bottom) / 2,
                        tint=box_tint,
                        complete=complete and not cut_off, band=ps.band, print_x0=print_x0,
                        notation_only=lone[si], top_y=int(round(ps.system.top)) - ps.y0,
                        colored_bar=a in ps.colored_bars or b in ps.colored_bars,
                        faint_bar=any(a - 1 <= x <= b + 1 for x in ps.unsure_bars))
            if header and ci == 0:
                mark_header(cell)
            ps.cells.append(cell)
    return out
