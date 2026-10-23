"""Find staffs (sets of evenly spaced horizontal lines) in a frame, light-on-dark or dark-on-light."""
from dataclasses import dataclass

import cv2
import numpy as np

# Lines must differ from the pixels just above and below them by this many gray levels
LINE_CONTRAST = 18
# A staff has 4 to 6 lines; banjo tab has 5, bass 4, guitar 6
MIN_LINES, MAX_LINES = 4, 6
# Gaps between neighboring lines of one staff agree within this share
GAP_TOLERANCE = 0.18
# Widest line spacing accepted in a crop, in pixels (a full frame allows a twelfth of its height)
MAX_GAP_PX = 60
# A row group taller than this share of the typical distance between row groups is two lines run together by notes
# filling the space between them; a line thickened by notes stays thinner than that
MERGED_SHARE = 0.9
# A row group taller than THICK_GROUP of the typical distance between row groups, whose fullest row covers PEAK_COVER
# of its width, holds a drawn line and ties or notes along it; the line's center is taken from the group's rows that
# cover at least LINE_PEAK of its fullest row. Strings in footage are thick rows that cover far less
THICK_GROUP, PEAK_COVER, LINE_PEAK = 0.5, 0.6, 0.75
# A line covering less than this share of its staff's typical line is a ledger line or beam row
WEAK_LINE = 0.5
# A weak line at a staff's edge is still a staff line, hidden under rows of digits, when it covers this share of the
# typical line, spans the staff's other lines, and is missing where digits sit on it. Digits reach this many line
# gaps above and below the line they sit on
HIDDEN_LINE, DIGIT_REACH = 0.3, 0.35
# In a run of more than six evenly spaced lines, a staff's own lines cover at least this share of the strongest
RUN_LINE = 0.8
# A staff's outer line has paper just past it, BEYOND_PX pixels or BEYOND_GAPS line gaps out, whichever is more; a
# panel's border can sit one even step past the staff, with footage right beyond it. A band can end in footage a third
# of a gap below its bottom line, so the test stays close to the line
BEYOND_GAPS, BEYOND_PX = 0.15, 3
# ...paper being within this many gray levels of the paper between the staff's lines, or of the gap next to the line
BEYOND_TOL = 40
# Text tab's lines are rows of typed dashes: dashes shorter than DASH_GAPS line gaps, with less paper between them than
# each dash is long, cover more than DASH_COVER of the line. Paper is within DASH_PAPER gray levels of the paper
# between the staff's lines
DASH_GAPS, DASH_COVER, DASH_PAPER = 1.0, 0.3, 40


@dataclass
class Staff:
    lines: list[float]      # y of each line, top to bottom
    x0: int
    x1: int
    gap: float               # line spacing
    polarity: str            # "dark" lines on light paper, or "light" lines on dark paper
    dashed: bool = False     # lines typed as dashes (text tab), not drawn

    @property
    def top(self) -> float:
        return self.lines[0]

    @property
    def bottom(self) -> float:
        return self.lines[-1]


def thin_response(gray: np.ndarray, polarity: str) -> np.ndarray:
    """Pixels on a thin horizontal stroke: darker (or lighter) than both the pixels 2-3 rows above and below.

    Thick marks such as rhythm beams and solid boxes fail the test, so they can't pose as staff lines.
    """
    g = gray.astype(np.int16)
    if polarity == "light":
        g = 255 - g
    best = np.zeros_like(g)
    for k in (2, 3):
        up = np.zeros_like(g)
        down = np.zeros_like(g)
        up[k:] = g[:-k] - g[k:]
        down[:-k] = g[k:] - g[:-k]
        best = np.maximum(best, np.minimum(up, down))
    return best > LINE_CONTRAST


def line_rows(gray: np.ndarray, polarity: str, min_cover: int) -> list[tuple[float, int, int, int, int]]:
    """Horizontal lines as (y, x0, x1, thickness, coverage): row groups where thin strokes cover at least `min_cover` columns.

    Coverage rather than run length, because fret digits cut tab lines into short pieces.
    """
    resp = cv2.dilate(thin_response(gray, polarity).astype(np.uint8), np.ones((3, 1), np.uint8))
    counts = resp.sum(axis=1)
    rows = np.flatnonzero(counts >= min_cover)
    out = []
    i = 0
    groups = []
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1] <= rows[j] + 1:
            j += 1
        groups.append((int(rows[i]), int(rows[j]) + 1))
        i = j + 1
    centers = [(a + b) / 2 for a, b in groups]
    steps = [q - p for p, q in zip(centers, centers[1:])]
    tallest = MERGED_SHARE * float(np.median(steps)) if steps else float(len(counts))
    thick = THICK_GROUP * float(np.median(steps)) if steps else float(len(counts))
    for a, b in groups:
        for a, b in split_rows(counts, a, b, tallest):
            cols = np.flatnonzero(resp[a:b].any(axis=0))
            weights = counts[a:b].astype(float)
            if b - a > thick and weights.max() >= PEAK_COVER * (np.ptp(cols) + 1):
                # Ties or note rows running along a line joined its row group; the line itself is the group's peak
                weights[weights < LINE_PEAK * weights.max()] = 0
            y = float((np.arange(a, b) * weights).sum() / weights.sum())
            x0, x1 = np.percentile(cols, [2, 98])
            out.append((y, int(x0), int(x1) + 1, b - a, int(counts[a:b].max())))
    return out


def split_rows(counts: np.ndarray, a: int, b: int, tallest: float) -> list[tuple[int, int]]:
    """A row group taller than `tallest` rows, cut between its peaks: two lines whose space is filled with notes."""
    if b - a <= tallest:
        return [(a, b)]
    c = counts[a:b].astype(float)
    peaks = [k for k in range(len(c)) if c[k] == c[max(0, k - 3):k + 4].max() and c[k] >= 0.8 * c.max()]
    # Neighboring rows of one flat top count as one peak
    tops = [k for n, k in enumerate(peaks) if n == 0 or k - peaks[n - 1] > 3]
    if len(tops) < 2:
        return [(a, b)]
    cuts = [a] + [a + (p + q) // 2 for p, q in zip(tops, tops[1:])] + [b]
    return list(zip(cuts, cuts[1:]))


def in_step(lines: list, firm: list[bool], k: int) -> bool:
    """True when weak line k sits one even step from the strong lines on both sides, and the same step repeats past
    one of them: a staff line half hidden under stacked fret digits, not a curve or beam between two lines."""
    ys = [ln[0] for ln, ok in zip(lines, firm) if ok]
    y = lines[k][0]
    above = [v for v in ys if v < y]
    below = [v for v in ys if v > y]
    if not above or not below:
        return False
    d1, d2 = y - above[-1], below[0] - y

    def near(a: float, b: float) -> bool:
        return abs(a - b) <= GAP_TOLERANCE * max(a, b)
    return near(d1, d2) and ((len(above) >= 2 and near(above[-1] - above[-2], d1))
                             or (len(below) >= 2 and near(below[1] - below[0], d2)))


def weak_end(gray: np.ndarray, polarity: str, lines: list, chain: list[int], end: int, strong: float) -> bool:
    """True when the chain's line at `end` (0 or -1) is a ledger line or beam row rather than a staff line hidden
    under digits: it covers less than WEAK_LINE of the typical line, and also covers less than HIDDEN_LINE, or
    doesn't span the median extent of the chain's other lines, or isn't missing just where digits cover it. A chain
    longer than a staff keeps none of its weak end lines."""
    ln = lines[chain[end]]
    if ln[4] >= WEAK_LINE * strong:
        return False
    if len(chain) > MAX_LINES:
        return True
    inner = [lines[c] for c in (chain[1:] if end == 0 else chain[:-1])]
    x0, x1 = int(np.median([v[1] for v in inner])), int(np.median([v[2] for v in inner]))
    across = ln[1] <= x0 + 0.05 * (x1 - x0) and ln[2] >= x1 - 0.05 * (x1 - x0)
    gap = abs(ln[0] - inner[0 if end == 0 else -1][0])
    return ln[4] < HIDDEN_LINE * strong or not across or not under_digits(gray, polarity, ln[0], x0, x1, gap)


def under_digits(gray: np.ndarray, polarity: str, y: float, x0: int, x1: int, gap: float) -> bool:
    """True when a line at row y is missing mostly where ink lies just above or below it, as digits sitting on it
    hide it, and shows mostly where there is none. A ledger row or a string in the footage shows where noteheads or
    fingers sit."""
    r, reach = int(round(y)), max(3, int(DIGIT_REACH * gap))
    a, b = max(0, r - reach), min(gray.shape[0], r + reach + 1)
    if b - a < 5 or x1 <= x0:
        return False
    g = gray[a:b, x0:x1].astype(np.int16)
    if polarity == "light":
        g = 255 - g
    shown = thin_response(gray[max(0, r - 4):r + 5, x0:x1], polarity)[min(r, 4) - 1:min(r, 4) + 2].any(axis=0)
    # Paper is the light end of the band; digits can cover most of it
    ink = g < np.percentile(g, 90) - 60
    # Rows off the line itself, where a digit's strokes are and the line's soft edge isn't
    off = np.ones(b - a, bool)
    off[max(0, r - a - 1):r - a + 2] = False
    near = ink[off].any(axis=0)
    missing = ~shown
    return bool(missing.any() and shown.any() and near[missing].mean() > 0.5 and near[shown].mean() < 0.5)


def paper_beyond(gray: np.ndarray, chain: list, side: int) -> bool:
    """True when the row just past the chain's last (side 1) or first (side -1) line looks like the paper between its
    lines, or like the gap next to that line. A row outside the image counts as paper."""
    ys = [ln[0] for ln in chain]
    gap = (ys[-1] - ys[0]) / (len(ys) - 1)
    x0, x1 = int(np.median([ln[1] for ln in chain])), int(np.median([ln[2] for ln in chain]))
    y = int(round((ys[-1] if side > 0 else ys[0]) + side * max(BEYOND_PX, BEYOND_GAPS * gap)))
    if not 0 <= y < gray.shape[0] or x1 <= x0:
        return True
    mids = [float(np.median(gray[int(round((a + b) / 2)), x0:x1])) for a, b in zip(ys, ys[1:])]
    # A highlight can darken the gap next to the end line, so the row past the line may match that gap, not the paper
    paper, nearest = float(np.median(mids)), mids[-1] if side > 0 else mids[0]
    beyond = float(np.median(gray[y, x0:x1]))
    return min(abs(beyond - paper), abs(beyond - nearest)) <= BEYOND_TOL


def without_border(gray: np.ndarray, st: Staff) -> Staff:
    """The staff without outer lines that have no paper past them in `gray`: a panel border one step past it."""
    rows = [(y, st.x0, st.x1) for y in st.lines]
    while len(rows) > MIN_LINES and not paper_beyond(gray, rows, 1):
        rows.pop()
    while len(rows) > MIN_LINES and not paper_beyond(gray, rows, -1):
        rows.pop(0)
    if len(rows) == len(st.lines):
        return st
    ys = [y for y, _, _ in rows]
    return Staff(ys, st.x0, st.x1, float(np.median(np.diff(ys))), st.polarity)


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and end (exclusive) of each run of True in a 1-D mask."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def dashed(gray: np.ndarray, st: Staff) -> bool:
    """True when most of a staff's lines are rows of typed dashes: short dashes, each with paper either side that is
    narrower than the dash. A drawn line runs on between the digits on it, a dashed or dotted one leaves more paper
    than ink, and the lines of a chord diagram are broken by its strings, which are ink."""
    a = max(0, int(st.top) - 3)
    resp = thin_response(gray[a:int(st.bottom) + 4, st.x0:st.x1], st.polarity)
    g = gray[:, st.x0:st.x1].astype(np.int16)
    paper = np.median([np.median(g[int(round((p + q) / 2))]) for p, q in zip(st.lines, st.lines[1:])])
    typed = 0
    for y in st.lines:
        r = resp[max(0, int(round(y)) - a - 1):int(round(y)) - a + 2].any(axis=0)
        blank = ~r & (np.abs(g[int(round(y))] - paper) <= DASH_PAPER)
        b0, b1 = runs(blank)
        before, after = dict(zip(b1, b1 - b0)), dict(zip(b0, b1 - b0))
        d0, d1 = runs(r)
        dash = d1 - d0
        ok = (dash < DASH_GAPS * st.gap) & np.array([before.get(s, 1e9) < w and after.get(e, 1e9) < w
                                                      for s, e, w in zip(d0, d1, dash)], bool)
        typed += dash[ok].sum() > DASH_COVER * len(r)
    return typed > len(st.lines) / 2


def find_staffs(gray: np.ndarray, min_len: int | None = None, keep_dashed: bool = False) -> list[Staff]:
    """Every staff of 4 to 6 evenly spaced lines that share a horizontal extent. Text tab's dashed lines are left
    out, unless `keep_dashed`.

    Lines are taken in order down the frame; a staff is a run of neighbors with equal gaps, so a
    line between two staff lines can't be skipped over to make a wider-spaced staff.
    """
    h, w = gray.shape
    min_len = min_len or max(w // 8, 20)
    max_gap = max(h / 12, MAX_GAP_PX)
    staffs = []
    for polarity in ("dark", "light"):
        lines = line_rows(gray, polarity, min_len)
        # A short line between strong ones (a bend curve, a beam, a ledger line) is not a staff line
        firm = [ln[4] >= WEAK_LINE * max([lines[m][4] for m in (k - 1, k + 1) if 0 <= m < len(lines)] or [0])
                  for k, ln in enumerate(lines)]
        lines = [ln for k, ln in enumerate(lines) if firm[k] or in_step(lines, firm, k)]
        i = 0
        while i < len(lines):
            chain = [i]
            j = i + 1
            while j < len(lines) and len(chain) <= MAX_LINES + 2:
                prev, cand = lines[chain[-1]], lines[j]
                gap = cand[0] - prev[0]
                ov = min(prev[2], cand[2]) - max(prev[1], cand[1])
                if ov < 0.7 * min(prev[2] - prev[1], cand[2] - cand[1]) or gap > max_gap:
                    break
                if len(chain) >= 2:
                    # The mean step so far: notes along a line shift its center by a pixel or two
                    ref = (lines[chain[-1]][0] - lines[chain[0]][0]) / (len(chain) - 1)
                    if abs(gap - ref) > GAP_TOLERANCE * ref:
                        # A row of fret digits can pose as a weak line between two staff lines; step over it
                        strong = np.median([lines[c][4] for c in chain])
                        if gap < ref * (1 - GAP_TOLERANCE) and cand[4] < 0.7 * strong:
                            j += 1
                            continue
                        break
                chain.append(j)
                j += 1
            # Ledger lines and beam rows next to a staff are short; drop weak lines from the ends
            strong = np.median([lines[c][4] for c in chain])
            while len(chain) > MIN_LINES and weak_end(gray, polarity, lines, chain, -1, strong):
                chain.pop()
            while len(chain) > MIN_LINES and weak_end(gray, polarity, lines, chain, 0, strong):
                chain.pop(0)
            while len(chain) > MIN_LINES and not paper_beyond(gray, [lines[c] for c in chain], 1):
                chain.pop()
            while len(chain) > MIN_LINES and not paper_beyond(gray, [lines[c] for c in chain], -1):
                chain.pop(0)
            if len(chain) < MIN_LINES:
                i += 1
                continue
            i = chain[-1] + 1
            if len(chain) > MAX_LINES:
                # Ledger and beam rows can carry on a staff's even spacing; its own lines are the ones drawn across
                # the whole staff. More than six such lines is a pattern (planks, blinds, ruled paper), not a staff.
                top = max(lines[c][4] for c in chain)
                run = [c for c in chain if lines[c][4] >= RUN_LINE * top]
                if not (MIN_LINES <= len(run) <= MAX_LINES and run == list(range(run[0], run[-1] + 1))):
                    continue
                chain = run
            ys = [lines[c][0] for c in chain]
            gap = float(np.median(np.diff(ys)))
            # The thinnest line: notes and ties drawn along a line thicken it, beams and boxes are thick throughout
            thick = float(min(lines[c][3] for c in chain))
            if thick > max(6, 0.35 * gap + 3):
                continue
            x0 = int(np.median([lines[c][1] for c in chain]))
            x1 = int(np.median([lines[c][2] for c in chain]))
            st = Staff(ys, x0, x1, gap, polarity)
            st.dashed = dashed(gray, st)
            if keep_dashed or not st.dashed:
                staffs.append(st)
    staffs.sort(key=lambda s: s.top)
    return staffs
