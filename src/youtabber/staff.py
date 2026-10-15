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
# A line covering less than this share of its staff's typical line is a ledger line or beam row
WEAK_LINE = 0.5


@dataclass
class Staff:
    lines: list[float]      # y of each line, top to bottom
    x0: int
    x1: int
    gap: float               # line spacing
    polarity: str            # "dark" lines on light paper, or "light" lines on dark paper

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
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1] <= rows[j] + 1:
            j += 1
        band = resp[rows[i]:rows[j] + 1].any(axis=0)
        cols = np.flatnonzero(band)
        weights = counts[rows[i]:rows[j] + 1].astype(float)
        y = float((np.arange(rows[i], rows[j] + 1) * weights).sum() / weights.sum())
        x0, x1 = np.percentile(cols, [2, 98])
        out.append((y, int(x0), int(x1) + 1, int(rows[j] - rows[i] + 1), int(counts[rows[i]:rows[j] + 1].max())))
        i = j + 1
    return out


def find_staffs(gray: np.ndarray, min_len: int | None = None) -> list[Staff]:
    """Every staff of 4 to 6 evenly spaced lines that share a horizontal extent.

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
        lines = [ln for k, ln in enumerate(lines)
                 if ln[4] >= WEAK_LINE * max([lines[m][4] for m in (k - 1, k + 1) if 0 <= m < len(lines)] or [0])]
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
                    ref = lines[chain[1]][0] - lines[chain[0]][0]
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
            while len(chain) > MIN_LINES and lines[chain[-1]][4] < WEAK_LINE * strong:
                chain.pop()
            while len(chain) > MIN_LINES and lines[chain[0]][4] < WEAK_LINE * strong:
                chain.pop(0)
            if len(chain) < MIN_LINES:
                i += 1
                continue
            i = chain[-1] + 1
            # More than six evenly spaced lines is a pattern (planks, blinds, ruled paper), not a staff
            if len(chain) > MAX_LINES:
                continue
            ys = [lines[c][0] for c in chain]
            gap = float(np.median(np.diff(ys)))
            thick = float(np.median([lines[c][3] for c in chain]))
            if thick > max(6, 0.35 * gap + 3):
                continue
            x0 = int(np.median([lines[c][1] for c in chain]))
            x1 = int(np.median([lines[c][2] for c in chain]))
            staffs.append(Staff(ys, x0, x1, gap, polarity))
    staffs.sort(key=lambda s: s.top)
    return staffs
