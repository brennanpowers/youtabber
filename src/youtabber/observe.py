"""Split a segment into still views, build a clean plate for each, and track the playhead over it."""
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from youtabber.video import frames

OBS_FPS = 6
# A view must stay still this long; shorter ones are page-turn animations and fades
MIN_VIEW_SECONDS = 1.2
# ...except a segment's first and last screens: one shown only briefly before the first page turn or after the last
# may hold music no other screen shows
EDGE_VIEW_SECONDS = 0.5
# A new picture must hold for this many frames to end the view
SETTLE_FRAMES = 2
# Frames kept per view for the median; more adds time, not quality
MAX_PLATE_FRAMES = 48
# Frames whose pictures a view keeps (64 s at 6 fps); past this, pictures are thinned so a page held for minutes
# doesn't fill memory with full-size frames
MAX_VIEW_FRAMES = 8 * MAX_PLATE_FRAMES
# Ink is darker than this share of the way from the darkest ink to the paper
INK_LEVEL = 0.55
# Pixels that differ from the plate by this much (any color channel) are cursor, not paper
CURSOR_DIFF = 45
# A playhead that sweeps at least this many line gaps, then jumps back as many into its sweep's left half, starts
# the page over
WRAP_GAPS = 6
# Just after the jump back, the playhead's next sample sits within this many line gaps of it, up or down: a box
# cursor's middle rides up and down a system with its notes
WRAP_STEADY_GAPS = 3.0
# On dark paper a staff area can show too little ink to measure; the ink is then taken to be at least this share
# of the way from the paper to white, so a see-through cursor bar a little lighter than the paper isn't ink
DARK_INK_REACH = 0.6
# Faint ink (light-gray measure numbers) is at least this much darker than paper, in levels or share of contrast
FAINT_MIN, FAINT_SHARE = 18, 0.1
# Color spread (max minus min channel) below which a pixel counts as gray
GRAY_SAT = 28
# Color spread above which a dark pixel is a cursor or highlight, not ink
COLOR_SAT = 70
# Colored pixels (and their surroundings) covering more than this share of the area a still view compares are
# tinted paper; the compare then keeps them
TINT_PAPER = 0.6
# Colored pixels covering more than this share of the ink are the ink's own color (music drawn in color); the
# compare keeps them too
TINT_INK = 0.85
# A pixel lighter than ink in the plate whose gray level spreads by more than this between a view's lighter and darker
# frames is footage moving behind a see-through panel
FOOTAGE_SPREAD = 12
# There the plate takes the pixel from the frame lighter than this share of the view's frames, where the footage
# hides the paper least
FOOTAGE_RANK = 0.95
# ...when that frame shows it within this many gray levels of the lightest pixel within FOOTAGE_NEAR_GAPS line gaps
# (and at least 2 pixels), and shows none of its eight neighbors more than FOOTAGE_EDGE levels darker than that
FOOTAGE_PAPER, FOOTAGE_NEAR_GAPS, FOOTAGE_EDGE = 10, 0.2, 20
# Masked pixels are ranked this many at a time, so the frame order of a large crop never takes a gigabyte
FOOTAGE_BLOCK = 1 << 16
# A pixel whose median is within this many gray levels of its lightest frame keeps its median
FOOTAGE_SHOWN = 6
# Views with fewer frames keep the plain median
FOOTAGE_MIN_FRAMES = 6
# Footage shows through the panel only when at least this share of its paper spreads by more than FOOTAGE_SPREAD
# between its lighter and darker frames; below it, the pixels that spread are faint lines and noise
FOOTAGE_SHARE = 0.2
# A pixel the median shows more than this many gray levels darker than the lightest of its eight neighbors keeps its
# median: a faint line up to two pixels wide has paper beside it. Footage at a hard edge with paper passes this too
FOOTAGE_DIP = 12
# A view whose frames' overall gray levels spread by more than this is fading in or out
FOOTAGE_FADE = 12


@dataclass
class View:
    t0: float
    t1: float
    # BGR median of the view's frames with footage behind the panel washed out, in crop coordinates
    plate: np.ndarray
    playhead: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)
    times: list[float] = field(default_factory=list)
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)   # panel in crop coordinates
    wrapped: bool = False                            # began where the playhead jumped back over part of the page
    print_rect: tuple[int, int, int, int] = (0, 0, 0, 0)   # paper area to print from, crop coordinates
    brief: bool = False                              # a first or last screen held under MIN_VIEW_SECONDS


def to_gray(frame: np.ndarray, polarity: str) -> np.ndarray:
    """Gray with ink dark, whatever the theme."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return 255 - g if polarity == "light" else g


def clear_footage(stack: np.ndarray, plate: np.ndarray, polarity: str, thr: float, fthr: float, gap: float,
                  rect: tuple[int, int, int, int]) -> np.ndarray:
    """The median plate of a view's frames with footage seen through the panel `rect` (x0, y0, x1, y1) washed out.

    The drawn page holds still while footage behind a see-through panel moves. A pixel lighter than ink whose level
    keeps changing, and which in its lightest frames is as light as the paper around it, is paper under footage: it
    takes its value from those frames. Faint lines drawn on the panel stay darker than the paper beside them even
    then, so they keep their median. Lightness is judged with ink dark, so on a light-ink theme it means darkness.
    Pixels outside the panel and pixels touching ink are left as they are: footage beside the panel isn't tab, and
    the soft edges of ink change from frame to frame. Only pixels darker than the faint-ink threshold `fthr` change:
    lighter ones already read as paper, and lifting them would only move the paper level the barline tests measure
    against. A view whose spreading pixels cover little of the panel's paper shows no footage and keeps its median.
    """
    n = len(stack)
    if n < FOOTAGE_MIN_FRAMES:
        return plate
    gray = np.stack([to_gray(f, polarity) for f in stack])
    # A page fading in or out changes the level of the whole picture: such a view keeps its median plate
    levels = np.median(gray[:, ::4, ::4].reshape(n, -1), axis=1)
    if levels.max() - levels.min() > FOOTAGE_FADE:
        return plate
    still = to_gray(plate, polarity)
    # The panel's paper: pixels inside it that neither are nor touch ink
    x0, y0, x1, y1 = rect
    paper = np.zeros(still.shape, bool)
    paper[max(0, y0):y1, max(0, x0):x1] = True
    paper &= cv2.dilate((still < thr).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
    # Max minus min is quick and never smaller than the spread between ranks, so it picks the pixels worth ranking
    mask = paper & (gray.max(axis=0) - gray.min(axis=0) > FOOTAGE_SPREAD)
    if not mask.any():
        return plate
    ys, xs = np.nonzero(mask)
    lo, hi = int(round((1 - FOOTAGE_RANK) * (n - 1))), int(round(FOOTAGE_RANK * (n - 1)))
    light = np.empty(len(ys), np.int16)
    spread = np.empty(len(ys), np.int16)
    frame = np.empty(len(ys), np.intp)
    for a in range(0, len(ys), FOOTAGE_BLOCK):
        g = gray[:, ys[a:a + FOOTAGE_BLOCK], xs[a:a + FOOTAGE_BLOCK]]
        order = np.argpartition(g, [lo, hi], axis=0)
        cols = np.arange(g.shape[1])
        light[a:a + FOOTAGE_BLOCK] = g[order[hi], cols]
        spread[a:a + FOOTAGE_BLOCK] = light[a:a + FOOTAGE_BLOCK] - g[order[lo], cols]
        frame[a:a + FOOTAGE_BLOCK] = order[hi]
    # Keep the median unless footage spreads over FOOTAGE_SHARE of the paper: faint lines that flicker on
    # low-resolution video spread too, but cover little of it
    if (spread > FOOTAGE_SPREAD).sum() < FOOTAGE_SHARE * paper.sum():
        return plate
    # Each pixel at its lightest (the median where the pixel holds still), and the lightest of those nearby
    lightest = still.copy()
    lightest[ys, xs] = np.maximum(lightest[ys, xs], light)
    r = max(2, int(round(FOOTAGE_NEAR_GAPS * gap)))
    near = cv2.dilate(lightest, np.ones((2 * r + 1, 2 * r + 1), np.uint8))[ys, xs].astype(np.int16)
    # The chosen frame must show the pixel's neighbors as paper too: a faint line that moved a pixel in that frame
    # leaves its old place light there, but its new place beside it is still dark
    h, w = plate.shape[:2]
    darkest = np.full(len(ys), 255, np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            darkest = np.minimum(darkest, gray[frame, np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)])
    # A pixel the median shows near its lightest needs nothing: paper a cursor crossed now and then stays as it is
    median = still[ys, xs].astype(np.int16)
    # A narrow faint line has paper beside it in the median; inside a patch of footage the neighbors are dark too
    beside = cv2.dilate(still, np.ones((3, 3), np.uint8))[ys, xs].astype(np.int16)
    keep = ((spread > FOOTAGE_SPREAD) & (light - median > FOOTAGE_SHOWN) & (light >= near - FOOTAGE_PAPER)
            & (darkest >= near - FOOTAGE_EDGE) & (median < fthr) & (beside - median <= FOOTAGE_DIP))
    out = plate.copy()
    out[ys[keep], xs[keep]] = stack[frame[keep], ys[keep], xs[keep]]
    return out


def ink_threshold(gray: np.ndarray, paper_bgr: np.ndarray, polarity: str) -> tuple[float, float]:
    """(ink threshold, faint-ink threshold): solid notation, and anything visibly darker than paper.

    Paper comes from the scan (the median between staff lines over the segment); ink is the darkest
    half percent of a frame's staff area.
    """
    b, g, r = (float(v) for v in paper_bgr)
    paper = 0.114 * b + 0.587 * g + 0.299 * r
    if polarity == "light":
        paper = 255 - paper
    ink = min(float(np.percentile(gray, 0.5)), paper - 60)
    if polarity == "light":
        ink = min(ink, (1 - DARK_INK_REACH) * paper)
    return ink + INK_LEVEL * (paper - ink), paper - max(FAINT_MIN, FAINT_SHARE * (paper - ink))


# Paper color of the segment being read (BGR). Color is judged relative to it, so yellow or gray paper and
# the black ink printed on it don't count as colored.
_paper = np.array([255.0, 255.0, 255.0])
# Paper whose darkest channel is at least this bright counts as light (white, yellow, gray)
LIGHT_PAPER = 128


def set_paper(paper: np.ndarray) -> None:
    global _paper
    _paper = np.maximum(np.asarray(paper, dtype=np.float32), 1.0)


def chroma(frame: np.ndarray) -> np.ndarray:
    """Color spread of each pixel after dividing out the paper color (0 for gray on any paper)."""
    if frame.size == 0:
        return np.zeros(frame.shape[:2], np.float32)
    if _paper.min() >= LIGHT_PAPER:
        # Tinted light paper: judge color against it. Dark paper is near black, so plain spread is used.
        # A table of every channel level divided by the paper gives the same numbers as dividing each pixel.
        table = (np.arange(256, dtype=np.float32)[:, None] / _paper * 255.0).reshape(256, 1, 3)
        b, g, r = cv2.split(cv2.LUT(frame, table))
    else:
        b, g, r = cv2.split(frame.astype(np.float32))
    # OpenCV's per-pixel max and min over three planes are much faster than numpy's reduce over the color axis
    return cv2.subtract(cv2.max(cv2.max(b, g), r), cv2.min(cv2.min(b, g), r))


# Wider images have their color judged this many columns at a time, so a long glide mosaic never needs float copies
# of the whole image at once
COLOR_PIECE = 4096


def color_test(frame: np.ndarray, test) -> np.ndarray:
    """test() applied to each pixel's color spread, a piece of columns at a time."""
    if frame.shape[1] <= COLOR_PIECE:
        return test(chroma(frame))
    return np.concatenate([test(chroma(frame[:, a:a + COLOR_PIECE])) for a in range(0, frame.shape[1], COLOR_PIECE)],
                          axis=1)


def colored(frame: np.ndarray) -> np.ndarray:
    """Strongly colored pixels: cursor boxes, highlights and recolored notes rather than printed ink."""
    return color_test(frame, lambda c: c > COLOR_SAT)


def ink_map(gray: np.ndarray, thr: float, gap: float, frame: np.ndarray | None = None) -> np.ndarray:
    """Ink without staff lines, which look the same wherever the music is, and without colored marks."""
    ink = (gray < thr).astype(np.uint8)
    if frame is not None:
        ink[colored(frame)] = 0
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(3 * gap), 1)))
    return (ink & ~lines.astype(bool)).astype(bool)


def gray_pixels(frame: np.ndarray) -> np.ndarray:
    """Pixels without strong color. Faint gray marks are numbers; faint colored ones are highlights."""
    return color_test(frame, lambda c: c < GRAY_SAT)


def alone(faint: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Faint marks away from solid ink: the soft edges of notes change color under a cursor; numbers don't."""
    near_ink = cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    return faint & ~near_ink


def unmatched(a: np.ndarray, b: np.ndarray, radius: int = 1) -> np.ndarray:
    """Marks in either map with nothing within `radius` pixels in the other."""
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    na = cv2.dilate(a.astype(np.uint8), k).astype(bool)
    nb = cv2.dilate(b.astype(np.uint8), k).astype(bool)
    return ((a & ~nb) | (b & ~na)).astype(np.uint8)


def tinted_paper(tint: np.ndarray, area: np.ndarray, ink: np.ndarray) -> bool:
    """True when colored pixels are the page's own colors: they cover so much of the compared area that they are
    the paper, or nearly all of its ink, so they are the ink. No cursor or highlight covers either."""
    inked = ink & area
    return bool((tint & area).sum() > TINT_PAPER * area.sum() or (tint & inked).sum() > TINT_INK * inked.sum())


def faint_change(a: np.ndarray, b: np.ndarray, gap: float) -> bool:
    """True when a small faint mark, the size of a light-gray measure number, differs between two faint-ink maps.

    Large faint areas are highlights and cursor boxes moving over the page, so they don't count.
    """
    diff = unmatched(a, b)
    if diff.sum() < 6:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= max(18, 0.01 * gap * gap) and h <= 1.3 * gap and w <= 2 * gap and area <= 1.5 * gap * gap \
                and h >= max(5, 0.15 * gap) and w >= 4:
            return True
    return False


# The row of measure numbers lies this many line gaps above the top staff line (from, to)
NUMBER_ROW_GAPS = (1.7, 0.15)
# The compare follows marks starting in the number row down to this many line gaps below the top staff line, so a
# digit sitting on the line is whole and a colored area coming up from the staff is seen reaching the row. A mark
# reaching down that far is a cursor or a highlight coming up from the staff, not a digit
NUMBER_STAFF_GAPS = 0.5
# A colored area at least this many line gaps wide reaching up from the staff into the number row is a cursor head or
# a highlight box
NUMBER_COVER_GAPS = 1.0
# A printed number that changes differs by more than this in some color channel; the faint edge of a pale cursor box
# differs less
NUMBER_DIFF = 80
# A changed mark in the number row is a digit when it covers this many square line gaps, is this many line gaps wide
# and between these many line gaps tall; a thin line cursor is narrower, and a piece of one is taller
NUMBER_AREA, NUMBER_W, NUMBER_H = 0.05, 0.25, (0.3, 1.0)
# A segment's views are told apart by their number rows once one of the first this many references whose number row
# changes and that show a staff with two measures prints measure numbers
NUMBER_CHECKS = 3
# A changed dark mark is the same mark moved when one frame's pixels, shifted by up to NUMBER_SHIFT_PX pixels each
# way, leave fewer than NUMBER_AREA square line gaps of it differing, or at most NUMBER_SHIFT_SHARE of the pixels that
# differ unshifted: video compression and a page drifting by part of a pixel move strokes a pixel or two. A digit that
# changes stays as different at every shift
NUMBER_SHIFT_PX, NUMBER_SHIFT_SHARE = 2, 0.5


def number_changed(a: np.ndarray, b: np.ndarray, rect, seed, gap: float, skip: np.ndarray | None = None,
                   tint: np.ndarray | None = None, keep: np.ndarray | None = None) -> bool:
    """True when a digit-sized mark in the row of measure numbers differs between two color frames, leaving out the
    pixels in `skip`, or counting only the pixels in `keep`. A mark counted through `keep` must also not be the same
    mark moved by a pixel or two. Pages of one riff can differ only in their numbers, which the ink compare misses
    when they are colored, faint, or change shape in place. False while a wide colored area of `tint` comes up from
    the staff into the row."""
    # Kept inside the panel, so footage above it doesn't count
    y0 = max(rect[1] + 2, int(seed[1] - NUMBER_ROW_GAPS[0] * gap))
    y1 = int(seed[1] - NUMBER_ROW_GAPS[1] * gap)
    y2 = min(rect[3] - 2, int(seed[1] + NUMBER_STAFF_GAPS * gap))
    x0, x1 = rect[0] + 2, rect[2] - 2
    if y1 <= y0 or y2 < y1 or x1 <= x0:
        return False
    if tint is not None:
        # A cursor head or highlight box over the row changes it as it moves, like a digit would; a thin line cursor
        # is left to the shape test below
        n, _, stats, _ = cv2.connectedComponentsWithStats(tint[y0:y2, x0:x1].astype(np.uint8), connectivity=8)
        if any(stats[i][1] < y1 - y0 and stats[i][1] + stats[i][3] >= y2 - y0
               and stats[i][2] >= NUMBER_COVER_GAPS * gap for i in range(1, n)):
            return False
    pa, pb = a[y0:y2, x0:x1], b[y0:y2, x0:x1]
    if keep is not None:
        # Dark marks are compared on brightness alone, so a pale highlight tinting them is no change
        pa, pb = cv2.cvtColor(pa, cv2.COLOR_BGR2GRAY), cv2.cvtColor(pb, cv2.COLOR_BGR2GRAY)
    d = channel_diff(pa, pb)
    if skip is not None:
        d[skip[y0:y2, x0:x1]] = 0
    if keep is not None:
        d[~keep[y0:y2, x0:x1]] = 0
    # Dilated, the 1-2 pixel strokes of a small digit join into one mark
    m = cv2.dilate((d > NUMBER_DIFF).astype(np.uint8), np.ones((3, 3), np.uint8))
    if not m[:y1 - y0].any():
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return any(stats[i][1] < y1 - y0 and stats[i][1] + stats[i][3] < y2 - y0
               and stats[i][4] >= NUMBER_AREA * gap * gap and stats[i][2] >= NUMBER_W * gap
               and NUMBER_H[0] * gap <= stats[i][3] <= NUMBER_H[1] * gap
               and (keep is None or not moved(pa, pb, stats[i][:4], NUMBER_AREA * gap * gap)) for i in range(1, n))


def moved(a: np.ndarray, b: np.ndarray, box, area: float) -> bool:
    """True when the box (x, y, w, h) of b matches a shifted by up to NUMBER_SHIFT_PX pixels each way, leaving under
    `area` pixels, or at most NUMBER_SHIFT_SHARE of those differing unshifted, differing by more than NUMBER_DIFF."""
    x, y, w, h = (int(v) for v in box)
    s = NUMBER_SHIFT_PX
    pad = cv2.copyMakeBorder(a, s, s, s, s, cv2.BORDER_REPLICATE)
    part = b[y:y + h, x:x + w]
    left = min(int((channel_diff(pad[y + s + dy:y + s + dy + h, x + s + dx:x + s + dx + w], part) > NUMBER_DIFF).sum())
               for dy in range(-s, s + 1) for dx in range(-s, s + 1))
    unshifted = int((channel_diff(a[y:y + h, x:x + w], part) > NUMBER_DIFF).sum())
    return left < area or left <= NUMBER_SHIFT_SHARE * unshifted


def channel_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per pixel, the largest difference between two images over their color channels, or of gray levels."""
    d = cv2.absdiff(a, b)
    return d.max(axis=2) if d.ndim == 3 else d


def significant_change(a: np.ndarray, b: np.ndarray, gap: float, radius: int = 1) -> bool:
    """True when some mark the size of a digit is in one ink map and not near the same place in the other.

    Thin tall marks are skipped: they are line cursors moving across a still page.
    """
    diff = unmatched(a, b, radius)
    if diff.sum() < 8:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(diff, connectivity=8)
    amin = max(12, 0.1 * gap * gap)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < amin:
            continue
        if h > 1.5 * gap and w < 0.5 * gap:
            continue
        # A sliver one or two pixels thick is an edge flickering, not a mark that changed
        if min(w, h) < max(3, 0.12 * gap):
            continue
        # A square about a line gap wide, filled or outlined, is a resting cursor marker. Fret digits are
        # taller than wide, so they never pass for one.
        if 0.85 < w / h < 1.4 and 0.5 * gap <= max(w, h) <= 2 * gap and (area > 0.85 * w * h or area < 0.45 * w * h):
            continue
        return True
    return False


PAPER_TOL = 40
# On dark paper, dark footage above a band passes PAPER_TOL, but the band is drawn flat. A row is flat when its middle
# color is within PAPER_TOL of the paper and its share of pixels within FLAT_TOL of that color is at least FLAT_SHARE
# of the staff rows' share; judged against each row's own color, a gradient band stays flat. The panel's rows above
# the staffs end at the first stretch of rows that aren't flat, FLAT_RUN_GAPS line gaps tall; chord names are shorter
FLAT_TOL, FLAT_SHARE, FLAT_RUN_GAPS = 12, 0.75, 1.0
# Every FLAT_STEP-th column is enough to judge a row's flatness
FLAT_STEP = 4
# A panel border is darker than this in every color channel (averaged over half a line gap left of the paper)
BORDER_LEVEL = 40
# A row (or column) belongs to the panel when this share of it is paper colored
PANEL_SHARE = 0.5
# The panel reaches at most this many tab line gaps above the top staff and below the tab staff
ABOVE_GAPS, BELOW_GAPS = 2.5, 3.0
# Marks printed with a system (chord names, section labels, tempo, beams, P.M.) sit within this many line gaps
# above its top staff line and below its tab staff
PRINT_ABOVE_GAPS, PRINT_BELOW_GAPS = 7.0, 5.0
# The print area keeps at least this many line gaps around the staffs, so digits on the outer lines are whole
PRINT_MARGIN_GAPS = 1.0


def grow(ok: np.ndarray, i: int, step: int, slack: int = 3) -> int:
    """Walk from index i in direction step while `ok` holds, stepping over up to `slack` rows that don't
    (a row of beams or a dense chord can dip below the paper share)."""
    n = len(ok)
    i = min(max(i, 0), n - 1)
    last = i
    misses = 0
    j = i
    while 0 <= j + step < n:
        j += step
        if ok[j]:
            last, misses = j, 0
        else:
            misses += 1
            if misses > slack:
                break
    return last


def flat_band_top(frame: np.ndarray, paper: np.ndarray, sx0: int, sx1: int, sy0: int, sy1: int, gap: float,
                  above: float) -> int:
    """The first row of a dark band above the staffs (rows sy0 to sy1, columns sx0 to sx1): the row below the stretch
    of rows that aren't flat, FLAT_RUN_GAPS line gaps tall, nearest the staffs, which starts within `above` line gaps
    of them. 0 when there is none."""
    run = max(1, int(FLAT_RUN_GAPS * gap))
    top = max(0, int(sy0 - above * gap) - run)
    if sy1 <= sy0 or sy0 <= top:
        return 0
    band = frame[top:sy1, sx0:sx1:FLAT_STEP].astype(np.int16)
    if band.shape[1] == 0:
        return 0
    middle = np.median(band, axis=1)
    share = (np.abs(band - middle[:, None, :]).max(axis=2) < FLAT_TOL).mean(axis=1)
    near = np.abs(middle - paper).max(axis=1) < PAPER_TOL
    staff = share[sy0 - top:][near[sy0 - top:]]
    if len(staff) == 0:
        return 0
    flat = near[:sy0 - top] & (share[:sy0 - top] >= FLAT_SHARE * float(np.median(staff)))
    lowest = int(sy0 - above * gap) - top
    k = sy0 - top - 1
    while k >= max(lowest, 0):
        if flat[k]:
            k -= 1
            continue
        end = k
        while k >= 0 and not flat[k]:
            k -= 1
        if end - k >= run:
            return top + end + 1
    return 0


def panel_rect(frame: np.ndarray, paper: np.ndarray, seed: tuple[int, int, int, int], gap: float,
               above: float = ABOVE_GAPS, below: float = BELOW_GAPS,
               for_print: bool = False, polarity: str = "dark") -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the paper panel around the seed box (the staffs), grown while rows and columns stay paper.

    With `for_print`, the area to print from, which keeps a margin around the staffs.
    """
    # Paper colored: every channel within PAPER_TOL of the paper's
    p = paper.astype(np.int16)
    lo = tuple(int(v) for v in np.clip(p - (PAPER_TOL - 1), 0, 255))
    hi = tuple(int(v) for v in np.clip(p + (PAPER_TOL - 1), 0, 255))
    m = cv2.inRange(frame, lo, hi) > 0
    h, w = m.shape
    sx0, sy0, sx1, sy1 = seed
    sx0, sx1 = max(0, sx0), min(w, sx1)
    sy0, sy1 = max(0, sy0), min(h, sy1)
    rows = m[:, sx0:sx1].mean(axis=1) >= PANEL_SHARE
    flat_top = flat_band_top(frame, p, sx0, sx1, sy0, sy1, gap, above) if polarity == "light" else 0
    slack = max(3, int(gap / 2))
    # Chord names, numbers and beams stay within a few line gaps of the staffs; past that, paper-colored
    # rows are footage that happens to match (dark rooms behind a dark theme)
    y0 = max(grow(rows, sy0, -1, slack), int(sy0 - above * gap), flat_top)
    y1 = min(grow(rows, sy1 - 1, 1, slack) + 1, int(sy1 + below * gap))
    if for_print:
        # Digits sit centered on the outer staff lines, so the print area always reaches past them
        y0 = max(0, min(y0, int(sy0 - PRINT_MARGIN_GAPS * gap)))
        y1 = min(h, max(y1, int(sy1 + PRINT_MARGIN_GAPS * gap)))
    cols = m[sy0:sy1].mean(axis=0) >= PANEL_SHARE
    # A seed that starts in footage is moved to the first paper column to its right before growing, but only
    # past a dark panel border: a highlight, a tinted clef panel or a clef glyph is music, not footage
    if not cols[sx0] and cols[sx0:].any():
        p = int(np.argmax(cols[sx0:])) + sx0
        border = frame[sy0:sy1, max(0, p - max(2, int(gap / 2))):p]
        if border.size and border.max(axis=2).mean() < BORDER_LEVEL:
            sx0 = p
    # On the left, the print area steps over up to half a line gap of non-paper columns when paper goes on past
    # them: the bracket joining a system's staffs, with the measure number printed above it
    x0 = grow(cols, sx0, -1, max(3, int(gap / 2)) if for_print else 3)
    x1 = grow(cols, sx1 - 1, 1) + 1
    return x0, y0, x1, y1


def within(rect: tuple[int, int, int, int], bound: tuple[int, int, int, int] | None,
           keep: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    """The rect cut to the bound (both x0, y0, x1, y1). `keep` gives rows (top, bottom) the cut never reaches."""
    if bound is None:
        return rect
    x0, y0, x1, y1 = rect
    bx0, by0, bx1, by1 = bound
    if keep is not None:
        by0, by1 = min(by0, keep[0]), max(by1, keep[1])
    x0, x1 = max(x0, bx0), min(x1, bx1)
    y0, y1 = max(y0, by0), min(y1, by1)
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def views_in(path: Path, crop: tuple[int, int, int, int], polarity: str, gap: float, paper: np.ndarray,
             seed_at, t_start: float, t_end: float, use_playhead: bool = True,
             progress: Callable[[float], None] | None = None, lead: float = 0.0,
             thresholds: tuple[float, float] | None = None,
             last_screen: bool = False, bound_at=None,
             prints_numbers: Callable[[np.ndarray, tuple, float, float], bool | None] | None = None
             ) -> tuple[list[View], float, float]:
    """Still views between t_start and t_end, and the ink threshold used. A brief screen in the `lead` seconds
    before t_start, which the scan's steps missed, comes first; it is read with the staffs' box at t_start and the
    threshold of the views after it. `thresholds` gives the ink thresholds instead of reading them from the first
    frame. With `last_screen`, a screen held briefly after the last view is kept even when a fade or a cut after it
    ends it before t_end. `bound_at(t)`, when given, is the box (crop coordinates) of the paper the tab is drawn on
    at time t, or None: panel rects stay inside it, except the rows of the staffs themselves.
    `prints_numbers(frame, panel rect, thr, fthr)`, when given, says whether a frame prints measure numbers, or None
    when it shows no staff with two measures; views are then also told apart by their rows of measure numbers.

    `seed_at(t)` gives the staffs' box at time t in crop coordinates, from the scan. `progress(t)` is called with
    each frame's time.
    """
    views: list[View] = []
    cur: list[tuple[float, np.ndarray, tuple, tuple]] = []   # (t, frame, panel rect, staffs' box)
    ref = None
    cand = None      # a new picture that may be the next view
    held: list = []  # frames that match cand, waiting to see whether it holds
    thr, fthr = thresholds if thresholds is not None else (None, None)
    # A long view keeps the picture of every `stride`-th frame only; the other frames keep their playhead,
    # found against the plate so far, in `points`
    stride = 1
    points: dict[float, tuple[float, float] | None] = {}
    so_far = None    # (plate, panel rect) of the view's frames when it was last thinned
    # The first screen held under MIN_VIEW_SECONDS since the last view. When no view follows it, it is the
    # last screen, even if a fade or a cut to black after it closed a chunk of its own
    tail: list = []
    numbered: list[bool] = []  # whether each reference judged so far prints measure numbers

    def plates_of(chunk, clear: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(median, plate) of the chunk's frames: the plate is the median with footage behind the panel washed out.
        The playhead and the panel are found on the median, which the frames are compared against."""
        kept = [f for _, f, _, _ in chunk if f is not None]
        step = max(1, len(kept) // MAX_PLATE_FRAMES)
        stack = np.stack(kept[::step])
        median = np.median(stack, axis=0).astype(np.uint8)
        return median, clear_footage(stack, median, polarity, thr, fthr, gap, rect_of(chunk)) if clear else median

    def rect_of(chunk) -> tuple[int, int, int, int]:
        return tuple(int(np.median([c[2][i] for c in chunk])) for i in range(4))

    def playhead_at(f, plate, rect) -> tuple[float, float] | None:
        x0, y0, x1, y1 = rect
        b, g, r = cv2.split(cv2.absdiff(f[y0:y1, x0:x1], plate[y0:y1, x0:x1]))
        dev = (cv2.max(cv2.max(b, g), r) > CURSOR_DIFF).astype(np.uint8)
        dev = cv2.morphologyEx(dev, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        ys, xs = np.nonzero(dev)
        if len(xs) < 0.3 * gap * gap:
            return None
        # The middle of everything that differs from the plate: a line, a box outline, a filled
        # highlight or recolored notes all center on the beat being played
        return float(np.median(xs)) + x0, float(np.median(ys)) + y0

    def add(c) -> None:
        """Append a frame to the current view, dropping pictures the view can't keep."""
        nonlocal stride, so_far
        cur.append(c)
        if (len(cur) - 1) % stride:
            drop(len(cur) - 1)
        elif -(-len(cur) // stride) > MAX_VIEW_FRAMES:
            # Keep every other picture of those kept so far
            stride *= 2
            so_far = plates_of(cur, clear=False)[0], rect_of(cur)
            for k in range(len(cur)):
                if k % stride and cur[k][1] is not None:
                    drop(k)

    def drop(k) -> None:
        t, f, rect, seed = cur[k]
        points[t] = playhead_at(f, *so_far)
        cur[k] = (t, None, rect, seed)

    def build(chunk, plates=None) -> tuple[View, tuple[np.ndarray, np.ndarray]]:
        """The view of the chunk's frames, and its (median, plate) for views that share them."""
        plates = plates_of(chunk) if plates is None else plates
        plate = plates[0]
        rect = rect_of(chunk)
        v = View(chunk[0][0], chunk[-1][0] + 1 / OBS_FPS, plates[1], times=[t for t, _, _, _ in chunk], rect=rect)
        mid_seed = chunk[len(chunk) // 2][3]
        # On a dark theme, footage in a dark room looks like paper, so the print area keeps the panel's reach
        above, below = (ABOVE_GAPS, BELOW_GAPS) if polarity == "light" else (PRINT_ABOVE_GAPS, PRINT_BELOW_GAPS)
        mid_bound = bound_at(chunk[len(chunk) // 2][0]) if bound_at is not None else None
        v.print_rect = within(panel_rect(plate, paper, mid_seed, gap, above, below, for_print=True,
                                         polarity=polarity), mid_bound,
                              (int(mid_seed[1] - PRINT_MARGIN_GAPS * gap), int(mid_seed[3] + PRINT_MARGIN_GAPS * gap)))
        for t, f, _, _ in chunk:
            p = points.get(t) if f is None else playhead_at(f, plate, rect)
            if p is not None:
                v.playhead.append((t, *p))
        return v, plates

    def wraps(v: View) -> list[float]:
        """Times where the playhead jumps back over part of the page: a page turn to a page that looks the same (a
        riff page whose only change is a faint measure number, or a jump by the riff's own length), or the same
        page played again.

        Only a playhead seen sweeping steadily counts: found in most frames since it last jumped back, for as long as
        a view lasts and across at least `WRAP_GAPS` line gaps, then jumping back at least as far, from the right
        quarter of its sweep into its left half.
        """
        width = v.rect[2] - v.rect[0]
        ph = v.playhead
        out = []
        since = 0
        for k in range(len(ph) - 2):
            (ta, xa, ya), (tb, xb, yb), (tc, xc, yc) = ph[k], ph[k + 1], ph[k + 2]
            if tb - ta > 0.5 or tc - tb > 0.5:
                continue
            sweep = ph[since:k + 1]
            lo, hi = min(x for _, x, _ in sweep), max(x for _, x, _ in sweep)
            jumped = xb < xa - WRAP_GAPS * gap and yb < ya + 2 * gap
            back = jumped and xa >= hi - 0.25 * (hi - lo) and xb <= lo + 0.5 * (hi - lo)
            steady = abs(yc - yb) < WRAP_STEADY_GAPS * gap and abs(xc - xb) < 0.2 * width
            frames_since = (ta - sweep[0][0]) * OBS_FPS + 1
            # A sweep lasts as long as a view must, so the cut leaves a view before it
            swept = (len(sweep) >= 0.6 * frames_since and hi - lo >= WRAP_GAPS * gap
                     and tb - sweep[0][0] >= MIN_VIEW_SECONDS)
            if back and steady and swept:
                out.append(tb)
            # Any jump back starts a new sweep: a beat cursor going over one measure again and again never sweeps
            if jumped:
                since = k + 1
        return out

    def close(last: bool = False) -> None:
        if not cur:
            return
        first = not views and cur[0][0] <= t_start + EDGE_VIEW_SECONDS
        seconds = held_for(cur)
        if seconds < (EDGE_VIEW_SECONDS if first or last else MIN_VIEW_SECONDS):
            if seconds >= EDGE_VIEW_SECONDS and not tail:
                tail[:] = cur
            return
        tail.clear()
        whole, plates = build(cur)
        whole.brief = seconds < MIN_VIEW_SECONDS
        cuts = wraps(whole) if use_playhead else []
        if not cuts:
            views.append(whole)
            return
        bounds = [cur[0][0]] + cuts + [cur[-1][0] + 1]
        # A thinned view keeps few pictures per pass; all passes show the same page, so they share its plate
        shared = plates if stride > 1 else None
        for n, (a, b) in enumerate(zip(bounds, bounds[1:])):
            chunk = [c for c in cur if a <= c[0] < b]
            if chunk and chunk[-1][0] - chunk[0][0] + 1 / OBS_FPS >= MIN_VIEW_SECONDS:
                v, _ = build(chunk, shared)
                v.wrapped = n > 0
                views.append(v)

    def held_for(chunk) -> float:
        # Rounded, since frame times carry float error and three frames are exactly EDGE_VIEW_SECONDS
        return round(chunk[-1][0] - chunk[0][0] + 1 / OBS_FPS, 3)

    def tinted(f):
        """Colored pixels and their surroundings: cursors, highlights and recolored notes are ignored."""
        return cv2.dilate(colored(f).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    def area(rect, seed, shape):
        """Where music is compared: the panel, limited to the band around the staffs, so cursor heads and
        banners above the music don't count."""
        x0, y0, x1, y1 = rect
        mask = np.zeros(shape, bool)
        top = max(y0 + 2, int(seed[1] - ABOVE_GAPS * gap))
        bottom = min(y1 - 2, int(seed[3] + BELOW_GAPS * gap))
        mask[top:bottom, x0 + 2:x1 - 2] = True
        return mask

    def start(t, f, g, rect, seed):
        """A new reference frame."""
        ink = ink_map(g, thr, gap)
        return {"t": t, "frame": f, "rect": rect, "seed": seed, "ink": ink & area(rect, seed, ink.shape),
                "tint": tinted(f)}

    def compare(r, f, g, rect):
        """(changed, frame) for a frame against reference r."""
        ink = ink_map(g, thr, gap)
        mask = area(r["rect"], r["seed"], ink.shape)
        tint = r["tint"] | tinted(f)
        if not tinted_paper(tint, mask, r["ink"]):
            mask &= ~tint
        # Two pixels of slack: a cursor passing over a note changes its soft edge
        changed = significant_change(r["ink"] & mask, ink & mask, gap, radius=2)
        if not changed and prints_numbers is not None and numbers_differ(r, f, ink, mask, tint):
            changed = shows_numbers(r)
        return changed, f

    def numbers_differ(r, f, ink, mask, tint) -> bool:
        """Whether the row of measure numbers changed between reference r and frame f. Ink in either frame, with a
        pixel of its soft edge, is left out, as solid marks above the staff shift by a pixel and change color under a
        highlight; colored and faint numbers are told apart that way. Dark ink with no ink within a pixel of it in
        the other frame is compared too, so a dark number that changes in place is seen; colored marks such as a
        moving line cursor are left out of that."""
        k = np.ones((3, 3), np.uint8)
        union = cv2.dilate(((r["ink"] | ink) & mask).astype(np.uint8), k).astype(bool)
        lone = cv2.dilate(unmatched(r["ink"] & mask, ink & mask), k).astype(bool) & ~tint
        return (number_changed(r["frame"], f, r["rect"], r["seed"], gap, union, tint)
                or number_changed(r["frame"], f, r["rect"], r["seed"], gap, tint=tint, keep=lone))

    def shows_numbers(r) -> bool:
        """Whether the segment prints measure numbers, judged on the first references whose number row changes and
        that show a staff with two measures, so count-in screens don't use up the checks."""
        if not any(numbered) and "numbered" not in r and len(numbered) < NUMBER_CHECKS:
            r["numbered"] = prints_numbers(r["frame"], r["rect"], thr, fthr)
            if r["numbered"] is not None:
                numbered.append(r["numbered"])
        return any(numbered)

    for t, f in frames(path, OBS_FPS, crop=crop, start=t_start, length=t_end - t_start):
        if progress is not None:
            progress(t)
        seed = seed_at(t)
        if seed is None:
            close()
            cur, ref, cand, held = [], None, None, []
            stride, points = 1, {}
            continue
        g = to_gray(f, polarity)
        if thr is None:
            thr, fthr = ink_threshold(g[max(0, seed[1]):seed[3], max(0, seed[0]):seed[2]], paper, polarity)
        rect = within(panel_rect(f, paper, seed, gap, polarity=polarity), bound_at(t) if bound_at is not None else None,
                      (seed[1], seed[3]))
        if ref is None:
            ref = start(t, f, g, rect, seed)
            cur, stride, points = [(t, f, rect, seed)], 1, {}
            continue
        changed, _ = compare(ref, f, g, rect)
        if not changed:
            # Frames held while a possible change was checked were blips (a cursor crossing a beam)
            for c in held + [(t, f, rect, seed)]:
                add(c)
            held, cand = [], None
            continue
        if cand is not None and not compare(cand, f, g, rect)[0]:
            held.append((t, f, rect, seed))
            if len(held) >= SETTLE_FRAMES:
                # The new picture held steady: the view ended where it began
                close()
                cur, ref, stride, points = held, cand, 1, {}
                held, cand = [], None
            continue
        # Changed from the view, and not yet steady: the start of a possible new view
        cand = start(t, f, g, rect, seed)
        held = [(t, f, rect, seed)]
    close(last=True)
    if tail and last_screen:
        v, _ = build(tail)
        v.brief = True
        views.append(v)
    if lead and thr is not None and seed_at(t_start) is not None:
        before, _, _ = views_in(path, crop, polarity, gap, paper, lambda _: seed_at(t_start), t_start - lead, t_start,
                                use_playhead, thresholds=(thr, fthr), bound_at=bound_at, prints_numbers=prints_numbers)
        views[:0] = before
    return views, (thr or 128.0), (fthr or 230.0)
