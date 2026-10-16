"""Split a segment into still views, build a clean plate for each, and track the playhead over it."""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from youtabber.video import frames

OBS_FPS = 6
# A view must stay still this long; shorter ones are page-turn animations and fades
MIN_VIEW_SECONDS = 1.2
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
# The playhead moving back by this share of the panel width starts the page over
WRAP_SHARE = 0.4
# Faint ink (light-gray measure numbers) is at least this much darker than paper, in levels or share of contrast
FAINT_MIN, FAINT_SHARE = 18, 0.1
# Color spread (max minus min channel) below which a pixel counts as gray
GRAY_SAT = 28
# Color spread above which a dark pixel is a cursor or highlight, not ink
COLOR_SAT = 70


@dataclass
class View:
    t0: float
    t1: float
    plate: np.ndarray                 # BGR median of the view's frames, in crop coordinates
    playhead: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)
    times: list[float] = field(default_factory=list)
    rect: tuple[int, int, int, int] = (0, 0, 0, 0)   # panel in crop coordinates
    wrapped: bool = False                            # began where the playhead jumped back to the page start
    print_rect: tuple[int, int, int, int] = (0, 0, 0, 0)   # paper area to print from, crop coordinates


def to_gray(frame: np.ndarray, polarity: str) -> np.ndarray:
    """Gray with ink dark, whatever the theme."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return 255 - g if polarity == "light" else g


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
    f = frame.astype(np.float32)
    if _paper.min() >= LIGHT_PAPER:
        # Tinted light paper: judge color against it. Dark paper is near black, so plain spread is used.
        f = f / _paper * 255.0
    return f.max(axis=2) - f.min(axis=2)


def colored(frame: np.ndarray) -> np.ndarray:
    """Strongly colored pixels: cursor boxes, highlights and recolored notes rather than printed ink."""
    return chroma(frame) > COLOR_SAT


def ink_map(gray: np.ndarray, thr: float, gap: float, frame: np.ndarray | None = None) -> np.ndarray:
    """Ink without staff lines, which look the same wherever the music is, and without colored marks."""
    ink = (gray < thr).astype(np.uint8)
    if frame is not None:
        ink[colored(frame)] = 0
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(3 * gap), 1)))
    return (ink & ~lines.astype(bool)).astype(bool)


def gray_pixels(frame: np.ndarray) -> np.ndarray:
    """Pixels without strong color. Faint gray marks are numbers; faint colored ones are highlights."""
    return chroma(frame) < GRAY_SAT


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


def panel_rect(frame: np.ndarray, paper: np.ndarray, seed: tuple[int, int, int, int], gap: float,
               above: float = ABOVE_GAPS, below: float = BELOW_GAPS,
               for_print: bool = False) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the paper panel around the seed box (the staffs), grown while rows and columns stay paper.

    With `for_print`, the area to print from, which keeps a margin around the staffs.
    """
    m = (np.abs(frame.astype(np.int16) - paper.astype(np.int16)).max(axis=2) < PAPER_TOL)
    h, w = m.shape
    sx0, sy0, sx1, sy1 = seed
    sx0, sx1 = max(0, sx0), min(w, sx1)
    sy0, sy1 = max(0, sy0), min(h, sy1)
    rows = m[:, sx0:sx1].mean(axis=1) >= PANEL_SHARE
    slack = max(3, int(gap / 2))
    # Chord names, numbers and beams stay within a few line gaps of the staffs; past that, paper-colored
    # rows are footage that happens to match (dark rooms behind a dark theme)
    y0 = max(grow(rows, sy0, -1, slack), int(sy0 - above * gap))
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


def views_in(path: Path, crop: tuple[int, int, int, int], polarity: str, gap: float, paper: np.ndarray,
             seed_at, t_start: float, t_end: float, use_playhead: bool = True) -> tuple[list[View], float, float]:
    """Still views between t_start and t_end, and the ink threshold used.

    `seed_at(t)` gives the staffs' box at time t in crop coordinates, from the scan.
    """
    views: list[View] = []
    cur: list[tuple[float, np.ndarray, tuple, tuple]] = []   # (t, frame, panel rect, staffs' box)
    ref = None
    cand = None      # a new picture that may be the next view
    held: list = []  # frames that match cand, waiting to see whether it holds
    thr = fthr = None
    # A long view keeps the picture of every `stride`-th frame only; the other frames keep their playhead,
    # found against the plate so far, in `points`
    stride = 1
    points: dict[float, tuple[float, float] | None] = {}
    so_far = None    # (plate, panel rect) of the view's frames when it was last thinned

    def plate_of(chunk) -> np.ndarray:
        kept = [f for _, f, _, _ in chunk if f is not None]
        step = max(1, len(kept) // MAX_PLATE_FRAMES)
        return np.median(np.stack(kept[::step]), axis=0).astype(np.uint8)

    def rect_of(chunk) -> tuple[int, int, int, int]:
        return tuple(int(np.median([c[2][i] for c in chunk])) for i in range(4))

    def playhead_at(f, plate, rect) -> tuple[float, float] | None:
        x0, y0, x1, y1 = rect
        dev = (np.abs(f[y0:y1, x0:x1].astype(np.int16) - plate[y0:y1, x0:x1].astype(np.int16)).max(axis=2)
               > CURSOR_DIFF).astype(np.uint8)
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
            so_far = plate_of(cur), rect_of(cur)
            for k in range(len(cur)):
                if k % stride and cur[k][1] is not None:
                    drop(k)

    def drop(k) -> None:
        t, f, rect, seed = cur[k]
        points[t] = playhead_at(f, *so_far)
        cur[k] = (t, None, rect, seed)

    def build(chunk, plate=None) -> View:
        plate = plate_of(chunk) if plate is None else plate
        rect = rect_of(chunk)
        v = View(chunk[0][0], chunk[-1][0] + 1 / OBS_FPS, plate, times=[t for t, _, _, _ in chunk], rect=rect)
        mid_seed = chunk[len(chunk) // 2][3]
        # On a dark theme, footage in a dark room looks like paper, so the print area keeps the panel's reach
        above, below = (ABOVE_GAPS, BELOW_GAPS) if polarity == "light" else (PRINT_ABOVE_GAPS, PRINT_BELOW_GAPS)
        v.print_rect = panel_rect(plate, paper, mid_seed, gap, above, below, for_print=True)
        for t, f, _, _ in chunk:
            p = points.get(t) if f is None else playhead_at(f, plate, rect)
            if p is not None:
                v.playhead.append((t, *p))
        return v

    def wraps(v: View) -> list[float]:
        """Times where the playhead jumps back to the start of the page: a page turn to a page that looks
        the same (a riff page whose only change is a faint measure number), or the same page played again.

        Only a playhead seen sweeping across the page counts: found in most frames since the last jump,
        and covering most of the page's width, then landing at its left part.
        """
        width = v.rect[2] - v.rect[0]
        left = v.rect[0]
        ph = v.playhead
        out = []
        since = 0
        for k in range(len(ph) - 2):
            (ta, xa, ya), (tb, xb, yb), (tc, xc, yc) = ph[k], ph[k + 1], ph[k + 2]
            if tb - ta > 0.5 or tc - tb > 0.5:
                continue
            back = (xb < xa - WRAP_SHARE * width and yb < ya + 2 * gap
                    and xa > left + 0.6 * width and xb < left + 0.35 * width)
            steady = abs(yc - yb) < 2 * gap and abs(xc - xb) < 0.2 * width
            sweep = ph[since:k + 1]
            frames_since = (ta - sweep[0][0]) * OBS_FPS + 1 if sweep else 0
            swept = (len(sweep) >= 0.6 * frames_since
                     and max(x for _, x, _ in sweep) - min(x for _, x, _ in sweep) >= 0.5 * width)
            if back and steady and swept:
                out.append(tb)
                since = k + 1
        return out

    def close() -> None:
        if not cur or cur[-1][0] - cur[0][0] + 1 / OBS_FPS < MIN_VIEW_SECONDS:
            return
        whole = build(cur)
        cuts = wraps(whole) if use_playhead else []
        if not cuts:
            views.append(whole)
            return
        bounds = [cur[0][0]] + cuts + [cur[-1][0] + 1]
        # A thinned view keeps few pictures per pass; all passes show the same page, so they share its plate
        shared = whole.plate if stride > 1 else None
        for n, (a, b) in enumerate(zip(bounds, bounds[1:])):
            chunk = [c for c in cur if a <= c[0] < b]
            if chunk and chunk[-1][0] - chunk[0][0] + 1 / OBS_FPS >= MIN_VIEW_SECONDS:
                v = build(chunk, shared)
                v.wrapped = n > 0
                views.append(v)

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
        mask &= ~(r["tint"] | tinted(f))
        # Two pixels of slack: a cursor passing over a note changes its soft edge
        changed = significant_change(r["ink"] & mask, ink & mask, gap, radius=2)
        return changed, f

    for t, f in frames(path, OBS_FPS, crop=crop, start=t_start, length=t_end - t_start):
        seed = seed_at(t)
        if seed is None:
            close()
            cur, ref, cand, held = [], None, None, []
            stride, points = 1, {}
            continue
        g = to_gray(f, polarity)
        if thr is None:
            thr, fthr = ink_threshold(g[max(0, seed[1]):seed[3], max(0, seed[0]):seed[2]], paper, polarity)
        rect = panel_rect(f, paper, seed, gap)
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
    close()
    return views, (thr or 128.0), (fthr or 230.0)
