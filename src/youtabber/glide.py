"""Turn a band that glides past a fixed playhead into one long still image (a mosaic), with a time map.

The band's movement is measured for every frame: from vertical lines (barlines, stems) crossing, else from the music
right of a fixed reading column (the slit) or across the whole staff. Each frame adds the strip of columns that
crossed the slit since the frame before, so the playhead and anything else fixed to the screen stay out
of those strips. The mosaic starts with the first clear frame from a few line gaps left of the playhead to the
slit (from the panel's left edge when no playhead is found), and ends with the last clear frame right of the
slit, so music on screen before the scroll starts or after it stops is read too.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from youtabber.observe import colored, ink_map, ink_threshold, to_gray
from youtabber.video import frames

GLIDE_FPS = 10
# The slit sits at this share of the panel width, right of a playhead drawn near the left
SLIT_SHARE = 0.7
# Largest movement between two sampled frames, as a share of the panel width
MAX_STEP_SHARE = 0.25
# A column whose colored share over the band is above this in most frames is a fixed playhead
PLAYHEAD_SHARE = 0.5
# Phase-correlation response below which a frame-to-frame shift is not trusted
MIN_RESPONSE = 0.1
# A whole-width shift is measured only on at least this many square line gaps of ink
LOW_NOTES_GAPS = 0.25
# Vertical lines on the staff at least MARK_GAPS line gaps tall and at most MARK_WIDTH pixels wide follow the band;
# a step from them needs MARK_AGREE lines and stays within MARK_CHANGE of the step before
MARK_GAPS, MARK_WIDTH, MARK_AGREE, MARK_CHANGE = 3.0, 8, 1, 0.4
# When the band was still, one pair of lines can match at any distance: a step then needs START_MARKS lines, or the
# frame before to have given a step within START_CHANGE of it
START_MARKS, START_CHANGE = 2, 0.1
# A staff line is a row whose median gray is this many levels below the rows LINE_HALF pixels above and below; a
# frame needs MIN_LINES such lines to be checked, and more than MAX_LINES is a dark or busy frame
LINE_DEPTH, LINE_HALF, MIN_LINES, MAX_LINES = 20, 3, 4, 20
# The staff has moved when MOVED_FRAMES checked frames within the last END_FRAMES frames, with no steady frame
# between them, keep under half of its lines as last seen
MOVED_FRAMES = 5
# Mean change of the staff's ink, on a 0-1 scale, under which a frame is the one before shown again
REPEATED_LEVEL = 0.01
# The band's recent speed is the median step of this many frames
SPEED_FRAMES = 5
# Slit and whole-width steps this many pixels apart agree
BOTH_PX = 2.0
# Frames sampled across the glide to find ink fixed to the screen, and the share of the staff's rows such a column
# covers in their median
FIXED_SAMPLES, FIXED_ROWS = 16, 0.3
# Frames sampled across the glide to find a playhead: colored votes and the median picture
PLAYHEAD_SAMPLES = 48
# A playhead that isn't colored stays in the median of the sampled frames while the music moves on: a column
# differing from the paper by this many gray levels over this share of the staff's rows
PLAYHEAD_LEVELS, PLAYHEAD_ROWS = 30, 0.9
# The playhead sits at least this many line gaps inside the staff; the staff's own first line is fixed too
PLAYHEAD_INSIDE_GAPS = 2.0
# The mosaic starts and ends on frames whose staff contrast reaches this share of the glide's typical frame, so a
# fade at either end is left out...
CLEAR_SHARE = 0.9
# ...looking this many frames in from each end
END_FRAMES = 60
# The band's speed at the end of a mosaic is measured over this many seconds of its last movement
LAST_SPEED_SECONDS = 2.0

# The canvas starts this many line gaps left of the playhead, so a first measure starting just before it is whole
SEED_BACK_GAPS = 4.0

# Short windows sampled across a segment to tell whether its tab glides, and the frames read in each
MOTION_WINDOWS, MOTION_FRAMES = 12, 10
# A frame pair moves when the band shifts left by at least this many pixels (5 px/s at GLIDE_FPS)
MIN_SHIFT_PX = 0.5
# A window moves when this share of its frame pairs do, and a segment glides when this share of its windows move.
# Still pages show no moving window; a glide's still intro, outro and bars of rest leave most windows moving.
MOVING_PAIRS, MOVING_WINDOWS = 2 / 3, 0.5
# A pixel inked in this share of a window's frames is fixed to the screen
STILL_SHARE = 0.8


@dataclass
class Mosaic:
    image: np.ndarray           # BGR, band rows of the crop, canvas columns
    y0: int                     # crop row of the mosaic's first row
    offsets: np.ndarray         # how far the band has moved since the mosaic's first frame, per frame
    times: np.ndarray           # time of each frame
    ref_x: float                # screen x (crop coordinates) where measures are said to be played
    lead: int                   # canvas columns that sat left of ref_x in the first frame
    seeded: int = 0             # canvas columns taken from the first frame alone
    tail_from: int = 0          # first canvas column taken from the last frame alone
    moved_at: float | None = None   # when the staff left its rows (the panel was redrawn), ending this piece
    until: float | None = None      # when the glide's frames end, past the last clear frame
    until_offset: float | None = None   # how far the band had moved by then, as measured frame to frame
    doubtful: list = field(default_factory=list)   # slit columns where the step left the speed unconfirmed

    def time_at(self, canvas_x: float) -> float:
        """When the canvas column reached ref_x on screen. Columns still right of it when the band last moved get
        the time the band's speed over its last LAST_SPEED_SECONDS of moving would bring them there."""
        reached = canvas_x - self.lead
        if self.offsets[-1] <= self.offsets[0]:
            return float(self.times[0])
        k = self.last_move()
        if reached <= self.offsets[k]:
            return float(np.interp(reached, self.offsets[:k + 1], self.times[:k + 1]))
        return float(self.times[k] + (reached - self.offsets[k]) / self.last_speed())

    def last_move(self) -> int:
        """The frame after which the band no longer moves (blank staff past the end, or the video holding still)."""
        return int(np.argmax(self.offsets >= self.offsets[-1] - 1.0))

    def last_speed(self) -> float:
        """Pixels a second the band moved over its last LAST_SPEED_SECONDS of moving. Apps ease the speed within
        each bar, so a single frame's step is not the speed."""
        k = self.last_move()
        j = int(np.searchsorted(self.times, self.times[k] - LAST_SPEED_SECONDS))
        j = min(j, max(k - 1, 0))
        return max(float(self.offsets[k] - self.offsets[j]) / max(float(self.times[k] - self.times[j]), 1e-6), 1e-6)

    def seen_at(self, canvas_x: float) -> tuple[float, float] | None:
        """(time, screen x) of a canvas column that never crosses ref_x while the band moves: on the first screen
        left of the play column, or still right of it when the band stops. None for a column that crosses it.
        When the mosaic ends on a band still moving (the picture fades out first), a column that its last speed
        brings to ref_x before the glide's frames end crosses it then, if the band truly moved that far by then."""
        reached = canvas_x - self.lead
        k = self.last_move()
        if (reached > self.offsets[k] and k == len(self.offsets) - 1 and self.until is not None
                and self.until_offset is not None and reached <= self.until_offset
                and self.time_at(canvas_x) <= self.until):
            return None
        if reached > self.offsets[k]:
            # Half a second after the band stops, the screen it stopped on is on show, within the glide's time
            return min(float(self.times[k]) + 0.5, float(self.times[-1])), self.ref_x + reached - float(self.offsets[k])
        if reached < self.offsets[0]:
            return float(self.times[0]), self.ref_x + reached
        return None


def glides(path: Path, crop: tuple[int, int, int, int], polarity: str, paper: np.ndarray, t0: float, t1: float,
           bands_at, gap: float) -> bool:
    """True when the tab moves left through most of t0..t1, judged on short windows spread across it.

    `bands_at(t)` gives the crop rows (top, bottom) of each staff system on screen at t. A window moves when any
    of its systems shifts left in most of its frame pairs; a cursor or highlight moving over a still page is
    colored, so it is left out of the comparison, and so are the staff lines, which match at any shift, and ink
    fixed to the screen, which matches at no shift.
    """
    sampled = moving = 0
    span = MOTION_FRAMES / GLIDE_FPS
    for start in np.linspace(t0, max(t0, t1 - span - 0.2), MOTION_WINDOWS):
        bands = bands_at(float(start))
        if not bands:
            continue
        shots = [f for _, f in frames(path, GLIDE_FPS, crop=crop, start=float(start), length=span)]
        if len(shots) < 2:
            continue
        sampled += 1
        for y0, y1 in bands:
            g0 = to_gray(shots[0][y0:y1], polarity)
            thr, _ = ink_threshold(g0, paper, polarity)
            contents = []
            for f in shots:
                band = f[y0:y1]
                g = to_gray(band, polarity)
                content = np.clip((thr + 40 - g.astype(np.float32)) / 80.0, 0, 1)
                content[colored(band)] = 0
                content[staff_lines(g, thr, gap)] = 0
                contents.append(content)
            # Ink in the same place in most of the window's frames is fixed to the screen (an uncolored playhead
            # line, a clef) and would match at no shift
            still = (np.stack(contents) > 0.5).mean(axis=0) >= STILL_SHARE
            pairs = 0
            for prev, content in zip(contents, contents[1:]):
                (dx, _), resp = cv2.phaseCorrelate(np.where(still, 0, prev).astype(np.float32),
                                                   np.where(still, 0, content).astype(np.float32))
                pairs += resp > MIN_RESPONSE and dx <= -MIN_SHIFT_PX
            if pairs >= MOVING_PAIRS * (len(shots) - 1):
                moving += 1
                break
    return sampled > 0 and moving >= MOVING_WINDOWS * sampled


def staff_lines(gray: np.ndarray, thr: float, gap: float) -> np.ndarray:
    """The staff lines' pixels, a row either side included: dark runs at least three line gaps long."""
    lines = cv2.morphologyEx((gray < thr).astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (int(3 * gap), 1)))
    return cv2.dilate(lines, np.ones((3, 1), np.uint8)).astype(bool)


def fixed_columns(path: Path, crop, staff_rows: tuple[int, int], polarity: str, thr: float, t0: float,
                  t1: float) -> np.ndarray | None:
    """Columns of ink that stays at one screen place through the glide (an uncolored playhead line, a clef),
    found in the median of frames spread across it, where moving music averages out to paper."""
    s0, s1 = staff_rows
    shots = []
    for t in np.linspace(t0, max(t0, t1 - 0.5), FIXED_SAMPLES):
        got = next(iter(frames(path, GLIDE_FPS, crop=crop, start=float(t), length=0.2)), None)
        if got is not None:
            shots.append(to_gray(got[1][s0:s1], polarity))
    if len(shots) < 3:
        return None
    med = np.median(np.stack(shots), axis=0)
    cols = (med < thr).mean(axis=0) >= FIXED_ROWS
    return cv2.dilate(cols.astype(np.uint8)[None, :], np.ones((1, 7), np.uint8))[0].astype(bool)


def still_playhead(samples: list[np.ndarray], staff_x: tuple[int, int], gap: float) -> float | None:
    """Screen x of a playhead that isn't colored: a column that stays in the median of frames sampled across the
    glide, where moving music averages out to paper, and crosses the staff top to bottom. None if there is none."""
    if len(samples) < 3:
        return None
    med = np.median(np.stack(samples), axis=0)
    dev = med - float(np.median(med))
    cover = (np.abs(dev) > PLAYHEAD_LEVELS).mean(axis=0)
    lo, hi = int(staff_x[0] + PLAYHEAD_INSIDE_GAPS * gap), int(staff_x[1] - PLAYHEAD_INSIDE_GAPS * gap)
    xs = [x for x in range(max(lo, 0), min(hi, med.shape[1])) if cover[x] >= PLAYHEAD_ROWS]
    if not xs:
        return None
    # The column furthest from the paper is the line's middle
    return float(max(xs, key=lambda x: abs(float(np.median(dev[:, x])))))


def build(path: Path, crop: tuple[int, int, int, int], rows: tuple[int, int], staff_rows: tuple[int, int],
          polarity: str, thr: float, gap: float, t0: float, t1: float, analysis_rows: tuple[int, int],
          staff_x: tuple[int, int] | None = None, redrawn: bool = False,
          tab_rows: tuple[float, float] | None = None) -> Mosaic | None:
    """Mosaic of the band rows `rows` (crop coordinates) between t0 and t1, or None if it never moves.

    The playhead and the motion are measured in the crop rows `analysis_rows` only; `staff_x` is the staff's
    left and right end on screen. The mosaic ends early, with `moved_at` set, where the panel is redrawn with its
    staff at other rows. `redrawn` says it starts at such a redraw; `tab_rows` are the tab staff's first and last
    rows, whose lines are followed.
    """
    y0, y1 = rows
    a0, a1 = analysis_rows
    s0, s1 = staff_rows
    prev = prev_marks = None
    offset = 0.0
    strips: list[tuple[int, np.ndarray]] = []    # (frame, columns that crossed the slit at that frame)
    offsets, times, contrast = [], [], []
    head: list[np.ndarray] = []                  # the first END_FRAMES bands
    tail: list[tuple[int, np.ndarray]] = []      # the last END_FRAMES bands, with their frame index
    color_votes = None
    samples: list[np.ndarray] = []
    voted = 0
    every = max(1, int((t1 - t0) * GLIDE_FPS / PLAYHEAD_SAMPLES))
    slit = None
    last_int = 0
    last_step = 0.0
    lone = None                  # the step one pair of lines gave in the frame before, while the band was still
    fixed = fixed_columns(path, crop, staff_rows, polarity, thr, t0, t1)
    ref_lines = None
    moved = []                   # frames, in a row, whose staff lines sit elsewhere than in the last steady frame
    moved_at = None
    doubtful: list[int] = []     # frames whose step left the recent speed unconfirmed
    recent: list[float] = []     # the last SPEED_FRAMES steps
    off = False                  # the frame before took a step off the recent speed
    for t, f in frames(path, GLIDE_FPS, crop=crop, start=t0, length=t1 - t0):
        # The panel can be redrawn with its staffs at other heights; this piece ends there and the next one is
        # built on the new rows
        lines = line_rows(to_gray(f, polarity), thr)
        checked = MIN_LINES <= len(lines) <= MAX_LINES
        # The piece's own tab staff lines, followed as they drift; another staff above it can stay where it was
        # while the tab staff moves
        lo, hi = (tab_rows or (s0, s1)) if ref_lines is None else (min(ref_lines), max(ref_lines))
        mine = [y for y in lines if lo - gap <= y <= hi + gap]
        if checked and ref_lines is not None and shared(lines, ref_lines) < 0.5:
            # Frames END_FRAMES or more back are forgotten, so the run's first frame is still in `tail`
            moved = [m for m in moved if len(times) - m < END_FRAMES]
            moved.append(len(times))
            if len(moved) >= MOVED_FRAMES:
                moved_at = float(times[moved[0]]) if moved[0] < len(times) else float(t)
                break
        elif checked and len(mine) >= MIN_LINES:
            # Lines that drift a pixel at a time (a slow zoom) are followed
            ref_lines, moved = mine, []
        band = f[y0:y1]
        w = band.shape[1]
        if slit is None:
            slit = int(SLIT_SHARE * w)
        k = len(times)
        if k < END_FRAMES:
            head.append(band.copy())
        tail.append((k, band.copy()))
        if len(tail) > END_FRAMES:
            tail.pop(0)
        g = to_gray(band, polarity)
        staff = g[s0 - y0:s1 - y0]
        if k % every == 0:
            votes = colored(f[a0:a1]).mean(axis=0) > PLAYHEAD_SHARE
            color_votes = votes.astype(np.int32) if color_votes is None else color_votes + votes
            voted += 1
            samples.append(staff.copy())
        # Staff lines against the paper between them: a fade dims both, sparse or dense music changes neither
        rows_level = np.median(staff[:, ::4], axis=1)
        contrast.append(float(rows_level.max() - rows_level.min()))
        # Motion is measured two ways. Right of the slit the upcoming music is drawn at full contrast, but it can be
        # sparse; the whole staff width, with colored marks and ink fixed to the screen (a playhead line, a clef) left
        # out, still holds music there. When both match, the step nearer the one before is taken: the band's speed
        # changes smoothly, and a stray match is far off
        g = np.clip((thr + 40 - staff.astype(np.float32)) / 80.0, 0, 1)
        # Staff lines match at any shift, and their cut-off end at the slit would pull the match towards no movement
        g[staff_lines(staff, thr, gap)] = 0
        g_slit = g.copy()
        g_slit[:, :slit] = 0
        g[colored(band[s0 - y0:s1 - y0])] = 0
        if fixed is not None:
            g[:, fixed] = 0
        content = (g_slit, g)
        marks = vertical_marks(staff, thr, gap, fixed)
        if prev is not None:
            # Vertical lines (barlines, stems) moving with the band settle it where sustained notes drawn as long
            # lines match at any shift
            crossed = line_step(prev_marks, marks, last_step, MAX_STEP_SHARE * w)
            crossed, lone = start_guard(crossed, last_step, lone)
            # Phase correlation is needed only where the lines settle nothing
            steps = []
            if crossed is None:
                (dx, _), resp = cv2.phaseCorrelate(prev[0], g_slit)
                if (resp > MIN_RESPONSE and abs(dx) < MAX_STEP_SHARE * w
                        and ink_map(staff[:, slit:], thr, gap).sum() >= gap * gap):
                    steps.append(max(-dx, 0.0))
                (dx, _), resp = cv2.phaseCorrelate(prev[1], g)
                if (resp > MIN_RESPONSE and abs(dx) < MAX_STEP_SHARE * w
                        and float(g.sum()) >= LOW_NOTES_GAPS * gap * gap):
                    steps.append(max(-dx, 0.0))
            was_off, off = off, False
            if crossed is not None:
                last_step = crossed[1]
            elif steps:
                last_step = min(steps, key=lambda d: abs(d - last_step))
                # A step far off the recent speed may have dropped or doubled columns, unless both phase
                # correlations agree on it; a run of such steps is marked once. A frame shown twice (a dropped
                # frame in the video) truly doesn't move
                speed = float(np.median(recent)) if recent else 0.0
                both = len(steps) == 2 and abs(steps[0] - steps[1]) <= max(BOTH_PX, START_CHANGE * last_step)
                repeated = last_step < MIN_SHIFT_PX and float(np.abs(g - prev[1]).mean()) < REPEATED_LEVEL
                off = (speed >= MIN_SHIFT_PX and abs(last_step - speed) > MARK_CHANGE * speed and not both
                       and not repeated)
                if off and not was_off:
                    doubtful.append(k)
            recent = (recent + [last_step])[-SPEED_FRAMES:]
            # No clear match (a bar of rests, a fade): the band keeps the speed it had
            step = last_step
            offset += step
            new_int = int(round(offset))
            if new_int > last_int:
                # Columns that crossed the slit since the frame before, in canvas order
                strips.append((k, band[:, slit - (new_int - last_int):slit].copy()))
                last_int = new_int
        offsets.append(offset)
        times.append(t)
        prev, prev_marks = content, marks
    if moved_at is not None:
        # Keep only the frames before the staff moved
        keep = moved[0]
        times, offsets, contrast = times[:keep], offsets[:keep], contrast[:keep]
        strips = [(k, s) for k, s in strips if k < keep]
        tail = [(k, b) for k, b in tail if k < keep]
        head = head[:keep]
    if not times or offsets[-1] < 2 * gap:
        return None
    n = len(times)
    # The first and last clear frames: a fade in or out at either end loses barlines
    typical = float(np.median(contrast))
    clear = [c >= CLEAR_SHARE * typical for c in contrast]
    # A piece that starts or ends where the panel was redrawn joins the next piece there, so it keeps every frame
    k0 = 0 if redrawn else next((k for k in range(min(END_FRAMES, n)) if clear[k]), 0)
    k1 = n - 1 if moved_at is not None else next((k for k, _ in reversed(tail) if clear[k]), n - 1)
    first, last = head[k0], dict(tail)[k1]
    if color_votes.max() > 0.5 * voted:
        ref_x = float(np.argmax(color_votes))
    else:
        ref_x = still_playhead(samples, staff_x or (0, w), gap)
    if ref_x is None:
        # No playhead: measures are said to be played at the panel's centre, and the whole first screen is read
        ref_x, start = w / 2, 0
    elif ref_x < slit:
        start = max(0, int(ref_x - SEED_BACK_GAPS * gap))
    else:
        start = slit
    kept = [s for k, s in strips if k0 < k <= k1]
    image = np.concatenate([first[:, start:slit]] + kept + [last[:, slit:]], axis=1)
    off = np.array(offsets[k0:k1 + 1]) - offsets[k0]
    # A frame's slit column sits on the canvas as far right of the first frame's as the band has moved since
    unsure = [slit - start + offsets[k] - offsets[k0] for k in doubtful if k0 < k <= k1]
    return Mosaic(image, y0, off, np.array(times[k0:k1 + 1]), ref_x, int(ref_x) - start, slit - start,
                  image.shape[1] - (last.shape[1] - slit), moved_at=moved_at, until=float(times[-1]),
                  until_offset=float(offsets[-1] - offsets[k0]), doubtful=unsure)


def vertical_marks(staff: np.ndarray, thr: float, gap: float, fixed: np.ndarray | None) -> np.ndarray:
    """Middle columns of the vertical lines at least MARK_GAPS line gaps tall on the staff rows, leaving out those
    fixed to the screen. Barlines are often drawn fainter than notes, so marks halfway to the paper count."""
    ink = (staff < (thr + float(np.median(staff[::4, ::4]))) / 2).astype(np.uint8)
    tall = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(MARK_GAPS * gap))))
    cols = tall.any(axis=0)
    if fixed is not None:
        cols &= ~fixed
    xs = np.flatnonzero(cols)
    if not len(xs):
        return xs
    # One line is a run of neighboring columns
    runs = np.split(xs, np.flatnonzero(np.diff(xs) > 1) + 1)
    return np.array([r.mean() for r in runs if len(r) <= MARK_WIDTH])


def line_step(before: np.ndarray, now: np.ndarray, last: float, most: float) -> tuple[int, float] | None:
    """(lines, step): the step that carries vertical lines of one frame onto those of the next, nearest the step
    before, and how many lines agree on it. It is the shift most lines agree on, when at least MARK_AGREE of them do
    and it is within MARK_CHANGE of the step before (any shift when the band was still)."""
    if len(before) == 0 or len(now) == 0:
        return None
    d = (before[:, None] - now[None, :]).ravel()
    d = np.sort(d[(d > -1.0) & (d < most)])
    if not len(d):
        return None
    # How many shifts lie within a pixel of each: the shift most lines agree on, nearest the step before on a tie
    agree = np.searchsorted(d, d + 1.0, side="right") - np.searchsorted(d, d - 1.0, side="left")
    top = agree.max()
    if top < MARK_AGREE:
        return None
    c = min(d[agree == top], key=lambda x: abs(x - last))
    best = (int(top), float(d[np.abs(d - c) <= 1.0].mean()))
    if last > 0 and abs(best[1] - last) > MARK_CHANGE * last:
        return None
    return best[0], max(best[1], 0.0)


def line_rows(gray: np.ndarray, thr: float) -> list[int]:
    """Middle rows of the thin lines running across the frame (staff lines): rows whose median gray is well
    below the rows a few pixels above and below. Footage and dark blocks are thick, so they are left out."""
    med = np.median(gray[:, ::16], axis=1).astype(np.float32)
    k = LINE_HALF
    rows = [y for y in range(k, len(med) - k) if med[y] < min(med[y - k], med[y + k]) - LINE_DEPTH]
    # One line is a run of neighboring rows; its middle stands for it
    lines: list[list[int]] = []
    for y in rows:
        if lines and y - lines[-1][-1] <= 1:
            lines[-1].append(y)
        else:
            lines.append([y])
    return [int(np.mean(run)) for run in lines]


def shared(lines: list[int], ref: list[int]) -> float:
    """Share of the reference lines found again within two pixels."""
    return sum(any(abs(y - z) <= 2 for z in lines) for y in ref) / len(ref)


def start_guard(crossed: tuple[int, float] | None, last_step: float,
                lone: float | None) -> tuple[tuple[int, float] | None, float | None]:
    """(crossed, lone): while the band is still, one pair of lines can match at any distance, so a line step of at
    least a pixel from fewer than START_MARKS lines is kept only when `lone`, the step one pair gave in the frame
    before, is within START_CHANGE of it. The second value is the `lone` for the next frame."""
    if crossed is not None and last_step == 0 and crossed[0] < START_MARKS and crossed[1] >= 1.0:
        agreed = lone is not None and abs(crossed[1] - lone) <= START_CHANGE * max(lone, crossed[1])
        return (crossed if agreed else None), crossed[1]
    return crossed, None
