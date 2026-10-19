"""Turn a band that glides past a fixed playhead into one long still image (a mosaic), with a time map.

The band's movement between frames is measured on the music to the right of a fixed reading column
(the slit), where the upcoming music is drawn at full contrast. Each frame adds the strip of columns that
crossed the slit since the frame before, so the playhead and anything else fixed to the screen stay out
of those strips. The mosaic starts with the first clear frame from a few line gaps left of the playhead to the
slit (from the panel's left edge when no playhead is found), and ends with the last clear frame right of the
slit, so music on screen before the scroll starts or after it stops is read too.
"""
from dataclasses import dataclass
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
# With only this many square line gaps of ink right of the slit, one frame's shift isn't trusted, but a run of frames
# with the same shift (within AGREE_SHARE) is: START_FRAMES of them start a band that isn't moving (a sparse intro),
# and CHANGE_FRAMES of them, further than CHANGE_SHARE from the speed it had, change its speed (a tempo change
# during bars of rest)
LOW_NOTES_GAPS, AGREE_SHARE = 0.25, 0.1
START_FRAMES, CHANGE_FRAMES, CHANGE_SHARE = 2, 5, 0.25
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

# The canvas starts this many line gaps left of the playhead, so a first measure starting just before it is whole
SEED_BACK_GAPS = 4.0

# Short windows sampled across a segment to tell whether its tab glides, and the frames read in each
MOTION_WINDOWS, MOTION_FRAMES = 12, 10
# A frame pair moves when the band shifts left by at least this many pixels (5 px/s at GLIDE_FPS)
MIN_SHIFT_PX = 0.5
# A window moves when this share of its frame pairs do, and a segment glides when this share of its windows move.
# Still pages show no moving window; a glide's still intro, outro and bars of rest leave most windows moving.
MOVING_PAIRS, MOVING_WINDOWS = 2 / 3, 0.5


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

    def time_at(self, canvas_x: float) -> float:
        """When the canvas column reached ref_x on screen. Columns still right of it when the band last moved get
        the time the band's average speed until then would bring them there."""
        reached = canvas_x - self.lead
        if self.offsets[-1] <= self.offsets[0]:
            return float(self.times[0])
        k = self.last_move()
        if reached <= self.offsets[k]:
            return float(np.interp(reached, self.offsets[:k + 1], self.times[:k + 1]))
        speed = (self.offsets[k] - self.offsets[0]) / max(float(self.times[k] - self.times[0]), 1e-6)
        return float(self.times[k] + (reached - self.offsets[k]) / speed)

    def last_move(self) -> int:
        """The frame after which the band no longer moves (blank staff past the end, or the video holding still)."""
        return int(np.argmax(self.offsets >= self.offsets[-1] - 1.0))

    def seen_at(self, canvas_x: float) -> tuple[float, float] | None:
        """(time, screen x) of a canvas column that never crosses ref_x while the band moves: on the first screen
        left of the play column, or still right of it when the band stops. None for a column that crosses it."""
        reached = canvas_x - self.lead
        k = self.last_move()
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
    colored, so it is left out of the comparison, and so are the staff lines, which match at any shift.
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
            prev = None
            pairs = 0
            for f in shots:
                band = f[y0:y1]
                g = to_gray(band, polarity)
                content = np.clip((thr + 40 - g.astype(np.float32)) / 80.0, 0, 1)
                content[colored(band)] = 0
                content[staff_lines(g, thr, gap)] = 0
                if prev is not None:
                    (dx, _), resp = cv2.phaseCorrelate(prev, content)
                    pairs += resp > MIN_RESPONSE and dx <= -MIN_SHIFT_PX
                prev = content
            if pairs >= MOVING_PAIRS * (len(shots) - 1):
                moving += 1
                break
    return sampled > 0 and moving >= MOVING_WINDOWS * sampled


def staff_lines(gray: np.ndarray, thr: float, gap: float) -> np.ndarray:
    """The staff lines' pixels, a row either side included: dark runs at least three line gaps long."""
    lines = cv2.morphologyEx((gray < thr).astype(np.uint8), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (int(3 * gap), 1)))
    return cv2.dilate(lines, np.ones((3, 1), np.uint8)).astype(bool)


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
          staff_x: tuple[int, int] | None = None) -> Mosaic | None:
    """Mosaic of the band rows `rows` (crop coordinates) between t0 and t1, or None if it never moves.

    The playhead and the motion are measured in the crop rows `analysis_rows` only; `staff_x` is the staff's
    left and right end on screen.
    """
    y0, y1 = rows
    a0, a1 = analysis_rows
    s0, s1 = staff_rows
    prev = None
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
    low: list[float] = []     # shifts of the frames just before, when each had too little ink to trust alone
    for t, f in frames(path, GLIDE_FPS, crop=crop, start=t0, length=t1 - t0):
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
        # Motion is measured on the staffs right of the slit only: lyrics, a fretboard or a title below them stay
        # still, and the playhead sits left of the slit. The columns left of the slit stay in, blank, so a
        # repeating pattern of barlines can't match at a shift that wraps round the edge.
        content = np.clip((thr + 40 - staff.astype(np.float32)) / 80.0, 0, 1)
        content[:, :slit] = 0
        # Staff lines are cut off at the slit, and that fixed edge would pull the match towards no movement
        content[staff_lines(staff, thr, gap)] = 0
        if prev is not None:
            (dx, _), resp = cv2.phaseCorrelate(prev, content)
            # Right of the slit there is nothing but staff lines (past the last measure): the lines match at any
            # shift, so the band keeps the speed it had
            notes = ink_map(staff[:, slit:], thr, gap).sum()
            matched = resp > MIN_RESPONSE and abs(dx) < MAX_STEP_SHARE * w
            if matched and notes >= gap * gap:
                last_step, low = max(-dx, 0.0), []
            elif matched and notes >= LOW_NOTES_GAPS * gap * gap:
                shift = max(-dx, 0.0)
                low.append(shift)
                need = START_FRAMES if last_step == 0 else CHANGE_FRAMES
                run = low[-need:]
                if (len(run) == need and shift > 0
                        and max(run) - min(run) <= AGREE_SHARE * shift
                        and (last_step == 0 or abs(shift - last_step) > CHANGE_SHARE * last_step)):
                    last_step = shift
            else:
                low = []
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
        prev = content
    if not times or offsets[-1] < 2 * gap:
        return None
    n = len(times)
    # The first and last clear frames: a fade in or out at either end loses barlines
    typical = float(np.median(contrast))
    clear = [c >= CLEAR_SHARE * typical for c in contrast]
    k0 = next((k for k in range(min(END_FRAMES, n)) if clear[k]), 0)
    k1 = next((k for k, _ in reversed(tail) if clear[k]), n - 1)
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
    return Mosaic(image, y0, off, np.array(times[k0:k1 + 1]), ref_x, int(ref_x) - start, slit - start,
                  image.shape[1] - (last.shape[1] - slit))
