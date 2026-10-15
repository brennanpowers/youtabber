"""Turn a band that glides past a fixed playhead into one long still image (a mosaic), with a time map.

The band's movement between frames is measured on the music to the right of a fixed reading column
(the slit), where the upcoming music is drawn at full contrast. Each frame adds the strip of columns that
crossed the slit since the frame before, so the playhead and anything else fixed to the screen stay out
of those strips. The mosaic starts with the first frame from a few line gaps left of the playhead to the
slit (or at the slit when no playhead is found), which holds the playhead and some of the dimmed played
zone, and ends with the last frame right of the slit.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from youtabber.observe import colored, ink_map, to_gray
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


# The canvas starts this many line gaps left of the playhead, so a first measure starting just before it is whole
SEED_BACK_GAPS = 4.0


@dataclass
class Mosaic:
    image: np.ndarray           # BGR, band rows of the crop, canvas columns
    y0: int                     # crop row of the mosaic's first row
    offsets: np.ndarray         # how far the band has moved since the first sampled frame, per frame
    times: np.ndarray           # time of each sampled frame
    ref_x: float                # screen x (crop coordinates) where measures are said to be played
    slit: int                   # screen x of the reading column
    lead: int                   # canvas columns that sat left of ref_x in the first frame
    seeded: int = 0             # canvas columns taken from the first frame alone
    tail_from: int = 0          # first canvas column taken from the last frame alone

    def time_at(self, canvas_x: float) -> float:
        """When the canvas column reached ref_x on screen. Columns still right of it when the band last moved get
        the time the band's average speed until then would bring them there."""
        reached = canvas_x - self.lead
        if self.offsets[-1] <= self.offsets[0]:
            return float(self.times[0])
        # Frames after the band's last move (blank staff past the end, or the video holding still) add no time
        k = int(np.argmax(self.offsets >= self.offsets[-1] - 1.0))
        if reached <= self.offsets[k]:
            return float(np.interp(reached, self.offsets[:k + 1], self.times[:k + 1]))
        speed = (self.offsets[k] - self.offsets[0]) / max(float(self.times[k] - self.times[0]), 1e-6)
        return float(self.times[k] + (reached - self.offsets[k]) / speed)


def build(path: Path, crop: tuple[int, int, int, int], rows: tuple[int, int], staff_rows: tuple[int, int],
          polarity: str, thr: float, gap: float, t0: float, t1: float) -> Mosaic | None:
    """Mosaic of the band rows `rows` (crop coordinates) between t0 and t1, or None if it never moves."""
    y0, y1 = rows
    prev = None
    offset = 0.0
    strips: list[np.ndarray] = []
    offsets, times = [], []
    color_votes = None
    slit = None
    last_int = 0
    last_step = 0.0
    first = last = None
    for t, f in frames(path, GLIDE_FPS, crop=crop, start=t0, length=t1 - t0):
        band = f[y0:y1]
        w = band.shape[1]
        if slit is None:
            slit = int(SLIT_SHARE * w)
            first = band.copy()
        votes = colored(band).mean(axis=0) > PLAYHEAD_SHARE
        color_votes = votes.astype(np.int32) if color_votes is None else color_votes + votes
        g = to_gray(band, polarity)
        content = (np.clip((thr + 40 - g.astype(np.float32)) / 80.0, 0, 1))
        content[:, :slit] = 0
        # Motion is measured on the staffs only: lyrics, a fretboard or a title below them stay still
        content[:staff_rows[0] - y0] = 0
        content[staff_rows[1] - y0:] = 0
        if prev is not None:
            (dx, _), resp = cv2.phaseCorrelate(prev, content)
            # Right of the slit there is nothing but staff lines (past the last measure): the lines match at any
            # shift, so the band keeps the speed it had
            notes = ink_map(g[staff_rows[0] - y0:staff_rows[1] - y0, slit:], thr, gap)
            if notes.sum() >= gap * gap and resp > MIN_RESPONSE and abs(dx) < MAX_STEP_SHARE * w:
                step = max(-dx, 0.0)
                last_step = step
            else:
                # No clear match (a bar of rests, a fade): the band keeps the speed it had
                step = last_step
            offset += step
            new_int = int(round(offset))
            if new_int > last_int:
                # Columns that crossed the slit since the frame before, in canvas order
                strips.append(band[:, slit - (new_int - last_int):slit].copy())
                last_int = new_int
        offsets.append(offset)
        times.append(t)
        prev = content
        last = band
    if first is None or offsets[-1] < 2 * gap:
        return None
    n = len(times)
    ref_x = float(np.argmax(color_votes)) if color_votes.max() > 0.5 * n else float(slit)
    # Music already left of the slit in the first frame, and still right of it in the last, never crossed it
    start = max(0, int(ref_x - SEED_BACK_GAPS * gap)) if ref_x < slit else slit
    image = np.concatenate([first[:, start:slit]] + strips + [last[:, slit:]], axis=1)
    return Mosaic(image, y0, np.array(offsets), np.array(times), ref_x, slit, int(ref_x) - start, slit - start,
                  image.shape[1] - (last.shape[1] - slit))
