"""Turn a band that glides past a fixed playhead into one long still image (a mosaic), with a time map.

The band's movement between frames is measured on the music to the right of a fixed reading column
(the slit), where the upcoming music is drawn at full contrast. Each frame adds the strip of columns that
passed the slit since the frame before, so the playhead, the dimmed played zone and anything else fixed
to the screen stay out of the mosaic.
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


@dataclass
class Mosaic:
    image: np.ndarray           # BGR, band rows of the crop, canvas columns
    y0: int                     # crop row of the mosaic's first row
    offsets: np.ndarray         # how far the band has moved since the first sampled frame, per frame
    times: np.ndarray           # time of each sampled frame
    ref_x: float                # screen x (crop coordinates) where measures are said to be played
    slit: int                   # screen x of the slit; canvas column 0 is what sat there in the first frame

    def time_at(self, canvas_x: float) -> float:
        """When the canvas column was at the reference x on screen."""
        pos = self.offsets + self.ref_x - self.slit   # canvas column under the reference, per frame
        if pos[-1] <= pos[0]:
            return float(self.times[0])
        return float(np.interp(canvas_x, pos, self.times))


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
    for t, f in frames(path, GLIDE_FPS, crop=crop, start=t0, length=t1 - t0):
        band = f[y0:y1]
        w = band.shape[1]
        if slit is None:
            slit = int(SLIT_SHARE * w)
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
            if resp > MIN_RESPONSE and abs(dx) < MAX_STEP_SHARE * w:
                step = max(-dx, 0.0)
                last_step = step
            else:
                # No clear match (a bar of rests, a fade): the band keeps the speed it had
                step = last_step
            offset += step
        new_int = int(round(offset))
        if prev is None:
            strips.append(band[:, slit:slit + 1].copy())
        elif new_int > last_int:
            # Columns that moved past the slit since the last frame, in canvas order
            strips.append(band[:, slit:slit + (new_int - last_int)].copy())
        last_int = new_int if prev is not None else 0
        offsets.append(offset)
        times.append(t)
        prev = content
    if not strips or offsets[-1] < 2 * gap:
        return None
    image = np.concatenate(strips, axis=1)
    n = max(len(times), 1)
    ref_x = float(np.argmax(color_votes)) if color_votes is not None and color_votes.max() > 0.5 * n else slit
    return Mosaic(image, y0, np.array(offsets), np.array(times), ref_x, slit)
