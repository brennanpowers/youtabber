"""Read a bar-counter widget ("BAR 12 / 27") that some play-along apps draw beside a glide, as a time series.

The counter names the measure under the playhead, so it numbers glided measures by when they are played, and it
tells a section played again from new music where the strip itself shows the same bars twice.
"""
import re
import threading
from pathlib import Path

import cv2
import numpy as np

from youtabber import ocr
from youtabber.video import frames

# Frames searched for the widget, spread over the glide, and how many must show it at one place
FIND_FRAMES, FIND_AGREE = 9, 2
# Readings per second of the counter over the glide
READ_FPS = 4
# The widget's text: the word, the measure, a slash and the count. OCR reads a 1 as I, l or | at times.
# "BAR" after a letter is part of another word ("Crowbar"), and a count followed by a digit is a longer number
PATTERN = re.compile(r"(?<![A-Z])BAR\s*([0-9Il|]{1,3})\s*/\s*([0-9Il|]{1,3})(?![0-9])", re.IGNORECASE)
# Readings of one measure further apart than this are separate showings of it
NEIGHBOR_SECONDS = 1.5
# A jump in the count is believed when it is shown for at least this share of a typical measure's time
JUMP_SHARE = 0.5


def _digits(text: str) -> str:
    return text.replace("I", "1").replace("l", "1").replace("|", "1")


def parse(text: str, total: int | None = None) -> tuple[int, int] | None:
    """(measure, count) from the widget's text, or None. With the count known, a slash read as a 1 ("5 127" for
    "5 / 27") and a stray letter before the measure ("BARA4") are allowed for."""
    t = text.replace(" ", "")
    m = PATTERN.search(t)
    if m:
        n, count = int(_digits(m.group(1))), int(_digits(m.group(2)))
        if 0 <= n <= count and (total is None or count == total):
            return n, count
    if total is None:
        return None
    m = re.search(r"(?<![A-Z])BAR\D?([0-9Il|]{1,3})[/1]" + str(total) + r"$", t, re.IGNORECASE)
    if m and 0 <= int(_digits(m.group(1))) <= total:
        return int(_digits(m.group(1))), total
    return None


def find(path: Path, t0: float, t1: float,
         stop: threading.Event | None = None) -> tuple[tuple[int, int, int, int], int] | None:
    """((x0, y0, x1, y1), count) of the number part of a bar counter shown through t0..t1, or None. Gives None
    once `stop` is set."""
    seen = []
    for t in np.linspace(t0 + 1, max(t0 + 1, t1 - 1), FIND_FRAMES):
        if stop is not None and stop.is_set():
            return None
        got = next(iter(frames(path, 2, start=float(t), length=0.5)), None)
        if got is None:
            continue
        result, _ = ocr.engine()(np.ascontiguousarray(got[1]))
        for box, text, score in result or []:
            read = parse(text)
            if read is not None and float(score) >= 0.5:
                pts = np.array(box)
                seen.append(((*pts.min(0).astype(int), *pts.max(0).astype(int)), read[1]))
    if not seen:
        return None
    # The same place and count in most of the searched frames
    box, total = seen[0]
    same = [b for b, n in seen if n == total and abs(b[1] - box[1]) < 10 and abs(b[0] - box[0]) < 40]
    if len(same) < FIND_AGREE:
        return None
    x0, y0, x1, y1 = (int(np.median([b[i] for b in same])) for i in range(4))
    return (x0, y0, x1, y1), total


def series(path: Path, t0: float, t1: float, stop: threading.Event | None = None) -> list[tuple[float, float, int]]:
    """(start, end, measure) runs of the bar counter over t0..t1; empty when no counter is shown, or once `stop`
    is set, so a caller that gave up on the result needn't wait for the whole read."""
    found = find(path, t0, t1, stop)
    if found is None:
        return []
    (x0, y0, x1, y1), total = found
    pad = (y1 - y0) // 4
    out = []
    for t, f in frames(path, READ_FPS, start=t0, length=t1 - t0):
        # Leaving the loop closes the frame reader, which stops its ffmpeg
        if stop is not None and stop.is_set():
            return []
        crop = cv2.cvtColor(f[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad], cv2.COLOR_BGR2GRAY)
        # The widget may be light on dark; the reader wants dark digits on light paper
        if np.median(crop) < 128:
            crop = 255 - crop
        text, score = ocr.read_line(crop)
        read = parse(text, total) if score >= 0.5 else None
        if read is not None and read[1] == total:
            out.append((float(t), read[0]))
    return steady(out)


def steady(readings: list[tuple[float, int]]) -> list[tuple[float, float, int]]:
    """(start, end, measure) runs of the counter that fit a count: one up from the run before, or a jump (a section
    played again, or a measure whose number went unread) held about as long as a measure and counted on from.
    OCR drops a digit at times ("11" read as "1"); such a run neither follows the count nor is counted on from."""
    runs: list[list] = []
    for t, n in readings:
        if runs and runs[-1][2] == n and t - runs[-1][1] <= NEIGHBOR_SECONDS:
            runs[-1][1] = t
        else:
            runs.append([t, t, n])
    lengths = [b - a for a, b, _ in runs if b > a]
    typical = float(np.median(lengths)) if lengths else 0.0
    kept: list[list] = []
    for k, (a, b, n) in enumerate(runs):
        if kept and n == kept[-1][2]:
            kept[-1][1] = b
            continue
        follows = not kept or n == kept[-1][2] + 1
        counted_on = any(m == n + 1 for _, _, m in runs[k + 1:k + 4])
        if follows or (b - a >= JUMP_SHARE * typical and counted_on):
            kept.append([a, b, n])
    return [tuple(r) for r in kept]


def at(runs: list[tuple[float, float, int]], t: float) -> int | None:
    """The measure the counter showed at time t, if a kept run covers it (to half a reading step either side)."""
    for a, b, n in runs:
        if a - 0.5 / READ_FPS <= t <= b + 0.5 / READ_FPS:
            return n
    return None
