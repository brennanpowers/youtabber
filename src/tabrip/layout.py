from pathlib import Path

import numpy as np
from PIL import Image

from tabrip.stitch import Placed

PAGE_ASPECT = 11 / 8.5
MARGIN = 80
ROW_GAP = 60
# Darker than paper; loose enough to catch light gray staff lines
INK = 200


def staff_lines(strip: np.ndarray) -> list[int]:
    """Rows of the strip holding staff lines, which run nearly its whole length."""
    rows = np.flatnonzero((strip < INK).mean(axis=1) > 0.5)
    lines: list[int] = []
    for r in rows:
        if not lines or r > lines[-1] + 2:
            lines.append(int(r))
    return lines


def barlines(strip: np.ndarray, lines: list[int]) -> list[tuple[int, int]]:
    """(first, last) columns of each barline, found where ink spans the bottom staff top to bottom."""
    if len(lines) < 2:
        return []
    # The bottom staff is the tab: the last run of evenly spaced lines
    gaps = np.diff(lines)
    spacing = int(np.median(gaps))
    top = len(lines) - 1
    while top > 0 and gaps[top - 1] <= spacing * 1.5:
        top -= 1
    y0, y1 = lines[top], lines[-1]
    full = (strip[y0:y1 + 1] < INK).mean(axis=0) > 0.95
    bars: list[tuple[int, int]] = []
    for x in np.flatnonzero(full):
        # Columns close together belong to one thick or double barline
        if bars and x <= bars[-1][1] + 12:
            bars[-1] = (bars[-1][0], int(x))
        else:
            bars.append((int(x), int(x)))
    return bars


def rows(strip: np.ndarray, placed: list[Placed], width: int) -> list[tuple[int, int]]:
    """Split the strip into (start, end) column ranges no wider than `width`."""
    bars = barlines(strip, staff_lines(strip))
    # Each page turn starts a new line of music with its own clef, so always cut there
    turns = [p.x for p in placed if p.new_from == 0] + [strip.shape[1]]
    out = []
    for seg_start, seg_end in zip(turns, turns[1:]):
        start = seg_start
        while start < seg_end:
            limit = min(start + width, seg_end)
            if limit == seg_end:
                out.append((start, seg_end))
                break
            fits = [b for b in bars if start < b[0] and b[1] < limit]
            if fits:
                # End the row after the barline and start the next one on it, so measure numbers stay whole
                first, last = fits[-1]
                out.append((start, last + 1))
                start = first
            else:
                out.append((start, limit))
                start = limit
    return out


def whiten(strip: np.ndarray) -> np.ndarray:
    """Scale brightness so the video's off-white paper prints as white."""
    paper = float(np.median(strip))
    out = np.clip(strip.astype(np.float32) * (255 / paper), 0, 255)
    out[out > 240] = 255
    return out.astype(np.uint8)


def write_pdf(strip: np.ndarray, row_ranges: list[tuple[int, int]], width: int, out: Path, title: str) -> None:
    strip = whiten(strip)
    page_w = width + 2 * MARGIN
    page_h = int(page_w * PAGE_ASPECT)
    row_h = strip.shape[0]
    pages: list[Image.Image] = []
    y = page_h
    for start, end in row_ranges:
        if y + row_h > page_h - MARGIN:
            pages.append(Image.new("L", (page_w, page_h), 255))
            y = MARGIN
        pages[-1].paste(Image.fromarray(strip[:, start:end]), (MARGIN, y))
        y += row_h + ROW_GAP
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=150, title=title)
