from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from tabrip.video import frames, probe

SCALE = 0.5
MAX_SAMPLES = 150
EDGE_TRIM = 4


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    w: int
    h: int


@dataclass
class RegionAnalysis:
    region: Region
    median: np.ndarray  # median frame at SCALE, for the debug image


def thin_dark_mask(gray: np.ndarray) -> np.ndarray:
    """Mask of features darker than the pixels just above and below them, such as staff lines."""
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    return blackhat > 20


def detect(path: Path) -> RegionAnalysis:
    info = probe(path)
    fps = min(1.0, MAX_SAMPLES / info.duration)
    sampled = [f for _, f in frames(path, fps=fps, scale=SCALE)]
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in sampled])
    h, w = gray.shape[1:]

    # Notes break staff lines in any one frame, but they move while the lines stay put,
    # so averaging over the samples fills the gaps before looking for long horizontal runs
    line_rate = np.mean([thin_dark_mask(g) for g in gray], axis=0)
    long_runs = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1))
    staff = cv2.morphologyEx((line_rate > 0.3).astype(np.uint8), cv2.MORPH_OPEN, long_runs).astype(bool)
    rows = np.flatnonzero(staff.sum(axis=1) > w // 8)
    if rows.size == 0:
        raise RuntimeError(f"No staff lines found in {path.name}; pass --region x,y,w,h to set the tab area")
    cols = np.flatnonzero(staff[rows].any(axis=0))
    x0, x1 = cols.min(), cols.max() + 1
    y0, y1 = rows.min(), rows.max() + 1

    # Paper is bright in the median frame and rarely changes between samples
    median = np.median(np.stack(sampled), axis=0).astype(np.uint8)
    change = (np.abs(np.diff(gray.astype(np.int16), axis=0)) > 25).mean(axis=0)
    paper = (cv2.cvtColor(median, cv2.COLOR_BGR2GRAY) > 190) & (change < 0.2)
    paper_rows = paper[:, x0:x1].mean(axis=1) > 0.9

    # Grow up and down from the staff band to take in chord names, measure numbers and section labels
    while y0 > 0 and paper_rows[y0 - 1]:
        y0 -= 1
    while y1 < h and paper_rows[y1]:
        y1 += 1

    # Then sideways, to take in clefs and brackets left of where the staff lines start
    paper_cols = paper[y0:y1].mean(axis=0) > 0.9
    while x0 > 0 and paper_cols[x0 - 1]:
        x0 -= 1
    while x1 < w and paper_cols[x1]:
        x1 += 1

    s = 1 / SCALE
    # Trim the edges, where video borders and box outlines would look like barlines
    m = EDGE_TRIM
    region = Region(int(x0 * s) + m, int(y0 * s) + m, int((x1 - x0) * s) - 2 * m, int((y1 - y0) * s) - 2 * m)
    return RegionAnalysis(region, median)


def debug_image(analysis: RegionAnalysis, out: Path, override: Region | None = None) -> None:
    """Save the median frame with the tab area outlined: red when detected, green when set by hand."""
    img = analysis.median.copy()
    r = override or analysis.region
    pt0 = (int(r.x * SCALE), int(r.y * SCALE))
    pt1 = (int((r.x + r.w) * SCALE) - 1, int((r.y + r.h) * SCALE) - 1)
    cv2.rectangle(img, pt0, pt1, (0, 255, 0) if override else (0, 0, 255), 2)
    cv2.imwrite(str(out), img)
