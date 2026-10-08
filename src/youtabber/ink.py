import cv2
import numpy as np

# Solid notation is darker than this; light gray staff lines and measure numbers can be lighter
DARK = 150
# Anything noticeably darker than paper, light gray lines and numbers included
FAINT = 200
# Long enough to be a staff line rather than part of a note or digit
STAFF_LINE_RUN = 51
# How much darker than paper a staff line must be. Some videos draw the tab lines very light.
LINE_CONTRAST = 25


def near(mask: np.ndarray) -> np.ndarray:
    """The mask grown by one pixel in every direction, to forgive compression shimmer at edges."""
    return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def difference(a: np.ndarray, b: np.ndarray) -> float:
    """Share of ink in either mask with no ink within one pixel in the other."""
    changed = max((a & ~near(b)).sum(), (b & ~near(a)).sum())
    return changed / max(a.sum(), b.sum(), 1)


def horizontal_lines(mask: np.ndarray, length: int) -> np.ndarray:
    """The parts of the mask that are horizontal runs at least `length` pixels long."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)


def without_staff_lines(image: np.ndarray) -> np.ndarray:
    """Everything darker than paper except staff lines, which look the same wherever you are."""
    ink = image < FAINT
    return ink & ~horizontal_lines(ink, STAFF_LINE_RUN)


def staff_lines(image: np.ndarray) -> list[int]:
    """Rows of the image holding staff lines, which run nearly its whole width."""
    paper = np.median(image)
    rows = np.flatnonzero((image < paper - LINE_CONTRAST).mean(axis=1) > 0.5)
    lines: list[int] = []
    prev = None
    for r in rows:
        # Rows next to each other, allowing a one-row gap, are one thick line
        if prev is None or r > prev + 2:
            lines.append(int(r))
        prev = r
    return lines


def tab_staff(lines: list[int]) -> list[int]:
    """Lines of the bottom staff, which is the tab: the last run of at least three evenly spaced lines.

    Needing three skips lone long lines below the tab, such as the bar of a rhythm bracket.
    """
    if len(lines) < 3:
        return []
    spacing = np.median(np.diff(lines))
    runs = [[lines[0]]]
    for prev, line in zip(lines, lines[1:]):
        if line - prev <= spacing * 1.5:
            runs[-1].append(line)
        else:
            runs.append([line])
    staffs = [run for run in runs if len(run) >= 3]
    return staffs[-1] if staffs else []
