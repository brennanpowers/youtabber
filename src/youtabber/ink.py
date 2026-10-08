import cv2
import numpy as np

# Solid notation is darker than this; light gray staff lines and measure numbers can be lighter
DARK = 150
# Anything noticeably darker than paper, light gray lines and numbers included
FAINT = 200
# Long enough to be a staff line rather than part of a note or digit
STAFF_LINE_RUN = 51


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
