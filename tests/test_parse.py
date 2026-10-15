import cv2
import numpy as np

from youtabber.parse import parse_view


def plate(light_line: bool) -> np.ndarray:
    """A tab-only staff with black barlines at x 100, 400 and 700, and optionally a light vertical line at 250."""
    img = np.full((200, 800, 3), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (20, 60 + i * 14), (780, 60 + i * 14), (60, 60, 60), 1)
    for x in (100, 400, 700):
        cv2.line(img, (x, 60), (x, 130), (0, 0, 0), 2)
    if light_line:
        # Only across the staff, as a playhead line is; a line running on below it is dropped as a stem anyway
        cv2.line(img, (250, 60), (250, 130), (190, 190, 190), 3)
    return img


def full_cells(img: np.ndarray) -> list[int]:
    # Ink threshold and faint threshold as observe.ink_threshold gives them for white paper and black ink
    systems = parse_view(0, img, (0, 0, 800, 200), "dark", 140.0, 229.0, 14.0)
    return [c.x0 for ps in systems for c in ps.cells if c.kind == "full"]


def test_barlines_make_measures():
    xs = full_cells(plate(light_line=False))
    assert len(xs) == 2 and abs(xs[0] - 100) <= 2 and abs(xs[1] - 400) <= 2


def test_a_light_line_is_not_a_barline():
    assert full_cells(plate(light_line=True)) == full_cells(plate(light_line=False))
