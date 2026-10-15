import cv2
import numpy as np

from youtabber.scan import systems_in
from youtabber.staff import find_staffs


def staff_image(dark: bool = True, notation: bool = False) -> np.ndarray:
    """A white page with a six-line tab staff (spacing 14 px), optionally a five-line staff (10 px) above it."""
    img = np.full((360, 800), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (50, 200 + i * 14), (750, 200 + i * 14), 60, 1)
    if notation:
        for i in range(5):
            cv2.line(img, (50, 80 + i * 10), (750, 80 + i * 10), 60, 1)
    return img if dark else 255 - img


def test_finds_six_evenly_spaced_lines():
    (st,) = find_staffs(staff_image())
    assert len(st.lines) == 6
    assert abs(st.gap - 14) < 0.6
    assert st.polarity == "dark"


def test_finds_light_lines_on_dark_paper():
    (st,) = find_staffs(staff_image(dark=False))
    assert len(st.lines) == 6
    assert st.polarity == "light"


def test_pairs_notation_over_tab():
    (system,) = systems_in(staff_image(notation=True))
    assert len(system.tab.lines) == 6
    assert system.notation is not None and len(system.notation.lines) == 5
