import cv2
import numpy as np
import pytest

from youtabber import scan as scan_module
from youtabber.observe import PRINT_ABOVE_GAPS

TOP, GAP = 400, 14


def still_tab(paper: int, ink: int):
    """Five seconds of one tab staff held still, as scan reads frames: (time, BGR frame) pairs."""
    img = np.full((720, 1280, 3), paper, np.uint8)
    for i in range(6):
        cv2.line(img, (100, TOP + i * GAP), (1180, TOP + i * GAP), (ink,) * 3, 1)

    def frames(path, fps):
        return [(k / fps, img) for k in range(5 * fps)]
    return frames


@pytest.mark.parametrize("paper, ink, polarity, gaps", [(255, 60, "dark", PRINT_ABOVE_GAPS), (0, 200, "light", 5)])
def test_crop_top_reaches_marks_above_the_staff(monkeypatch, paper, ink, polarity, gaps):
    monkeypatch.setattr(scan_module, "frames", still_tab(paper, ink))
    segments, _ = scan_module.scan(None, 1280, 720)
    assert len(segments) == 1 and segments[0].polarity == polarity
    # The crop top is rounded down to an even row
    assert TOP - gaps * GAP - 2 <= segments[0].y0 <= TOP - gaps * GAP
