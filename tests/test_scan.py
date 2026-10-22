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
    segments, _, _ = scan_module.scan(None, 1280, 720)
    assert len(segments) == 1 and segments[0].polarity == polarity
    # The crop top is rounded down to an even row
    assert TOP - gaps * GAP - 2 <= segments[0].y0 <= TOP - gaps * GAP


def staff_at(top: float) -> scan_module.System:
    return scan_module.System(scan_module.Staff([top + i * GAP for i in range(6)], 100, 1180, GAP, "dark"), None)


def test_a_page_moving_steadily_up_is_a_vertical_scroll():
    # 20 px up every half second for 12 s, then held still for 5 s
    found = {k / 2: [staff_at(800 - 20 * k)] for k in range(24)}
    found.update({12 + k / 2: [staff_at(320)] for k in range(10)})
    assert scan_module.vertical_scrolls(found) == [(0.0, 12.0)]


def test_a_page_that_jumps_now_and_then_is_no_vertical_scroll():
    # Still for 5 s, a jump of 4 gaps, still again
    found = {k / 2: [staff_at(400 if k < 10 else 344)] for k in range(40)}
    assert scan_module.vertical_scrolls(found) == []


def test_a_short_upward_drift_is_no_vertical_scroll():
    found = {k / 2: [staff_at(800 - 20 * k)] for k in range(12)}
    assert scan_module.vertical_scrolls(found) == []


def test_a_title_bar_above_the_tab_does_not_set_the_segment_s_theme(monkeypatch):
    img = np.full((720, 1280, 3), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (100, TOP + i * GAP), (1180, TOP + i * GAP), (60, 60, 60), 1)
    # A dark title bar with light lines in it, found as a close staff of the other polarity above the tab
    cv2.rectangle(img, (100, 280), (1180, 340), (20, 20, 20), -1)
    for i in range(5):
        cv2.line(img, (110, 296 + i * 5), (1170, 296 + i * 5), (200, 200, 200), 1)
    monkeypatch.setattr(scan_module, "frames", lambda path, fps: [(k / fps, img) for k in range(5 * fps)])
    segments = scan_module.scan(None, 1280, 720)[0]
    assert [(s.polarity, s.gap) for s in segments] == [("dark", 14.0)]


def staff(top: float, n: int, gap: float, polarity: str) -> scan_module.System:
    return scan_module.System(scan_module.Staff([top + i * gap for i in range(n)], 100, 1180, gap, polarity), None)


def fake_frames(monkeypatch, shown):
    """Five seconds of frames at the scan's rate; `shown(k)` lists the staffs the finder reports in frame k. Each
    frame is white paper, with black between the lines of its light staffs (strings over dark footage)."""
    images = []
    for k in range(5 * scan_module.SCAN_FPS):
        img = np.full((720, 1280, 3), 255, np.uint8)
        img[0, 0] = k
        for s in shown(k):
            if s.tab.polarity == "light":
                img[int(s.tab.top):int(s.tab.bottom), 100:1180] = 0
        images.append(img)
    monkeypatch.setattr(scan_module, "frames", lambda path, fps: [(k / fps, im) for k, im in enumerate(images)])
    # The frame number is in the top left pixel; each call builds new staffs, as the real finder does
    monkeypatch.setattr(scan_module, "systems_in", lambda g, whole_lines=False: shown(int(g[0, 0])))
    return scan_module.scan(None, 1280, 720)


def test_strings_of_the_other_polarity_above_the_tab_do_not_break_its_segment(monkeypatch):
    # Dark tab all along; from 1 s, eight light "lines" of guitar strings above it
    def shown(k):
        tab = [staff(400, 6, 14.0, "dark")]
        return ([staff(100, 8, 12.0, "light")] if k >= 2 else []) + tab
    segments, found, _ = fake_frames(monkeypatch, shown)
    assert [(s.polarity, s.t0, s.t1) for s in segments] == [("dark", 0.0, 5.0)]
    assert all(s.tab.polarity == "dark" for syss in found.values() for s in syss)


def test_a_segment_started_by_footage_keeps_its_polarity_over_the_real_tab_after_it(monkeypatch):
    # A known limit: a dark false staff starts the segment, then the real light tab shows above it from 1 s and is
    # dropped as footage
    def shown(k):
        footage = [staff(600, 6, 22.0, "dark")]
        return ([staff(300, 6, 22.5, "light")] if k >= 2 else []) + footage
    segments, found, _ = fake_frames(monkeypatch, shown)
    assert [(s.polarity, s.t0, s.t1) for s in segments] == [("dark", 0.0, 5.0)]
    assert all(s.tab.polarity == "dark" for syss in found.values() for s in syss)


def test_a_segment_too_short_to_keep_does_not_hide_the_tab_beside_it(monkeypatch):
    # A dark false staff for 1.5 s, too short to keep, with the real light tab above it from 0.5 s on
    def shown(k):
        footage = [staff(600, 6, 22.0, "dark")] if k <= 2 else []
        return ([staff(300, 6, 22.5, "light")] if k >= 1 else []) + footage
    segments, found, _ = fake_frames(monkeypatch, shown)
    assert [(s.polarity, s.t0, s.t1) for s in segments] == [("light", 0.5, 5.0)]
    assert all(any(s.tab.polarity == "light" for s in found[t]) for t in (0.5, 1.0))
