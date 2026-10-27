from types import SimpleNamespace

import numpy as np

from youtabber import parts
from youtabber.parts import Part, find_parts, lanes, merge_order, paper_rows_share, stream_of
from youtabber.scan import Segment


def sys(top, bottom, x0=0, x1=1000):
    return SimpleNamespace(top=top, tab=SimpleNamespace(bottom=bottom, x0=x0, x1=x1))


def seg():
    return Segment(t0=0, t1=20, gap=10, polarity="dark", y0=0, y1=1080, x0=0, x1=1920)


def test_lanes_merge_staffs_that_overlap_over_the_segment():
    found = {0.0: [sys(100, 200)], 0.5: [sys(104, 204), sys(600, 700)]}
    assert lanes(found, seg()) == [[100, 204], [600, 700]]


def test_a_staff_within_one_gap_of_a_lane_joins_it():
    assert lanes({0.0: [sys(100, 204), sys(212, 300)]}, seg()) == [[100, 300]]
    assert lanes({0.0: [sys(100, 204), sys(215, 300)]}, seg()) == [[100, 204], [215, 300]]


def test_paper_rows_share_counts_rows_that_are_mostly_paper():
    strip = np.full((100, 50, 3), 255, np.uint8)
    strip[0:40] = 0
    assert paper_rows_share(strip, np.array([255, 255, 255])) == 0.6


def two_lanes(lower_times):
    return {k * 0.5: [sys(100, 204)] + ([sys(600, 700)] if k < lower_times else []) for k in range(20)}


def test_two_lanes_with_footage_between_are_two_bands(monkeypatch):
    monkeypatch.setattr(parts, "paper_between", lambda *a: 0.02)
    ps = find_parts(None, seg(), 0, two_lanes(20), 0.5)
    assert [p.kind for p in ps] == ["band", "band"]
    assert [p.rows for p in ps] == [(100, 204), (600, 700)]


def test_two_lanes_on_one_paper_are_one_page(monkeypatch):
    monkeypatch.setattr(parts, "paper_between", lambda *a: 1.0)
    s = seg()
    ps = find_parts(None, s, 0, two_lanes(20), 0.5)
    assert [(p.kind, p.rows) for p in ps] == [("page", (s.y0, s.y1))]


def test_a_lane_seen_only_briefly_is_not_a_part(monkeypatch):
    monkeypatch.setattr(parts, "paper_between", lambda *a: 0.02)
    ps = find_parts(None, seg(), 0, two_lanes(6), 0.5)
    assert [p.kind for p in ps] == ["page"]


def test_the_longest_seen_part_is_stream_0_and_others_get_their_own():
    a = Part(4, 0, (100, 204), "band", seen=10.0)
    b = Part(5, 0, (600, 700), "band", seen=30.0)
    assert stream_of(b, [a, b]) == 0
    assert stream_of(a, [a, b]) == 5


def test_measures_first_seen_on_one_screen_print_top_first():
    assert merge_order([(10.2, 0, 0), (10.0, 500, 1)]) == [0, 1]
    assert merge_order([(10.4, 0, 0), (10.0, 500, 1)]) == [1, 0]


def test_a_screen_starts_again_after_the_same_screen_window():
    assert merge_order([(10.0, 0, 0), (10.2, 500, 1), (10.6, 0, 2)]) == [0, 1, 2]


def split_screen():
    """Scan times 0-19.5 s: the staff across the width for 6 s, then a left and a right panel side by side."""
    return {k * 0.5: [sys(100, 204, 100, 1820)] if k < 12 else [sys(100, 204, 100, 950), sys(100, 204, 970, 1820)]
            for k in range(40)}


def test_a_split_screen_is_the_full_width_part_then_its_left_and_right_panels():
    ps = find_parts(None, seg(), 0, split_screen(), 0.5)
    assert [p.id for p in ps] == [0, 1, 2]
    whole, left, right = ps
    assert whole.cols is None and whole.seen == 6.0
    assert left.cols == (0, 960) and right.cols == (960, 1920)
    assert all(ss == [] for t, ss in whole.found.items() if t >= 6)
    assert all(s.tab.x1 <= 960 for ss in left.found.values() for s in ss)
    assert all(s.tab.x0 >= 960 for ss in right.found.values() for s in ss)
    # All three come from the page the scan saw, which is read across the full width to check that it holds still
    assert whole.cut_from is left.cut_from is right.cut_from and whole.cut_from.kind == "page"


def test_the_left_panel_joins_the_full_width_part_s_stream_and_the_right_one_starts_its_own():
    ps = find_parts(None, seg(), 0, split_screen(), 0.5)
    whole, left, right = ps
    assert left.continues is whole and right.continues is None
    # The full-width part and its left panel are seen longer together than the right panel alone
    assert [stream_of(p, ps) for p in ps] == [0, 0, 3]


def test_panels_shown_briefly_leave_the_part_whole():
    found = {k * 0.5: [sys(100, 204, 100, 1820)] if k < 34 else [sys(100, 204, 100, 950), sys(100, 204, 970, 1820)]
             for k in range(40)}
    ps = find_parts(None, seg(), 0, found, 0.5)
    assert [(p.kind, p.cols) for p in ps] == [("page", None)]
