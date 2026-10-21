from types import SimpleNamespace

import numpy as np

from youtabber import parts
from youtabber.parts import Part, find_parts, lanes, merge_order, paper_rows_share, stream_of
from youtabber.scan import Segment


def sys(top, bottom):
    return SimpleNamespace(top=top, tab=SimpleNamespace(bottom=bottom))


def seg():
    return Segment(t0=0, t1=20, gap=10, polarity="dark", y0=0, y1=1080)


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
