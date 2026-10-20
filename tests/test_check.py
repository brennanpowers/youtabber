import numpy as np

from youtabber.check import (dwell, empty_stretches, glide_gaps, glide_number_gaps, glide_repeats, number_checks,
                             odd_widths, pieces, too_few)
from youtabber.assemble import Obs
from youtabber.parse import Cell


class FakeMosaic:
    """Canvas x / 100 is the play time."""
    def time_at(self, x: float) -> float:
        return x / 100.0


class FakeView:
    def __init__(self, cells):
        self.cells = cells
        self.mosaic = FakeMosaic()


def cell(x0: int, x1: int, pattern: int = 0) -> Cell:
    rng = np.random.default_rng(pattern)
    ink = rng.random((60, x1 - x0)) > 0.9
    return Cell(0, 0, 0, x0, x1, "full", ink, np.zeros_like(ink), 0, gap=10.0, ref_y=10,
                tint=np.zeros_like(ink), glided=True)


def test_pieces():
    assert pieces(0, 70) == [(0, 30), (30, 60), (60, 70)]


def test_a_repeated_run_of_four_is_named():
    patterns = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
    view = FakeView([cell(100 * i, 100 * i + 90, p) for i, p in enumerate(patterns)])
    (w,) = glide_repeats(view, 10.0)
    assert w["kind"] == "glide_repeat" and w["t0"] == 6.45 and w["t1"] == 9.45


def test_no_repeat_without_four_in_a_row():
    patterns = [1, 2, 3, 4, 5, 1, 2, 3, 6, 7]
    view = FakeView([cell(100 * i, 100 * i + 90, p) for i, p in enumerate(patterns)])
    assert glide_repeats(view, 10.0) == []


def test_a_long_pause_between_glided_measures_is_flagged():
    xs = [0, 100, 200, 300, 600, 700]
    (w,) = glide_gaps(FakeView([cell(x, x + 90) for x in xs]))
    assert (w["t0"], w["t1"]) == (3.45, 6.45)


def test_empty_stretches_come_in_pieces():
    ws = empty_stretches([(0.0, 100.0)], [1.0, 2.0, 3.0, 60.0])
    assert [(w["t0"], w["t1"]) for w in ws] == [(3.0, 33.0), (33.0, 60.0), (60.0, 90.0), (90.0, 100.0)]


def test_number_checks():
    labels = [(1.0, 1), (2.0, 2), (3.0, 5), (4.0, 5)]
    assert [w["kind"] for w in number_checks(labels, True, 10.0)] == ["numbers_skipped", "number_again"]
    assert [w["kind"] for w in number_checks(labels, False, 10.0)] == ["numbers_skipped"]


def test_unread_numbers_in_between_are_not_a_skip():
    labels = [(1.0, 19), (2.0, None), (3.0, None), (4.0, None), (5.0, 23)]
    assert number_checks(labels, True, 10.0) == []


def test_unread_last_numbers_are_not_counted_over():
    labels = [(float(t), t + 1) for t in range(10)] + [(float(t), None) for t in range(10, 15)]
    assert [w["kind"] for w in number_checks(labels, True, 20.0)] == []


def test_more_measures_than_numbers():
    labels = [(0.0, 1)] + [(float(t), None) for t in range(1, 5)] + [(5.0, 2)]
    kinds = [w["kind"] for w in number_checks(labels, True, 10.0)]
    assert "count_over" in kinds


def test_numbers_that_skip_in_a_glide_warn():
    cells = [cell(x, x + 100) for x in range(0, 500, 100)]
    # Measures 7 and 8 print as one cell, so the numbers read go 6, 7, 9; across an unread number, 9 to 11 is no skip
    for c, n in zip(cells, [6, 7, 9, None, 11]):
        c.read = n
    (w,) = glide_number_gaps(FakeView(cells))
    assert (w["kind"], w["t0"], w["t1"]) == ("glide_number_gap", 1.5, 2.5)
    assert "from 7 to 9" in w["text"]


def still(index: int, gap: float, widths: list[int]) -> Obs:
    """A still view held 5 s whose measures are drawn at line gap gap, with these widths in pixels."""
    cells, x = [], 0
    for w in widths:
        c = cell(x, x + w)
        c.gap = gap
        cells.append(c)
        x += w
    return Obs(index, 5.0 * index, 5.0 * index + 5, cells, [], 0, (0, 0))


def test_wide_measures_are_judged_in_line_gaps_with_one_notice_per_view():
    # Tab drawn at twice the size has measures twice as wide in pixels, which is no missed barline
    small = [still(i, 8.0, [120, 130, 125]) for i in range(3)]
    big = [still(i, 16.0, [250, 240, 260]) for i in range(3, 5)]
    assert odd_widths(small + big) == []
    # Two wide measures on one screen make one notice
    missed = still(5, 16.0, [500, 510, 250])
    (w,) = odd_widths(small + big + [missed])
    assert (w["kind"], w["t0"], w["t1"]) == ("wide_measure", 25.0, 30.0)


class HeldView:
    def __init__(self, t0: float, t1: float, n: int):
        self.t0, self.t1, self.cells = t0, t1, [None] * n


def test_a_screen_held_long_for_one_measure_says_measure():
    obs = [HeldView(4.0 * i, 4.0 * i + 4.0, 4) for i in range(5)] + [HeldView(20.0, 30.0, 1)]
    (w,) = dwell(obs)
    assert w["kind"] == "few_measures"
    assert "for the 1 measure read from it;" in w["text"]


def test_one_measure_from_a_long_video_says_measure():
    assert too_few([HeldView(0.0, 60.0, 1)], 1).startswith("Read only 1 measure from 60 s of tab")
