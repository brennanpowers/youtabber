import numpy as np

from youtabber.check import empty_stretches, glide_gaps, glide_number_gaps, glide_repeats, number_checks, pieces
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
