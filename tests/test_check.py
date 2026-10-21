from types import SimpleNamespace

import numpy as np

from youtabber import track
from youtabber.check import (dwell, empty_stretches, glide_fade, glide_gaps, glide_number_gaps, glide_piece_cuts,
                             glide_repeats, glide_rows, glide_steps, number_checks, odd_widths, pieces, redraw_gaps,
                             too_few)
from youtabber.assemble import Obs
from youtabber.glide import Mosaic
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


def test_a_glide_whose_picture_fades_out_while_moving_warns_from_the_last_clear_frame():
    def view(offsets, until):
        times = np.arange(11, dtype=float)
        return SimpleNamespace(mosaic=Mosaic(np.zeros((10, 200, 3), np.uint8), 0, np.array(offsets, float), times,
                                             50.0, 10, until=until))
    rising = [10.0 * k for k in range(11)]
    (w,) = glide_fade(view(rising, 14.0), 2.0)
    assert (w["kind"], w["t0"], w["t1"]) == ("glide_end", 10.0, 14.0)
    assert glide_fade(view(rising, 11.0), 2.0) == []
    assert glide_fade(view(rising[:-1] + [rising[-2]], 14.0), 2.0) == []


def piece(cells, t0: float, t1: float, width: int = 2000, doubtful=()) -> SimpleNamespace:
    """A glide piece whose canvas x / 100 is its play time."""
    m = FakeMosaic()
    m.image, m.doubtful = np.zeros((10, width, 3), np.uint8), list(doubtful)
    return SimpleNamespace(cells=cells, all_cells=cells, systems=[SimpleNamespace(cells=cells)], mosaic=m, t0=t0,
                           t1=t1, move=None, part=0)


def test_unconfirmed_speed_changes_close_together_are_one_notice():
    ws = glide_steps(piece([], 0.0, 20.0, doubtful=[300, 400, 1200]))
    assert [(w["kind"], w["t0"], w["t1"]) for w in ws] == [("glide_step", 2.0, 5.0), ("glide_step", 11.0, 13.0)]


def test_a_piece_with_little_staff_found_names_the_stretch_without_it():
    cells = [cell(x, x + 100) for x in (0, 100, 200, 300, 1500, 1600)]
    (w,) = glide_rows(piece(cells, 0.0, 20.0, width=1700))
    assert (w["kind"], w["t0"], w["t1"]) == ("glide_rows", 4.0, 15.0)
    assert glide_rows(piece([cell(x, x + 100) for x in range(0, 1700, 100)], 0.0, 20.0)) == []


def test_a_narrow_measure_where_a_glide_is_redrawn_warns():
    a = piece([cell(x, x + 100) for x in (0, 100, 200)] + [cell(300, 330)], 0.0, 3.3)
    b = piece([cell(x, x + 100) for x in (400, 500, 600)], 3.3, 7.0)
    b.move = track.Move("glide")
    (w,) = glide_piece_cuts([a, b])
    assert (w["kind"], w["t0"], w["t1"]) == ("glide_piece", 2.3, 4.5)
    # A still view of another part shown between the pieces doesn't part them
    other = piece([cell(0, 100)], 3.0, 4.0)
    other.part = 1
    assert [(w["t0"], w["t1"]) for w in glide_piece_cuts([a, other, b])] == [(2.3, 4.5)]
    a.all_cells = a.cells = a.cells[:3]
    assert glide_piece_cuts([a, b]) == []


def test_a_long_wait_across_a_redraw_of_a_glide_is_flagged():
    a = piece([cell(x, x + 100) for x in (0, 100, 200)], 0.0, 3.0)
    b = piece([cell(x, x + 100) for x in (900, 1000)], 3.0, 12.0)
    b.move = track.Move("glide")
    (w,) = redraw_gaps([a, b])
    assert (w["kind"], w["t0"], w["t1"]) == ("glide_gap", 2.5, 9.5)
    b.cells = b.all_cells = [cell(x, x + 100) for x in (300, 400)]
    assert redraw_gaps([a, b]) == []


def test_empty_stretches_come_in_pieces():
    ws = empty_stretches([(0.0, 100.0)], [1.0, 2.0, 3.0, 60.0])
    assert [(w["t0"], w["t1"]) for w in ws] == [(3.0, 33.0), (33.0, 60.0), (60.0, 90.0), (90.0, 100.0)]


def test_number_checks():
    labels = [(1.0, 1), (2.0, 2), (3.0, 5), (4.0, 5)]
    assert [w["kind"] for w in number_checks(labels, True, 10.0)] == ["numbers_skipped", "number_again"]
    assert [w["kind"] for w in number_checks(labels, False, 10.0)] == ["numbers_skipped"]


def test_unread_numbers_in_between_are_not_a_skip():
    labels = [(1.0, 1), (2.0, None), (3.0, None), (4.0, None), (5.0, 5)]
    assert number_checks(labels, True, 10.0) == []


def test_a_first_number_far_past_the_count_points_at_missing_opening_measures():
    labels = [(14.0, 7), (16.0, 8), (18.0, 9)]
    (w,) = number_checks(labels, True, 60.0)
    assert (w["kind"], w["t0"], w["t1"]) == ("numbers_start", 9.0, 14.0)
    assert "Measures 1-6 weren't printed" in w["text"]
    # A pickup numbered 0, or a first measure numbered 1 to 3, is no sign of missing measures
    assert number_checks([(14.0, 3), (16.0, 4)], True, 60.0) == []


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
    def __init__(self, t0: float, t1: float, n: int, wrapped: bool = False, index: int = 0):
        self.t0, self.t1, self.cells, self.wrapped, self.seg, self.index = t0, t1, [None] * n, wrapped, 0, index


def held_twice(last_slots: list[int]) -> list[dict]:
    """Notices for five ordinary views, then one measure held 10 s in two views, the second cut where the playhead
    jumped back and placed in `last_slots`."""
    obs = [HeldView(4.0 * i, 4.0 * i + 4.0, 4, index=i) for i in range(5)] + [
        HeldView(20.0, 25.0, 1, index=5), HeldView(25.0, 30.0, 1, wrapped=True, index=6)]
    return dwell(obs, SimpleNamespace(view_slots={5: [20], 6: last_slots}))


def test_a_screen_cut_where_the_playhead_jumped_back_is_judged_whole():
    # The second view shares no measure with the first, so it shows the same screen again
    assert [(w["kind"], w["t0"], w["t1"]) for w in held_twice([21])] == [("few_measures", 20.0, 30.0)]


def test_a_screen_matched_again_to_the_same_measures_is_judged_whole():
    # The second view holds just the first's measure, so it is the same page shown again
    assert [(w["t0"], w["t1"]) for w in held_twice([20])] == [(20.0, 30.0)]


def test_a_view_that_scrolled_on_after_the_playhead_jumped_back_is_its_own_screen():
    # The second view shares a measure with the first and adds a new one, so the page scrolled on to new music
    assert [(w["t0"], w["t1"]) for w in held_twice([20, 21])] == [(20.0, 25.0), (25.0, 30.0)]


def test_a_screen_held_long_for_one_measure_says_measure():
    obs = [HeldView(4.0 * i, 4.0 * i + 4.0, 4, index=i) for i in range(5)] + [HeldView(20.0, 30.0, 1, index=5)]
    (w,) = dwell(obs, SimpleNamespace(view_slots={}))
    assert w["kind"] == "few_measures"
    assert "for the 1 measure read from it;" in w["text"]


def test_one_measure_from_a_long_video_says_measure():
    assert too_few([HeldView(0.0, 60.0, 1)], 1).startswith("Read only 1 measure from 60 s of tab")
