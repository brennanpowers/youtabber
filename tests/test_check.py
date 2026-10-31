from types import SimpleNamespace

import numpy as np

from youtabber import track
from youtabber.check import (dwell, empty_stretches, glide_fade, glide_gaps, glide_number_gaps, glide_piece_cuts,
                             glide_repeats, glide_rows, glide_steps, judged_width, no_barlines, number_checks,
                             numbers_restart, odd_widths, pieces, redraw_gaps, same_columns, settled, shared_barlines,
                             slow_measures, tab_in_pieces, too_few, unread_stretches)
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


def test_measures_printed_again_in_an_unresolved_fold_are_no_new_music():
    played = [341.5, 357.5, 380.17, 383.5, 386.5, 389.12]
    assert empty_stretches([(340.0, 396.5)], played) == []
    # The measures played at 380.17 and 383.5 may repeat earlier ones; the one at 386.5 is new
    ws = empty_stretches([(340.0, 396.5)], played, [380.17, 383.5])
    assert [(w["kind"], w["t0"], w["t1"]) for w in ws] == [("no_measures", 357.5, 386.5)]
    # A measure at 380.17 left out of `again` breaks the stretch
    assert empty_stretches([(340.0, 396.5)], played, [383.5]) == []


def test_unread_stretches_need_four_seconds_with_no_view_read():
    spans = [(0.0, 40.0)]
    read = [(0.0, 10.0), (13.0, 20.0), (25.0, 34.0)]
    assert [(w["t0"], w["t1"]) for w in unread_stretches(spans, read)] == [(20.0, 25.0), (34.0, 40.0)]


def test_an_unread_stretch_of_exactly_four_seconds_is_named_and_one_just_shorter_is_not():
    assert [(w["t0"], w["t1"]) for w in unread_stretches([(0.0, 20.0)], [(0.0, 10.0), (14.0, 20.0)])] == [(10.0, 14.0)]
    assert unread_stretches([(0.0, 20.0)], [(0.0, 10.0), (13.9, 20.0)]) == []


def test_a_long_unread_stretch_comes_in_pieces():
    ws = unread_stretches([(0.0, 80.0)], [(0.0, 5.0), (75.0, 80.0)])
    assert [(w["t0"], w["t1"]) for w in ws] == [(5.0, 35.0), (35.0, 65.0), (65.0, 75.0)]


def test_number_checks():
    labels = [(1.0, 1), (2.0, 2), (3.0, 5), (4.0, 5)]
    assert [w["kind"] for w in number_checks(labels, True)] == ["numbers_skipped", "number_again"]
    assert [w["kind"] for w in number_checks(labels, False)] == ["numbers_skipped"]


def test_unread_numbers_in_between_are_not_a_skip():
    labels = [(1.0, 1), (2.0, None), (3.0, None), (4.0, None), (5.0, 5)]
    assert number_checks(labels, True) == []


def test_a_first_number_far_past_the_count_points_at_missing_opening_measures():
    labels = [(14.0, 7), (16.0, 8), (18.0, 9)]
    (w,) = number_checks(labels, True)
    assert (w["kind"], w["t0"], w["t1"]) == ("numbers_start", 9.0, 14.0)
    assert "Measures 1-6 weren't printed" in w["text"]
    # A pickup numbered 0, or a first measure numbered 1 to 3, is no sign of missing measures
    assert number_checks([(14.0, 3), (16.0, 4)], True) == []


def test_unread_last_numbers_are_not_counted_over():
    labels = [(float(t), t + 1) for t in range(10)] + [(float(t), None) for t in range(10, 15)]
    assert [w["kind"] for w in number_checks(labels, True)] == []


def test_more_measures_than_numbers():
    labels = [(0.0, 1)] + [(float(t), None) for t in range(1, 5)] + [(5.0, 2)]
    kinds = [w["kind"] for w in number_checks(labels, True)]
    assert "count_over" in kinds


def test_a_pickup_numbered_0_is_not_counted_over():
    labels = [(1.0, 0), (2.0, 1), (3.0, 2), (4.0, 3)]
    assert [w for w in number_checks(labels, True) if w["kind"] == "count_over"] == []


def test_a_measure_numbered_0_after_three_unnumbered_measures_is_not_counted_over():
    labels = [(1.0, None), (2.0, None), (3.0, 0)]
    assert [w for w in number_checks(labels, True) if w["kind"] == "count_over"] == []


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


def test_judged_width_measures_a_tail_to_its_last_mark_and_a_cut_lead_whole():
    still_view = SimpleNamespace(mosaic=None, systems=[SimpleNamespace(bars=[(250, 252)])])
    glide_view = SimpleNamespace(mosaic=FakeMosaic())
    tail = cell(0, 400)
    tail.kind, tail.ink = "tail", np.zeros((60, 400), bool)
    tail.ink[20:30, 340:350] = True
    assert judged_width(still_view, tail) == 35.0
    lead = cell(0, 250)
    lead.kind, lead.complete = "lead", False
    assert judged_width(still_view, lead) == 25.0
    lead.complete = True
    assert judged_width(still_view, lead) is None
    assert judged_width(glide_view, tail) is None
    lead.complete = False
    assert judged_width(glide_view, lead) is None
    # A cut staff with no barline is one lead cell holding any number of measures
    bare_view = SimpleNamespace(mosaic=None, systems=[SimpleNamespace(bars=[])])
    assert judged_width(bare_view, lead) is None


def played(index: int, widths: list[int], times: list[float], seg: int = 0) -> Obs:
    """A still screen of one system whose playhead enters each measure just past its barline at the given times."""
    o = still(index, 10.0, widths)
    for c in o.cells:
        c.mid_y, c.y0, c.ref_y = 30.0, 0, 10
    o.seg = seg
    o.t0, o.t1 = times[0], times[-1] + 2.0
    o.track = [(t, c.x0 + 5, 30.0) for t, c in zip(times, o.cells)]
    return o


def steady_views(seg: int = 0, first: int = 0) -> list[Obs]:
    """Three screens whose measures each take 2 s."""
    return [played(i, [200, 200, 200, 200], [10.0 * i + k * 2.0 for k in range(4)], seg)
            for i in range(first, first + 3)]


def test_a_measure_played_for_two_measures_time_is_named():
    steady = steady_views()
    assert slow_measures(steady) == []
    # The second cell holds two measures: the playhead takes 4 s over it against the usual 2 s
    merged = played(3, [200, 400, 200], [40.0, 42.0, 46.0])
    (w,) = slow_measures(steady + [merged])
    assert (w["kind"], w["t0"], w["t1"]) == ("long_measure", 42.0, 46.0)
    # A view that opens with the playhead deep inside a measure gives that measure no timing
    late = played(4, [200, 400, 200], [50.0, 52.0, 56.0])
    late.track[1] = (52.0, late.cells[1].x0 + 200, 30.0)
    assert slow_measures(steady + [late]) == []


def test_a_view_with_two_slow_measures_gets_one_notice():
    both = played(3, [400, 400, 400], [40.0, 44.0, 48.0])
    (w,) = slow_measures(steady_views() + [both])
    assert (w["t0"], w["t1"]) == (40.0, 44.0)


def test_a_slow_first_measure_of_a_segment_is_a_count_in():
    count_in = played(0, [400, 200, 200], [0.0, 4.0, 6.0], seg=1)
    assert slow_measures([count_in] + steady_views(seg=1, first=1)) == []


def test_a_slow_measure_narrower_than_the_views_typical_measure_is_not_named():
    narrow = played(3, [400, 200, 400, 400], [40.0, 42.0, 46.0, 48.0])
    assert slow_measures(steady_views() + [narrow]) == []


class HeldView:
    def __init__(self, t0: float, t1: float, n: int, wrapped: bool = False, index: int = 0):
        self.t0, self.t1, self.cells, self.wrapped, self.seg, self.index = t0, t1, [None] * n, wrapped, 0, index
        self.mosaic, self.track = None, []


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
    unsettled = SimpleNamespace(decisions=[SimpleNamespace(kind="join", resolved=False)])
    assert too_few([lines_view(0.0, 60.0, [[300]])], 1, unsettled).startswith("Read only 1 measure from 60 s of tab")


def test_a_lesson_holding_few_measures_long_prints_when_its_reading_is_settled():
    view = HeldView(0.0, 120.0, 0)
    view.cells = [cell(0, 300), cell(300, 620)]
    settled = SimpleNamespace(decisions=[SimpleNamespace(kind="move", resolved=True)])
    assert too_few([view], 2, settled) is None
    # A view that reads no measure leaves time on screen unaccounted for
    assert too_few([view, HeldView(120.0, 130.0, 0)], 2, settled) is not None


def lesson_view() -> HeldView:
    view = HeldView(0.0, 120.0, 0)
    view.cells = [cell(0, 300), cell(300, 620)]
    return view


def test_settled_is_false_while_a_fold_is_unresolved():
    A = SimpleNamespace(decisions=[SimpleNamespace(kind="move", resolved=True),
                                   SimpleNamespace(kind="fold", resolved=False)])
    assert not settled([lesson_view()], A)


def test_settled_is_false_when_a_measure_is_much_wider_than_the_others():
    obs = [still(0, 10.0, [100, 100]), still(1, 10.0, [100, 300])]
    A = SimpleNamespace(decisions=[SimpleNamespace(kind="join", resolved=True)])
    assert odd_widths(obs)
    assert not settled(obs, A)


def test_settled_is_true_when_every_view_reads_measures_and_every_decision_is_resolved():
    A = SimpleNamespace(decisions=[SimpleNamespace(kind="move", resolved=True),
                                   SimpleNamespace(kind="join", resolved=True)])
    assert settled([lesson_view()], A)


def test_a_segment_seen_too_little_to_time_its_measures_names_no_slow_measure():
    # The steady screens time their own segment only; the other segment has too few crossings to time
    merged = played(3, [200, 400, 200], [40.0, 42.0, 46.0], seg=1)
    assert slow_measures(steady_views() + [merged]) == []


def test_count_over_points_at_where_the_count_runs_ahead():
    # Measures 1-10 at 1 s each, then three unnumbered measures at 40-42 s, then 11-20 at 50-59 s, then three
    # more at 70-72 s
    labels = [(float(n), n) for n in range(1, 11)] + [(40.0 + k, None) for k in range(3)]
    labels += [(39.0 + n, n) for n in range(11, 21)] + [(70.0 + k, None) for k in range(3)] + [(80.0, 21)]
    ws = [w for w in number_checks(labels, True) if w["kind"] == "count_over"]
    assert [(w["t0"], w["t1"]) for w in ws] == [(10.0, 40.0), (40.0, 50.0), (59.0, 80.0)]


def staves_view(k: int, lower_read: bool, paired: bool = True):
    """A view of two tab staves, the lower one's barlines 2 px off the upper's; unpaired, it shows one staff."""
    bars = [(100, 102), (400, 402), (700, 702)]
    cells = [SimpleNamespace(system=s, read=(k + i if s == 0 or lower_read else None)) for s in (0, 1)
             for i in range(2)]
    systems = [SimpleNamespace(bars=bars), SimpleNamespace(bars=[(a + 2, b + 2) for a, b in bars])]
    return SimpleNamespace(mosaic=None, seg=0, part=0, systems=systems if paired else systems[:1], cells=cells,
                           t0=10.0 * k, t1=10.0 * k + 9)


def test_two_staves_with_lined_up_barlines_and_numbers_over_one_are_pointed_out():
    ws = shared_barlines([staves_view(k, False) for k in range(4)])
    assert [(w["t0"], w["t1"]) for w in ws] == [(0.0, 30.0), (30.0, 39.0)]
    # One part's systems in turn carry numbers of their own
    assert shared_barlines([staves_view(k, True) for k in range(4)]) == []


def test_shared_barlines_need_half_the_views_paired_and_three_numbered():
    # Four paired views among nine are fewer than half
    assert shared_barlines([staves_view(k, False, paired=k < 4) for k in range(9)]) == []
    # Numbers over the upper staff in only two views are too few to go by
    assert shared_barlines([staves_view(k, False) for k in range(2)]) == []


def test_same_columns_matches_barlines_a_few_pixels_apart():
    upper = [(10, 12), (100, 102), (200, 202)]
    assert same_columns(upper, [(11, 13), (99, 101), (202, 204)])
    assert not same_columns(upper, [(11, 13), (202, 204)])
    assert not same_columns([], [])


def seg(t0: float, t1: float, gap: float = 23.5, polarity: str = "dark") -> SimpleNamespace:
    return SimpleNamespace(t0=t0, t1=t1, gap=gap, polarity=polarity)


def test_tab_found_in_pieces_warns_each_long_gap_between_them():
    # The 2 s break at 1:58 is too short to warn
    segments = [seg(9.0, 19.0), seg(55.0, 64.5), seg(108.5, 118.0), seg(120.0, 125.0)]
    ws = tab_in_pieces(segments, [10.0, 60.0])
    assert [(w["t0"], w["t1"]) for w in ws] == [(19.0, 49.0), (49.0, 55.0), (64.5, 94.5), (94.5, 108.5)]
    assert {w["kind"] for w in ws} == {"tab_in_pieces"}
    assert "before 1:04 and after 1:48" in ws[3]["text"]


def test_tab_in_pieces_needs_gaps_to_take_most_of_the_time():
    # 30 s of gaps over 50 s is enough, 29 s isn't
    assert len(tab_in_pieces([seg(0.0, 10.0), seg(40.0, 50.0)], [5.0])) == 1
    assert tab_in_pieces([seg(0.0, 10.0), seg(39.0, 50.0)], [5.0]) == []


def test_tab_in_pieces_needs_one_layout_and_printed_measures():
    assert tab_in_pieces([seg(0.0, 10.0), seg(50.0, 60.0, gap=30.0)], [5.0]) == []
    assert tab_in_pieces([seg(0.0, 10.0), seg(50.0, 60.0, polarity="light")], [5.0]) == []
    assert tab_in_pieces([seg(0.0, 10.0), seg(50.0, 60.0)], [30.0]) == []


def view(index: int, segment: int, t0: float, t1: float, start=None, measures: int = 4, mosaic=None) -> SimpleNamespace:
    return SimpleNamespace(index=index, seg=segment, t0=t0, t1=t1, start=start, cells=[None] * measures, mosaic=mosaic)


def numbered_to_25(switches=()) -> SimpleNamespace:
    return SimpleNamespace(slots=[SimpleNamespace(number=n, obs=0 if n < 22 else 1) for n in range(1, 26)],
                           switches=list(switches))


def test_a_screen_numbered_below_what_the_last_screen_showed_is_warned():
    # The last screen showed 22-25, so a start at 21 is lower than any ordinary overlap
    obs = [view(0, 0, 2.0, 30.0, 1), view(1, 0, 30.0, 50.0, 22), view(2, 1, 55.0, 59.0, 21), view(3, 1, 59.0, 70.0, 25)]
    (w,) = numbers_restart(numbered_to_25(), obs)
    assert (w["t0"], w["t1"], w["kind"]) == (50.0, 59.0, "numbers_restart")
    assert "numbered from 21" in w["text"] and "up to 25" in w["text"]


def test_a_screen_showing_the_last_screens_measures_again_is_not_warned():
    obs = [view(0, 0, 2.0, 30.0, 1), view(1, 0, 30.0, 50.0, 22), view(2, 1, 55.0, 59.0, 22)]
    assert numbers_restart(numbered_to_25(), obs) == []


def test_a_segment_without_numbers_is_passed_over_and_the_next_compared_with_its_screen():
    # The unnumbered screen at 0:55 shows 2 measures, so the next segment may start at 24 but not at 23
    obs = [view(0, 0, 2.0, 30.0, 1), view(1, 0, 30.0, 50.0, 22), view(2, 1, 55.0, 59.0, None, measures=2)]
    assert numbers_restart(numbered_to_25(), obs + [view(3, 2, 62.0, 66.0, 24)]) == []
    (w,) = numbers_restart(numbered_to_25(), obs + [view(3, 2, 62.0, 66.0, 23)])
    assert (w["t0"], w["t1"]) == (59.0, 66.0)


def test_a_screen_already_read_as_another_part_is_not_warned_as_a_restart():
    obs = [view(0, 0, 2.0, 30.0, 1), view(1, 0, 30.0, 50.0, 22), view(2, 1, 55.0, 59.0, 21)]
    assert numbers_restart(numbered_to_25(switches=[(55.0, 59.0)]), obs) == []


def test_a_blank_last_view_is_passed_over_for_the_screen_before_it():
    # The faded view at 0:50 shows no measures; the screen before it showed 22-25, so a start at 22 is an overlap
    obs = [view(0, 0, 2.0, 30.0, 1), view(1, 0, 30.0, 50.0, 22), view(2, 0, 50.0, 51.0, None, measures=0),
           view(3, 1, 55.0, 59.0, 22)]
    assert numbers_restart(numbered_to_25(), obs) == []
    (w,) = numbers_restart(numbered_to_25(), obs[:3] + [view(3, 1, 55.0, 59.0, 21)])
    assert (w["t0"], w["t1"]) == (50.0, 59.0)


def test_a_glide_before_the_segment_is_passed_over_for_the_still_screen_before_it():
    # The glide at 0:50 holds 20 measures; the still screen before it showed 22-25, so a start at 21 is warned
    obs = [view(0, 0, 2.0, 30.0, 1), view(1, 0, 30.0, 50.0, 22),
           view(2, 0, 50.0, 54.0, None, measures=20, mosaic=object()), view(3, 1, 55.0, 59.0, 21)]
    (w,) = numbers_restart(numbered_to_25(), obs)
    assert (w["t0"], w["t1"]) == (50.0, 59.0)


def page_views(held: float, entries: list[float]) -> list[Obs]:
    """Four pages of three measures held 8 s each, then one held `held` s whose playhead enters measures 0, 1, 2, 0,
    1, 2 at the times in `entries` (fewer for a single sweep)."""
    obs = [played(i, [200, 200, 200], [8.0 * i, 8.0 * i + 2.7, 8.0 * i + 5.3]) for i in range(4)]
    for o in obs:
        o.t1 = o.t0 + 8.0
    last = played(4, [200, 200, 200], [32.0, 34.7, 37.3])
    last.t1 = 32.0 + held
    last.track = [(t, last.cells[k % 3].x0 + 5, 30.0) for k, t in enumerate(entries)]
    return obs + [last]


def test_a_page_held_twice_as_long_as_usual_and_played_twice_warns():
    twice = [32.0, 34.7, 37.3, 40.0, 42.7, 45.3]
    (w,) = dwell(page_views(16.0, twice), SimpleNamespace(view_slots={}))
    assert (w["kind"], w["t0"], w["t1"]) == ("few_measures", 32.0, 48.0)
    assert "the playhead goes back over it" in w["text"]
    # 1.8 times the usual 8 s is 14.4 s
    assert len(dwell(page_views(14.5, twice), SimpleNamespace(view_slots={}))) == 1
    assert dwell(page_views(14.3, twice), SimpleNamespace(view_slots={})) == []


def test_a_long_page_the_playhead_crosses_once_does_not_warn():
    assert dwell(page_views(16.0, [32.0, 37.3, 42.7]), SimpleNamespace(view_slots={})) == []


def test_a_playhead_that_goes_back_more_often_than_the_page_time_allows_is_lost_and_does_not_warn():
    # Held 16 s against the usual 8 s leaves room for one return to the first measure; this one comes back three times
    lost = [32.0 + 1.25 * k for k in range(12)]
    assert dwell(page_views(16.0, lost), SimpleNamespace(view_slots={})) == []
    # Played three times in 16 s, it returns twice: one more than the page time leaves room for
    thrice = [32.0 + 1.75 * k for k in range(9)]
    assert dwell(page_views(16.0, thrice), SimpleNamespace(view_slots={})) == []
    # Held 24 s, three times the usual, a page played three times returns twice
    thrice = [32.0 + 2.7 * k for k in range(9)]
    (w,) = dwell(page_views(24.0, thrice), SimpleNamespace(view_slots={}))
    assert (w["t0"], w["t1"]) == (32.0, 56.0) and "the playhead goes back over it" in w["text"]


def lines_view(t0: float, t1: float, rows: list[list[int]]) -> HeldView:
    """A view on a panel 1000 px wide whose systems hold cells of the given widths, one list a system."""
    view = HeldView(t0, t1, 0)
    view.rect = (0, 0, 1000, 600)
    view.cells = []
    for s, widths in enumerate(rows):
        x = 0
        for w in widths:
            c = cell(x, x + w)
            c.system, x = s, x + w
            view.cells.append(c)
    return view


def test_tab_without_barlines_reads_a_cell_a_line_and_prints_with_a_notice():
    # 186 s for 3 cells is over the 15 s a measure may take, but each cell is a whole line
    unsettled = SimpleNamespace(decisions=[SimpleNamespace(kind="join", resolved=False)])
    views = [lines_view(0.0, 90.0, [[985], [900]]), lines_view(90.0, 186.0, [[1000]])]
    assert too_few(views, 3, unsettled) is None
    (w,) = no_barlines(views)
    assert (w["kind"], w["t0"], w["t1"]) == ("no_barlines", 0.0, 186.0)
    assert w["text"].startswith("No barlines were found in this tab, so")


def test_a_cell_short_of_the_panel_or_two_cells_on_a_line_are_measures():
    unsettled = SimpleNamespace(decisions=[SimpleNamespace(kind="join", resolved=False)])
    for rows in ([[985], [899]], [[500, 500]]):
        views = [lines_view(0.0, 90.0, rows), lines_view(90.0, 186.0, [[1000]])]
        assert too_few(views, 3, unsettled) is not None
        assert no_barlines(views) == []
