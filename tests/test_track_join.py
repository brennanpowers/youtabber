from dataclasses import replace

import numpy as np
from test_numbering import band_view, music

from youtabber.assemble import (Assembly, Decision, Obs, Slot, assemble, cell_at, choose_shift, find_run, measure_time,
                                place_rest, playhead_at, playhead_join, playhead_restarts, playhead_track, register,
                                repeat_played, replay_evidence, rest_of, same_cell, scroll_shifts, split_measure,
                                still_links, track_join)
from youtabber.parse import Cell
from youtabber.pipeline import Measure, Notice, edge_notices, join_notices, readings, repeat_notices
from youtabber.track import Move

EVIDENCE = {"numbers", "playhead", "navigation"}


def test_the_playhead_track_keeps_only_samples_on_the_music():
    # 700 jumps away from both neighbors, y 300 is far below the staff, x 900 is right of the last measure
    o = band_view(0, 1)
    o.playhead = [(0.0, 50, 30), (0.5, 100, 30), (1.0, 700, 30), (1.5, 120, 30), (2.0, 200, 300), (2.5, 150, 30),
                  (3.0, 900, 30)]
    assert playhead_track(o) == [(0.0, 50, 30), (0.5, 100, 30), (1.5, 120, 30), (2.5, 150, 30)]


def test_a_playhead_first_seen_well_into_a_view_does_not_say_where_it_started():
    o = band_view(0, 1)
    o.playhead = [(o.t0 + 2.5, 100, 30)]
    assert playhead_at(o, True) is None


def held(first: int = 1, p_end: int | None = None, o_start: int | None = None):
    """Two views of a band that held still (a still Move), the second showing measures from `first`, with the
    playhead at p_end at the end of the first view and at o_start at the start of the second."""
    p, o = band_view(0, 1), band_view(1, first)
    o.move = Move("still", steps=5)
    if p_end is not None:
        p.playhead = [(p.t1 - 0.5, p_end, 30)]
    if o_start is not None:
        o.playhead = [(o.t0 + 0.1, o_start, 30)]
    return p, o


def test_a_slide_registers_at_the_first_lowest_mismatch_near_its_measured_shift():
    # Every shift within the 2 px match radius of 400 lines the plates up, and the scan starts at dx - reach
    assert register(band_view(0, 1), band_view(1, 3), 400, 10.0, 6) == (398, 0.0)
    assert register(band_view(0, 1), band_view(1, 1), 0, 10.0, 2) == (-2, 0.0)


def test_a_band_that_held_still_links_each_measure_to_the_one_in_its_place():
    assert still_links(band_view(0, 1), band_view(1, 1)) == {0: 0, 1: 1, 2: 2, 3: 3}


def test_a_held_page_whose_playhead_carries_on_joins_in_place():
    p, o = held(p_end=300, o_start=320)
    assert track_join(p, o, 10.0) == ({0: 0, 1: 1, 2: 2, 3: 3}, "still")


def test_a_held_page_with_no_playhead_seen_joins_in_place_unconfirmed():
    p, o = held()
    assert track_join(p, o, 10.0) == ({0: 0, 1: 1, 2: 2, 3: 3}, "still?")


def test_a_held_page_whose_playhead_starts_over_is_not_joined():
    p, o = held(p_end=700, o_start=20)
    assert track_join(p, o, 10.0) is None
    assert o.restarted is True


def test_a_still_band_showing_other_music_is_not_joined():
    p, o = held(first=5)
    assert track_join(p, o, 10.0) is None


def test_a_view_cut_at_a_playhead_jump_joins_when_the_playhead_on_the_music_carries_on():
    p, o = held(p_end=300, o_start=320)
    o.wrapped = True
    assert track_join(p, o, 10.0) == ({0: 0, 1: 1, 2: 2, 3: 3}, "still")


def test_assembly_places_a_held_page_on_the_same_measures_and_notes_the_join_is_unconfirmed():
    p, o = held()
    A = assemble([p, o], 10.0, False, {"numbers", "playhead", "navigation"}, lambda v: [])
    assert A.view_slots[1] == A.view_slots[0]
    joins = [d for d in A.decisions if d.kind == "join"]
    assert len(joins) == 1 and joins[0].resolved is False


def restarted(loose: bool, frames: int = 0):
    """A held band whose playhead starts over (700 -> 20), the second view showing measures 3-6 on plates like the
    first's: the same plates, or with the last measure's columns blank when `loose`. More frames than playhead
    samples leave the playhead untracked."""
    p, o = band_view(0, 1), band_view(1, 3)
    o.move = Move("still", steps=5)
    p.playhead = [(p.t1 - 0.5, 700, 30)]
    o.playhead = [(o.t0 + 0.1, 20, 30)]
    o.tall = p.tall.copy()
    if loose:
        o.tall[:, 600:800] = 0
    o.frames = frames
    return p, o


def test_a_restarted_page_printed_as_new_music_leaves_its_join_unresolved():
    p, o = restarted(loose=False)
    A = assemble([p, o], 10.0, False, EVIDENCE, lambda v: [])
    assert o.restarted is True
    assert A.view_slots[1] == [4, 5, 6, 7]
    assert [d for d in A.decisions if d.kind == "join"] == [Decision(10.0, "join", "new", ["played again"], "restart",
                                                                     False)]


def test_a_restarted_page_whose_plates_differ_more_may_join_by_an_unresolved_overlap():
    p, o = restarted(loose=True, frames=10)
    assert round(register(p, o, 0, 10.0, 2)[1], 3) == 0.142
    A = assemble([p, o], 10.0, False, EVIDENCE, lambda v: [])
    assert A.view_slots[1] == [2, 3, 4, 5]
    assert [d for d in A.decisions if d.kind == "join"] == [Decision(10.0, "join", "overlap 2", ["none"], "prior",
                                                                     False)]


def test_a_restarted_page_the_playhead_keeps_from_an_overlap_leaves_its_join_unresolved():
    # The tracked playhead going back over the page rules the overlap out as playback carrying on
    p, o = restarted(loose=True)
    A = assemble([p, o], 10.0, False, EVIDENCE, lambda v: [])
    assert A.view_slots[1] == [4, 5, 6, 7]
    assert [d for d in A.decisions if d.kind == "join"] == [Decision(10.0, "join", "none", ["overlap 2"], "playhead",
                                                                     False)]


def test_an_unresolved_join_after_a_restart_is_noticed_from_the_page_before():
    p, o = restarted(loose=False)
    A = Assembly(decisions=[Decision(10.0, "join", "new", ["played again"], "restart", False, view=1)])
    text = ("Couldn't tell how this screen continues from the one before; check that no measure is missing or "
            "printed twice here.")
    assert join_notices(A, [p, o]) == [Notice(9.0, 20.0, "unresolved_join", text, True)]
    o.restarted = True
    assert join_notices(A, [p, o]) == [Notice(0.0, 20.0, "unresolved_join", text, True)]


def riff_view(index: int, first: int):
    """`band_view` with every measure drawn alike, as on the pages of a riff that repeats."""
    o = band_view(index, first)
    for c in o.cells:
        c.ink = music("a", 1)
    o.band = np.hstack([c.ink for c in o.cells])
    o.tall, o.fixed = o.band, np.zeros_like(o.band)
    return o


def slid(shift: int, drawn: bool, p_end: int | None = None, o_start: int | None = None, riff: bool = False):
    """Two views of a band that slid by `shift` between them, the first showing measures 1-4 and the second 3-6,
    with the playhead at p_end at the end of the first view and at o_start at the start of the second."""
    make = riff_view if riff else band_view
    p, o = make(0, 1), make(1, 3)
    o.move = Move("slide", shifts=[shift], drawn=[drawn], steps=5)
    if p_end is not None:
        p.playhead = [(p.t1 - 0.5, p_end, 30)]
    if o_start is not None:
        o.playhead = [(o.t0 + 0.1, o_start, 30)]
    return p, o


def test_a_slide_the_playhead_carries_on_across_joins_at_the_registered_shift():
    p, o = slid(400, True, p_end=700, o_start=320)
    assert track_join(p, o, 10.0) == ({0: 2, 1: 3}, "slide 398")


def test_a_slide_measured_one_measure_short_of_where_the_playhead_went_is_not_joined():
    # The plates of a riff line up at 198 too, but the playhead would have gone back 182 px
    p, o = slid(200, True, p_end=700, o_start=320, riff=True)
    assert track_join(p, o, 10.0) is None


def test_a_slide_is_not_joined_when_another_shift_the_plates_register_at_fits_the_playhead_too():
    p, o = slid(400, False, p_end=700, o_start=320, riff=True)
    cands = scroll_shifts(p, o, 10.0)
    assert cands == [(198, 0.0), (398, 0.0), (598, 0.0)]
    assert track_join(p, o, 10.0, cands) is None
    assert track_join(p, o, 10.0, [(400, 0.0)]) == ({0: 2, 1: 3}, "slide 398")


def test_a_slide_under_a_box_cursor_that_stayed_in_one_column_joins_unconfirmed():
    # The cursor sat on p's last measure and, after the slide, past that measure's start
    p, o = slid(400, True, p_end=700, o_start=705, riff=True)
    assert track_join(p, o, 10.0) == ({0: 2, 1: 3}, "slide? 398")


def test_with_no_playhead_seen_only_a_slide_drawn_in_several_steps_joins():
    p, o = slid(400, True)
    assert track_join(p, o, 10.0) == ({0: 2, 1: 3}, "slide? 398")
    p, o = slid(400, False)
    assert track_join(p, o, 10.0) is None


def test_assembly_leaves_a_slide_the_plates_refuse_to_the_registered_shifts():
    # The plates disagree at the measured 200 px, so the playhead picks among the shifts they register at
    p, o = slid(200, True, p_end=700, o_start=320)
    A = assemble([p, o], 10.0, False, {"numbers", "playhead", "navigation"}, lambda v: [])
    joins = [d for d in A.decisions if d.kind == "join"]
    assert [(d.chosen, d.basis, d.resolved) for d in joins] == [("shift 398", "playhead", True)]
    assert A.view_slots[1] == [2, 3, 4, 5]


def usual_pick(move, usual=400.0):
    p, o = band_view(0, 1), band_view(1, 3)
    o.move = move
    return choose_shift(p, o, [(398, 0.0), (598, 0.1)], 0.5, set(), 10.0, usual=usual)


def test_a_jump_takes_the_shift_nearest_the_usual_slide_settled_when_the_move_slid_that_far_in_steps():
    assert usual_pick(Move("slide", shifts=[400], drawn=[True], steps=5)) == (398, "usual_slide", True)


def test_a_jump_near_the_usual_slide_is_unresolved_when_the_move_was_not_a_slide_drawn_that_far():
    for move in (Move("cut"), Move("still", steps=5), Move("slide", shifts=[200], drawn=[True], steps=5),
                 Move("slide", shifts=[400], drawn=[False], steps=1)):
        assert usual_pick(move) == (398, "usual_slide", False)


def test_a_jump_whose_drawn_slide_lands_on_another_shift_skips_the_usual_slide():
    # The band was seen sliding 46 px, near the 44 px shift, while the usual slide is near the 709 px one
    p, o = band_view(0, 1), band_view(1, 3)
    o.move = Move("slide", shifts=[46], drawn=[True], steps=5)
    shifts = [(44, 0.0), (886, 0.1), (797, 0.1), (623, 0.2), (709, 0.2)]
    assert choose_shift(p, o, shifts, 0.9, set(), 10.0, usual=712.0) == (44, "segment_scrolls", False)
    assert choose_shift(p, o, shifts, 0.5, set(), 10.0, usual=712.0) == (None, "prior", False)


def test_with_no_usual_slide_a_jump_falls_back_on_the_segment_share():
    assert usual_pick(Move("slide", shifts=[400], drawn=[True], steps=5), usual=None) == (None, "prior", False)


def played(o, *pts):
    """Set a view's playhead samples, (seconds after its start, x, y)."""
    o.playhead = [(o.t0 + t, x, y) for t, x, y in pts]
    o.track = None


def played_views(n: int):
    """n views of bars 1-4, 5-8, ..., 8 s each, 10 s apart, the playhead entering a measure every 2 s."""
    out = []
    for i in range(n):
        o = band_view(i, 1 + 4 * i)
        o.t0, o.t1 = 10.0 * i, 10.0 * i + 8
        played(o, *[(0.5 + 0.5 * k, 10 + 50 * k, 30) for k in range(16)])
        out.append(o)
    return out


def cut_last(o, bar: int, w: int) -> None:
    """Make a view's last measure, `bar`, a cell w px wide, cut by the page edge."""
    c = o.cells[-1]
    o.cells[-1] = replace(c, kind="tail", complete=False, x1=c.x0 + w, ink=music("a", bar)[:, :w])


def cut_pair(w: int, entered: float, first: int = 4, whole_track: bool = False):
    """A page of bars 1-4 played 0-10 s, its last measure cut to w px and played from `entered`, then a page
    showing bars from `first` with the playhead on its first measure. With `whole_track` the playhead is also seen
    entering each earlier measure every 2 s, enough to measure a measure's time from the pair alone."""
    p = band_view(0, 1)
    cut_last(p, 4, w)
    p.t0, p.t1 = 0.0, 10.0
    earlier = [(0.5 + 2 * k, 10 + 200 * k, 30) for k in range(3)] if whole_track else []
    played(p, *earlier, (entered - 0.5, 550, 30), (entered, 610, 30), (9.9, 600 + w - 5, 30))
    o = band_view(1, first)
    o.t0, o.t1 = 10.0, 20.0
    played(o, *[(0.1 + 2 * k, 30 + 200 * k, 30) for k in range(4 if whole_track else 1)])
    return p, o


def joins_at(A, t: float) -> list:
    return [(d.chosen, d.alternatives, d.resolved) for d in A.decisions if d.kind == "join" and d.t == t]


def test_measure_time_is_the_median_time_from_entering_one_measure_to_the_next():
    assert measure_time(played_views(3)) == 2.0
    # Three measures played through are too few to say
    assert measure_time(played_views(1)) is None


def test_a_cut_last_measure_being_played_when_the_page_changed_joins_the_next_pages_first():
    assert playhead_join(*cut_pair(120, 9.0), 2.0, 10.0) == ({0: 3}, True, False)


def test_a_cut_measure_played_long_is_joined_and_flagged_as_near_a_whole_one():
    assert playhead_join(*cut_pair(120, 8.6), 2.0, 10.0) == ({0: 3}, True, True)


def test_a_wide_cut_measure_is_joined_and_flagged_as_near_a_whole_one():
    assert playhead_join(*cut_pair(160, 9.0), 2.0, 10.0) == ({0: 3}, True, True)


def test_a_page_with_no_whole_measure_joins_a_cut_measure_of_any_width_and_flags_it():
    # A cut measure as wide as a whole one gives no join while the page shows whole measures to compare with
    assert playhead_join(*cut_pair(196, 9.0), 2.0, 10.0) is None
    for w in (120, 196):
        # The page before shows only the end of bar 3, cut by its left edge, and bar 4 cut by its right edge
        p, o = cut_pair(w, 9.0)
        p.cells = [replace(p.cells[2], kind="lead", complete=False), p.cells[3]]
        assert playhead_join(p, o, 2.0, 10.0) == ({0: 1}, True, True)


def test_a_next_page_opening_with_other_music_is_joined_without_continuing_the_cut_measure():
    assert playhead_join(*cut_pair(120, 9.0, first=1), 2.0, 10.0) == ({0: 3}, False, False)


def test_a_playhead_join_to_a_page_that_opens_a_replay_is_unresolved():
    # Bars 1-12 on three pages, the last one's bar 12 cut by the edge, then a page opening on bar 1 again
    obs = played_views(3)
    cut_last(obs[2], 12, 120)
    played(obs[2], *[(0.5 + 0.5 * k, 10 + 50 * k, 30) for k in range(12)], (7.0, 610, 30), (7.9, 700, 30))
    o = band_view(3, 1)
    played(o, (0.1, 30, 30))
    obs.append(o)
    for v in obs[1:]:
        v.move = Move("cut")
    A = assemble(obs, 10.0, False, EVIDENCE, lambda v: [])
    assert joins_at(A, o.t0) == [("playhead [(0, 3)]", [], False)]
    assert A.view_slots[3][0] == A.view_slots[2][3]


def test_a_playhead_join_to_a_page_like_no_earlier_music_is_resolved_though_its_first_cell_differs():
    p, o = cut_pair(120, 9.0, first=12, whole_track=True)
    assert playhead_join(p, o, 2.0, 10.0)[1] is False
    A = assemble([p, o], 10.0, False, EVIDENCE, lambda v: [])
    assert joins_at(A, o.t0) == [("playhead [(0, 3)]", [], True)]
    assert A.view_slots[1][0] == A.view_slots[0][3]


def test_no_playhead_join_for_a_whole_measure_too_long_a_dwell_or_an_unknown_measure_time():
    assert playhead_join(*cut_pair(196, 9.0), 2.0, 10.0) is None
    assert playhead_join(*cut_pair(120, 7.5), 2.0, 10.0) is None
    assert playhead_join(*cut_pair(120, 9.0), None, 10.0) is None


def test_a_playhead_join_near_a_whole_measure_gets_a_notice_from_the_page_before():
    p, o = cut_pair(120, 8.6, first=12, whole_track=True)
    A = assemble([p, o], 10.0, False, EVIDENCE, lambda v: [])
    assert joins_at(A, o.t0) == [("playhead [(0, 3)]", ["edge"], True)]
    assert A.edge_joins == [(0, 1)]
    assert [(w.kind, w.t0, w.t1) for w in edge_notices(A, [p, o])] == [("edge_join", 0.0, 20.0)]


def test_a_playhead_join_to_a_next_page_opening_mid_measure_is_flagged():
    p, o = cut_pair(120, 9.0)
    assert o.cells[0].complete is True
    assert playhead_join(p, o, 2.0, 10.0) == ({0: 3}, True, False)
    # The next page's first cell starts at the panel edge, inside the measure
    o.cells[0] = replace(o.cells[0], kind="lead", complete=False)
    assert playhead_join(p, o, 2.0, 10.0) == ({0: 3}, True, True)


def test_an_edge_notice_covers_the_view_the_joined_measure_prints_from():
    p, o = cut_pair(120, 8.6, first=12, whole_track=True)
    A = assemble([p, o], 10.0, False, EVIDENCE, lambda v: [])
    q = band_view(2, 12)
    q.t0, q.t1 = 20.0, 30.0
    shown = Measure(A.view_slots[1][0], 2, q.cells[0], 25.0, 25.0, 0.0, 0.0, None, None, False)
    assert [(w.t0, w.t1) for w in edge_notices(A, [p, o, q], [shown])] == [(0.0, 30.0)]


def restart_evidence(p_end: int | None, sliding: bool = True):
    """Replay evidence for a cut from a page of bars 9-12, the playhead at x p_end at its end (unseen when None), to
    bars 1-4 with the playhead on the first measure, in a segment that does or doesn't slide."""
    A = Assembly()
    A.sliding = {0: sliding}
    p, o = band_view(0, 9), band_view(1, 1)
    o.move = Move("cut")
    played(p, *([(9.5, p_end, 30)] if p_end is not None else []))
    played(o, (0.1, 10, 30))
    return replay_evidence(o, o.cells[:2], 0, A, 0.0, p, set(), lambda x: [])


def test_a_sliding_band_cut_from_its_last_measure_to_the_opening_restarts():
    assert restart_evidence(790) == (True, "track_restart", True)


def test_a_restart_cut_from_inside_the_page_folds_unresolved():
    assert restart_evidence(300) == (True, "track_restart", False)


def test_a_restart_cut_with_the_playhead_unseen_on_the_page_before_folds_unresolved():
    assert restart_evidence(None) == (True, "track_restart", False)


def test_a_cut_back_to_the_opening_in_a_band_that_does_not_slide_is_no_restart():
    assert restart_evidence(790, sliding=False) == (False, "none", False)


def numbered_slide(p_end: int | None = None, o_start: int | None = None):
    """A slide of 400 px from a view numbered 1-4 to one whose numbers were misread as 11-14."""
    p, o = slid(400, True, p_end, o_start)
    for n, c in enumerate(p.cells, 1):
        c.number = c.read = n
    for n, c in enumerate(o.cells, 11):
        c.number = c.read = n
    A = assemble([p, o], 10.0, False, EVIDENCE, lambda v: [])
    return A, o


def test_a_slide_the_playhead_confirmed_clears_numbers_that_disagree_with_the_measures_it_joins():
    A, o = numbered_slide(p_end=700, o_start=320)
    assert A.view_slots[1] == [2, 3, 4, 5]
    assert [c.number for c in o.cells] == [None] * 4
    assert [c.read for c in o.cells] == [11, 12, 13, 14]
    joins = [d for d in A.decisions if d.kind == "join"]
    assert [(d.chosen, d.basis) for d in joins] == [("slide 398", "track")]


def test_an_unconfirmed_slide_keeps_the_numbers_as_read():
    A, o = numbered_slide()
    assert [c.number for c in o.cells] == [11, 12, 13, 14]
    assert A.view_slots[1] == [4, 5, 6, 7]


def test_the_folded_reading_keeps_the_numbers_of_a_view_that_slid_on_from_a_fold_without_evidence():
    # Views numbered 1-4 and 5-8, then a view like 1-4 with no numbers or evidence, then a confirmed slide to a view
    # numbered 11-14 whose first measures look like 3-4. The folded reading puts the third view on 1-4, which
    # would make the slide disagree with 11-14
    views = [band_view(0, 1), band_view(1, 5), band_view(2, 1), band_view(3, 3)]
    for v, first in ((views[0], 1), (views[1], 5), (views[3], 11)):
        for n, c in enumerate(v.cells, first):
            c.number = c.read = n
    views[3].move = Move("slide", shifts=[400], drawn=[True], steps=5)
    views[2].playhead = [(views[2].t1 - 0.5, 700, 30)]
    views[3].playhead = [(views[3].t0 + 0.1, 320, 30)]
    folded = assemble(views, 10.0, True, EVIDENCE, lambda v: [])
    assert folded.view_slots[2] == [0, 1, 2, 3]
    assert [c.number for c in views[3].cells] == [11, 12, 13, 14]
    assert folded.view_slots[3] == [8, 9, 10, 11]
    assert [folded.slots[s].number for s in folded.view_slots[3]] == [11, 12, 13, 14]


def test_the_folded_reading_places_by_the_numbers_read_while_the_main_reading_clears_them():
    # A confirmed slide from measures 1-4 to a view misread as 11-14, then a view like 1-4 with no evidence
    p, o = slid(400, True, p_end=700, o_start=320)
    for n, c in enumerate(p.cells, 1):
        c.number = c.read = n
    for n, c in enumerate(o.cells, 11):
        c.number = c.read = n
    views = [p, o, band_view(2, 1)]
    by_stream, main, folded = readings({0: views}, [0], views, 10.0, {}, {}, lambda v: [])
    assert main.view_slots[1] == [2, 3, 4, 5]
    assert main.unresolved_folds == [(20.0, 30.0, 0, 3)]
    assert folded.view_slots == {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [0, 1, 2, 3]}
    assert [folded.slots[s].number for s in folded.view_slots[1]] == [11, 12, 13, 14]
    assert [c.number for c in o.cells] == [None] * 4


def piece(c, kind: str, a: int, b: int):
    """Columns a..b of cell c, as a piece of its measure cut by the page edge."""
    return replace(c, kind=kind, complete=False, x0=c.x0 + a, x1=c.x0 + b, ink=c.ink[:, a:b], faint=c.faint[:, a:b],
                   tint=c.tint[:, a:b])


def slots_of(*views):
    """One slot per cell of the views in order, each first seen as that cell."""
    return [Slot(k, c, v.index, None) for k, (v, c) in enumerate((v, c) for v in views for c in v.cells)]


def test_a_run_matches_a_slot_by_any_earlier_sighting_when_the_first_was_cut_otherwise():
    # Bars 1-4 with bar 4 cut to 120 px by the page edge, then bars 4-7 whole; a third page shows bars 2-5 with
    # bar 4 cut to 160 px, so its bar 4 matches only the whole sighting
    p, w, o = band_view(0, 1), band_view(1, 4), band_view(2, 2)
    p.cells[3] = piece(p.cells[3], "tail", 0, 120)
    slots = slots_of(p) + slots_of(w)[1:]
    sightings = {3: [(0, p.cells[3]), (1, w.cells[0])]}
    run = [o.cells[1], piece(o.cells[2], "tail", 0, 160)]
    assert find_run(slots, run, 10.0, 2, sightings=sightings) == 2
    assert find_run(slots, run, 10.0, 2) is None


def test_a_run_is_matched_against_the_first_sighting_of_a_slot_never_seen_whole():
    # Bars 1-4 with bar 4 cut to its first 120 px, then bar 4 cut to its last 40 px: bar 4 was never seen whole,
    # so only the first piece counts
    p, w, o = band_view(0, 1), band_view(1, 4), band_view(2, 3)
    p.cells[3] = piece(p.cells[3], "tail", 0, 120)
    w.cells[0] = piece(w.cells[0], "lead", 160, 200)
    sightings = {3: [(0, p.cells[3]), (1, w.cells[0])]}
    assert find_run(slots_of(p), [o.cells[0], piece(o.cells[1], "tail", 0, 120)], 10.0, 2, sightings=sightings) == 2
    assert find_run(slots_of(p), [o.cells[0], piece(o.cells[1], "lead", 160, 200)], 10.0, 2,
                    sightings=sightings) is None


def test_a_run_is_not_matched_by_a_cut_or_highlighted_sighting_of_a_slot_seen_whole():
    # Bars 1-4 whole, then bar 4 cut to its last 40 px by the page edge, then bar 4 with the highlight over all
    # but its last 40 px. A new measure ends like bar 4 but is other music: either later sighting alone matches it
    p, w, h = band_view(0, 1), band_view(1, 4), band_view(2, 4)
    w.cells[0] = piece(w.cells[0], "lead", 160, 200)
    tint = np.zeros_like(h.cells[0].ink)
    tint[:, :160] = True
    h.cells[0] = replace(h.cells[0], tint=tint)
    other = band_view(3, 8).cells[1]
    ink = other.ink.copy()
    ink[:, 160:] = p.cells[3].ink[:, 160:]
    new = replace(other, ink=ink)
    assert same_cell(w.cells[0], new, 10.0) and same_cell(h.cells[0], new, 10.0)
    sightings = {3: [(0, p.cells[3]), (1, w.cells[0]), (2, h.cells[0])]}
    assert find_run(slots_of(p), [new], 10.0, 3, sightings=sightings) is None


def test_a_run_matched_only_by_a_cut_sighting_prints_as_new_in_both_readings_with_a_notice():
    # Bars 1-4 whole, then bar 4 cut to its last 40 px by the page edge. A later page shows bar 3 and a measure that
    # ends like bar 4 but differs from its whole sighting: only the cut piece matches it
    p, w = band_view(0, 1), band_view(1, 4)
    w.cells[0] = piece(w.cells[0], "lead", 160, 200)
    o = band_view(2, 8)
    ink = o.cells[1].ink.copy()
    ink[:, 160:] = p.cells[3].ink[:, 160:]
    o.cells = [replace(o.cells[0], ink=p.cells[2].ink), replace(o.cells[1], ink=ink)]
    for fold_unresolved in (False, True):
        A = Assembly(slots=slots_of(p), sightings={3: [(0, p.cells[3]), (1, w.cells[0])]})
        slot_of = [None, None]
        place_rest(o, slot_of, A, 10.0, 0.0, None, EVIDENCE, lambda v: [], fold_unresolved)
        assert slot_of == [4, 5] and A.unresolved_folds == []
        assert A.maybe_repeats == [(o.t0, o.t1, 2)]


def test_a_view_with_an_unresolved_fold_gets_no_may_repeat_notice():
    # Two later pages may repeat bars 1-4; the first of them also folds onto bars 1-4 unresolved
    p, o, q = band_view(0, 1), band_view(1, 5), band_view(2, 9)
    A = Assembly(slots=slots_of(p), maybe_repeats=[(o.t0, o.t1, 0), (q.t0, q.t1, 0)],
                 unresolved_folds=[(o.t0, o.t1, 0, 3)])
    assert [(n.t0, n.t1, n.kind) for n in repeat_notices(A, [p, o, q])] == [(q.t0, q.t1, "possible_duplicate")]


def test_after_a_repeat_sign_the_run_looked_for_is_the_latest_that_matches():
    # Bars 1-4 written out twice, then a page showing bars 1-2 again
    slots = slots_of(band_view(0, 1), band_view(1, 1))
    run = band_view(2, 1).cells[:2]
    assert find_run(slots, run, 10.0, 2) == 0
    assert find_run(slots, run, 10.0, 2, latest=True) == 4


def signed(sign_x: int, at_x: int | None):
    """A one-system page of bars 1-4 (x 0-800, line gap 10) with an end-repeat sign at sign_x and the playhead last
    seen at at_x (unseen when None)."""
    p = band_view(0, 1)
    p.repeat_end, p.repeat_bars = True, [(0, sign_x)]
    played(p, *([(9.5, at_x, 30)] if at_x is not None else []))
    return p


def test_a_repeat_sign_sends_the_playhead_back_only_while_it_lies_ahead():
    assert repeat_played(signed(800, 650))      # the sign ends the measure being played
    assert repeat_played(signed(800, 250))      # the sign is further on
    assert not repeat_played(signed(400, 650))  # the sign ends bar 2, already played
    assert repeat_played(signed(400, None))     # no playhead seen


def two_systems(sign: tuple[int, int], at: tuple[int, int]):
    """A page of bars 1-2 on an upper system and bars 3-4 on a lower one (x 0-400 each, line gap 10), with an
    end-repeat sign at (system, x) and the playhead last seen at (x, y)."""
    p = band_view(0, 1)
    for c in p.cells[2:]:
        c.system, c.x0, c.x1, c.y0, c.mid_y = 1, c.x0 - 400, c.x1 - 400, 100, 130.0
    p.rect, p.single = (0, 0, 400, 160), False
    p.repeat_end, p.repeat_bars = True, [sign]
    played(p, (9.5, *at))
    return p


def test_a_repeat_sign_on_another_system_is_ahead_only_when_that_system_is_lower():
    # The sign ends the upper system and the playhead is in bar 4, on the lower one
    assert not repeat_played(two_systems((0, 400), (250, 130)))
    # The sign ends bar 3, on the lower system, and the playhead is in bar 2, on the upper one
    assert repeat_played(two_systems((1, 200), (350, 30)))
    assert repeat_played(two_systems((1, 400), (350, 130)))


def sign_evidence(j: int, landed: int | None):
    """Replay evidence for a fold at slot j after a page with an end-repeat sign, that page itself placed by the
    sign at slot `landed` (None when it wasn't)."""
    A = Assembly()
    p, o = band_view(0, 1), band_view(1, 1)
    p.repeat_end = True
    if landed is not None:
        A.sign_folds[p.index] = landed
    return replay_evidence(o, o.cells[:2], j, A, 0.0, p, {"navigation"}, lambda v: [])


def test_a_page_placed_by_a_repeat_sign_sends_the_next_one_back_only_to_the_same_place():
    assert sign_evidence(22, None) == (True, "repeat_sign", True)
    assert sign_evidence(22, 20) == (True, "repeat_sign", True)
    assert sign_evidence(18, 20) == (True, "repeat_sign", True)
    assert sign_evidence(23, 20) == (False, "none", False)
    assert sign_evidence(17, 20) == (False, "none", False)


def test_a_page_placed_by_a_repeat_sign_records_where_it_landed_and_the_next_page_must_land_near_it():
    # Bars 1-4, then bars 5-8 ending on an end-repeat sign shown twice, then a page like bars 1-4: four slots back
    # from where the repeat landed, so the sign doesn't send the playhead there
    obs = [band_view(0, 1), band_view(1, 5), band_view(2, 5), band_view(3, 1)]
    for o in obs[1:3]:
        o.repeat_end, o.repeat_bars = True, [(0, 800)]
        played(o, (9.5, 650, 30))
    obs[2].wrapped = obs[3].wrapped = True
    A = assemble(obs, 10.0, False, EVIDENCE, lambda v: [])
    assert A.sign_folds == {2: 4}
    assert [(d.view, d.basis, d.resolved) for d in A.decisions if d.kind == "fold"] == [(2, "repeat_sign", True),
                                                                                       (3, "none", False)]


def split_pair(head: np.ndarray):
    """A page of bars 1-4 whose bar 4 is cut to its first 120 px, and a next page whose first cell, cut by its left
    edge, holds `head`."""
    p, o = band_view(0, 1), band_view(1, 5)
    p.cells[3] = piece(p.cells[3], "tail", 0, 120)
    o.cells[0] = replace(o.cells[0], kind="lead", complete=False, x1=head.shape[1], ink=head,
                         faint=np.zeros_like(head), tint=np.zeros_like(head))
    return p, o


def test_a_measure_split_by_a_page_turn_is_shown_whole_again_or_stitched_from_its_two_pieces():
    p, o = split_pair(music("clef", 4))
    assert split_measure(p, 3, o, 10.0) == "whole"
    p, o = split_pair(music("a", 4)[:, 120:])
    assert split_measure(p, 3, o, 10.0) == "stitch"
    # 120 + 210 px is wider than 1.6 measures of 200 px: the cut measure may have been whole
    p, o = split_pair(np.hstack([music("b", 9), np.zeros((60, 10), bool)]))
    assert split_measure(p, 3, o, 10.0) is None


def rest_pair(t_w: int, h_w: int, h_complete: bool = False):
    """A page of 200 px measures whose last cell is cut to t_w px, and a next page's first cell h_w px wide."""
    p, o = band_view(0, 1), band_view(1, 5)
    t = replace(p.cells[3], kind="tail", complete=False, x1=p.cells[3].x0 + t_w)
    h = replace(o.cells[0], kind="lead", complete=h_complete, x1=h_w)
    return t, h, p


def test_a_narrow_cut_first_cell_holds_the_rest_of_the_cut_measure_when_the_two_make_about_one_measure():
    assert rest_of(*rest_pair(120, 80))
    assert not rest_of(*rest_pair(120, 100))   # half a measure or more is no sliver
    assert not rest_of(*rest_pair(60, 80))     # together under 0.85 of a measure
    assert not rest_of(*rest_pair(190, 95))    # together over 1.4 measures
    assert not rest_of(*rest_pair(120, 80, h_complete=True))


def test_a_cut_measure_played_long_joins_a_next_page_that_opens_on_its_sliver():
    # Played from 7.5 s, too long for a cut measure, so only the sliver rule joins it
    p, o = cut_pair(120, 7.5)
    assert playhead_join(p, o, 2.0, 10.0) is None
    o.cells[0] = replace(piece(o.cells[0], "lead", 120, 200), x0=0, x1=80)
    assert playhead_join(p, o, 2.0, 10.0)[0] == {0: 3}


def system_view() -> Obs:
    """A view of one measure, x 100-200, on a system of a notation staff (lines at y 100-140) over a tab staff
    (lines at y 200-250), line gap 10, its cell box starting at y 90."""
    ink = np.zeros((180, 100), bool)
    c = Cell(0, 0, 0, 100, 200, "full", ink, ink, 90, gap=10.0, ref_y=110, mid_y=225.0, top_y=10)
    return Obs(0, 0.0, 10.0, [c], [], 0, (0, 0))


def test_a_playhead_between_the_notation_and_tab_staffs_is_on_the_measure():
    assert cell_at(system_view(), (150, 170)) == 0


def test_a_playhead_more_than_a_gap_outside_the_system_is_on_no_measure():
    assert cell_at(system_view(), (150, 85)) is None
    assert cell_at(system_view(), (150, 265)) is None


def paged(p_end, o_start, single: bool = True):
    """Two views, the playhead at p_end (x, y) as the first ends and at o_start as the second starts."""
    p = Obs(0, 0.0, 10.0, [], [], 0, (0, 0), single=single, track=[(9.5, *p_end)])
    o = Obs(1, 12.0, 22.0, [], [], 0, (0, 0), single=single, track=[(12.1, *o_start)])
    return p, o


def test_on_pages_of_one_system_a_playhead_gone_back_restarts_though_a_tall_cursor_rose():
    assert playhead_restarts(*paged((900, 400), (50, 250)), 20.0) is True


def test_on_pages_of_one_system_a_playhead_gone_on_does_not_restart_though_a_tall_cursor_rose():
    assert playhead_restarts(*paged((300, 400), (500, 250)), 20.0) is False


def test_on_pages_of_several_systems_a_playhead_gone_up_a_system_restarts():
    assert playhead_restarts(*paged((300, 400), (500, 250), single=False), 20.0) is True
