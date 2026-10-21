import numpy as np
from test_numbering import band_view, music

from youtabber.assemble import (assemble, playhead_at, playhead_track, register, scroll_shifts, still_links,
                                track_join)
from youtabber.track import Move


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
