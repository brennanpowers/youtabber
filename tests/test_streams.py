from types import SimpleNamespace

import numpy as np
from test_check import FakeMosaic

from youtabber.assemble import Assembly, Decision, Obs, Slot, assemble
from youtabber.parse import Cell
from youtabber.parts import Part, stream_of
from youtabber.pipeline import (Measure, carry_on, first_shown, join_notices, joined, number_streams, on_screen,
                                part_notices, part_pairs, split_duplicates, stream_number_checks)


def still(index: int, part: int, seg: int = 0) -> Obs:
    o = Obs(index, float(index), index + 1.0, [], [], seg, (0, 0))
    o.part = part
    return o


def test_views_of_parts_shown_at_once_pair_within_their_own_part():
    obs = [still(i, p) for i, p in enumerate([0, 1, 0, 1, 0])]
    assert {(p.index, o.index) for p, o in part_pairs(obs)} == {(0, 2), (2, 4), (1, 3)}


def test_views_of_different_segments_never_pair():
    assert part_pairs([still(0, 0, seg=0), still(1, 0, seg=1)]) == []


def screen(index: int, t0: float, t1: float, xs: list[tuple[int, int]], part: int = 0) -> Obs:
    """A view of part `part` whose cells span the columns xs, on a panel 1000 px wide."""
    ink = np.zeros((10, 10), bool)
    cells = [Cell(index, 0, i, x0, x1, "full", ink, ink, 0) for i, (x0, x1) in enumerate(xs)]
    o = Obs(index, t0, t1, cells, [], 0, (0, 0), rect=(0, 0, 1000, 200))
    o.all_cells = list(cells)
    o.part = part
    return o


def placed(view_slots: dict[int, list[int]], obs: list[Obs]) -> Assembly:
    """An assembly whose views hold the slots given, each slot first read in the first view holding it."""
    first: dict[int, int] = {}
    for v, sl in view_slots.items():
        for s in sl:
            first.setdefault(s, v)
    A = Assembly(slots=[Slot(s, obs[first[s]].cells[view_slots[first[s]].index(s)], first[s], None)
                        for s in sorted(first)], view_slots=view_slots)
    A.order = [s.id for s in A.slots]
    A.sightings = {s.id: [(s.obs, obs[s.obs].cells[view_slots[s.obs].index(s.id)])] for s in A.slots}
    return A


THREE = [(0, 300), (300, 600), (600, 900)]


def test_the_assemblies_of_two_streams_join_with_slot_ids_of_their_own():
    obs = [screen(0, 0.0, 5.0, THREE, part=0), screen(1, 0.0, 5.0, THREE[:2], part=1)]
    second = placed({1: [0, 1]}, obs)
    second.switches = [(12.0, 15.0)]
    second.jump_repeats, second.unsure_slots = [(12.0, 15.0, 1)], {0: (12.0, 15.0)}
    out = joined([placed({0: [0, 1, 2]}, obs), second], obs, {0: 100, 1: 600}, {0: 1200, 1: 1200}, [0, 3])
    assert [s.id for s in out.slots] == [0, 1, 2, 3, 4]
    assert [s.stream for s in out.slots] == [0, 0, 0, 3, 3]
    assert out.stream_orders == {0: [0, 1, 2], 3: [3, 4]}
    assert out.view_slots == {0: [0, 1, 2], 1: [3, 4]}
    assert out.switches == [(12.0, 15.0)]
    assert out.jump_repeats == [(12.0, 15.0, 4)] and out.unsure_slots == {3: (12.0, 15.0)}


def test_the_assembly_of_one_stream_is_kept_as_it_is():
    obs = [screen(0, 0.0, 5.0, THREE)]
    A = placed({0: [0, 1, 2]}, obs)
    assert joined([A], obs, {0: 0}, {0: 1200}, [0]) is A
    assert A.stream_orders == {0: A.order}
    assert [s.stream for s in A.slots] == [0, 0, 0]


CROP = {0: 1200}


def test_a_measure_scrolled_on_first_showed_in_the_view_before_when_the_shift_puts_it_inside_the_crop():
    # The view before ends its last measure on its panel's right edge, but the crop runs 200 px further
    obs = [screen(0, 0.0, 5.0, [(100, 400), (400, 700), (700, 1000)]), screen(1, 5.5, 10.0, THREE)]
    assert first_shown(placed({0: [0, 1, 2], 1: [1, 2, 3]}, obs), obs, CROP)[3] == 0.0


def test_a_measure_at_a_left_panel_s_edge_first_showed_there_when_the_full_width_part_goes_on_from_it():
    # A left panel 960 px wide (part 1) shows the start of slot 3 past its last barline; the full width (part 0) follows
    for stream, when in ((0, 0.0), (3, 5.5)):
        obs = [screen(0, 0.0, 5.0, THREE, part=1), screen(1, 5.5, 10.0, THREE)]
        obs[0].stream = stream
        assert first_shown(placed({0: [0, 1, 2], 1: [1, 2, 3]}, obs), obs, {0: 1920, 1: 960})[3] == when


def test_a_measure_peeking_from_a_screen_cut_into_two_views_first_showed_at_the_first_one():
    obs = [screen(0, 0.0, 2.5, THREE), screen(1, 2.5, 5.0, THREE), screen(2, 5.5, 10.0, THREE)]
    assert first_shown(placed({0: [0, 1, 2], 1: [0, 1, 2], 2: [1, 2, 3]}, obs), obs, CROP)[3] == 0.0


def test_a_measure_scrolled_on_that_the_shift_puts_at_the_crop_edge_first_showed_in_its_own_view():
    obs = [screen(0, 0.0, 5.0, [(298, 598), (598, 898), (898, 1198)]), screen(1, 5.5, 10.0, THREE)]
    assert first_shown(placed({0: [0, 1, 2], 1: [1, 2, 3]}, obs), obs, CROP)[3] == 5.5


def test_the_shift_skips_cell_ends_cut_by_the_panel_edge():
    # Slot 1 enters view 1 cut by the left edge and slot 2 leaves view 0 cut by the panel's right edge, inside the
    # crop: only their inner barlines show the 910 px slide, which leaves slot 3 starting past the crop
    obs = [screen(0, 0.0, 5.0, [(200, 500), (500, 1000), (1000, 1050)]), screen(1, 5.5, 10.0, [(0, 90), (90, 400),
                                                                                                (400, 700)])]
    obs[0].cells[2].kind = "tail"
    obs[1].cells[0].kind = "lead"
    assert first_shown(placed({0: [0, 1, 2], 1: [1, 2, 3]}, obs), obs, CROP)[3] == 5.5


def test_after_a_page_turn_a_new_measure_first_showed_in_the_view_before_when_it_dropped_a_tail_cell():
    ink = np.zeros((10, 10), bool)
    for tail, when in ((True, 0.0), (False, 5.5)):
        obs = [screen(0, 0.0, 5.0, THREE), screen(1, 5.5, 10.0, THREE)]
        if tail:
            obs[0].all_cells.append(Cell(0, 0, 3, 900, 990, "tail", ink, ink, 0))
        assert first_shown(placed({0: [0, 1, 2], 1: [3, 4, 5]}, obs), obs, CROP)[3] == when, tail


def test_views_further_apart_than_a_page_turn_keep_their_own_start():
    obs = [screen(0, 0.0, 5.0, THREE), screen(1, 6.5, 10.0, THREE)]
    assert first_shown(placed({0: [0, 1, 2], 1: [1, 2, 3]}, obs), obs, CROP)[3] == 6.5


def test_a_glided_measure_prints_by_when_its_start_reached_the_play_column():
    # A glide piece of part 0 plays from 0 to 100 s, its measures reaching the play column at 10, 50 and 90 s; part 1
    # shows one still screen at 50-55 s
    glide = screen(0, 0.0, 100.0, [(1000, 1400), (5000, 5400), (9000, 9400)])
    glide.mosaic = FakeMosaic()
    obs = [glide, screen(1, 50.0, 55.0, THREE, part=1)]
    out = joined([placed({0: [0, 1, 2]}, obs), placed({1: [0, 1, 2]}, obs)], obs, {0: 100, 1: 600},
                 {0: 1200, 1: 1200}, [0, 1])
    assert out.order == [0, 1, 3, 4, 5, 2]


def test_a_second_glide_piece_times_its_first_measure_by_its_own_mosaic():
    # Two pieces of one glide: 0-50 s (measures reach the play column at 10 and 30 s) and 50-100 s (at 60 and 90 s).
    # The redraw cut a narrow tail cell off the first piece, which looks like a page turn but isn't one
    p = screen(0, 0.0, 50.0, [(1000, 1400), (3000, 3400)])
    o = screen(1, 50.0, 100.0, [(6000, 6400), (9000, 9400)])
    for v in (p, o):
        v.mosaic = FakeMosaic()
    ink = np.zeros((10, 10), bool)
    p.all_cells = p.cells + [Cell(0, 0, 2, 4800, 4900, "tail", ink, ink, 0)]
    assert first_shown(placed({0: [0, 1], 1: [2, 3]}, [p, o]), [p, o], {0: 10000})[2] == 60.0


def test_a_part_beside_the_main_one_misses_no_numbers_from_before_it_came_on_screen():
    view = screen(0, 61.5, 66.0, THREE, part=1)
    for stream, warned in ((1, False), (0, True)):
        part = Part(1, 0, (0, 100), "band", found={61.0: ["system"], 63.0: ["system"]}, stream=stream)
        primary = [Measure(i, 0, view.cells[i], 62.0 + i, 62.0 + i, 0, 0, None, 22 + i, False, stream)
                   for i in range(3)]
        kinds = [w["kind"] for w in stream_number_checks(stream, primary, [part], [view])]
        assert ("numbers_start" in kinds) == warned, stream


def test_skipped_numbers_are_pointed_out_only_when_the_part_stayed_on_screen_through_the_skip():
    w = {"kind": "numbers_skipped", "t0": 10.0, "t1": 14.0}
    assert on_screen(w, [10.0 + 0.5 * k for k in range(9)], 0.0)
    # Off screen for 4 s, longer than a page turn
    assert not on_screen(w, [10.0, 14.0], 0.0)


def music(part: str, number: int) -> np.ndarray:
    """Ink of one measure, different for each part and measure."""
    rng = np.random.default_rng([ord(part), number])
    ink = np.zeros((60, 200), bool)
    for y, x in zip(rng.integers(5, 55, 12), rng.integers(10, 190, 12)):
        ink[y - 3:y + 3, x - 4:x + 4] = True
    return ink


def numbered_view(index: int, part: str, numbers: list[int]) -> Obs:
    """A screen of `part` whose measures read as `numbers`, on the stream of its own name."""
    cells = []
    for i, n in enumerate(numbers):
        ink = music(part, n)
        cells.append(Cell(index, 0, i, 200 * i, 200 * (i + 1), "full", ink, np.zeros_like(ink), 0, gap=10.0,
                          mid_y=30.0, tint=np.zeros_like(ink), number_crop=np.full((5, 5), n)))
    o = Obs(index, 5.0 * index, 5.0 * index + 5, cells, [], 0, (0, 0))
    o.part = o.stream = 0 if part == "a" else 1
    return o


def test_two_streams_shown_at_once_are_numbered_and_placed_apart():
    # Both parts count 1-4, and their screens alternate in time
    shown = [("a", [1, 2]), ("b", [1, 2]), ("a", [2, 3]), ("b", [2, 3]), ("a", [3, 4]), ("b", [3, 4])]
    obs = [numbered_view(i, part, numbers) for i, (part, numbers) in enumerate(shown)]
    stream_obs = {k: [o for o in obs if o.stream == k] for k in (0, 1)}
    assert number_streams(stream_obs, lambda crop: int(crop[0, 0])) == "numbered"
    by_stream = {k: assemble(views, 10.0, False, {"numbers"}, lambda o: []) for k, views in stream_obs.items()}
    for k, A in by_stream.items():
        assert [s.number for s in A.slots] == [1, 2, 3, 4]
    out = joined([by_stream[0], by_stream[1]], obs, {0: 0, 1: 100}, {0: 1200, 1: 1200}, [0, 1])
    assert out.stream_orders == {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    assert [s.stream for s in out.slots] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert all(obs[s.obs].stream == s.stream for s in out.slots)


def two_parts(seg: int, t: float, first: int) -> tuple[list[Part], list[Obs]]:
    """Two parts of segment `seg` from time t: the first shows at t+0-6 and t+6-12, the second at t+2-8 and t+8-14.
    View indexes and part ids start at `first`."""
    parts = [Part(first, seg, (0, 100), "band"), Part(first + 1, seg, (200, 300), "band")]
    obs = []
    for i, (part, t0) in enumerate([(0, 0), (1, 2), (0, 6), (1, 8)]):
        o = Obs(first * 2 + i, t + t0, t + t0 + 6, [], [], seg, (0, 0))
        o.part = first + part
        obs.append(o)
    return parts, obs


def test_parts_that_print_get_one_notice_over_the_first_view_of_the_second_to_show():
    parts, obs = two_parts(0, 0.0, 0)
    notices = part_notices(parts, obs, [SimpleNamespace(view=0), SimpleNamespace(view=3)])
    assert [(n.kind, n.t0, n.t1) for n in notices] == [("parts", 2.0, 8.0)]


def test_a_part_that_prints_nothing_gets_no_notice():
    parts, obs = two_parts(0, 0.0, 0)
    assert part_notices(parts, obs, [SimpleNamespace(view=0), SimpleNamespace(view=2)]) == []


def test_each_segment_with_two_printing_parts_gets_its_own_notice():
    parts, obs = two_parts(0, 0.0, 0)
    more_parts, more_obs = two_parts(1, 100.0, 2)
    primary = [SimpleNamespace(view=v) for v in (0, 1, 4, 5)]
    notices = part_notices(parts + more_parts, obs + more_obs, primary)
    assert [(n.t0, n.t1) for n in notices] == [(2.0, 8.0), (102.0, 108.0)]


def test_an_unresolved_join_notice_covers_the_view_its_decision_was_made_for():
    obs = [Obs(0, 0.0, 10.0, [], [], 0, (0, 0)), Obs(1, 10.0, 15.0, [], [], 0, (0, 0)),
           Obs(2, 10.0, 30.0, [], [], 0, (0, 0))]
    obs[2].part = obs[2].stream = 1
    # Both parts' views start at 10 s, so only the decision's view tells them apart
    for view, end in ((1, 15.0), (2, 30.0)):
        A = Assembly(decisions=[Decision(10.0, "join", "page turn", [], "track", False, view=view)])
        assert [(n.kind, n.t0, n.t1) for n in join_notices(A, obs)] == [("unresolved_join", 9.0, end)]


def panel_view(index: int, t0: float, t1: float, part: int, inks: list[np.ndarray], seg: int = 0) -> Obs:
    """A still view of part `part` with one full cell per ink pattern given."""
    cells = [Cell(index, 0, i, 300 * i, 300 * i + 300, "full", ink, np.zeros_like(ink), 0, gap=10.0,
                  tint=np.zeros_like(ink))
             for i, ink in enumerate(inks)]
    o = Obs(index, t0, t1, cells, [], seg, (0, 0))
    o.all_cells = list(cells)
    o.part = part
    return o


def marks(seed: int) -> np.ndarray:
    """A cell's ink: fret-sized blobs on a grid, each drawn or not as the seed sets."""
    ink = np.zeros((60, 300), bool)
    for k, on in enumerate(np.random.default_rng(seed).random(40) < 0.5):
        if on:
            y, x = 4 + 14 * (k // 10), 6 + 29 * (k % 10)
            ink[y:y + 10, x:x + 14] = True
    return ink


def split_parts(whole: bool) -> list[Part]:
    """A segment's split screen: the full width (when `whole`), then a left and a right panel."""
    left = Part(1, 0, (0, 100), "band", cols=(0, 960), carries=not whole)
    right = Part(2, 0, (0, 100), "band", cols=(960, 1920))
    parts = [Part(0, 0, (0, 100), "band", seen=10.0), left, right]
    left.continues = parts[0] if whole else None
    return parts


def numbered(obs: list[Obs], numbers: list[list[int]]) -> None:
    """Give each view's cells number crops that stand for the numbers read from them."""
    for o, ns in zip(obs, numbers):
        for c, n in zip(o.cells, ns):
            c.number_crop = n


def test_the_panel_whose_numbers_the_full_width_goes_on_from_carries_the_stream():
    for whole in (True, False):
        parts = split_parts(whole)
        if not whole:
            parts[0].seg = 1
        obs = [panel_view(0, 0.0, 5.0, 1, [marks(1), marks(2)]), panel_view(1, 0.0, 5.0, 2, [marks(3), marks(4)]),
               panel_view(2, 6.0, 9.0, 0, [marks(5), marks(6), marks(7)], seg=parts[0].seg)]
        # The number over a measure cut at the edge reads short
        numbered(obs, [[12, 13], [53, 54], [4, 55, 56]])
        carry_on(parts, obs, lambda crop: crop)
        assert parts[2].stream == 0 and parts[1].stream != 0
        assert [o.stream for o in obs[:2]] == [parts[1].stream, 0]


def test_the_left_panel_carries_the_stream_when_the_numbers_single_no_panel_out():
    # No numbers, numbers going on from neither panel, from both, too few readings to count from, and a panel
    # whose numbers read nothing
    for numbers in ([[], [], []], [[12, 13], [53, 54], [30, 31]], [[12, 13], [12, 13], [13, 14]],
                    [[12, 13], [53, 54], [55, None]], [[None, None], [53, 54], [55, 56]]):
        parts = split_parts(False)
        parts[0].seg = 1
        for p in parts[1:]:
            p.stream = stream_of(p, parts[1:])
        obs = [panel_view(0, 0.0, 5.0, 1, [marks(1), marks(2)]), panel_view(1, 0.0, 5.0, 2, [marks(3), marks(4)]),
               panel_view(2, 6.0, 9.0, 0, [marks(5), marks(6)], seg=1)]
        numbered(obs, numbers)
        carry_on(parts, obs, lambda crop: crop)
        assert (parts[1].stream, parts[2].stream) == (0, 3)


def test_a_split_screen_s_parts_notice_covers_every_panel_view():
    parts = split_parts(True)
    obs = [panel_view(0, 0.0, 4.0, 0, []), panel_view(1, 4.0, 9.0, 1, []), panel_view(2, 4.0, 7.0, 2, []),
           panel_view(3, 9.0, 15.0, 1, []), panel_view(4, 7.0, 20.0, 2, [])]
    primary = [SimpleNamespace(view=v) for v in (0, 1, 2)]
    assert [(n.kind, n.t0, n.t1) for n in part_notices(parts, obs, primary)] == [("parts", 4.0, 20.0)]


def test_the_same_measure_read_in_both_panels_at_once_gets_a_duplicate_notice():
    for other, notices in ((marks(1), 1), (marks(9), 0)):
        obs = [panel_view(0, 0.0, 5.0, 1, [marks(1)]), panel_view(1, 0.0, 5.0, 2, [other])]
        A = placed({0: [0], 1: [1]}, obs)
        found = split_duplicates(A, obs, split_parts(True), 10.0)
        assert [n.kind for n in found] == ["possible_duplicate"] * notices
