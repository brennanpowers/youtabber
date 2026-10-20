import numpy as np

from youtabber.assemble import Obs, assemble, number_views
from youtabber.parse import Cell


def music(part: str, number: int) -> np.ndarray:
    """Ink of one measure, the same each time a part's measure is drawn and different between parts. Part "clef"
    draws part "a" at the start of a system, with a clef before it; part "cut" draws part "a" with its top line gap
    cut off by the screen edge, so its staff is placed a line gap too low."""
    if part == "clef":
        clef = np.zeros((60, 30), bool)
        clef[10:50, 8:20] = True
        return np.hstack([clef, music("a", number)])
    if part == "cut":
        return music("a", number)[10:]
    rng = np.random.default_rng([ord(part[0]), number])
    ink = np.zeros((60, 200), bool)
    for y, x in zip(rng.integers(5, 55, 12), rng.integers(10, 190, 12)):
        ink[y - 3:y + 3, x - 4:x + 4] = True
    return ink


def view(index: int, measures: list[tuple[str, int]], seg: int = 0) -> Obs:
    """A screen of numbered measures side by side on one staff, in layout seg."""
    cells, x = [], 0
    for i, (p, n) in enumerate(measures):
        ink = music(p, n)
        w = ink.shape[1]
        cells.append(Cell(index, 0, i, x, x + w, "full", ink, np.zeros_like(ink), 0, gap=10.0, mid_y=30.0,
                          tint=np.zeros_like(ink), number=n))
        x += w
    return Obs(index, 10.0 * index, 10.0 * index + 10, cells, [], seg, (0, 0))


def slots(*screens: list[tuple[str, int]], segs: tuple[int, ...] = ()) -> list[list[int]]:
    obs = [view(i, s, segs[i] if segs else 0) for i, s in enumerate(screens)]
    A = assemble(obs, 10.0, False, {"numbers"}, lambda o: [])
    return [A.view_slots[i] for i in range(len(screens))]


def run(part: str, first: int, last: int) -> list[tuple[str, int]]:
    return [(part, n) for n in range(first, last + 1)]


def test_a_replay_with_printed_numbers_folds_onto_the_first_showing():
    assert slots(run("a", 1, 4), run("a", 5, 8), run("a", 1, 4)) == [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3]]


def test_another_part_numbered_the_same_way_gets_measures_of_its_own():
    # The second part's screen also shows 3-4, which the screen before showed for the first part
    assert slots(run("a", 1, 4), run("a", 3, 6), run("b", 1, 4)) == [[0, 1, 2, 3], [2, 3, 4, 5], [6, 7, 8, 9]]


def test_a_number_past_the_screen_before_carries_on_whatever_it_looked_like_before():
    # Measure 5 was glimpsed drawn differently on the first screen, then left the screen
    first = run("a", 1, 4) + [("x", 5)]
    assert slots(first, run("a", 2, 4), run("a", 5, 7)) == [[0, 1, 2, 3, 4], [1, 2, 3], [4, 5, 6]]


def test_one_measure_drawn_differently_on_a_replayed_screen_still_folds():
    replay = run("a", 1, 4)
    replay[0] = ("x", 1)
    assert slots(run("a", 1, 4), run("a", 5, 8), replay) == [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3]]


def test_a_replay_in_another_layout_folds_though_it_is_drawn_differently():
    assert slots(run("a", 1, 4), run("a", 5, 8), run("b", 1, 4), segs=(0, 0, 1)) == [[0, 1, 2, 3], [4, 5, 6, 7],
                                                                                    [0, 1, 2, 3]]


def test_a_measure_shown_again_at_the_start_of_a_system_folds_though_a_clef_is_drawn_before_it():
    assert slots(run("a", 1, 4), run("a", 5, 8), [("clef", 3)] + run("a", 9, 10)) == [[0, 1, 2, 3], [4, 5, 6, 7],
                                                                                         [2, 8, 9]]


def test_another_part_showing_the_numbers_of_the_screen_before_gets_measures_of_its_own():
    assert slots(run("a", 1, 4), run("a", 3, 6), run("b", 5, 8)) == [[0, 1, 2, 3], [2, 3, 4, 5], [6, 7, 8, 9]]


def test_measures_of_the_screen_before_cut_at_the_top_by_the_screen_edge_are_the_same_music():
    assert slots(run("a", 1, 4), run("a", 3, 6), run("cut", 5, 6) + run("a", 7, 8)) == [[0, 1, 2, 3], [2, 3, 4, 5],
                                                                                         [4, 5, 6, 7]]


def test_one_measure_drawn_differently_where_the_screens_overlap_is_not_another_part():
    # Measure 6, the only one this screen shares with the screen before, is drawn differently there
    assert slots(run("a", 1, 4), run("a", 3, 6), [("x", 6)] + run("a", 7, 9)) == [[0, 1, 2, 3], [2, 3, 4, 5],
                                                                                   [5, 6, 7, 8]]


def band_view(index: int, first: int) -> Obs:
    """A screen of a scrolling band without readable numbers: measures first to first + 3 of part "a"."""
    o = view(index, run("a", first, first + 3))
    for c in o.cells:
        c.number = None
    o.band = np.hstack([c.ink for c in o.cells])
    o.tall, o.fixed = o.band, np.zeros_like(o.band)
    o.rect, o.single = (0, 0, o.band.shape[1], o.band.shape[0]), True
    o.bars = sorted({c.x0 for c in o.cells} | {c.x1 for c in o.cells})
    return o


def test_music_like_earlier_music_after_a_scroll_prints_again_without_evidence():
    # The band scrolls two measures at a time through measures 1-12, then a screen shows measures like 1-4 again:
    # a riff written out again looks the same as a jump back, so only the folded reading folds it
    obs = [band_view(i, 1 + 2 * i) for i in range(5)] + [band_view(5, 1)]
    main = assemble(obs, 10.0, False, {"numbers", "playhead", "navigation"}, lambda o: [])
    assert [main.view_slots[i] for i in (1, 4)] == [[2, 3, 4, 5], [8, 9, 10, 11]]
    assert main.view_slots[5] == [12, 13, 14, 15]
    folded = assemble(obs, 10.0, True, {"numbers", "playhead", "navigation"}, lambda o: [])
    assert folded.view_slots[5] == [0, 1, 2, 3]


def read_view(index: int, measures: list[tuple[str, int]], shown: list[int | None]) -> Obs:
    """A screen whose measure numbers read as `shown` (None where no number could be read)."""
    o = view(index, measures)
    for c, n in zip(o.cells, shown):
        c.number, c.number_crop = None, None if n is None else np.full((5, 5), n)
    return o


def test_a_screen_keeps_its_own_read_number_over_a_look_alike_neighbor():
    # The second screen repeats the first screen's riff as measures 4-6, and only its "6" can be read
    obs = [read_view(0, run("a", 1, 3), [1, 2, 3]), read_view(1, run("a", 1, 3), [None, None, 6]),
           read_view(2, run("a", 7, 9), [7, 8, 9]), read_view(3, run("a", 10, 12), [10, 11, 12])]
    assert number_views(obs, lambda crop: int(crop[0, 0])) == "numbered"
    assert obs[1].start is None and [c.number for c in obs[1].cells] == [None, None, None]
    # Without a read of its own, or with one far off (a misread such as 30 over measure 2), the screen still borrows
    # the numbers of the screen it looks like
    for shown in ([None, None, None], [None, 30, None]):
        obs[1] = read_view(1, run("a", 1, 3), shown)
        number_views(obs, lambda crop: int(crop[0, 0]))
        assert [c.number for c in obs[1].cells] == [1, 2, 3], shown


def test_a_screen_that_began_where_the_playhead_jumped_back_still_joins_by_a_scroll():
    # The band scrolls on by two measures, then by two more of a riff, so the playhead jumps back to where the same
    # music now sits
    obs = [band_view(0, 1), band_view(1, 3), band_view(2, 5)]
    obs[2].wrapped = True
    A = assemble(obs, 10.0, False, {"numbers", "playhead", "navigation"}, lambda o: [])
    assert A.view_slots[2] == [4, 5, 6, 7]


def test_a_screen_shown_again_after_the_playhead_jumped_back_is_an_unresolved_fold():
    # The same page twice, the second time from where the playhead went back, with every measure marked by cursor
    # residue so none matches
    obs = [band_view(0, 1), band_view(1, 1)]
    obs[1].wrapped = True
    for c in obs[1].cells:
        c.ink = c.ink.copy()
        c.ink[20:40, 80:120] = True
    main = assemble(obs, 10.0, False, {"numbers", "playhead", "navigation"}, lambda o: [])
    assert main.view_slots[1] == [4, 5, 6, 7] and len(main.unresolved_folds) == 1
    folded = assemble(obs, 10.0, True, {"numbers", "playhead", "navigation"}, lambda o: [])
    assert folded.view_slots[1] == [0, 1, 2, 3]
