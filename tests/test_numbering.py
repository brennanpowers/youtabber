import numpy as np

from youtabber.assemble import Obs, assemble
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
