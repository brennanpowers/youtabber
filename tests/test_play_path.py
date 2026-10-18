import numpy as np

from youtabber.assemble import Assembly, Obs
from youtabber.parse import Cell
from youtabber.pipeline import play_path


class FakeMosaic:
    """Canvas x / 100 is the play time."""
    def time_at(self, x: float) -> float:
        return x / 100.0


def cells(n: int) -> list[Cell]:
    """n cells 100 px wide side by side on one staff."""
    ink = np.zeros((10, 100), bool)
    return [Cell(0, 0, i, i * 100, (i + 1) * 100, "full", ink, ink, 0, gap=10.0, mid_y=50.0) for i in range(n)]


def view(index: int, n: int, playhead=(), glide=False) -> Obs:
    """A view of n cells; playhead lists the x positions the playhead was seen at, one second apart."""
    o = Obs(index, float(index * 10), float(index * 10 + 10), cells(n),
            [(index * 10 + k, x, 50) for k, x in enumerate(playhead)], 0, (0, 0))
    if glide:
        o.mosaic = FakeMosaic()
    return o


def slots(path) -> list[int]:
    return [s for s, _, _, _ in path]


def test_cells_tied_to_the_view_before_are_not_listed_again():
    obs = [view(0, 5), view(1, 2)]
    A = Assembly(view_slots={0: [15, 16, 17, 18, 19], 1: [17, 18]}, view_links={1: {0, 1}})
    assert slots(play_path(A, obs)) == [15, 16, 17, 18, 19]


def test_cells_not_tied_to_the_view_before_are_listed():
    obs = [view(0, 3), view(1, 3)]
    A = Assembly(view_slots={0: [0, 1, 2], 1: [1, 2, 3]}, view_links={1: {0}})
    assert slots(play_path(A, obs)) == [0, 1, 2, 2, 3]


def test_a_view_without_join_links_drops_the_repeat_of_the_path_end():
    obs = [view(0, 3), view(1, 3)]
    A = Assembly(view_slots={0: [0, 1, 2], 1: [1, 2, 3]})
    assert slots(play_path(A, obs)) == [0, 1, 2, 3]


def test_a_left_out_cell_between_two_placed_ones_takes_the_slot_between():
    obs = [view(0, 6, glide=True)]
    A = Assembly(view_slots={0: [10, None, None, 13, None, 20]})
    assert slots(play_path(A, obs)) == [10, 11, 12, 13, 20]


def test_cells_left_of_the_first_playhead_sample_are_dropped_when_already_played():
    obs = [view(0, 3, playhead=[50, 150, 250]), view(1, 3, playhead=[150, 250])]
    A = Assembly(view_slots={0: [0, 1, 2], 1: [0, 1, 2]}, view_links={1: set()})
    assert slots(play_path(A, obs)) == [0, 1, 2, 1, 2]


def test_cells_left_of_the_first_playhead_sample_are_kept_when_one_was_not_played():
    obs = [view(0, 2, playhead=[50, 150]), view(1, 3, playhead=[250])]
    A = Assembly(view_slots={0: [1, 2], 1: [0, 1, 2]}, view_links={1: set()})
    assert slots(play_path(A, obs)) == [1, 2, 0, 1, 2]
