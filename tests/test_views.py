from types import SimpleNamespace

import numpy as np

from youtabber.assemble import Assembly, Obs, number_views
from youtabber.check import numbers_against_cells
from youtabber.parse import Cell
from youtabber.pipeline import covered, measures_of


def cell(x0: int, x1: int, number: int | None = None, **kw) -> Cell:
    ink = np.zeros((40, x1 - x0), bool)
    return Cell(0, 0, 0, x0, x1, "full", ink, ink, 0, number_crop=None if number is None else np.full((5, 5), number),
                gap=10.0, ref_y=10, **kw)


def view(index: int, cells, t0: float = 0.0, t1: float = 5.0) -> Obs:
    return Obs(index, t0, t1, cells, [], 0, (0, 0))


def test_a_number_read_against_the_count_is_a_conflict():
    # Three views count up from 1, 4 and 7; the last one's third cell is read as 10, where the count says 9.
    # The second view's last cell reads 60, a misread too far off to point at a barline
    views = [view(0, [cell(0, 90, 1), cell(100, 190, 2), cell(200, 290, 3)]),
             view(1, [cell(0, 90, 4), cell(100, 190, 5), cell(200, 290, 6), cell(300, 390, 60)]),
             view(2, [cell(0, 90, 7), cell(100, 190, 8), cell(200, 290, 10), cell(300, 390, 10)], 5.0, 9.0)]
    assert number_views(views, lambda crop: int(crop[0, 0])) == "numbered"
    assert [c.number for c in views[2].cells] == [7, 8, 9, 10]
    assert [n for _, n in views[2].number_conflicts] == [10]
    assert views[0].number_conflicts == [] and views[1].number_conflicts == []
    (w,) = numbers_against_cells(views)
    assert w["kind"] == "numbers_disagree" and (w["t0"], w["t1"]) == (5.0, 9.0)


def systems(*tops: float):
    return [SimpleNamespace(system=SimpleNamespace(tab=SimpleNamespace(top=t, gap=10.0))) for t in tops]


def banded(ink: np.ndarray, tops=(50.0,)) -> Obs:
    o = view(0, [])
    o.band, o.systems = ink, systems(*tops)
    return o


def test_a_view_with_a_card_over_the_page_is_covered():
    page = np.zeros((100, 200), bool)
    page[40:60, 10:190:10] = True                     # the music, the same on both screens
    card = page.copy()
    card[10:30, 20:180] = True                         # a title card laid over the first screen
    assert covered(banded(card), banded(page))
    assert not covered(banded(page), banded(card))   # the later, busier view is not the covered one
    assert not covered(banded(page), banded(page))
    assert not covered(banded(card), banded(page, (80.0,)))   # the staffs moved: another page
    # A fade-out after a clean view also loses marks, but it is fainter: the clean view is not covered
    clean, fading = banded(card), banded(page)
    clean.contrast, fading.contrast = 255.0, 76.0
    assert not covered(clean, fading)


def test_the_best_sighting_shows_the_tab_then_follows_the_view_before():
    A = Assembly()
    # Slot 0 is seen on view 1 as notation only and on view 2 whole; slot 1 is seen alike on both
    a, b = cell(0, 100), cell(0, 100, notation_only=True)
    c1, c2 = cell(100, 200), cell(100, 200)
    obs = [view(0, []), view(1, [b, c1]), view(2, [a, c2])]
    A.sightings = {0: [(1, b), (2, a)], 1: [(1, c1), (2, c2)]}
    picked = measures_of(A, obs, [0, 1])
    assert [m.view for m in picked] == [2, 2]
