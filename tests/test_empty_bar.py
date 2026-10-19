import numpy as np

from youtabber.parse import Cell
from youtabber.pipeline import empty_bar, kept_cells

GAP = 10.0


def cell(width: int, marks: bool, kind: str = "full", complete: bool = True, above: bool = False,
         clef: bool = False) -> Cell:
    """A cell between barlines at x 100 and 100 + width whose ink box starts 15 px left of its left barline. Its
    box holds a notation staff (rows 15-49) above the tab staff (rows 60-110)."""
    ink = np.zeros((120, width + 16), bool)
    ink[60:111, 15:17] = True            # left barline
    ink[60:111, -1] = True               # first column of the right barline
    if marks:
        ink[80:92, 15 + width // 2:15 + width // 2 + 12] = True   # a chord of fret digits
    if above:
        ink[2:10, 30:42] = True          # a measure number
        ink[22:27, 15 + width // 2:15 + width // 2 + 12] = True   # a whole rest on the notation staff
    if clef:
        ink[16:48, 20:40] = True         # a clef and key signature on the notation staff
    return Cell(0, 0, 1, 100, 100 + width, kind, ink, np.zeros_like(ink), 0, gap=GAP, ref_y=60, mid_y=85.0,
                complete=complete, top_y=15)


def test_a_bar_with_only_staff_lines_and_barlines_is_empty():
    assert empty_bar(cell(60, marks=False))


def test_a_rest_bar_with_a_number_and_a_notation_rest_is_empty():
    assert empty_bar(cell(60, marks=False, above=True))


def test_a_clef_and_key_signature_cell_is_not_an_empty_bar():
    assert not empty_bar(cell(60, marks=False, clef=True))


def test_a_bar_with_fret_digits_is_not_empty():
    assert not empty_bar(cell(60, marks=True))


def test_lead_cells_cut_cells_and_notation_only_cells_are_never_empty_bars():
    assert not empty_bar(cell(60, marks=False, kind="lead"))
    assert not empty_bar(cell(60, marks=False, complete=False))
    lone = cell(60, marks=False)
    lone.notation_only = True
    assert not empty_bar(lone)


def test_a_blank_sliver_between_a_barline_and_a_stem_is_not_an_empty_bar():
    assert not empty_bar(cell(25, marks=False))


def test_a_short_empty_bar_is_kept_and_a_short_bar_with_notes_is_not():
    short_empty, short_notes, wide = cell(60, marks=False, above=True), cell(60, marks=True), cell(300, marks=True)
    assert kept_cells([short_empty, short_notes, wide], typical=300) == [short_empty, wide]
