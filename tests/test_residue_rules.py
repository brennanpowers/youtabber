import numpy as np

from youtabber.assemble import Obs
from youtabber.parse import Cell, colored_bars
from youtabber.pipeline import colored_bar_notices, counted_header

GAP = 10.0


class Staff:
    top, bottom, gap = 20.0, 70.0, 12.5


def test_a_colored_box_side_is_marked_but_a_gray_barline_and_a_thin_colored_cursor_are_not():
    dark = np.zeros((100, 200), bool)
    tint = np.zeros((100, 200), bool)
    dark[20:71, 50:53] = True       # gray barline
    dark[20:71, 120:126] = True     # box side: dark gray anti-aliased edge columns around a colored core
    tint[20:71, 121:125] = True
    dark[20:71, 160:162] = tint[20:71, 160:162] = True   # line cursor
    assert colored_bars(dark, tint, Staff, [(50, 52), (120, 125), (160, 161)]) == {120}


def cell(x0: int, x1: int, number: int | None, frets: bool = False) -> Cell:
    """A cell whose ink box starts 15 px left of its left barline, tab staff rows 60-110, TAB clef letters at its
    left and, with `frets`, a fret digit in its middle."""
    w = x1 - x0
    ink = np.zeros((120, w + 16), bool)
    ink[62:108, 18:30] = True            # T, A, B letters
    if frets:
        ink[80:92, 15 + w // 2:15 + w // 2 + 8] = True
    c = Cell(0, 0, 0, x0, x1, "lead", ink, np.zeros_like(ink), 0, gap=GAP, ref_y=60, mid_y=85.0, top_y=15)
    c.number = number
    return c


def view(cells: list[Cell]) -> Obs:
    return Obs(index=0, t0=4.0, t1=12.0, cells=cells, playhead=[], seg=0, offset=(0, 0))


def measures(first: Cell) -> Obs:
    rest = [cell(100 + 120 * i, 220 + 120 * i, i + 1, frets=True) for i in range(4)]
    for c in rest:
        c.kind = "full"
    return view([first] + rest)


def test_a_cell_counted_as_measure_0_holding_only_the_clef_is_a_header():
    assert counted_header(measures(cell(20, 100, 0)))


def test_a_pickup_bar_with_notes_is_kept():
    assert not counted_header(measures(cell(20, 100, 0, frets=True)))


def test_a_wide_blank_cell_is_a_rest_pickup():
    assert not counted_header(measures(cell(-100, 100, 0)))


def test_a_first_cell_counted_as_measure_1_is_no_header():
    assert not counted_header(measures(cell(20, 100, 1)))


def test_a_glided_cell_counted_as_measure_0_is_no_header():
    first = cell(20, 100, 0)
    first.glided = True
    assert not counted_header(measures(first))


def test_one_notice_for_a_view_whose_measure_ends_at_a_colored_stroke():
    c = cell(20, 100, None)
    c.colored_bar = True
    notes = colored_bar_notices([view([c])])
    assert len(notes) == 1 and notes[0].kind == "colored_bar"
    assert colored_bar_notices([view([cell(20, 100, None)])]) == []


def test_a_pickup_of_open_string_digits_on_the_top_line_is_kept():
    c = cell(20, 100, 0)
    c.ink[55:65, 60:70] = c.ink[55:65, 80:90] = True     # two digits centred on the top line at row 60
    assert not counted_header(measures(c))


def test_the_upper_half_of_a_digit_on_the_top_line_counts_as_tab_ink():
    c = cell(20, 100, 0)
    c.ink[56:59, 60:63] = True     # 0.09 square gaps just above the top line
    assert not counted_header(measures(c))


class WideStaff:
    top, bottom, gap = 20.0, 70.0, 20.0


def test_a_colored_stroke_is_a_box_side_only_between_a_cursor_and_a_highlight_edge_wide():
    dark = np.zeros((100, 200), bool)
    dark[20:71, 20:25] = True       # 0.25 gap: a line cursor
    dark[20:71, 80:89] = True       # 0.45 gap: a box side
    dark[20:71, 140:154] = True     # 0.7 gap: a highlight's edge
    assert colored_bars(dark, dark.copy(), WideStaff, [(20, 24), (80, 88), (140, 153)]) == {80}
