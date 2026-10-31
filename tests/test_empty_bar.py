import numpy as np

from youtabber.parse import Cell, staff_ink
from youtabber.pipeline import empty_bar, kept_cells, rest_run, short_bar

GAP = 10.0


def cell(width: int, marks: bool, kind: str = "full", complete: bool = True, above: bool = False,
         clef: bool = False, tied: bool = False) -> Cell:
    """A cell between barlines at x 100 and 100 + width whose ink box starts 15 px left of its left barline. Its
    box holds a notation staff (rows 15-49) above the tab staff (rows 60-110). A clef cell is its system's first."""
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
    if tied:
        ink[25:35, 30:44] = ink[38:48, 30:44] = True   # whole notes tied over from the bar before
    return Cell(0, 0, 0 if clef else 1, 100, 100 + width, kind, ink, np.zeros_like(ink), 0, gap=GAP, ref_y=60,
                mid_y=85.0, complete=complete, top_y=15)


def test_a_bar_with_only_staff_lines_and_barlines_is_empty():
    assert empty_bar(cell(60, marks=False))


def test_a_rest_bar_with_a_number_and_a_notation_rest_is_empty():
    assert empty_bar(cell(60, marks=False, above=True))


def test_a_clef_and_key_signature_cell_is_not_an_empty_bar():
    assert not empty_bar(cell(60, marks=False, clef=True))


def test_a_bar_of_notes_tied_over_with_nothing_on_its_tab_staff_is_empty():
    assert empty_bar(cell(60, marks=False, tied=True))


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


def test_a_short_bar_with_notes_is_kept_at_a_fifth_of_a_measure_and_five_gaps():
    # A still-screen bar at least SHORT_SHARE of a typical measure and SHORT_MIN_GAPS line gaps wide holds a whole
    # or tied note; a narrower piece lies between a barline and a stem read as one
    short_notes, sliver = cell(60, marks=True), cell(45, marks=True)
    short_empty, wide = cell(60, marks=False, above=True), cell(300, marks=True)
    assert kept_cells([short_notes, sliver, short_empty, wide], typical=300) == [short_notes, short_empty, wide]


def test_a_short_bar_is_a_complete_still_measure_between_two_barlines():
    assert short_bar(cell(60, marks=True), typical=300)
    glided = cell(60, marks=True)
    glided.glided = True
    assert not short_bar(glided, typical=300)
    assert not short_bar(cell(60, marks=True, complete=False), typical=300)
    assert not short_bar(cell(45, marks=True), typical=300)


def in_system(cells: list[Cell]) -> list[Cell]:
    """The cells side by side on one system, numbered left to right."""
    for i, c in enumerate(cells):
        c.index = i
    return cells


def test_three_narrow_empty_bars_in_a_row_are_a_rest_run():
    run = in_system([cell(w, marks=False) for w in (30, 32, 31)])
    assert all(rest_run(run, i) for i in range(3))
    broken = in_system([cell(30, marks=False), cell(32, marks=False), cell(45, marks=False), cell(31, marks=False)])
    assert not any(rest_run(broken, i) for i in (0, 1, 3))
    assert not rest_run(in_system([cell(30, marks=False)]), 0)


def test_a_rest_run_of_narrow_bars_is_kept_beside_a_full_measure():
    cells = in_system([cell(300, marks=True)] + [cell(30, marks=False) for _ in range(3)])
    assert kept_cells(cells, typical=300) == cells


def test_a_fret_digit_on_the_tab_staff_s_top_line_counts_whole():
    c = cell(60, marks=False)
    c.ink[60:111, 15:17] = False
    c.ink[55:65, 45:55] = True
    assert staff_ink(c, 7) == (1.0, 0.0)


def test_a_clef_cell_of_a_system_cut_by_the_panel_top_is_not_an_empty_bar():
    cut = cell(60, marks=False, clef=True)
    cut.top_y = -5
    assert not empty_bar(cut)


def tail(inked: int) -> Cell:
    """A 300 px cell past the system's last barline holding `inked` pixels of ink."""
    ink = np.zeros((120, 316), bool)
    ink.flat[:inked] = True
    return Cell(0, 0, 5, 100, 400, "tail", ink, np.zeros_like(ink), 0, gap=GAP, ref_y=60, mid_y=85.0, top_y=15)


def test_a_tail_past_the_last_barline_with_next_to_no_ink_is_no_measure():
    # Under 0.02 of its width times the line gap, 60 pixels, it is blank staff
    blank, inked = tail(59), tail(60)
    assert kept_cells([blank, inked], typical=300) == [inked]


def test_a_barline_left_at_the_start_of_a_blank_tail_is_no_tall_mark():
    # The cell's left edge is ink column 15 and the barline's skipped columns reach column 22, so a 30 px column at
    # 18 is read as that barline; at column 200 it would be a mark
    left = tail(0)
    left.ink[40:70, 18] = True
    assert kept_cells([left], typical=300) == []
    inside = tail(0)
    inside.ink[40:70, 200] = True
    assert kept_cells([inside], typical=300) == [inside]


def test_a_lone_thin_digit_in_a_wide_tail_is_a_measure():
    # 1000 px wide, so under 200 pixels of ink passes for blank staff; the staff lines left 150 flat pixels
    ink = np.zeros((120, 1001), bool)
    ink[70, 100:250] = True
    blank = Cell(0, 0, 5, 100, 1100, "tail", ink, np.zeros_like(ink), 0, gap=GAP, ref_y=60, mid_y=85.0, top_y=15)
    # A "1" drawn 6 px tall and 2 px wide stands taller than a quarter gap
    digit = ink.copy()
    digit[64:70, 500:502] = True
    one = Cell(0, 0, 5, 100, 1100, "tail", digit, np.zeros_like(digit), 0, gap=GAP, ref_y=60, mid_y=85.0, top_y=15)
    assert kept_cells([blank, one], typical=300) == [one]
