from youtabber.assemble import Obs
from youtabber.parse import bare_header, header_cell, mark_header
from youtabber.pipeline import header_kept_notices, header_notices, kept_cells

from test_empty_bar import cell

BARS = [(96, 99), (160, 166), (280, 281), (400, 401)]
EDGES = [(100, 160, "full"), (160, 280, "full"), (280, 400, "full")]


def test_a_narrow_first_cell_closed_by_a_wide_barline_is_a_header():
    assert header_cell(EDGES, BARS, [], 100, 10.0)


def test_a_first_cell_closed_by_a_thin_double_barline_is_no_header():
    assert not header_cell(EDGES, [(96, 99), (160, 163), (280, 281), (400, 401)], [], 100, 10.0)


def test_a_first_cell_closed_by_an_end_repeat_is_no_header():
    assert not header_cell(EDGES, BARS, [160], 100, 10.0)


def test_a_first_cell_fifteen_gaps_wide_is_no_header():
    edges = [(100, 250, "full"), (250, 400, "full"), (400, 550, "full")]
    bars = [(96, 99), (250, 256), (400, 401), (550, 551)]
    assert not header_cell(edges, bars, [], 100, 10.0)


def test_a_system_with_no_measure_beside_its_first_cell_has_no_header():
    assert not header_cell([(100, 160, "full"), (160, 280, "tail")], BARS[:2], [], 100, 10.0)


def tab_clef(digit: bool):
    """A 60 px system's first cell with a tab clef on its tab staff in its first 4 line gaps, and optionally a fret
    digit past them."""
    c = cell(60, marks=False)
    c.index = 0
    c.ink[62:108, 20:50] = True
    if digit:
        c.ink[80:92, 65:77] = True
    return c


def test_a_first_cell_holding_only_a_clef_is_a_bare_header():
    assert bare_header(tab_clef(digit=False))


def test_a_first_cell_with_a_fret_digit_past_the_clef_is_a_pickup():
    assert not bare_header(tab_clef(digit=True))


def test_a_header_shaped_cell_with_marks_past_the_clef_is_a_possible_header():
    c = tab_clef(digit=True)
    mark_header(c)
    assert c.maybe_header and not c.header


def test_a_header_shaped_cell_holding_only_a_clef_is_a_header():
    c = tab_clef(digit=False)
    mark_header(c)
    assert c.header and not c.maybe_header


def test_a_header_cell_is_no_measure_however_wide():
    header, measure = cell(300, marks=True), cell(300, marks=True)
    header.header = True
    assert kept_cells([header, measure], typical=300) == [measure]


def screen(t0: float, t1: float, header: bool) -> Obs:
    c = cell(300, marks=True)
    c.header = header
    return Obs(0, t0, t1, [], [], 0, (0, 0), all_cells=[c])


def test_neighbouring_screens_with_a_header_make_one_notice():
    (n,) = header_notices([screen(0.0, 5.0, True), screen(5.5, 10.0, True), screen(20.0, 25.0, False)])
    assert (n.kind, n.t0, n.t1) == ("header", 0.0, 10.0)


def kept_screen(t0: float, t1: float, printed: bool) -> Obs:
    c = cell(300, marks=True)
    c.maybe_header = True
    return Obs(0, t0, t1, [c] if printed else [], [], 0, (0, 0), all_cells=[c])


def test_neighbouring_screens_printing_a_possible_header_make_one_notice():
    (n,) = header_kept_notices([kept_screen(0.0, 5.0, True), kept_screen(5.5, 10.0, True)])
    assert (n.kind, n.t0, n.t1, n.doubt) == ("header_kept", 0.0, 10.0, True)


def test_a_possible_header_left_unprinted_makes_no_notice():
    assert header_kept_notices([kept_screen(0.0, 5.0, False)]) == []
