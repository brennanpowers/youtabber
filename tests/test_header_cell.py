from youtabber.assemble import Assembly, Obs
from youtabber.parse import bare_header, clef_only, header_cell, header_lead, mark_header, tab_marks, tab_sign
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


def screen(index: int, t0: float, t1: float, header: bool, width: int = 300) -> Obs:
    """A screen whose first cell, `width` px wide and cut by the frame's left edge, may be left out as a header,
    before a 300 px measure."""
    c, measure = cell(width, marks=True, kind="lead", complete=False), cell(300, marks=True)
    c.header = header
    return Obs(index, t0, t1, [measure], [], 0, (0, 0), all_cells=[c, measure])


def test_neighbouring_screens_with_a_header_make_one_notice():
    obs = [screen(0, 0.0, 5.0, True), screen(1, 5.5, 10.0, True), screen(2, 20.0, 25.0, False)]
    (n,) = header_notices(Assembly(), obs, 300.0)
    assert (n.kind, n.t0, n.t1) == ("header", 0.0, 10.0)


def test_a_header_too_narrow_to_print_as_a_measure_makes_no_notice():
    assert header_notices(Assembly(), [screen(0, 0.0, 5.0, True, width=60)], 300.0) == []


def test_a_header_cut_from_the_measure_the_screen_before_ended_on_makes_no_notice():
    # Screen 0 ends on slot 0; screen 1's first measure is slot 1, so its left-out first cell is the rest of slot 0
    before, now = screen(0, 0.0, 5.0, False), screen(1, 5.5, 10.0, True)
    A = Assembly(order=[0, 1], view_slots={0: [0], 1: [1]},
                 sightings={0: [(0, before.cells[0])], 1: [(1, now.cells[0])]})
    assert header_notices(A, [before, now], 300.0) == []


def test_a_header_after_a_screen_that_ended_elsewhere_makes_a_notice_whatever_an_earlier_screen_ended_on():
    # Screen 0 ended on slot 0, but screen 1, the one just before, ended on slot 5; screen 2's first measure is slot 1
    first, before, now = screen(0, 0.0, 5.0, False), screen(1, 5.5, 10.0, False), screen(2, 10.5, 15.0, True)
    A = Assembly(order=[0, 1, 5], view_slots={0: [0], 1: [5], 2: [1]},
                 sightings={0: [(0, first.cells[0])], 5: [(1, before.cells[0])], 1: [(2, now.cells[0])]})
    (n,) = header_notices(A, [first, before, now], 300.0)
    assert (n.kind, n.t0) == ("header", 10.5)


def test_a_header_before_a_page_that_starts_the_song_makes_a_notice():
    now = screen(0, 0.0, 5.0, True)
    A = Assembly(order=[0], view_slots={0: [0]}, sightings={0: [(0, now.cells[0])]})
    (n,) = header_notices(A, [now], 300.0)
    assert n.kind == "header"


def kept_screen(t0: float, t1: float, printed: bool) -> Obs:
    c = cell(300, marks=True)
    c.maybe_header = True
    return Obs(0, t0, t1, [c] if printed else [], [], 0, (0, 0), all_cells=[c])


def test_neighbouring_screens_printing_a_possible_header_make_one_notice():
    (n,) = header_kept_notices([kept_screen(0.0, 5.0, True), kept_screen(5.5, 10.0, True)])
    assert (n.kind, n.t0, n.t1, n.doubt) == ("header_kept", 0.0, 10.0, True)


def test_a_possible_header_left_unprinted_makes_no_notice():
    assert header_kept_notices([kept_screen(0.0, 5.0, False)]) == []


LEAD_EDGES = [(0, 80, "lead"), (80, 280, "full"), (280, 480, "full")]


def test_a_narrow_lead_cell_at_the_frame_edge_may_be_a_kept_clef():
    assert header_lead(LEAD_EDGES, False, 10.0)


def test_a_lead_cell_whose_staff_starts_inside_the_panel_or_past_the_frame_edge_is_no_kept_clef():
    assert not header_lead(LEAD_EDGES, True, 10.0)
    assert not header_lead([(120, 200, "lead"), (200, 400, "full")], False, 10.0)


def test_a_lead_cell_twelve_gaps_wide_is_no_kept_clef():
    assert not header_lead([(0, 120, "lead"), (120, 320, "full")], False, 10.0)


def test_a_lead_cell_alone_on_its_staff_is_no_kept_clef():
    assert not header_lead([(0, 80, "lead")], False, 10.0)


def lead(marks: bool = False, clef: bool = False, tied: bool = False, sign: bool = False):
    """A 70 px first cell cut by the frame's left edge, so with no left barline; `sign` draws the letters of a tab
    sign one above another in the spaces of its tab staff, in its first line gap and a half."""
    c = cell(70, marks=marks, kind="lead", complete=False, clef=clef, tied=tied)
    c.ink[60:111, 15:17] = False
    if sign:
        for top in (62, 72, 82):
            c.ink[top:top + 7, 18:26] = True
    return c


def test_a_cut_cell_with_a_clef_and_a_tab_sign_on_a_bare_tab_staff_holds_only_the_clef():
    assert clef_only(lead(clef=True, sign=True))


def test_a_cut_cell_with_no_notation_ink_or_with_a_fret_digit_is_a_cut_measure():
    assert not clef_only(lead(sign=True))
    assert not clef_only(lead(marks=True, clef=True, sign=True))


def test_a_cut_measure_of_notes_tied_over_a_bare_tab_staff_is_no_kept_clef():
    # Tied notes are not repeated on the tab, so its staff is bare; a kept clef's tab staff shows the tab sign
    assert not clef_only(lead(tied=True))
    assert clef_only(lead(tied=True, sign=True))


def test_a_fret_digit_at_the_cut_edge_is_no_tab_sign():
    c = lead(clef=True)
    c.ink[85:96, 18:25] = True     # a digit on the fourth line, cut in two by it
    assert not tab_sign(c) and not clef_only(c)


def test_bits_of_staff_line_left_on_the_tab_staff_are_no_marks():
    c = lead(clef=True, sign=True)
    c.ink[[70, 91], 40:80] = True
    assert tab_marks(c, 15) == 0.0 and clef_only(c)


def test_a_narrow_header_closed_by_a_barline_keeps_its_notice():
    c, measure = cell(40, marks=True), cell(300, marks=True)
    c.header = True
    o = Obs(0, 0.0, 5.0, [measure], [], 0, (0, 0), all_cells=[c, measure])
    (n,) = header_notices(Assembly(), [o], 300.0)
    assert n.kind == "header"
