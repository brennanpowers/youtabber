from dataclasses import replace

import numpy as np

from youtabber.assemble import (Assembly, Obs, Slot, below_top, held_lines, place_numbered, seen_before, strings,
                               switched_part)
from youtabber.parse import Cell


def cell(number: int, lines: int, gap: float = 10.0, width: int = 100, index: int = 0,
         complete: bool = True) -> Cell:
    ink = np.zeros((90, width), bool)
    ink[30:60:5, 20:80] = True
    # The staff's top line sits 20 rows into the box and its middle half its height lower
    return Cell(0, 0, index, 0, width, "full", ink, ink, 0, gap=gap, ref_y=20, mid_y=20 + (lines - 1) * gap / 2,
                number=number, complete=complete)


def view(index: int, cells, seg: int) -> Obs:
    return Obs(index, 10.0 * index, 10.0 * index + 5, cells, [], seg, (0, 0))


def test_measures_numbered_again_on_a_staff_of_other_lines_are_another_part():
    guitar = view(0, [cell(1, 6), cell(2, 6), cell(3, 6)], 0)
    A = Assembly(slots=[Slot(k, c, 0, k + 1) for k, c in enumerate(guitar.cells)],
                 sightings={k: [(0, c)] for k, c in enumerate(guitar.cells)})
    by_number = {k + 1: k for k in range(3)}
    bass = view(1, [cell(1, 4), cell(2, 4), cell(3, 4)], 1)
    assert switched_part(bass, A, by_number, guitar, [0, 1, 2], 10.0, {0: 0, 1: 1}, {0: True, 1: True})
    # A staff read with fewer lines on a screen or two is a misread, not another part
    assert not switched_part(bass, A, by_number, guitar, [0, 1, 2], 10.0, {0: 0, 1: 1}, {0: True, 1: False})
    # The same staff drawn again after a card is the same part played again
    again = view(1, [cell(1, 6), cell(2, 6), cell(3, 6)], 1)
    assert not switched_part(again, A, by_number, guitar, [0, 1, 2], 10.0, {0: 0, 1: 1}, {0: True, 1: True})


def test_a_staff_line_count_is_held_over_four_screens_in_a_row():
    def screens(first: int, lines: list[int]) -> list[Obs]:
        return [view(first + k, [cell(1, n), cell(2, n)], 0) for k, n in enumerate(lines)]

    assert held_lines(screens(0, [6] * 4)) == {k: True for k in range(4)}
    assert held_lines(screens(0, [6] * 3)) == {k: False for k in range(3)}
    assert held_lines(screens(0, [6] * 4 + [4] * 4)) == {k: True for k in range(8)}
    # A screen with no whole measure ends the run, so three screens either side of it are each too few
    cut = view(3, [cell(1, 6, complete=False)], 0)
    obs = screens(0, [6] * 3) + [cut] + screens(4, [6] * 3)
    assert held_lines(obs) == {k: False for k in range(7)}


def test_staffs_two_lines_apart_are_another_part_and_one_line_apart_are_not():
    guitar = view(0, [cell(1, 6), cell(2, 6), cell(3, 6)], 0)
    A = Assembly(slots=[Slot(k, c, 0, k + 1) for k, c in enumerate(guitar.cells)],
                 sightings={k: [(0, c)] for k, c in enumerate(guitar.cells)})
    by_number = {k + 1: k for k in range(3)}
    held = {0: True, 1: True}
    # A staff one line short is more often a line lost at a small scale
    five = view(1, [cell(1, 5), cell(2, 5), cell(3, 5)], 1)
    assert not switched_part(five, A, by_number, guitar, [0, 1, 2], 10.0, {0: 0, 1: 1}, held)
    four = view(1, [cell(1, 4), cell(2, 4), cell(3, 4)], 1)
    assert switched_part(four, A, by_number, guitar, [0, 1, 2], 10.0, {0: 0, 1: 1}, held)
    # The earlier staff was shown on too few screens in a row to believe its line count
    assert not switched_part(four, A, by_number, guitar, [0, 1, 2], 10.0, {0: 0, 1: 1}, {0: False, 1: True})


def test_measures_seen_only_on_the_notation_staff_count_no_staff_lines():
    # A five-line notation staff seen with its tab out of view, then a seven-line tab staff at the same spacing
    notation = replace(cell(1, 5), notation_only=True)
    A = Assembly(slots=[Slot(0, notation, 0, 1)], sightings={0: [(0, notation)]})
    tab = view(1, [cell(1, 7)], 0)
    assert not switched_part(tab, A, {1: 0}, view(0, [notation], 0), [0], 10.0, {0: 0, 1: 0}, {0: True, 1: True})
    # Notation-only measures outnumber the tab one on the first two screens; the tab's six lines hold on all four
    obs = [view(k, [cell(1, 6), replace(cell(2, 5), notation_only=True), replace(cell(3, 5), notation_only=True)], 0)
           for k in range(2)] + [view(k, [cell(1, 6)], 0) for k in (2, 3)]
    assert held_lines(obs) == {k: True for k in range(4)}


def test_strings_counts_a_staffs_lines_from_its_spacing_and_middle():
    ink = np.zeros((90, 100), bool)
    six = Cell(0, 0, 0, 0, 100, "full", ink, ink, 0, gap=10, ref_y=10, mid_y=35)
    four = Cell(0, 0, 0, 0, 100, "full", ink, ink, 0, gap=10, ref_y=10, mid_y=25)
    assert (strings(six), strings(four)) == (6, 4)


def test_measures_numbered_again_at_another_width_with_no_sighting_in_their_layout_are_another_part():
    first = view(0, [cell(1, 6, width=200, index=1), cell(2, 6, width=200, index=2)], 0)
    A = Assembly(slots=[Slot(k, c, 0, k + 1) for k, c in enumerate(first.cells)],
                 sightings={k: [(0, c)] for k, c in enumerate(first.cells)})
    by_number = {1: 0, 2: 1}

    def again(width: int) -> Obs:
        return view(5, [cell(1, 6, width=width, index=1), cell(2, 6, width=width, index=2)], 1)

    assert switched_part(again(300), A, by_number, None, [], 10.0, {0: 0, 5: 1}, {})
    # A few pixels wider is the same measure drawn again
    assert not switched_part(again(205), A, by_number, None, [], 10.0, {0: 0, 5: 1}, {})


def guitar_then(cells, held=True, complete=True):
    """switched_part for a screen of `cells` in a new segment after a held 6-line staff numbered 1-3."""
    guitar = view(0, [cell(n, 6, complete=complete) for n in (1, 2, 3)], 0)
    A = Assembly(slots=[Slot(k, c, 0, k + 1) for k, c in enumerate(guitar.cells)],
                 sightings={k: [(0, c)] for k, c in enumerate(guitar.cells)})
    return switched_part(view(1, cells, 1), A, {1: 0, 2: 1, 3: 2}, guitar, [0, 1, 2], 10.0, {0: 0, 1: 1},
                         {0: held, 1: held})


def test_a_staff_of_other_lines_is_no_switch_at_another_spacing_or_against_a_cut_whole_measure():
    assert guitar_then([cell(n, 4) for n in (1, 2, 3)])
    # A 4-line staff at gap 13 against a 6-line one at gap 10
    assert not guitar_then([cell(n, 4, gap=13.0) for n in (1, 2, 3)])
    # The earlier staff was seen only in whole-width cells the panel's edges cut, which may hide lines
    assert not guitar_then([cell(n, 4) for n in (1, 2, 3)], complete=False)


def width_switch(cells) -> bool:
    """switched_part for a screen of `cells`, in a layout of its own, after measures 1-2 drawn 200 px wide."""
    first = view(0, [cell(1, 6, width=200, index=1), cell(2, 6, width=200, index=2)], 0)
    A = Assembly(slots=[Slot(k, c, 0, k + 1) for k, c in enumerate(first.cells)],
                 sightings={k: [(0, c)] for k, c in enumerate(first.cells)})
    return switched_part(view(5, cells, 1), A, {1: 0, 2: 1}, None, [], 10.0, {0: 0, 5: 1}, {})


def test_a_width_is_no_switch_on_a_systems_first_measure_or_at_another_spacing():
    assert width_switch([cell(1, 6, width=300, index=1), cell(2, 6, width=300, index=2)])
    # Each measure starts a system, where a clef and key widen it
    assert not width_switch([cell(1, 6, width=300, index=0), cell(2, 6, width=300, index=0)])
    assert not width_switch([cell(1, 6, gap=13.0, width=300, index=1), cell(2, 6, gap=13.0, width=300, index=2)])


def test_a_number_past_the_screen_before_is_judged_by_its_width_not_its_earlier_look():
    # Measures 1-4 seen once, then a screen of 1-2, then 3-4 at their earlier width but looking different
    first = view(0, [cell(n, 6, width=200, index=n) for n in (1, 2, 3, 4)], 0)
    for c in first.cells[2:]:
        c.ink = np.zeros_like(c.ink)
        c.ink[60:85, 120:190] = True
        c.faint = c.ink
    A = Assembly(slots=[Slot(k, c, 0, k + 1) for k, c in enumerate(first.cells)],
                 sightings={k: [(0, c)] for k, c in enumerate(first.cells)})
    prev = view(1, [cell(1, 6, width=200, index=1), cell(2, 6, width=200, index=2)], 0)
    o = view(2, [cell(3, 6, width=200, index=1), cell(4, 6, width=200, index=2)], 0)
    for c in first.cells + prev.cells + o.cells:
        c.tint = np.zeros_like(c.ink)
    by_number = {n: n - 1 for n in (1, 2, 3, 4)}
    assert not switched_part(o, A, by_number, prev, [0, 1], 10.0, {0: 0, 1: 0, 2: 0}, {})


def test_a_screen_read_as_another_part_is_recorded_as_a_switch():
    guitar = view(0, [cell(1, 6), cell(2, 6), cell(3, 6)], 0)
    A = Assembly(slots=[Slot(k, c, 0, k + 1) for k, c in enumerate(guitar.cells)],
                 sightings={k: [(0, c)] for k, c in enumerate(guitar.cells)})
    bass = view(1, [cell(1, 4), cell(2, 4), cell(3, 4)], 1)
    slot_of: dict = {}
    place_numbered(bass, slot_of, A, guitar, [0, 1, 2], 10.0, {0: 0, 1: 1}, {0: True, 1: True})
    assert A.switches == [(10.0, 15.0)]
    # Its measures are printed as new ones
    assert sorted(slot_of.values()) == [3, 4, 5]


def test_a_caption_laid_over_the_rows_above_the_tab_staff_is_cut_off():
    plain = cell(1, 6)
    plain.tint = np.zeros_like(plain.ink)
    cut = below_top(plain, 10.0)
    # The cut starts a quarter gap below the top line at row 20
    assert cut.ink.shape[0] == plain.ink.shape[0] - 22 and cut.ref_y == -2
    captioned = cell(1, 6)
    captioned.ink = captioned.ink.copy()
    captioned.ink[2:21, 10:90] = True
    captioned.tint = np.zeros_like(captioned.ink)
    assert not seen_before(captioned, [plain], 10.0)
    assert seen_before(below_top(captioned, 10.0), [cut], 10.0)


def test_sightings_of_the_notation_staff_alone_cast_no_vote_for_a_switch():
    def plain(n: int) -> Cell:
        c = cell(n, 6, index=n)
        c.tint = np.zeros_like(c.ink)
        return c

    def notation(n: int) -> Cell:
        c = plain(n)
        c.ink = np.zeros_like(c.ink)
        c.ink[25:70, 10:30] = True
        c.faint, c.notation_only = c.ink, True
        return c

    def switch(earlier: list[Cell], o: list[Cell], before: bool) -> bool:
        first = view(0, earlier, 0)
        A = Assembly(slots=[Slot(k, c, 0, k + 1) for k, c in enumerate(earlier)],
                     sightings={k: [(0, c)] for k, c in enumerate(earlier)})
        by_number = {n: n - 1 for n in (1, 2, 3)}
        prev, prev_slots = (first, [0, 1, 2]) if before else (None, [])
        return switched_part(view(1, o, 0), A, by_number, prev, prev_slots, 10.0, {0: 0, 1: 0}, {})

    # Notation alone on this screen after tab on the screen before, or the other way round
    assert not switch([plain(n) for n in (1, 2, 3)], [notation(n) for n in (1, 2, 3)], True)
    assert not switch([notation(n) for n in (1, 2, 3)], [plain(n) for n in (1, 2, 3)], True)
    # An earlier glimpse in the same layout seen on notation alone
    assert not switch([notation(n) for n in (1, 2, 3)], [plain(n) for n in (1, 2, 3)], False)
    # Tab drawn with other music on both screens is a switch
    assert switch([plain(n) for n in (1, 2, 3)], [replace(notation(n), notation_only=False) for n in (1, 2, 3)],
                  True)
