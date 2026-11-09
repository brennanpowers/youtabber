import numpy as np

from youtabber.assemble import Obs, assemble, glide_jumps, repeat_sign
from youtabber.parse import Cell

GAP = 10.0
EVIDENCE = {"numbers", "playhead", "navigation"}


class FakeMosaic:
    """Canvas x / 100 is the play time."""
    def time_at(self, x: float) -> float:
        return x / 100.0


def notes(bar: int) -> np.ndarray:
    """Ink of one measure on a staff from row 10 to row 40: fret marks that are the same each time bar `bar` is
    drawn and differ between bars, kept clear of the barlines at either end."""
    rng = np.random.default_rng(bar)
    ink = np.zeros((50, 200), bool)
    for y, x in zip(rng.integers(12, 38, 10), rng.integers(40, 160, 10)):
        ink[y - 3:y + 3, x - 3:x + 3] = True
    return ink


def start_repeat(ink: np.ndarray, dots: bool = True) -> np.ndarray:
    """The measure opening with a thick and a thin barline, and the two dots of a start-repeat sign after them."""
    ink = ink.copy()
    ink[10:41, 0:4] = True
    ink[10:41, 5:7] = True
    if dots:
        ink[17:20, 9:12] = True
        ink[31:34, 9:12] = True
    return ink


def end_dots(ink: np.ndarray) -> np.ndarray:
    """The measure ending with the two dots of an end-repeat sign before its closing barline."""
    ink = ink.copy()
    ink[17:20, -6:-3] = True
    ink[31:34, -6:-3] = True
    return ink


def cell(i: int, ink: np.ndarray, read: int | None) -> Cell:
    w = ink.shape[1]
    c = Cell(0, 0, i, i * w, (i + 1) * w, "full", ink, np.zeros_like(ink), 0, gap=GAP, ref_y=10, mid_y=25.0,
             tint=np.zeros_like(ink), glided=True)
    c.read = read
    return c


def glide(bars: list[int], reads: list[int | None], signs: dict | None = None) -> Obs:
    """One glide view whose cells draw `bars` with printed numbers read as `reads`; `signs` maps a cell index to a
    function that draws a sign into its ink."""
    cells = []
    for i, (b, n) in enumerate(zip(bars, reads)):
        ink = notes(b)
        if signs and i in signs:
            ink = signs[i](ink)
        cells.append(cell(i, ink, n))
    o = Obs(0, 0.0, 20.0, cells, [], 0, (0, 0))
    o.mosaic = FakeMosaic()
    return o


# Bars 1-5 then a jump back to bar 3: the cell after bar 5 is bar 3 with its number unread, then 4, 5 and on to 6, 7
BARS = [1, 2, 3, 4, 5, 3, 4, 5, 6, 7]
READS = [1, 2, 3, 4, 5, None, 4, 5, 6, 7]


def test_a_confirmed_drop_in_the_numbers_on_a_cell_like_the_earlier_one_is_a_jump_back():
    jumps = glide_jumps(glide(BARS, READS).cells, GAP)
    assert jumps == [({5: 2, 6: 3, 7: 4}, False)]


def test_a_drop_in_the_numbers_on_a_cell_unlike_the_earlier_one_is_no_jump():
    bars = [1, 2, 3, 4, 5, 13, 14, 15, 16, 17]
    assert glide_jumps(glide(bars, READS).cells, GAP) == []


def test_a_later_pass_repeats_the_cells_the_first_pass_repeated():
    bars = [1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 6]
    reads = [1, 2, 3, 4, 5, None, 4, 5, None, 4, 5, 6]
    jumps = glide_jumps(glide(bars, reads).cells, GAP)
    assert [m for m, _ in jumps] == [{5: 2, 6: 3, 7: 4}, {8: 2, 9: 3, 10: 4}]


def test_a_start_repeat_sign_where_the_pass_lands_is_found():
    cells = glide(BARS, READS, {2: start_repeat}).cells
    assert repeat_sign(cells[1], cells[2], GAP)
    assert glide_jumps(cells, GAP)[0][1] is True


def test_an_end_repeat_sign_where_the_pass_jumps_from_is_found():
    cells = glide(BARS, READS, {4: end_dots, 5: start_repeat}).cells
    # The barline strokes open the cell after the sign's dots
    cells[5].ink[17:20, 9:12] = False
    cells[5].ink[31:34, 9:12] = False
    assert repeat_sign(cells[4], cells[5], GAP)


def test_a_thick_barline_without_dots_is_no_repeat_sign():
    cells = glide(BARS, READS, {2: lambda ink: start_repeat(ink, dots=False)}).cells
    assert not repeat_sign(cells[1], cells[2], GAP)
    assert not repeat_sign(cells[0], cells[1], GAP)


def test_a_jump_back_over_a_repeat_sign_folds_in_the_main_reading():
    A = assemble([glide(BARS, READS, {2: start_repeat})], GAP, False, EVIDENCE, lambda o: [])
    assert A.view_slots[0] == [0, 1, 2, 3, 4, 2, 3, 4, 5, 6]
    assert not A.unresolved_folds
    assert [(d.basis, d.chosen, d.resolved) for d in A.decisions if d.kind == "fold"] == [("repeat_sign", "fold", True)]


def test_a_jump_back_without_a_repeat_sign_prints_again_in_the_main_reading_and_folds_in_the_folded_one():
    main = assemble([glide(BARS, READS)], GAP, False, EVIDENCE, lambda o: [])
    assert main.view_slots[0] == list(range(10))
    assert main.unresolved_folds == [(11.0, 15.0, 2, 4)]
    assert sorted(main.unsure_slots) == [5, 6, 7]
    folded = assemble([glide(BARS, READS)], GAP, True, EVIDENCE, lambda o: [])
    assert folded.view_slots[0] == [0, 1, 2, 3, 4, 2, 3, 4, 5, 6]
