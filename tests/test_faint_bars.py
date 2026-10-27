import cv2
import numpy as np
import pytest

from youtabber import parse
from youtabber.assemble import Obs
from youtabber.parse import (BAR_COVER, Cell, ParsedSystem, faint_bars, faint_candidates, faint_rule, joined_staffs,
                             parse_view)
from youtabber.pipeline import faint_bar_notices
from youtabber.scan import System
from youtabber.staff import Staff


def page(bar_gray: int, stems: bool = False) -> np.ndarray:
    """A light-gray tab staff whose barlines at x 100, 250, 400 and 700 are drawn lighter than the faint-ink level;
    with `stems`, the line at 250 is a stem that stops a line short of the bottom."""
    img = np.full((200, 800, 3), 245, np.uint8)
    for i in range(6):
        cv2.line(img, (20, 60 + i * 14), (780, 60 + i * 14), (150, 150, 150), 1)
    for x in (100, 250, 400, 700):
        bottom = 116 if stems and x == 250 else 130
        cv2.line(img, (x, 60), (x, bottom), (bar_gray,) * 3, 1)
    return img


def full_cells(img: np.ndarray, hidden_bars: bool = False) -> list[int]:
    systems = parse_view(0, img, (0, 0, 800, 200), "dark", 120.0, 200.0, 14.0, hidden_bars=hidden_bars)
    return [c.x0 for ps in systems for c in ps.cells if c.kind == "full"]


def test_barlines_lighter_than_the_faint_level_still_make_measures():
    xs = full_cells(page(205))
    assert [round(x / 10) for x in xs] == [10, 25, 40]


def test_a_stem_stopping_short_of_the_bottom_line_is_no_barline():
    xs = full_cells(page(205, stems=True))
    assert [round(x / 10) for x in xs] == [10, 40]


# Gray pictures for faint_candidates: paper 245, staff lines 150 at rows 60, 74, ... 130, columns 20-780
FTHR = 200.0
NONE = np.zeros((200, 800), bool)


def staff(gap: int = 14) -> Staff:
    return Staff([60 + i * gap for i in range(6)], 20, 780, float(gap), "dark")


def gray(st: Staff, cols=(400,), shade: int = 205, y0: int | None = None, y1: int | None = None) -> np.ndarray:
    """Vertical lines of `shade` at `cols` over rows y0 to y1 (default the staff's top to bottom line)."""
    g = np.full((200, 800), 245, np.uint8)
    for x in cols:
        g[int(st.top) if y0 is None else y0:int(st.bottom) + 1 if y1 is None else y1, x] = shade
    for y in st.lines:
        g[int(y), 20:780] = 150
    return g


def candidates(g: np.ndarray, st: Staff | None = None, bars=(), notation=None, colored=NONE, dropped=(), fthr=FTHR):
    return faint_candidates(g, st or staff(), list(bars), notation, fthr, colored, list(dropped))


def columns(g: np.ndarray, **kw) -> list[int]:
    return [x for x, _ in candidates(g, **kw)]


def test_a_light_line_across_the_staff_is_a_candidate_with_its_measurements():
    ((x, f),) = candidates(gray(staff()))
    # The darker of columns 399 and 400 is the line, so the first column of the pair is the candidate
    assert x == 399
    assert f["run"] == 1.0 and f["ends"] == 1.0
    assert f["dark"] == pytest.approx(40 / 95)
    # Only the six staff-line rows of the 71 are under the faint-ink level
    assert f["fcover"] == pytest.approx(6 / 71) and f["side"] == pytest.approx(6 / 71)
    assert f["below"] == 0.0 and f["above"] == 0.0 and f["colored"] == 0.0 and f["notation"] == -1.0


def test_a_row_counts_when_the_line_is_a_quarter_of_the_staff_lines_contrast_darker_than_the_paper():
    # Contrast is 245 - 150 = 95: 24 below the paper is 0.253 of it, 23 is 0.242
    assert columns(gray(staff(), shade=221)) == [399]
    assert columns(gray(staff(), shade=222)) == []


def test_a_column_is_a_candidate_when_half_the_rows_between_the_lines_count():
    # Rows 60-95 hold 28 of the 55 rows clear of the staff lines, rows 60-94 hold 27
    assert columns(gray(staff(), y1=96)) == [399]
    assert columns(gray(staff(), y1=95)) == []


def test_the_paper_beside_a_candidate_is_looked_for_a_share_of_a_gap_away_and_two_pixels_at_least():
    # A stroke 5 pixels wide: paper lies 3 pixels out from its middle columns, so a gap of 20 (0.15 of it is 3) sees
    # it and a gap of 14 (2 pixels) does not
    assert columns(gray(staff(20), cols=range(400, 405)), st=staff(20)) == [401]
    assert columns(gray(staff(), cols=range(400, 405))) == []
    # A stroke 3 pixels wide on a gap of 8: 0.15 of the gap rounds to 1, raised to 2, which sees paper beside it
    assert columns(gray(staff(8), cols=range(400, 403)), st=staff(8)) == [400]


def test_one_candidate_per_half_gap():
    # Half a gap of 14 is 7 pixels: a line at 406 is within that of 399, and one at 407 is a candidate in its own column
    assert columns(gray(staff(), cols=(400, 406))) == [399]
    assert columns(gray(staff(), cols=(400, 407))) == [399, 407]


@pytest.mark.parametrize("key", ["bars", "dropped"])
def test_no_candidate_within_0_7_gaps_of_a_found_or_dropped_barline(key):
    # 0.7 of a gap of 14 is 9.8 pixels; the line darkens columns 399 and 400
    assert columns(gray(staff()), **{key: [(391, 391)]}) == []
    assert columns(gray(staff()), **{key: [(390, 390)]}) == [400]
    assert columns(gray(staff()), **{key: [(408, 408)]}) == []
    assert columns(gray(staff()), **{key: [(409, 409)]}) == [399]


def test_no_candidate_that_would_leave_a_cell_narrower_than_a_measure_at_the_staff_ends():
    # The staff runs from column 20 to 780; a measure is at least 1.5 gaps (21 pixels) wide
    assert columns(gray(staff(), cols=(41,))) == [41]
    assert columns(gray(staff(), cols=(40,))) == []
    assert columns(gray(staff(), cols=(760,))) == [759]
    assert columns(gray(staff(), cols=(761,))) == []


def test_run_is_the_longest_unbroken_stretch_of_counted_rows():
    # Rows 60-117 hold 44 of the 55 rows clear of the staff lines
    ((_, f),) = candidates(gray(staff(), y1=118))
    assert f["run"] == pytest.approx(0.8)
    # A break across rows 80-89 leaves 15 counted rows above it and 33 below
    g = gray(staff())
    g[80:90, 400] = 245
    ((_, f),) = candidates(g)
    assert f["run"] == pytest.approx(33 / 55)


def test_ends_is_the_lesser_share_of_the_top_and_bottom_line_spaces_the_line_fills():
    # The bottom space is rows 118-128; a line ending at row 123 fills 6 of its 11 rows
    ((_, f),) = candidates(gray(staff(), y1=124))
    assert f["ends"] == pytest.approx(6 / 11)
    ((_, f),) = candidates(gray(staff(), y0=67))
    assert f["ends"] == pytest.approx(6 / 11)


@pytest.mark.parametrize("beside", [397, 402])
def test_side_is_the_faint_ink_on_the_columns_where_the_paper_beside_is_looked_for(beside):
    # With the faint-ink level at 230, a lighter stroke 2 pixels left or right of the pair 399-400 is faint ink
    g = gray(staff(), shade=170)
    g[60:131, beside] = 225
    ((x, f),) = candidates(g, fthr=230.0)
    assert x == 399 and f["side"] == 1.0
    ((_, f),) = candidates(gray(staff(), shade=170), fthr=230.0)
    assert f["side"] == pytest.approx(6 / 71)


def test_below_and_above_are_the_shares_of_a_line_running_on_past_the_staff():
    ((_, f),) = candidates(gray(staff(), y1=160))
    assert f["below"] == 1.0 and f["above"] == 0.0
    ((_, f),) = candidates(gray(staff(), y0=20))
    assert f["above"] == 1.0 and f["below"] == 0.0


def test_fcover_dark_colored_and_notation_measurements():
    ((_, f),) = candidates(gray(staff(), shade=190))
    assert f["fcover"] == 1.0
    assert f["dark"] == pytest.approx(55 / 95)
    tint = NONE.copy()
    tint[60:131, 400] = True
    ((_, f),) = candidates(gray(staff()), colored=tint)
    assert f["colored"] == 1.0
    # A notation staff over rows 20-50: a line through all of it, and one that starts at row 35
    ((_, f),) = candidates(gray(staff(), y0=20), notation=(20.0, 50.0))
    assert f["notation"] == 1.0
    ((_, f),) = candidates(gray(staff(), y0=35), notation=(20.0, 50.0))
    assert f["notation"] == pytest.approx(0.5)


def sure(**kw) -> dict:
    """Measurements of a faint line that is a barline, with `kw` changed."""
    f = {"run": 0.8, "ends": 0.9, "fcover": 0.1, "dark": 1.0, "side": 0.0, "below": 0.0, "above": 0.0,
         "colored": 0.0, "notation": -1.0}
    return f | kw


def test_a_faint_line_running_unbroken_and_reaching_both_outer_spaces_is_a_barline():
    assert faint_rule(sure()) == "bar"
    assert faint_rule(sure(run=0.6)) == "bar"
    assert faint_rule(sure(run=0.55)) == "unsure"
    assert faint_rule(sure(ends=0.7)) == "bar"
    assert faint_rule(sure(ends=0.65)) == "unsure"


def test_a_line_short_of_the_barline_tests_but_close_is_unsure_and_one_further_off_is_not_a_barline():
    assert faint_rule(sure(run=0.5)) == "unsure"
    assert faint_rule(sure(run=0.45)) == ""
    assert faint_rule(sure(ends=0.6)) == "unsure"
    assert faint_rule(sure(ends=0.55)) == ""


@pytest.mark.parametrize("key, kept, turned_down", [
    ("side", 0.25, 0.3),
    ("below", 0.45, 0.5),
    ("above", 0.45, 0.5),
    ("dark", 1.55, 1.6),
    ("colored", 0.45, 0.5),
    ("fcover", 0.919, 0.92),
    ("notation", 0.9, 0.85),
])
def test_each_test_turns_down_sure_and_unsure_lines_alike(key, kept, turned_down):
    assert BAR_COVER == 0.92
    assert faint_rule(sure(**{key: kept})) == "bar"
    assert faint_rule(sure(**{key: turned_down})) == ""
    assert faint_rule(sure(run=0.55, **{key: kept})) == "unsure"
    assert faint_rule(sure(run=0.55, **{key: turned_down})) == ""


def test_faint_bars_returns_the_barlines_and_the_unsure_columns():
    st = staff()
    # A full line at 400 is a barline; one at 550 ending at row 124 fills 7 of the bottom space's 11 rows: unsure
    g = gray(st)
    g[60:125, 550] = 205
    # One at 650 ending at row 123 fills 6 of them: neither
    g[60:124, 650] = 205
    assert faint_bars(g, st, [], None, FTHR, NONE, []) == ([(399, 400)], {549})
    assert faint_bars(gray(st, cols=()), st, [], None, FTHR, NONE, []) == ([], set())


def unsure_page() -> np.ndarray:
    """Dark barlines at 100, 250, 400 and 700, and a faint line at 550 that stops short of filling the bottom space."""
    img = page(60)
    cv2.line(img, (550, 60), (550, 124), (205,) * 3, 1)
    return img


def test_an_unsure_line_is_left_out_and_marks_the_cell_it_lies_in():
    (ps,) = parse_view(0, unsure_page(), (0, 0, 800, 200), "dark", 120.0, 200.0, 14.0)
    assert ps.unsure_bars == {549}
    assert [(round(c.x0 / 10), c.faint_bar) for c in ps.cells if c.kind == "full"] == [(10, False), (25, False),
                                                                                        (40, True)]


@pytest.mark.parametrize("x, flags", [(99, [True, False, False]), (98, [False, False, False]),
                                      (251, [True, True, False]), (252, [False, True, False])])
def test_a_cell_is_marked_for_an_unsure_line_up_to_a_pixel_past_its_ends(monkeypatch, x, flags):
    monkeypatch.setattr(parse, "faint_bars", lambda *args: ([], {x}))
    (ps,) = parse_view(0, page(60), (0, 0, 800, 200), "dark", 120.0, 200.0, 14.0)
    # The full cells run 100-250, 250-400 and 400-700
    assert [c.faint_bar for c in ps.cells if c.kind == "full"] == flags


def spy(monkeypatch) -> list:
    calls = []
    real = parse.faint_bars

    def recorded(*args):
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(parse, "faint_bars", recorded)
    return calls


def test_glide_mosaics_are_not_searched_for_faint_barlines(monkeypatch):
    calls = spy(monkeypatch)
    assert full_cells(page(205), hidden_bars=True) == []
    assert calls == []


def test_a_line_dropped_as_lighter_than_the_page_barlines_is_passed_on_so_it_is_never_a_candidate(monkeypatch):
    calls = spy(monkeypatch)
    img = page(60)
    # Under the faint-ink level, so the cover rule finds it, but lighter than the staff lines and the other barlines
    cv2.line(img, (550, 60), (550, 130), (190,) * 3, 1)
    assert [round(x / 10) for x in full_cells(img)] == [10, 25, 40]
    ((_, _, bars, _, _, _, dropped),) = calls
    assert [round(p / 10) for p, _ in bars] == [10, 25, 40, 70]
    assert [round(p / 10) for p, _ in dropped] == [55]


def test_joined_staffs_keep_both_staffs_unsure_columns():
    upper = System(Staff([20.0, 30.0, 40.0], 0, 400, 10.0, "dark"), None)
    lower = System(Staff([80.0, 90.0, 100.0], 0, 400, 10.0, "dark"), None)
    bars = [(50, 51), (200, 201), (350, 351)]
    struct = np.zeros((120, 400), bool)
    for p, _ in bars:
        struct[:, p] = True
    out, _ = joined_staffs([ParsedSystem(upper, bars, 0, 60, unsure_bars={120}),
                            ParsedSystem(lower, bars, 60, 120, unsure_bars={280})], [False, False], struct)
    (ps,) = out
    assert ps.unsure_bars == {120, 280}


def screen(t0: float, t1: float, faint: bool) -> Obs:
    c = Cell(0, 0, 0, 100, 200, "full", np.zeros((10, 10), bool), np.zeros((10, 10), bool), 0)
    c.faint_bar = faint
    return Obs(0, t0, t1, [c], [], 0, (0, 0))


def test_one_faint_bar_notice_per_stretch_of_screens_a_turn_apart_or_less():
    notes = faint_bar_notices([screen(0.0, 5.0, True), screen(6.0, 9.0, True), screen(10.01, 12.0, True),
                               screen(12.5, 20.0, False)])
    assert [(n.t0, n.t1, n.kind, n.doubt) for n in notes] == [(0.0, 9.0, "faint_bar", True),
                                                             (10.01, 12.0, "faint_bar", True)]
    assert notes[1].text == ("A faint line on the tab at 0:10 was not read as a barline, but may be one. If it is, "
                             "two measures there are printed as one.")
    assert faint_bar_notices([screen(0.0, 5.0, False)]) == []
