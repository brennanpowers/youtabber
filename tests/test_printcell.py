import cv2
import numpy as np

from youtabber.observe import panel_rect
from youtabber.parse import Cell
from youtabber.printcell import clean, joined_cells, mark_groups, owner
from youtabber.scan import Segment


def test_a_mark_belongs_to_the_measure_its_left_edge_is_in():
    assert owner(150, 190, [0, 100, 200], 10.0) == 1


def test_a_number_centered_on_a_barline_belongs_to_the_measure_starting_there():
    assert owner(95, 110, [0, 100, 200], 10.0) == 1


def test_a_long_chord_name_crossing_far_from_its_start_stays_with_its_measure():
    assert owner(150, 260, [0, 100, 200], 10.0) == 1


def test_a_mark_before_the_first_barline_belongs_to_the_first_measure():
    assert owner(-5, 8, [0, 100], 10.0) == 0


def test_both_digits_of_a_number_centered_on_a_barline_print_in_the_measure_starting_there():
    # "12" drawn across the barline at x = 200, its "1" ending before the barline
    marks = np.zeros((30, 400), np.uint8)
    cv2.putText(marks, "12", (188, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 2)
    groups = mark_groups(marks, 0, [0, 200, 400], 20.0)
    assert [k for *_, k in groups] == [1]
    assert groups[0][0].sum() == marks.sum()


def test_a_yellow_box_is_lifted_but_its_ink_stays():
    band = np.full((100, 200, 3), 250, np.uint8)
    band[20:80, 50:150] = (120, 230, 250)      # BGR pale yellow highlight
    band[45:55, 90:100] = (20, 20, 20)         # a digit under it
    seg = Segment(0.0, 1.0, 10.0, "dark", paper=np.array([250.0, 250.0, 250.0]))
    out = clean(band, seg, 10.0)
    assert out[30, 60] == 255                  # the box is paper
    assert out[50, 95] < 60                    # the digit is ink


def test_a_small_red_number_stays():
    band = np.full((100, 200, 3), 250, np.uint8)
    band[10:18, 10:16] = (40, 40, 220)         # a red digit, smaller than two square gaps
    seg = Segment(0.0, 1.0, 10.0, "dark", paper=np.array([250.0, 250.0, 250.0]))
    assert clean(band, seg, 10.0)[14, 12] < 200


def test_a_dark_barline_tinted_by_yellow_paper_stays():
    band = np.full((100, 200, 3), (172, 255, 253), np.uint8)   # BGR yellow paper
    band[10:90, 100:102] = (12, 69, 68)        # a barline that compression tinted yellow
    seg = Segment(0.0, 1.0, 10.0, "dark", paper=np.array([172.0, 255.0, 253.0]))
    assert clean(band, seg, 10.0)[50, 100] < 100


def cell(x0: int, x1: int, kind: str = "full", complete: bool = True, print_x0: int | None = None) -> Cell:
    return Cell(0, 0, 0, x0, x1, kind, np.zeros((1, 1), bool), np.zeros((1, 1), bool), 0, gap=22.0,
                complete=complete, print_x0=x0 if print_x0 is None else print_x0)


def test_a_time_signature_cell_left_out_prints_with_the_measure_after_it():
    sig, first, mid, second = cell(45, 79, print_x0=12), cell(79, 516), cell(516, 550), cell(550, 900)
    parts = joined_cells([sig, first, mid, second], [first, second])
    assert [(x is first or x is second, at, col) for x, at, col in parts] == [(True, 45, 12), (True, 516, 516)]


def test_a_sliver_cut_by_the_panel_edge_stays_out_of_the_next_measure():
    sliver, first = cell(0, 20, "lead", complete=False), cell(20, 400)
    parts = joined_cells([sliver, first], [first])
    assert [(at, col) for _, at, col in parts] == [(0, 0), (20, 20)]


def test_dropped_cells_side_by_side_or_wide_print_on_their_own():
    # Two dropped cells in a row, then one wider than a clef or time signature (110 px is 5 line gaps)
    a, b, first, wide, second = cell(0, 40), cell(40, 80), cell(80, 500), cell(500, 610), cell(610, 1000)
    parts = joined_cells([a, b, first, wide, second], [first, second])
    assert [(at, col) for x, at, col in parts if x is first or x is second] == [(80, 80), (610, 610)]


def test_an_empty_column_range_prints_white():
    band = np.full((40, 100, 3), 30, np.uint8)
    seg = Segment(0.0, 1.0, 10.0, "dark", paper=np.array([250.0, 250.0, 250.0]))
    assert (clean(band, seg, 10.0, (60, 60)) == 255).all()


def strip_frame(left_strip: bool) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Paper with staffs between x 26 and 274, line gap 14, and a 5 px dark strip just left or right of them."""
    frame = np.full((200, 300, 3), 250, np.uint8)
    if left_strip:
        frame[40:160, 16:21] = 0
    else:
        frame[40:160, 279:284] = 0
    return frame, (26, 50, 274, 150)


def test_the_print_area_steps_over_a_bracket_left_of_the_staffs():
    frame, seed = strip_frame(True)
    paper = np.array([250, 250, 250], np.uint8)
    assert panel_rect(frame, paper, seed, 14.0)[0] == 21
    assert panel_rect(frame, paper, seed, 14.0, for_print=True)[0] == 0


def test_the_print_area_stops_at_a_dark_strip_right_of_the_staffs():
    frame, seed = strip_frame(False)
    paper = np.array([250, 250, 250], np.uint8)
    assert panel_rect(frame, paper, seed, 14.0, for_print=True)[2] == 279
