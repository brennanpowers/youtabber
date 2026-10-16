import cv2
import numpy as np

from youtabber.printcell import clean, mark_groups, owner
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
