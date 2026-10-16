import cv2
import numpy as np

from youtabber.printcell import mark_groups, owner


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
