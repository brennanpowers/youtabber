from youtabber.printcell import owner


def test_a_mark_belongs_to_the_measure_its_left_edge_is_in():
    assert owner(150, 190, [0, 100, 200], 10.0) == 1


def test_a_number_centered_on_a_barline_belongs_to_the_measure_starting_there():
    assert owner(95, 110, [0, 100, 200], 10.0) == 1


def test_a_long_chord_name_crossing_far_from_its_start_stays_with_its_measure():
    assert owner(150, 260, [0, 100, 200], 10.0) == 1


def test_a_mark_before_the_first_barline_belongs_to_the_first_measure():
    assert owner(-5, 8, [0, 100], 10.0) == 0
