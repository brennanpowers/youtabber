from youtabber.counter import at, dropped, parse, steady, widget_boxes


def test_parse_reads_the_widget():
    assert parse("BAR 12 / 27") == (12, 27)
    assert parse("Bar 91/421") == (91, 421)
    assert parse("BAR 1l / 27") == (11, 27)       # an l read for a 1
    assert parse("BARA4/27", 27) == (4, 27)       # a stray letter before the measure
    assert parse("BAR5 127", 27) == (5, 27)       # the slash read as a 1


def test_parse_skips_other_text():
    for text in ["Bar 12 of 40", "12/40", "Measure 3/4", "Barely 1/2 of me", "Crowbar 7/8 tour", "BAR 30 / 27",
                 "BAR 12/4210"]:
        assert parse(text) is None, text
    assert parse("BAR 12 / 28", 27) is None


def test_steady_keeps_a_replay_and_drops_a_lone_misread():
    replay = [(0.0, 1), (0.25, 1), (0.5, 1), (2.0, 2), (2.25, 2), (4.0, 3), (4.25, 3), (6.0, 1), (6.25, 1),
              (6.5, 1), (8.0, 2), (8.25, 2)]
    assert steady(replay) == [(0.0, 0.5, 1), (2.0, 2.25, 2), (4.0, 4.25, 3), (6.0, 6.5, 1), (8.0, 8.25, 2)]
    misread = [(0.0, 5), (0.25, 5), (2.0, 6), (2.25, 6), (4.0, 1), (6.0, 8), (6.25, 8)]
    assert steady(misread) == [(0.0, 0.25, 5), (2.0, 2.25, 6)]


def test_at_reaches_half_a_reading_step():
    runs = [(0.0, 0.5, 1), (2.0, 2.25, 2)]
    assert at(runs, 0.6) == 1
    assert at(runs, 1.2) is None


def test_parse_reads_an_o_for_a_zero_and_a_count_cut_short():
    assert parse("BAR IO/41") == (10, 41)
    assert parse("BAR22/4", 41) == (22, 41)
    assert parse("BAR22/4") is None


def test_a_dropped_digit_stands_for_the_next_measure_when_the_count_carries_on():
    # 11 read as "1", then 12 read whole
    reads = [(0.0, 10), (0.25, 10), (2.0, 1), (2.25, 1), (4.0, 12), (4.25, 12), (6.0, 13)]
    assert [n for _, _, n in steady(reads)] == [10, 11, 12, 13]
    # 11 unread, 12 and 13 read as "2" and "3", then 14 read whole
    reads = [(0.0, 10), (0.25, 10), (2.0, 2), (2.25, 2), (4.0, 3), (4.25, 3), (6.0, 14), (6.25, 14)]
    assert [n for _, _, n in steady(reads)] == [10, 12, 13, 14]


def test_a_section_played_again_after_measure_10_stays_a_jump_back():
    reads = [(0.0, 9), (2.0, 10), (2.25, 10)] + [(2.0 * k + 2, k) for k in range(1, 11)]
    reads = sorted(reads + [(2.0 * k + 2.25, k) for k in range(1, 11)])
    assert [n for _, _, n in steady(reads)] == [9, 10] + list(range(1, 11))


def test_widget_boxes_joins_a_widget_split_on_one_row():
    box = lambda x0, x1: [[x0, 970], [x1, 970], [x1, 990], [x0, 990]]
    result = [(box(1435, 1470), "BAR", 0.99), (box(1475, 1485), "9", 0.95), (box(1488, 1520), "/41", 0.9),
              (box(1600, 1660), "BEAT1/4", 0.9)]
    assert widget_boxes(result) == [((1435, 970, 1520, 990), (9, 41))]


def test_a_dropped_digit_before_the_next_measure_read_whole_is_the_measure_still_shown():
    # 22 read whole, then as "2" while still shown, then 23 and 24
    reads = [(43.0, 22), (43.25, 2), (44.75, 2), (45.0, 23), (46.75, 23), (47.0, 24), (48.5, 24)]
    assert [n for _, _, n in steady(reads)] == [22, 23, 24]
    assert parse("BAR24/124", 24) == (24, 24)


def test_dropped_matches_a_number_with_one_digit_left_out():
    assert dropped(1, 11) and dropped(2, 20) and dropped(4, 41)
    # A one-digit measure has no digit to drop, and 12 can't lose a digit to read 3
    assert not dropped(1, 1) and not dropped(3, 12)


def test_parse_reads_a_missing_slash_and_a_one_read_as_a_cjk_stroke():
    assert parse("BARI26", 26) == (1, 26)
    assert parse("BAR 丨2/27") == (12, 27)


def test_a_dropped_digit_between_two_readings_of_the_whole_measure_is_that_measure():
    # 17 read as "1" for one reading while still shown
    reads = [(0.0, 17), (0.25, 17), (0.5, 1), (0.75, 17), (2.0, 18), (2.25, 18), (4.0, 19), (4.25, 19)]
    assert steady(reads) == [(0.0, 0.75, 17), (2.0, 2.25, 18), (4.0, 4.25, 19)]


def test_widget_boxes_joins_only_near_boxes_on_one_row():
    box = lambda x0, x1, y0=970: [[x0, y0], [x1, y0], [x1, y0 + 20], [x0, y0 + 20]]
    far = [(box(1435, 1470), "BAR", 0.99), (box(1600, 1610), "9", 0.95), (box(1612, 1640), "/41", 0.9)]
    assert widget_boxes(far) == []
    lower = [(box(1435, 1470), "BAR", 0.99), (box(1475, 1485, 1010), "9", 0.95), (box(1488, 1520), "/41", 0.9)]
    assert widget_boxes(lower) == []
