from youtabber.counter import at, parse, steady


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
