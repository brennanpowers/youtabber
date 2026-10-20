from youtabber.check import fmt
from youtabber.ui import spot_lines
from youtabber.pipeline import Notice


def gap(t0, t1, text="Tab is on screen at {}, but no new measure was printed from it."):
    return Notice(t0, t1, "no_measures", text.format(f"{fmt(t0)}-{fmt(t1)}"), True)


def test_a_repeated_notice_prints_once_in_time_order():
    dwell = Notice(70.0, 75.0, "dwell", "Check here.", True)
    join = Notice(10.0, 14.0, "unresolved_join", "Couldn't tell.", True)
    quiet = Notice(20.0, 25.0, "unresolved_fold", "Not doubtful.", False)
    assert spot_lines([dwell, join, dwell, quiet, join]) == ["0:10-0:14  Couldn't tell.", "1:10-1:15  Check here."]


def test_touching_no_measures_pieces_print_as_one_line():
    lines = spot_lines([gap(100.0, 130.0), gap(130.0, 160.0), gap(160.0, 172.5),
                        Notice(140.0, 141.0, "dwell", "Check here.", True), gap(200.0, 230.0)])
    assert lines == ["1:40-2:52  Tab is on screen at 1:40-2:52, but no new measure was printed from it.",
                     "2:20-2:21  Check here.",
                     "3:20-3:50  Tab is on screen at 3:20-3:50, but no new measure was printed from it."]
