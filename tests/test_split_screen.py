from pathlib import Path
from types import SimpleNamespace

import numpy as np
from test_streams import THREE, marks, numbered, panel_view, placed, screen, split_parts

from youtabber import glide, pipeline
from youtabber import scan as scan_module
from youtabber.observe import View
from youtabber.panel import panel_bounds
from youtabber.parts import Part, find_parts, panels, side_by_side, stream_of
from youtabber.pipeline import (carry_on, first_shown, part_crop, part_notices, seed_function, split_duplicates,
                                unsplit_held)
from youtabber.scan import Segment
from youtabber.video import VideoInfo

W, H = 1920, 1080
TOP, GAP = 400, 14


def tab(top: int, x0: int = 100, x1: int = 1820) -> scan_module.Staff:
    return scan_module.Staff([top + i * GAP for i in range(6)], x0, x1, GAP, "dark")


def frame(paper: tuple[int, int], tops: list[int], divider: tuple[int, int] = (959, 962),
          rows: tuple[int, int] = (0, H), x0: int = 100, x1: int = 1820) -> np.ndarray:
    """A gray frame of flat footage with paper on `paper` rows holding a tab staff at each of `tops`, and a dark
    column over `divider` columns on `rows`."""
    g = np.full((H, W), 128, np.uint8)
    g[paper[0]:paper[1]] = 255
    for top in tops:
        for i in range(6):
            g[top + i * GAP, x0:x1] = 0
    g[rows[0]:rows[1], divider[0]:divider[1]] = 0
    return g


def one_staff(**kw) -> tuple[np.ndarray, scan_module.Staff]:
    return frame((TOP - 3 * GAP, TOP + 8 * GAP), [TOP], **kw), tab(TOP)


# The divider is looked for on rows 372-497 (2 gaps above and below the staff) and must run on through rows 0-371
# above or rows 498 on below


def test_a_divider_is_at_most_three_line_gaps_wide():
    g, st = one_staff(divider=(940, 940 + 42))
    assert scan_module.divider_at(g, st, TOP) == (940 + 981) // 2
    g, st = one_staff(divider=(940, 940 + 43))
    assert scan_module.divider_at(g, st, TOP) is None


def test_a_divider_must_stand_out_from_the_paper_on_95_percent_of_the_rows_around_the_staff():
    # 126 rows: 120 dark is 0.952, 119 is 0.944
    for hole, found in ((6, True), (7, False)):
        g, st = one_staff()
        g[375:375 + hole, 959:962] = 255
        assert (scan_module.divider_at(g, st, TOP) is not None) == found


def test_a_divider_must_lie_in_the_middle_three_fifths_of_the_staff():
    # The staff spans 1720 columns from x 100, so the middle runs from x 444 to x 1475
    for x, found in ((443, False), (444, True), (1475, True), (1476, False)):
        g, st = one_staff(divider=(x, x + 1))
        assert (scan_module.divider_at(g, st, TOP) == x) == found, x


def test_a_divider_must_run_through_the_footage_on_half_the_rows_above_or_below():
    # 372 rows above the system: 186 dark rows are half of them, 185 too few; the column stops at the staff's band
    for start, found in ((186, True), (187, False)):
        g, st = one_staff(rows=(start, 498))
        assert (scan_module.divider_at(g, st, TOP) is not None) == found


def test_a_divider_needs_seven_columns_of_frame_left_of_it_and_six_right_of_it():
    # Short staffs at the frame's left and right edges, each crossed by a one-pixel column
    for x, (x0, x1), found in ((7, (0, 20), True), (6, (0, 20), False),
                               (1913, (1900, W), True), (1914, (1900, W), False)):
        g = frame((TOP - 3 * GAP, TOP + 8 * GAP), [TOP], divider=(x, x + 1), x0=x0, x1=x1)
        assert (scan_module.divider_at(g, tab(TOP, x0, x1), TOP) == x) == found, x


def test_a_divider_must_differ_from_the_footage_beside_it_by_more_than_25_gray_levels():
    # The footage is gray 128; the column is dark across the paper and 26 or 25 levels darker through the footage
    for level, found in ((102, True), (103, False)):
        g, st = one_staff()
        g[:TOP - 3 * GAP, 959:962] = level
        g[TOP + 8 * GAP:, 959:962] = level
        assert (scan_module.divider_at(g, st, TOP) is not None) == found


def test_a_divider_is_not_judged_on_fewer_rows_of_frame_than_the_system_takes():
    # The system takes 126 rows from 2 gaps above the staff. The column runs through all the rows above it, which
    # are 126 for a staff at row 154 but only 125 at row 153, and none below
    for top, found in ((154, True), (153, False)):
        g = frame((top - 3 * GAP, top + 8 * GAP), [top], rows=(0, top + 7 * GAP))
        assert (scan_module.divider_at(g, tab(top), top) is not None) == found


def test_a_divider_through_a_whole_stack_of_staffs_cuts_each_of_them():
    upper, lower = TOP, TOP + 10 * GAP
    g = frame((upper - 3 * GAP, lower + 8 * GAP), [upper, lower])
    out = scan_module.split_panels(g, [scan_module.System(tab(upper), None), scan_module.System(tab(lower), None)])
    assert [(s.tab.top, s.tab.x0, s.tab.x1, s.panel) for s in out] == [
        (upper, 100, 953, True), (upper, 967, 1820, True), (lower, 100, 953, True), (lower, 967, 1820, True)]


def test_a_divider_must_run_on_beyond_its_own_staff_as_well_as_beyond_the_stack():
    # The column runs through the footage above the upper staff and through both staffs' paper, but not through the
    # footage between them: only the upper staff has it running on beyond its own rows
    upper, lower = 160, 800
    g = frame((upper - 3 * GAP, upper + 8 * GAP), [upper, lower], rows=(0, upper + 7 * GAP))
    g[lower - 3 * GAP:lower + 8 * GAP] = 255
    for i in range(6):
        g[lower + i * GAP, 100:1820] = 0
    g[lower - 3 * GAP:lower + 8 * GAP, 959:962] = 0
    out = scan_module.split_panels(g, [scan_module.System(tab(upper), None), scan_module.System(tab(lower), None)])
    assert [(s.tab.top, s.panel) for s in out] == [(upper, True), (upper, True), (lower, False)]


def test_a_staff_in_other_columns_does_not_join_the_stack_a_divider_must_run_past():
    # The column runs from the staff's band down through the footage to the bottom of a staff below it. That staff
    # starts at column 1820, where the upper one ends, so it is in other columns; starting one column sooner it joins
    # the stack, and the column no longer runs on past the stack
    for x0, cut in ((1820, True), (1819, False)):
        g = frame((TOP - 3 * GAP, TOP + 8 * GAP), [TOP], rows=(372, 870))
        for i in range(6):
            g[800 + i * GAP, x0:1900] = 0
        systems = [scan_module.System(tab(TOP), None), scan_module.System(tab(800, x0, 1900), None)]
        assert scan_module.split_panels(g, systems)[0].panel == cut


def test_a_divider_cuts_the_notation_staff_above_the_tab_too():
    notation = scan_module.Staff([TOP - 9 * GAP + i * GAP for i in range(5)], 100, 1820, GAP, "dark")
    g = frame((TOP - 12 * GAP, TOP + 8 * GAP), [TOP])
    left, right = scan_module.split_panels(g, [scan_module.System(tab(TOP), notation)])
    assert [(s.tab.x0, s.tab.x1, s.notation.x0, s.notation.x1) for s in (left, right)] == [
        (100, 953, 100, 953), (967, 1820, 967, 1820)]


def test_a_strip_of_chord_diagrams_is_not_cut(monkeypatch):
    g, st = one_staff()
    monkeypatch.setattr(scan_module, "broken_up", lambda gray, staff: True)
    s = scan_module.System(st, None)
    assert scan_module.split_panels(g, [s]) == [s]


def sys(top: int, bottom: int, x0: int, x1: int) -> SimpleNamespace:
    return SimpleNamespace(top=top, tab=SimpleNamespace(bottom=bottom, x0=x0, x1=x1))


def seg() -> Segment:
    return Segment(t0=0, t1=20, gap=10, polarity="dark", y0=0, y1=1080, x0=0, x1=1920)


def test_systems_side_by_side_share_rows_but_not_columns():
    a = sys(100, 204, 100, 950)
    assert side_by_side({1.0: [a, sys(204, 300, 970, 1820)]}) == [(1.0, 960.0)]
    assert side_by_side({1.0: [a, sys(205, 300, 970, 1820)]}) == []
    assert side_by_side({1.0: [a, sys(0, 100, 970, 1820)]}) == [(1.0, 960.0)]
    assert side_by_side({1.0: [a, sys(0, 99, 970, 1820)]}) == []
    assert side_by_side({1.0: [a, sys(100, 204, 950, 1820)]}) == []
    assert side_by_side({1.0: [a, sys(100, 204, 951, 1820)]}) == [(1.0, 950.5)]


def split_at(n_whole: int, n_split: int) -> dict:
    """Scan times every half second: the staff across the width n_whole times, then two panels n_split times."""
    whole = [sys(100, 204, 100, 1820)]
    split = [sys(100, 204, 100, 950), sys(100, 204, 970, 1820)]
    return {k * 0.5: whole if k < n_whole else split for k in range(n_whole + n_split)}


def test_a_split_screen_must_show_for_four_seconds():
    page = Part(0, 0, (0, 1080), "page", split_at(20, 8), 14.0)
    assert [p.cols for p in panels(page, seg(), 0.5)] == [None, (0, 960), (960, 1920)]
    page = Part(0, 0, (0, 1080), "page", split_at(20, 7), 13.5)
    assert panels(page, seg(), 0.5) == [page]


def test_a_split_screen_s_panels_meet_on_an_even_column():
    for left_x1, div in ((951, 960), (950, 960), (953, 962)):
        found = {k * 0.5: [sys(100, 204, 100, left_x1), sys(100, 204, left_x1 + 20, 1820)] for k in range(10)}
        page = Part(0, 0, (0, 1080), "page", found, 5.0)
        assert [p.cols for p in panels(page, seg(), 0.5)] == [(0, div), (div, 1920)], left_x1


def test_a_split_screen_shown_across_the_width_under_four_seconds_has_its_left_panel_carry_the_stream():
    page = Part(0, 0, (0, 1080), "page", split_at(8, 20), 14.0)
    whole, left, right = panels(page, seg(), 0.5)
    assert (whole.seen, left.continues, left.carries) == (4.0, whole, False)
    page = Part(0, 0, (0, 1080), "page", split_at(7, 20), 13.5)
    left, right = panels(page, seg(), 0.5)
    assert (left.seen, left.continues, left.carries, right.carries) == (10.0, None, True, False)
    assert left.cut_from is right.cut_from is page


def test_a_page_part_keeps_the_systems_of_its_own_segment_only():
    found = {k * 0.5: [sys(100, 204, 100, 1820)] for k in range(50)}
    (page,) = find_parts(None, seg(), 0, found, 0.5)
    assert page.kind == "page" and sorted(page.found) == [k * 0.5 for k in range(41)]


def test_the_stream_goes_on_with_the_carrying_panel_or_the_part_seen_longest_with_its_panel():
    whole = Part(0, 0, (0, 100), "band", seen=2.0)
    left = Part(1, 0, (0, 100), "band", seen=3.0, cols=(0, 960), continues=whole)
    for seen, streams in ((5.0, [0, 0, 3]), (5.5, [1, 1, 0])):
        right = Part(2, 0, (0, 100), "band", seen=seen, cols=(960, 1920))
        assert [stream_of(p, [whole, left, right]) for p in (whole, left, right)] == streams
    left = Part(1, 0, (0, 100), "band", seen=1.0, cols=(0, 960), carries=True)
    right = Part(2, 0, (0, 100), "band", seen=30.0, cols=(960, 1920))
    assert [stream_of(p, [left, right]) for p in (left, right)] == [0, 3]


def carried(right_numbers: list, next_numbers: list, next_t0: float = 6.0, mosaic: bool = False) -> tuple:
    """Streams of the left and right panels after `carry_on`, the full width showing in the next segment. The
    left panel's numbers never go on into it."""
    parts = split_parts(False)
    parts[0].seg = 1
    for p in parts[1:]:
        p.stream = stream_of(p, parts[1:])
    obs = [panel_view(0, 0.0, 5.0, 1, [marks(1), marks(2)]),
           panel_view(1, 0.0, 5.0, 2, [marks(k) for k in range(3, 3 + len(right_numbers))]),
           panel_view(2, next_t0, 9.0, 0, [marks(k) for k in range(10, 10 + len(next_numbers))], seg=1)]
    obs[2].mosaic = object() if mosaic else None
    numbered(obs, [[100, 101], right_numbers, next_numbers])
    carry_on(parts, obs, lambda crop: crop)
    return parts[1].stream, parts[2].stream


LEFT, RIGHT = (0, 3), (2, 0)


def test_a_panel_carries_the_stream_when_the_full_width_shows_its_last_measure_again_or_one_after_it():
    # The full width shows measures 20-22, so a panel ending on measure 19 to 22 goes on into it
    for last, streams in ((18, LEFT), (19, RIGHT), (22, RIGHT), (23, LEFT)):
        assert carried([last - 1, last], [20, 21, 22]) == streams, last


def test_a_view_s_first_number_needs_two_readings_that_count_up_from_it():
    assert carried([20, None], [20, 21, 22]) == LEFT
    assert carried([20, 30], [20, 21, 22]) == LEFT
    assert carried([19, 99, 21], [20, 21, 22]) == RIGHT
    assert carried([18, 19], [20, None, 99]) == LEFT


def test_the_full_width_screen_after_the_split_starts_at_most_a_second_before_the_split_ends():
    assert carried([18, 19], [20, 21, 22], next_t0=4.0) == RIGHT
    assert carried([18, 19], [20, 21, 22], next_t0=3.99) == LEFT


def test_a_glide_mosaic_is_not_taken_for_the_full_width_screen_after_the_split():
    assert carried([18, 19], [20, 21, 22], mosaic=True) == LEFT


def test_a_panel_with_no_measures_leaves_the_left_panel_carrying_the_stream():
    parts = split_parts(False)
    parts[0].seg = 1
    for p in parts[1:]:
        p.stream = stream_of(p, parts[1:])
    obs = [panel_view(0, 0.0, 5.0, 1, [marks(1), marks(2)]), panel_view(1, 0.0, 5.0, 2, []),
           panel_view(2, 6.0, 9.0, 0, [marks(5), marks(6)], seg=1)]
    numbered(obs, [[18, 19], [], [20, 21]])
    carry_on(parts, obs, lambda crop: crop)
    assert (parts[1].stream, parts[2].stream) == LEFT


def test_a_split_screen_s_parts_notice_leaves_out_a_panel_that_prints_nothing():
    parts = split_parts(True)
    obs = [panel_view(0, 0.0, 4.0, 0, []), panel_view(1, 4.0, 9.0, 1, []), panel_view(2, 4.0, 7.0, 2, []),
           panel_view(3, 9.0, 15.0, 1, []), panel_view(4, 7.0, 20.0, 2, [])]
    primary = [SimpleNamespace(view=v) for v in (0, 1)]
    assert [(n.kind, n.t0, n.t1) for n in part_notices(parts, obs, primary)] == [("parts", 4.0, 15.0)]


def duplicates(monkeypatch, likeness: float, t0: float = 0.0, numbers=(None, None), parts=(1, 2),
               segs=(0, 0)) -> list:
    """split_duplicates over a measure first read in each of two views at 0-5 s and from t0 on, of the parts and
    segments given, whose cells compare as `likeness`."""
    monkeypatch.setattr(pipeline, "resemble", lambda a, b, gap: likeness)
    obs = [panel_view(0, 0.0, 5.0, parts[0], [marks(1)], seg=segs[0]),
           panel_view(1, t0, t0 + 5.0, parts[1], [marks(1)], seg=segs[1])]
    A = placed({0: [0], 1: [1]}, obs)
    for s, n in zip(A.slots, numbers):
        s.number = n
    return [(n.kind, n.t0) for n in split_duplicates(A, obs, split_parts(True), 10.0)]


def test_a_duplicate_looks_less_different_than_the_check_s_limit(monkeypatch):
    assert duplicates(monkeypatch, 0.0299) == [("possible_duplicate", 0.0)]
    assert duplicates(monkeypatch, 0.03) == []


def test_a_duplicate_is_first_read_within_a_second_of_the_other_panel_s_view_ending(monkeypatch):
    assert duplicates(monkeypatch, 0.0, t0=6.0) == [("possible_duplicate", 6.0)]
    assert duplicates(monkeypatch, 0.0, t0=6.01) == []


def test_a_duplicate_has_the_same_measure_number_when_both_are_read(monkeypatch):
    assert duplicates(monkeypatch, 0.0, numbers=(5, 5)) == [("possible_duplicate", 0.0)]
    assert duplicates(monkeypatch, 0.0, numbers=(5, None)) == [("possible_duplicate", 0.0)]
    assert duplicates(monkeypatch, 0.0, numbers=(5, 6)) == []


def test_a_duplicate_is_read_in_another_panel_of_the_same_segment(monkeypatch):
    assert duplicates(monkeypatch, 0.0, parts=(1, 1)) == []
    assert duplicates(monkeypatch, 0.0, parts=(0, 2)) == []
    assert duplicates(monkeypatch, 0.0, segs=(0, 1)) == []


def panel_system(x0: int, x1: int) -> SimpleNamespace:
    """A system drawn at columns x0-x1 on a still frame."""
    lines = np.zeros(W, bool)
    lines[x0:x1] = True
    st = SimpleNamespace(top=400, bottom=470, x0=x0, x1=x1, gap=14)
    return SimpleNamespace(top=400, tab=st, lines_at=lines, motion=np.zeros((H // 8, W // 8)))


def test_a_panel_s_bound_must_reach_into_its_own_columns():
    s = Segment(t0=0, t1=10, gap=14, polarity="dark", y0=0, y1=H, x0=0, x1=W)
    for x1, bound in ((939, None), (940, None), (1050, None), (1051, (0, 0, 112, H))):
        found = {k * 0.5: [panel_system(100, x1)] for k in range(20)}
        bound_at = panel_bounds(found, s, (960, W))
        assert (bound_at if bound_at is None else bound_at(5.0)) == bound, x1


def test_a_panel_is_read_in_its_own_columns():
    s = Segment(t0=0, t1=10, gap=14, polarity="dark", y0=100, y1=600, x0=50, x1=1850)
    assert part_crop(s, Part(0, 0, (100, 600), "band")) == (50, 100, 1800, 500)
    assert part_crop(s, Part(0, 0, (100, 600), "band", cols=(960, 1850))) == (960, 100, 890, 500)
    found = {1.0: [SimpleNamespace(top=200, tab=SimpleNamespace(x0=1000, x1=1700, bottom=270))]}
    assert seed_function(found, s)(1.0) == (950, 100, 1650, 170)
    assert seed_function(found, s, 960)(1.0) == (40, 100, 740, 170)


def test_a_panel_view_s_cells_are_placed_from_its_own_first_column(monkeypatch):
    monkeypatch.setattr(pipeline, "parse_view", lambda *a, **kw: [])
    s = Segment(t0=0, t1=10, gap=14, polarity="dark", y0=100, y1=300, x0=50, x1=1850)
    v = View(0.0, 5.0, np.full((200, 400, 3), 255, np.uint8), rect=(0, 0, 400, 200), print_rect=(0, 0, 400, 200))
    assert pipeline.plate_obs(v, s, 0, 128.0, 230.0, 0).offset == (50, 100)
    assert pipeline.plate_obs(v, s, 0, 128.0, 230.0, 0, 960).offset == (960, 100)


def held_reading(monkeypatch, whole: Part, glides: bool, views: list[tuple[float, float]]) -> tuple:
    """unsplit_held's answer for a part read as `views` (start, end) across its width, and the views_in calls."""
    calls = []

    def views_in(path, crop, *a, **kw):
        calls.append((crop, a[3], kw["lead"], kw["last_screen"]))
        return [SimpleNamespace(t0=t0, t1=t1) for t0, t1 in views], 128.0, 230.0
    monkeypatch.setattr(glide, "glides", lambda *a: glides)
    monkeypatch.setattr(pipeline, "views_in", views_in)
    monkeypatch.setattr(pipeline, "panel_bounds", lambda found, s, cols=None: None)
    s = Segment(t0=10, t1=40, gap=14, polarity="dark", y0=0, y1=H, x0=0, x1=W)
    return unsplit_held(Path("unused.mp4"), (0, 0, W, H), s, whole, "seed", 0.5, True), calls


def test_a_split_part_read_across_its_width_holds_still_for_its_views_or_all_of_it_when_it_glides(monkeypatch):
    found = {10.0: [panel_system(100, 1800)]}
    band = Part(0, 0, (0, H), "band", found, 20.0)
    assert held_reading(monkeypatch, band, True, [(10.0, 12.0)])[0] == (20.0, 20.0)
    band = Part(0, 0, (0, H), "band", found, 19.5)
    held, calls = held_reading(monkeypatch, band, True, [(10.0, 12.0), (15.0, 18.5)])
    assert held == (5.5, 19.5)
    assert [(c[0], c[2], c[3]) for c in calls] == [((0, 0, W, H), 0.5, True)]
    # A page shows for its whole segment, 10-40 s
    page = Part(0, 0, (0, H), "page", found, 25.0)
    held, calls = held_reading(monkeypatch, page, False, [(10.0, 20.0)])
    assert held == (10.0, 30.0) and calls[0][1] == "seed"


def guarded(monkeypatch, held: float, shown: float, stacked: bool = False):
    """A reading of one segment whose part is cut into two panels, read across its full width as holding still for
    `held` of `shown` seconds; with `stacked`, a longer-seen part above it isn't split. Returns the reading, the
    crops read and the parts left out."""
    s = Segment(0.0, 30.0, 10.0, "dark", paper=np.array([255.0, 255.0, 255.0]), x1=640, y1=360)
    whole = Part(1 if stacked else 0, 0, (200, 360) if stacked else (0, 360), "band", {}, 5.0)
    split = [Part(0, 0, whole.rows, "band", {}, 2.0, cols=(0, 320), carries=True, cut_from=whole),
             Part(0, 0, whole.rows, "band", {}, 2.0, cols=(320, 640), cut_from=whole)]
    above = [Part(0, 0, (0, 150), "band", {}, 10.0)] if stacked else []
    crops, dropped = [], []

    def views_in(path, crop, *a, **kw):
        crops.append(crop)
        return [], 128.0, 230.0

    def finish(path, info, r, segments, scrolls, clips, obs, gaps, left_out, *a):
        dropped.extend(left_out)
        return r
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(640, 360, 30.0))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None, progress=None: ([s], {}, []))
    monkeypatch.setattr(pipeline, "find_parts", lambda path, seg, si, found, step: above + split)
    monkeypatch.setattr(pipeline, "unsplit_held", lambda path, crop, seg, part, *a: (held, shown))
    monkeypatch.setattr(glide, "glides", lambda *a: False)
    monkeypatch.setattr(pipeline, "views_in", views_in)
    monkeypatch.setattr(pipeline, "finish", finish)
    return pipeline.read(Path("unused.mp4")), crops, dropped


def test_a_split_screen_whose_full_width_holds_still_under_two_fifths_of_the_time_is_refused(monkeypatch):
    r, crops, _ = guarded(monkeypatch, 7.99, 20.0)
    assert (r.outcome, r.fail_reason) == ("fail", "check_failed") and crops == []
    assert r.message == ("The tab at 0:00-0:30 holds still for only 8 of 20 s and doesn't scroll smoothly either, so "
                         "most of it can't be read; no PDF was written.")
    for held, shown in ((8.0, 20.0), (0.0, 19.9)):
        r, crops, _ = guarded(monkeypatch, held, shown)
        assert r.outcome != "fail" and crops == [(0, 0, 320, 360), (320, 0, 320, 360)]


def test_a_split_screen_beside_the_main_part_that_does_not_hold_still_is_left_out(monkeypatch):
    r, crops, dropped = guarded(monkeypatch, 7.99, 20.0, stacked=True)
    assert r.outcome != "fail" and crops == [(0, 0, 640, 360)]
    assert dropped == [(0.0, 30.0, "it holds still for only 8 of 20 s")]


def test_a_measure_past_the_left_panel_s_right_edge_first_showed_in_the_full_width_screen_after_it():
    # The left panel (part 1) is 960 px wide; moved into its columns, the next measure starts at x 960, past its edge
    obs = [screen(0, 0.0, 5.0, [(60, 360), (360, 660), (660, 957)], part=1), screen(1, 5.5, 10.0, THREE)]
    assert first_shown(placed({0: [0, 1, 2], 1: [1, 2, 3]}, obs), obs, {0: 1920, 1: 960})[3] == 5.5


def test_a_staff_cut_by_a_divider_gives_the_paper_color_as_often_as_a_whole_staff(monkeypatch):
    # A five-line staff on white paper is cut at a divider at x 640; a six-line staff right of the divider sits on
    # gray 120, as does the rest of the frame. Whole, the white staff gives 4 samples to the gray one's 5
    img = np.full((720, 1280, 3), 120, np.uint8)
    img[258:399] = 255
    img[:, 638:642] = 0
    white = scan_module.Staff([300 + i * GAP for i in range(5)], 100, 1180, GAP, "dark")
    gray = scan_module.Staff([500 + i * GAP for i in range(6)], 1000, 1180, GAP, "dark")
    for st in (white, gray):
        for y in st.lines:
            img[y, st.x0:st.x1] = 0
    monkeypatch.setattr(scan_module, "frames", lambda path, fps: [(k / fps, img) for k in range(5 * fps)])
    monkeypatch.setattr(scan_module, "systems_in",
                        lambda g, whole_lines=False: [scan_module.System(white, None), scan_module.System(gray, None)])
    segments, found, _ = scan_module.scan(None, 1280, 720)
    assert sum(s.panel for s in found[0.0]) == 2
    assert [list(s.paper) for s in segments] == [[120.0, 120.0, 120.0]]


def test_a_staff_cut_by_a_divider_gives_the_paper_color_when_the_region_holds_only_its_left_panel(monkeypatch):
    # Dark paper (gray 40) with light lines, cut at a divider at x 640. The region covers x 0-600, so the staff's
    # middle at x 640 is outside it while its left panel is kept
    img = np.full((720, 1280, 3), 40, np.uint8)
    img[:, 638:642] = 255
    light = scan_module.Staff([300 + i * GAP for i in range(6)], 100, 1180, GAP, "light")
    for y in light.lines:
        img[y, light.x0:light.x1] = 200
    monkeypatch.setattr(scan_module, "frames", lambda path, fps: [(k / fps, img) for k in range(5 * fps)])
    monkeypatch.setattr(scan_module, "systems_in", lambda g, whole_lines=False: [scan_module.System(light, None)])
    segments, found, _ = scan_module.scan(None, 1280, 720, region=(0, 0, 600, 720))
    assert [(s.tab.x0, s.panel) for s in found[0.0]] == [(100, True)]
    assert [list(s.paper) for s in segments] == [[40.0, 40.0, 40.0]]
