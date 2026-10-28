"""The panel bound: the box of paper a part's tab is drawn on, found from where the scan saw staff lines and where
the picture moved, and the rects, seeds and edge cells kept inside it."""
import cv2
import numpy as np
import pytest

from synth import Song, draw_measures
from youtabber import observe, parse
from youtabber import scan as scan_module
from youtabber.observe import within
from youtabber.panel import layout_bound, layout_of, panel_bounds, runs
from youtabber.pipeline import bounded_seed
from youtabber.scan import Segment, Staff, System, line_columns

# Scanned frames are 800 x 480 with motion in 8-pixel cells; tab line gap 10
W, H, CELL, GAP = 800, 480, 8, 10.0
SIDES = [(0, 152), (648, 800)]   # footage columns on both sides of the paper
ROWS = [(0, 96), (400, 480)]     # footage rows above and below it


def motion(cols=(), rows=(), level=20.0) -> np.ndarray:
    """A scanned frame's change per cell: `level` over the pixel column and row ranges given, still elsewhere."""
    m = np.zeros((H // CELL, W // CELL), np.float32)
    for a, b in cols:
        m[:, a // CELL:b // CELL] = level
    for a, b in rows:
        m[a // CELL:b // CELL] = level
    return m


def system(top: float, lines=((200, 600),), m=None, gap: float = GAP, n: int = 6) -> System:
    """A tab staff of `n` lines at `top`, its lines drawn over the pixel column ranges `lines`."""
    s = System(Staff([top + i * gap for i in range(n)], 100, 700, gap, "dark"), None)
    s.lines_at = np.zeros(W, bool)
    for a, b in lines:
        s.lines_at[a:b] = True
    s.motion = m
    return s


def segment(**kw) -> Segment:
    return Segment(0.0, 20.0, GAP, "dark", **({"x0": 0, "y0": 0, "x1": W, "y1": H} | kw))


def frames_of(n: int, **kw) -> list[list[System]]:
    m = kw.pop("m", motion(SIDES, ROWS))
    return [[system(200, m=m, **kw)] for _ in range(n)]


def test_runs_gives_each_stretch_of_true():
    assert runs(np.array([1, 1, 0, 1, 0, 0, 1], bool)) == [(0, 2), (3, 4), (6, 7)]
    assert runs(np.zeros(4, bool)) == []


def test_the_bound_reaches_past_the_lines_and_stops_at_the_footage_rows():
    # Lines at 200-600 reach 2.5 gaps left and 1.5 right; rows stop where the footage above and below begins
    assert layout_bound(frames_of(10), segment()) == (175, 96, 615, 400)


def test_the_bound_is_in_crop_coordinates():
    assert layout_bound(frames_of(10), segment(x0=40, y0=20, x1=780, y1=470)) == (135, 76, 575, 380)


def test_the_reach_past_the_lines_stops_at_a_moving_column():
    # A cursor one cell wide at 184-192 is too narrow to split the staff, but the bound stays right of it
    assert layout_bound(frames_of(10, m=motion(SIDES + [(184, 192)], ROWS)), segment()) == (192, 96, 615, 400)


@pytest.mark.parametrize("hole, box", [
    ((392, 400), (175, 96, 615, 400)),   # under a line gap of footage: still one staff
    ((392, 408), (175, 96, 392, 400)),   # a line gap or more: two pictures, and the one with more lines is kept
])
def test_footage_between_staff_columns_splits_them_from_a_line_gap_wide(hole, box):
    m = motion(SIDES + [hole], ROWS)
    assert layout_bound(frames_of(10, lines=((200, hole[0]), (hole[1], 600)), m=m), segment()) == box


@pytest.mark.parametrize("moving, box", [(32, (175, 96, 615, 400)), (40, None)])
def test_staff_lines_over_four_line_gaps_of_footage_give_no_bound(moving, box):
    assert layout_bound(frames_of(10, m=motion(SIDES + [(200, 200 + moving)], ROWS)), segment()) == box


@pytest.mark.parametrize("end, box", [(279, None), (280, (175, 96, 295, 400))])
def test_a_staff_under_eight_line_gaps_wide_gives_no_bound(end, box):
    assert layout_bound(frames_of(10, lines=((200, end),)), segment()) == box


@pytest.mark.parametrize("seen, x1", [(1, 615), (2, 655)])
def test_columns_count_as_staff_where_lines_show_in_a_tenth_of_the_frames(seen, x1):
    m = motion([(0, 152), (704, 800)], ROWS)
    systems = [[system(200, lines=((200, 640),) if k < seen else ((200, 600),), m=m)] for k in range(20)]
    assert layout_bound(systems, segment()) == (175, 96, x1, 400)


@pytest.mark.parametrize("top", [50, 410])
@pytest.mark.parametrize("seen, rows", [(2, (96, 400)), (3, (0, H))])
def test_a_staff_seen_in_a_few_frames_does_not_hold_the_rows_open(top, seen, rows):
    # A staff drawn over the footage above or below the paper: in under a tenth of the frames the rows still stop
    # at the footage; in more, the rows take in the footage rows, so no row bound is used
    m = motion(SIDES, ROWS)
    systems = [[system(top, m=m)] * (k < seen) + [system(200, m=m)] for k in range(20)]
    assert layout_bound(systems, segment()) == (175, rows[0], 615, rows[1])


@pytest.mark.parametrize("moving, x0", [(4, 175), (6, 184)])
def test_footage_moving_in_over_half_the_frames_stops_the_reach(moving, x0):
    # Footage left of 184 moves in only some of the ten frames; the reach past the lines (to 175) stops at it only
    # when it moves in most of them
    systems = [[system(200, m=motion([(0, 184), SIDES[1]] if k < moving else [SIDES[1]], ROWS))] for k in range(10)]
    assert layout_bound(systems, segment()) == (x0, 96, 615, 400)


@pytest.mark.parametrize("moving, y0", [(4, 96), (6, 120)])
def test_footage_rows_moving_in_over_half_the_frames_stop_the_rows(moving, y0):
    # The footage above reaches down to 120 in only some of the ten frames, and to 96 in the rest
    systems = [[system(200, m=motion(SIDES, [(0, 120) if k < moving else ROWS[0], ROWS[1]]))] for k in range(10)]
    assert layout_bound(systems, segment()) == (175, y0, 615, 400)


def test_a_cursor_across_the_staff_rows_leaves_the_rows_unbounded():
    # A highlight over one cell row inside the staffs (rows 224-232) across the whole frame: per column it averages
    # 3 levels over the staffs' seven cell rows, so the columns stay still, but the staff rows move and get no
    # row bound
    m = motion(SIDES, ROWS)
    m[28] = np.maximum(m[28], 21.0)
    assert layout_bound(frames_of(10, m=m), segment()) == (175, 0, 615, H)


def test_frames_without_motion_give_no_bound():
    assert layout_bound(frames_of(10, m=None), segment()) is None


def test_a_layout_is_the_rows_of_its_staffs_at_the_segment_s_spacing():
    # Tops in half line gaps; a staff spaced over 35% away from the segment's is left out
    assert layout_of([system(203), system(300, gap=13.0)], GAP) == (41, 60)
    assert layout_of([system(203), system(300, gap=14.0)], GAP) == (41,)


def found_over(times, top: float, lines=((200, 600),)) -> dict:
    m = motion(SIDES, ROWS)
    return {t: [system(top, lines=lines, m=m)] for t in times}


def half_seconds(a: float, n: int) -> list[float]:
    return [a + k / 2 for k in range(n)]


@pytest.mark.parametrize("n, box", [(7, None), (8, (175, 96, 615, 400))])
def test_a_layout_needs_eight_frames_for_a_bound(n, box):
    bound_at = panel_bounds(found_over(half_seconds(0.0, n), 200), segment())
    assert (bound_at(1.0) if bound_at is not None else None) == box


def test_each_layout_gets_its_own_box():
    found = found_over(half_seconds(0.0, 10), 200) | found_over(half_seconds(5.0, 10), 300, ((250, 550),))
    bound_at = panel_bounds(found, segment())
    assert bound_at(2.0) == (175, 96, 615, 400)
    assert bound_at(7.0) == (225, 96, 565, 400)


def test_a_brief_layout_borrows_the_box_shown_within_two_seconds():
    # Ten frames of one page to 4.5 s, then seven of another, too few for a box of its own
    found = found_over(half_seconds(0.0, 10), 200) | found_over(half_seconds(5.0, 7), 300)
    bound_at = panel_bounds(found, segment())
    box = (175, 96, 615, 400)
    assert (bound_at(6.5), bound_at(7.0)) == (box, None)
    assert bound_at.layouts == [(0.0, 4.5, 10, box), (5.0, 8.0, 7, None)]


def test_frames_with_no_staff_at_the_segment_s_spacing_get_no_box_of_their_own():
    # Ten frames to 4.5 s, then ten whose only staff is spaced 14 apart: past 6.5 s none is near enough to borrow
    found = found_over(half_seconds(0.0, 10), 200)
    m = motion(SIDES, ROWS)
    found |= {t: [system(200, m=m, gap=14.0)] for t in half_seconds(5.0, 10)}
    bound_at = panel_bounds(found, segment())
    assert (bound_at(6.5), bound_at(7.0)) == ((175, 96, 615, 400), None)


@pytest.mark.parametrize("short, y0", [(8, 96), (12, 216)])
def test_a_staff_missing_its_top_lines_keeps_their_rows_open(short, y0):
    # Ten frames show all six lines from 200; then the top two lines are lost under digits that change at 200-216.
    # While six lines are the usual count, the lost lines' rows stay open and the bound stops at the
    # footage above (96); when four lines are usual, the changing digits are footage
    found = found_over(half_seconds(0.0, 10), 200)
    m = motion(SIDES, ROWS + [(200, 216)])
    found |= {t: [system(220, m=m, n=4)] for t in half_seconds(5.0, short)}
    assert panel_bounds(found, segment())(7.0) == (175, y0, 615, 400)


def test_frames_outside_the_segment_are_left_out():
    found = found_over(half_seconds(0.0, 10), 200) | found_over(half_seconds(25.0, 10), 300)
    assert panel_bounds(found, segment()).layouts == [(0.0, 4.5, 10, (175, 96, 615, 400))]


def test_systems_without_line_columns_give_no_bound():
    found = found_over(half_seconds(0.0, 10), 200)
    del found[1.0][0].lines_at
    assert panel_bounds(found, segment()) is None


def test_within_cuts_a_rect_to_the_bound_but_keeps_the_staff_rows():
    rect, bound = (10, 20, 300, 200), (50, 60, 250, 150)
    assert within(rect, bound) == (50, 60, 250, 150)
    assert within(rect, bound, (40, 180)) == (50, 40, 250, 180)
    assert within(rect, None) == rect
    # A bound beside the rect leaves one column
    assert within(rect, (400, 60, 500, 150)) == (400, 60, 401, 150)


def test_a_bounded_seed_keeps_the_staffs_box_inside_the_bound_s_columns():
    def seed_at(t):
        return None if t < 0 else (100, 50, 700, 90)
    assert bounded_seed(seed_at, lambda t: (150, 0, 600, 200))(1.0) == (150, 50, 600, 90)
    assert bounded_seed(seed_at, lambda t: None)(1.0) == (100, 50, 700, 90)
    # A bound that misses the seed's columns leaves the seed as it is
    assert bounded_seed(seed_at, lambda t: (800, 0, 900, 200))(1.0) == (100, 50, 700, 90)
    assert bounded_seed(seed_at, lambda t: (150, 0, 600, 200))(-1.0) is None


def test_line_columns_are_where_at_least_half_the_lines_are_drawn():
    g = np.full((200, 600), 255, np.uint8)
    st = Staff([60 + i * 14 for i in range(6)], 0, 600, 14.0, "dark")
    for i, y in enumerate(st.lines):
        g[y, 100:300] = 60
        if i < 3:
            g[y, 300:400] = 60
        if i < 2:
            g[y, 400:500] = 60
    assert np.array_equal(np.flatnonzero(line_columns(g, st)), np.arange(100, 400))


def test_the_scan_keeps_each_frame_s_motion_and_line_columns(monkeypatch):
    # A block at 400-416, 80-96 flips between black and white each frame; staff lines are drawn at 200-900
    tab = [300 + i * 14 for i in range(6)]
    images = []
    for k in range(4):
        img = np.full((720, 1280, 3), 255, np.uint8)
        img[0, 0] = k
        for y in tab:
            img[y, 200:900] = 60
        if k % 2:
            img[80:96, 400:416] = 0
        images.append(img)
    monkeypatch.setattr(scan_module, "frames", lambda path, fps: [(k / 2, im) for k, im in enumerate(images)])
    monkeypatch.setattr(scan_module, "systems_in",
                        lambda g, whole_lines=False: [System(Staff(list(map(float, tab)), 100, 1180, 14.0, "dark"),
                                                             None)])
    found = scan_module.scan(None, 1280, 720)[1]
    first, second = found[0.0][0], found[0.5][0]
    assert first.motion is None
    assert second.motion.shape == (90, 160)
    assert np.array_equal(np.argwhere(second.motion == 255), [[10, 50], [10, 51], [11, 50], [11, 51]])
    assert np.array_equal(np.flatnonzero(second.lines_at), np.arange(200, 900))


PAPER = np.array([255, 255, 255], np.uint8)
PAGE = draw_measures(Song(notation=False, measures=4), 0, 4, 1280)
STAFFS = (0, 70, 1280, 160)


def page_frames(path, fps, crop=None, start=0.0, length=None):
    for i in range(int(round(length * fps))):
        yield start + i / fps, PAGE.copy()


def test_views_keep_their_panels_inside_the_bound_except_the_staff_rows(monkeypatch):
    monkeypatch.setattr(observe, "frames", page_frames)
    observe.set_paper(PAPER)
    # A bound whose rows stop inside the staffs (70-160): the panel keeps the staffs' rows, the print area a line
    # gap more. The brief screen read in the lead before 2 s is bounded as well
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 2.0, 6.0, lead=1.0,
                             bound_at=lambda t: (200, 100, 1000, 120))[0]
    assert [(v.t0, v.rect, v.print_rect) for v in views] == [(1.0, (200, 70, 1000, 160), (200, 52, 1000, 178)),
                                                            (2.0, (200, 70, 1000, 160), (200, 52, 1000, 178))]
    unbounded = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 2.0, 6.0)[0]
    assert [(v.rect, v.print_rect) for v in unbounded] == [((0, 25, 1280, 214), (0, 0, 1280, 250))]


def dashed_tail(on: int, measures_on: int = 10) -> np.ndarray:
    """A tab staff with barlines at 100, 400 and 700 and a fret digit after the last. Past 710 its lines are dashes
    `on` pixels long every 10; inside the measures, dashes `measures_on` long."""
    img = np.full((200, 800, 3), 255, np.uint8)
    for i in range(6):
        y = 60 + i * 14
        for x in range(20, 711, 10):
            img[y, x:x + measures_on] = 60
        for x in range(711, 781, 10):
            img[y, x:x + on] = 60
    for x in (100, 400, 700):
        cv2.line(img, (x, 60), (x, 130), (0, 0, 0), 2)
    cv2.putText(img, "5", (740, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def tail_cells(img: np.ndarray) -> list[str]:
    return [c.kind for ps in parse.parse_view(0, img, (0, 0, 800, 200), "dark", 140.0, 229.0, 14.0)
            for c in ps.cells if c.kind == "tail"]


@pytest.mark.parametrize("on, tail", [(1, []), (5, ["tail"])])
def test_an_edge_cell_where_under_half_the_staff_s_line_columns_show_is_no_measure(on, tail):
    # Lines show in 99% of the measures' columns, so the cut is 49.5%; in the tail cell they show in 34% of its
    # columns with 1-pixel dashes and 64% with 5-pixel ones
    assert tail_cells(dashed_tail(on)) == tail


def test_an_edge_cell_is_held_to_how_fully_the_lines_show_in_the_measures():
    # 3-pixel dashes all along: lines show in 29% of the measures' columns and 32% of the tail's
    assert tail_cells(dashed_tail(3, measures_on=3)) == ["tail"]


def test_a_system_with_no_whole_measure_keeps_its_edge_cells():
    # One barline at 400, solid lines before it and 1-pixel dashes after it: the tail shows lines in 10% of its
    # columns, but there is no whole measure to hold it to
    img = np.full((200, 800, 3), 255, np.uint8)
    for i in range(6):
        y = 60 + i * 14
        img[y, 20:401] = 60
        for x in range(401, 781, 10):
            img[y, x] = 60
    cv2.line(img, (400, 60), (400, 130), (0, 0, 0), 2)
    cv2.putText(img, "5", (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "7", (600, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cells = [(c.kind, c.x0) for ps in parse.parse_view(0, img, (0, 0, 800, 200), "dark", 140.0, 229.0, 14.0)
             for c in ps.cells]
    assert cells == [("lead", 20), ("tail", 401)]
