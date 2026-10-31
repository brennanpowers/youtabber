"""Still views from a fake frame stream: a long-held page keeps a bounded number of frames in memory."""
import weakref

import numpy as np

from synth import Song, draw_measures
from youtabber import observe

PAPER = np.array([255, 255, 255], np.uint8)
SONG = Song(notation=False, measures=4)
PAGE = draw_measures(SONG, 0, 4, 1280)
STAFFS = (0, 70, 1280, 160)


def held_page(n: int, sweep: int, alive: list[int], span: tuple[int, int] = (20, 1220)):
    """A fake `frames`: one page held for n frames, with a cursor crossing the columns in `span` every `sweep`
    frames.

    At each new frame, appends to `alive` how many of the frames handed out so far are still in memory.
    """
    handed = []

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i in range(n):
            f = PAGE.copy()
            x = span[0] + int((span[1] - span[0]) * (i % sweep) / sweep)
            f[:, x:x + 8] = (f[:, x:x + 8] * 0.5 + np.array(SONG.cursor) * 0.5).astype(np.uint8)
            handed.append(weakref.ref(f))
            alive.append(sum(r() is not None for r in handed))
            yield i / fps, f
    return frames


def held_views(monkeypatch, n: int, bound: int, alive: list[int]):
    monkeypatch.setattr(observe, "frames", held_page(n, 90, alive))
    monkeypatch.setattr(observe, "MAX_VIEW_FRAMES", bound)
    observe.set_paper(PAPER)
    return observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, n / 6)[0]


def test_long_held_page_keeps_few_frames(monkeypatch):
    alive: list[int] = []
    views = held_views(monkeypatch, 400, 40, alive)
    # The bound, plus the frame being read and the reference frames compared against
    assert max(alive) <= 40 + 3
    unbounded = held_views(monkeypatch, 400, 1000, [])
    # The same passes over the page, each with a clean plate and the playhead in every frame
    assert len(views) == len(unbounded) == 5
    for v, w in zip(views, unbounded):
        assert (v.t0, v.t1, v.wrapped, v.times) == (w.t0, w.t1, w.wrapped, w.times)
        assert np.array_equal(v.plate, PAGE)
        assert v.playhead == w.playhead and len(v.playhead) == len(v.times)


def test_a_cursor_going_back_over_part_of_the_page_starts_a_new_view(monkeypatch):
    # The cursor plays the same two measures again and again, as when a riff page turns to one that looks the same
    monkeypatch.setattr(observe, "frames", held_page(180, 30, [], span=(600, 900)))
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 30.0)[0]
    assert [round(v.t0, 2) for v in views] == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    assert [v.wrapped for v in views] == [False] + [True] * 5


def test_a_cursor_going_back_after_less_than_a_view_does_not_cut(monkeypatch):
    # A beat cursor stepping over one measure every 5 frames, under a second, never sweeps as long as a view lasts
    monkeypatch.setattr(observe, "frames", held_page(60, 5, [], span=(600, 900)))
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 10.0)[0]
    assert [(v.t0, v.wrapped) for v in views] == [(0.0, False)]


def test_a_view_that_opens_with_the_cursor_left_at_the_right_end_is_not_cut_there(monkeypatch):
    # The first frame still shows the cursor where the page before ended; then it sweeps the page once
    sweep = held_page(80, 80, [])

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i, (t, f) in enumerate(sweep(path, fps)):
            if i == 0:
                f = PAGE.copy()
                f[:, 1200:1208] = (f[:, 1200:1208] * 0.5 + np.array(SONG.cursor) * 0.5).astype(np.uint8)
            yield t, f
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 80 / 6)[0]
    assert [(v.t0, v.wrapped) for v in views] == [(0.0, False)]


def test_color_spread_is_judged_against_the_paper():
    observe.set_paper(np.array([200.0, 240.0, 250.0]))
    px = np.array([[[200, 240, 250], [100, 240, 250], [0, 0, 255]]], np.uint8)
    # Paper itself is gray; blue at half the paper's level spreads 255 - 127.5; pure red on it 255 * 255 / 250
    assert np.allclose(observe.chroma(px), [[0.0, 127.5, 260.1]])
    observe.set_paper(np.array([20.0, 20.0, 20.0]))
    # Dark paper: the plain spread between channels
    assert observe.chroma(px).tolist() == [[50.0, 150.0, 255.0]]
    observe.set_paper(PAPER)


def test_paper_rect_counts_pixels_within_the_tolerance():
    frame = np.full((40, 60, 3), 255, np.uint8)
    # Rows 0-9 sit just inside the paper tolerance, rows 30-39 just outside it
    frame[:10] = 255 - (observe.PAPER_TOL - 1)
    frame[30:] = 255 - observe.PAPER_TOL
    assert observe.panel_rect(frame, PAPER.astype(np.float64), (10, 15, 50, 25), 10.0) == (0, 0, 60, 30)


def dark_footage(rows: int, width: int, seed: int) -> np.ndarray:
    """Dark footage: every channel 0-45, within PAPER_TOL of black paper in most pixels but far from flat."""
    return np.random.default_rng(seed).integers(0, 46, (rows, width, 3), dtype=np.uint8)


def test_dark_paper_rows_above_the_staffs_end_at_a_line_gap_of_rows_that_are_not_flat():
    dark = np.array([19.0, 19.0, 19.0])
    frame = np.full((100, 60, 3), 19, np.uint8)
    frame[:40] = dark_footage(40, 60, 1)
    seed = (10, 50, 50, 70)
    assert observe.panel_rect(frame, dark, seed, 10.0) == (0, 25, 60, 100)
    assert observe.panel_rect(frame, dark, seed, 10.0, polarity="light") == (0, 40, 60, 100)
    # Rows of chord names under a line gap tall don't end the band
    frame[:40] = 19
    frame[35:42, ::3] = 230
    assert observe.panel_rect(frame, dark, seed, 10.0, polarity="light") == (0, 25, 60, 100)


def test_a_dark_band_lightening_toward_its_top_stays_flat():
    dark = np.array([19.0, 19.0, 19.0])
    frame = np.full((100, 60, 3), 19, np.uint8)
    # Each row is one color, up to 30 levels lighter than the paper at the top
    frame[:50] = (19 + np.linspace(30, 0, 50)).astype(np.uint8)[:, None, None]
    assert observe.panel_rect(frame, dark, (10, 50, 50, 70), 10.0, polarity="light") == (0, 25, 60, 100)


def test_views_on_dark_paper_keep_footage_above_the_band_out_of_the_panel(monkeypatch):
    page = 255 - PAGE

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i in range(30):
            f = page.copy()
            f[:56] = dark_footage(56, f.shape[1], i)
            yield i / fps, f
    monkeypatch.setattr(observe, "frames", frames)
    black = np.array([0.0, 0.0, 0.0])
    observe.set_paper(black)
    views = observe.views_in(None, (0, 0, 1280, 270), "light", 18.0, black, lambda t: STAFFS, 0.0, 5.0)[0]
    observe.set_paper(PAPER)
    assert [v.rect[1] for v in views] == [56]


def test_dark_paper_with_too_little_ink_keeps_the_threshold_below_a_cursor_bar():
    # Inverted gray of paper at level 19, with a few white digit pixels, too few to set the ink level
    gray = np.full((100, 100), 236, np.uint8)
    gray[0, :20] = 0
    thr, fthr = observe.ink_threshold(gray, np.array([19.0, 19.0, 19.0]), "light")
    assert round(thr, 2) == 172.28 and round(fthr, 2) == 218.0
    # A see-through amber cursor bar at gray 49.5 is lighter than the threshold, so it isn't ink
    assert 255 - 49.5 > thr
    # Light paper takes its ink to be at least 60 levels darker than the paper
    thr, fthr = observe.ink_threshold(gray, np.array([236.0, 236.0, 236.0]), "dark")
    assert round(thr, 2) == 209.0 and round(fthr, 2) == 218.0


def test_color_is_judged_the_same_on_a_wide_image_in_pieces():
    rng = np.random.default_rng(3)
    wide = rng.integers(0, 256, (20, observe.COLOR_PIECE * 2 + 37, 3), dtype=np.uint8)
    observe.set_paper(np.array([200.0, 240.0, 250.0]))
    assert np.array_equal(observe.colored(wide), observe.chroma(wide) > observe.COLOR_SAT)
    assert np.array_equal(observe.gray_pixels(wide), observe.chroma(wide) < observe.GRAY_SAT)
    observe.set_paper(PAPER)


def test_colored_pixels_are_kept_when_they_are_the_paper_or_nearly_all_the_ink():
    area = np.ones((10, 100), bool)
    ink = np.zeros_like(area)
    ink[4:6] = True
    # A cursor: a colored band over a fifth of the page and of its ink
    cursor = np.zeros_like(area)
    cursor[:, :20] = True
    assert not observe.tinted_paper(cursor, area, ink)
    # Music drawn in color: the colored pixels are the ink and its surroundings
    drawn = np.zeros_like(area)
    drawn[3:7] = True
    assert observe.tinted_paper(drawn, area, ink)
    # Colored paper
    assert observe.tinted_paper(np.ones_like(area), area, ink)


def test_colored_ink_counts_as_the_ink_s_own_color_past_its_share():
    area = np.ones((10, 100), bool)
    ink = np.zeros_like(area)
    ink[4:6] = True
    # 80 and 90 of the 100 inked columns colored, a fifth of the area or less
    for share, kept in ((0.8, False), (0.9, True)):
        tint = np.zeros_like(area)
        tint[4:6, :int(share * 100)] = True
        assert observe.tinted_paper(tint, area, ink) == kept


# The panel covers all of a 20x60 test frame
WHOLE = (0, 0, 60, 20)


def footage_stack(polarity: str = "dark") -> np.ndarray:
    """Ten 20x60 frames of paper (240) behind a see-through panel: a dark stripe of footage (200) covers columns 10-35
    in the first seven frames and columns 40-49 in the last three. Column 30 holds a faint line (215 on bare paper,
    180 under the stripe) and column 12 solid ink (40). With `polarity` "light", the levels are inverted."""
    frames = []
    for i in range(10):
        f = np.full((20, 60), 240, np.uint8)
        f[:, 10:36] = 200 if i < 7 else 240
        f[:, 40:50] = 200 if i >= 7 else 240
        f[:, 30] = 180 if i < 7 else 215
        f[:, 12] = 40
        frames.append(np.dstack([f, f, f]))
    stack = np.stack(frames)
    return 255 - stack if polarity == "light" else stack


def test_footage_moving_behind_the_panel_is_washed_out_of_the_plate():
    stack = footage_stack()
    plate = np.median(stack, axis=0).astype(np.uint8)
    assert plate[5, 20, 0] == 200
    out = observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, WHOLE)
    # Paper under the stripe takes its lightest frames; the stripe seen in only three frames was never in the median
    assert out[5, 20, 0] == 240 and out[5, 45, 0] == 240
    # A faint line stays darker than the paper beside it in its lightest frames, so it keeps its median
    assert out[5, 30, 0] == 180
    # Solid ink is left as it is
    assert out[5, 12, 0] == 40


def test_footage_is_washed_out_toward_dark_paper_on_a_light_ink_theme():
    stack = footage_stack("light")
    plate = np.median(stack, axis=0).astype(np.uint8)
    out = observe.clear_footage(stack, plate, "light", 120.0, 225.0, 10.0, WHOLE)
    assert out[5, 20, 0] == 15 and out[5, 30, 0] == 75 and out[5, 12, 0] == 215


def test_a_view_of_few_frames_keeps_its_median_plate():
    stack = footage_stack()[:5]
    plate = np.median(stack, axis=0).astype(np.uint8)
    assert np.array_equal(observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, WHOLE), plate)


def test_a_faint_line_that_moves_a_pixel_in_one_frame_keeps_its_median():
    # Footage at 200 over columns 40-59 in seven of ten frames covers more than FOOTAGE_SHARE of the paper
    frames = []
    for i in range(10):
        f = np.full((20, 60), 240, np.uint8)
        f[:, 40:] = 200 if i < 7 else 240
        f[:, 31 if i == 4 else 30] = 180
        frames.append(np.dstack([f, f, f]))
    stack = np.stack(frames)
    plate = np.median(stack, axis=0).astype(np.uint8)
    out = observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, WHOLE)
    assert out[5, 30, 0] == 180 and out[5, 31, 0] == 240 and out[5, 50, 0] == 240


def test_footage_over_too_little_of_the_paper_keeps_the_median_plate():
    # Paper at 240 over 20x60 pixels; a stripe at 200 in seven of ten frames covers 11 columns (0.18 of the paper) in
    # one stack and 13 columns (0.22) in the other
    stack = np.full((10, 20, 60, 3), 240, np.uint8)
    stack[:7, :, 10:21] = 200
    plate = np.median(stack, axis=0).astype(np.uint8)
    assert np.array_equal(observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, WHOLE), plate)
    stack[:7, :, 21:23] = 200
    plate = np.median(stack, axis=0).astype(np.uint8)
    assert observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, WHOLE)[5, 15, 0] == 240


def test_footage_lighter_than_faint_ink_keeps_its_median():
    # Paper at 240 under footage at 222 in seven of ten frames over columns 10-35: with the faint-ink threshold at 220
    # the median's 222 already reads as paper and stays; at 225 it is washed to 240
    stack = np.full((10, 20, 60, 3), 240, np.uint8)
    stack[:7, :, 10:36] = 222
    plate = np.median(stack, axis=0).astype(np.uint8)
    assert observe.clear_footage(stack, plate, "dark", 120.0, 220.0, 10.0, WHOLE)[5, 20, 0] == 222
    assert observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, WHOLE)[5, 20, 0] == 240


def test_paper_under_footage_takes_its_second_lightest_frame_of_21():
    # Footage over columns 10-35 of paper at 240: eighteen frames at 200, then one each at 220, 230 and 250. The frame
    # lighter than 95% of them is the 230
    levels = [200] * 18 + [220, 230, 250]
    stack = np.full((21, 20, 60, 3), 240, np.uint8)
    for i, v in enumerate(levels):
        stack[i, :, 10:36] = v
    plate = np.median(stack, axis=0).astype(np.uint8)
    assert plate[5, 20, 0] == 200
    assert observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, WHOLE)[5, 20, 0] == 230


def test_a_page_fading_in_keeps_its_median_plate():
    # The whole frame brightens from 150 to 240 over ten frames
    stack = np.stack([np.full((20, 60, 3), 150 + 10 * i, np.uint8) for i in range(10)])
    plate = np.median(stack, axis=0).astype(np.uint8)
    assert observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, WHOLE)[5, 20, 0] == 195


def test_a_view_ending_in_a_fade_keeps_its_median_plate():
    # Paper at 240 with footage over columns 10-35 at 200 in eight frames and 230 in two; then four frames where the
    # whole picture fades to 120
    stack = np.full((14, 20, 60, 3), 240, np.uint8)
    for i in range(10):
        stack[i, :, 10:36] = 230 if i >= 8 else 200
    stack[10:] = 120
    plate = np.median(stack, axis=0).astype(np.uint8)
    assert observe.clear_footage(stack, plate, "dark", 100.0, 225.0, 10.0, WHOLE)[5, 20, 0] == 200


def test_paper_the_median_shows_near_its_lightest_keeps_its_median():
    # A cursor darkens columns 10-35 to 200 in three of ten frames; the median paper there sits 5 levels under the
    # lightest frame, which is noise, not footage
    stack = np.full((10, 20, 60, 3), 240, np.uint8)
    stack[:, :, 10:36] = 235
    stack[9, :, 10:36] = 240
    stack[:3, :, 10:36] = 200
    plate = np.median(stack, axis=0).astype(np.uint8)
    assert observe.clear_footage(stack, plate, "dark", 100.0, 238.0, 10.0, WHOLE)[5, 20, 0] == 235


def test_paper_beside_ink_under_footage_keeps_its_median():
    # The neighbor test can't tell solid ink from a faint line that moved a pixel, and such a line must survive, so
    # the one-pixel strip beside the ink keeps the footage's 200
    stack = footage_stack()
    plate = np.median(stack, axis=0).astype(np.uint8)
    assert observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, WHOLE)[5, 13, 0] == 200


def test_footage_outside_the_panel_keeps_its_median():
    # The panel covers columns 0-24 only: paper under the stripe at column 20 is washed, at column 25 it isn't
    stack = footage_stack()
    plate = np.median(stack, axis=0).astype(np.uint8)
    out = observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, (0, 0, 25, 20))
    assert out[5, 20, 0] == 240 and out[5, 25, 0] == 200


def test_paper_touching_ink_keeps_its_median_even_where_the_ink_is_light_in_the_lightest_frames():
    # Footage at 200 over columns 10-35 in seven of ten frames; ink at column 12 shows 40 in those frames and paper in
    # the other three, so its median is ink. Column 13 touches it and keeps 200; column 14 is washed to 240
    stack = np.full((10, 20, 60, 3), 240, np.uint8)
    stack[:7, :, 10:36] = 200
    stack[:7, :, 12] = 40
    plate = np.median(stack, axis=0).astype(np.uint8)
    out = observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, WHOLE)
    assert out[5, 12, 0] == 40 and out[5, 13, 0] == 200 and out[5, 14, 0] == 240


def test_footage_ranked_a_few_pixels_at_a_time_gives_the_same_plate(monkeypatch):
    stack = footage_stack()
    plate = np.median(stack, axis=0).astype(np.uint8)
    whole = observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, WHOLE)
    monkeypatch.setattr(observe, "FOOTAGE_BLOCK", 7)
    out = observe.clear_footage(stack, plate, "dark", 120.0, 225.0, 10.0, WHOLE)
    assert np.array_equal(out, whole)
    assert out[5, 20, 0] == 240 and out[5, 30, 0] == 180


def faint_line_stack(line: int, light: int = 246) -> np.ndarray:
    """Ten 20x60 frames of paper (253): footage at 225 over columns 0-23 in seven of them, and a faint line along row
    10 of columns 30-59 at `line` in eight frames and `light` in the other two."""
    stack = np.full((10, 20, 60, 3), 253, np.uint8)
    stack[:7, :, :24] = 225
    stack[:8, 10, 30:] = line
    stack[8:, 10, 30:] = light
    return stack


def test_a_faint_line_beside_footage_keeps_its_median():
    stack = faint_line_stack(228)
    plate = np.median(stack, axis=0).astype(np.uint8)
    out = observe.clear_footage(stack, plate, "dark", 120.0, 235.3, 10.0, WHOLE)
    # The line is 25 levels darker than the paper beside it in the median; the footage is washed to paper
    assert out[10, 45, 0] == 228 and out[5, 10, 0] == 253


def test_a_pixel_up_to_twelve_levels_darker_than_its_neighbors_is_washed():
    # The paper beside the line is 253: a line at 241 is washed to its lightest frames' 254, one at 240 keeps its median
    for line, want in ((241, 254), (240, 240)):
        stack = faint_line_stack(line, 254)
        plate = np.median(stack, axis=0).astype(np.uint8)
        assert observe.clear_footage(stack, plate, "dark", 120.0, 245.0, 10.0, WHOLE)[10, 45, 0] == want


PANEL = (0, 25, 1280, 214)


def number_frames(x: int, level: tuple[int, int, int] = (0, 0, 230)) -> tuple[np.ndarray, np.ndarray]:
    """PAGE, and PAGE with a 10x15 mark of the given color at column x in the row of measure numbers above STAFFS
    (rows 39-67 at an 18-pixel line gap)."""
    b = PAGE.copy()
    b[45:60, x:x + 10] = level
    return PAGE, b


def test_a_colored_digit_changing_in_the_number_row_is_a_change():
    assert observe.number_changed(*number_frames(300), PANEL, STAFFS, 18.0)


def test_the_number_row_counts_a_channel_differing_by_more_than_80():
    # White paper against gray 175 differs by 80, against 174 by 81
    assert not observe.number_changed(*number_frames(300, (175, 175, 175)), PANEL, STAFFS, 18.0)
    assert observe.number_changed(*number_frames(300, (174, 174, 174)), PANEL, STAFFS, 18.0)


def test_a_line_cursor_or_a_speck_in_the_number_row_is_no_change():
    a = PAGE
    line = PAGE.copy()
    line[40:67, 300:302] = (0, 0, 230)
    speck = PAGE.copy()
    speck[50:53, 300:303] = (0, 0, 230)
    # Dilated, the line is 4 pixels wide against 4.5 for a digit, the speck 5 tall against 5.4
    assert not observe.number_changed(a, line, PANEL, STAFFS, 18.0)
    assert not observe.number_changed(a, speck, PANEL, STAFFS, 18.0)


def test_a_cursor_reaching_up_from_the_staff_into_the_number_row_is_no_change():
    # 10 pixels wide, from row 60 in the number row down past the top staff line at row 70: the compare follows it to
    # row 79, taller than a line gap
    b = PAGE.copy()
    b[60:100, 300:310] = (230, 60, 30)
    assert not observe.number_changed(PAGE, b, PANEL, STAFFS, 18.0)
    # Ending at row 72, just past the top staff line, it is a digit sitting on the line
    b = PAGE.copy()
    b[60:72, 300:310] = (230, 60, 30)
    assert observe.number_changed(PAGE, b, PANEL, STAFFS, 18.0)


def test_a_number_row_change_in_skipped_pixels_is_no_change():
    a, b = number_frames(300)
    skip = np.zeros(PAGE.shape[:2], bool)
    skip[40:65, 295:315] = True
    assert not observe.number_changed(a, b, PANEL, STAFFS, 18.0, skip)


def test_a_colored_box_coming_up_from_the_staff_over_the_number_row_stops_the_number_compare():
    # A pale highlight box over rows 50-120, 55 levels from the paper, recolors a gray number under it by 90 levels
    a = PAGE.copy()
    a[50:62, 300:308] = (150, 150, 150)
    b = a.copy()
    b[50:120, 280:500] = (200, 240, 245)
    b[50:62, 300:308] = (60, 100, 110)
    tint = np.zeros(PAGE.shape[:2], bool)
    tint[48:122, 278:502] = True
    assert observe.number_changed(a, b, PANEL, STAFFS, 18.0)
    assert not observe.number_changed(a, b, PANEL, STAFFS, 18.0, tint=tint)
    # A colored line cursor 17 pixels wide, under the 18 of a line gap, leaves the number row compared
    tint[48:122, 278:502] = False
    tint[30:122, 600:617] = True
    assert observe.number_changed(a, b, PANEL, STAFFS, 18.0, tint=tint)


def test_a_mark_taller_than_a_line_gap_in_the_number_row_is_no_digit():
    # 6 pixels wide; dilated a pixel each way, 16 rows measure the 18 of a line gap and 17 rows are a piece of a cursor
    for rows, want in ((16, True), (17, False)):
        b = PAGE.copy()
        b[42:42 + rows, 300:306] = (0, 0, 230)
        assert observe.number_changed(PAGE, b, PANEL, STAFFS, 18.0) is want


def test_a_mark_changing_below_the_number_row_is_no_number_change():
    b = PAGE.copy()
    b[68:83, 300:310] = (0, 0, 230)
    assert not observe.number_changed(PAGE, b, PANEL, STAFFS, 18.0)


def test_a_mark_changing_above_the_panel_is_no_number_change():
    # The mark covers rows 45-59; the panel starts at row 60, inside the number row
    assert not observe.number_changed(*number_frames(300), (0, 60, 1280, 214), STAFFS, 18.0)


def look_alike_views(monkeypatch, printed: bool, calls: list) -> list:
    """Views of two pages held 5 s each that differ only in a light-gray measure number, lighter than ink."""
    first, second = number_frames(300, (170, 170, 170))[1], number_frames(600, (170, 170, 170))[1]

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i in range(60):
            yield i / fps, (first if i < 30 else second).copy()

    def prints_numbers(*a):
        calls.append(a)
        return printed
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    return observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 10.0,
                            prints_numbers=prints_numbers)[0]


def test_pages_differing_only_in_their_numbers_are_told_apart_when_the_segment_prints_numbers(monkeypatch):
    calls: list = []
    views = look_alike_views(monkeypatch, True, calls)
    assert [round(v.t0, 2) for v in views] == [0.0, 5.0]
    assert len(calls) == 1


def test_a_number_row_change_is_no_change_where_no_measure_numbers_are_printed(monkeypatch):
    calls: list = []
    views = look_alike_views(monkeypatch, False, calls)
    assert [round(v.t0, 2) for v in views] == [0.0]
    # Each reference whose number row changed is checked once, up to NUMBER_CHECKS of them
    assert 1 <= len(calls) <= observe.NUMBER_CHECKS



def test_without_a_numbers_check_the_number_row_is_not_compared(monkeypatch):
    first, second = number_frames(300, (170, 170, 170))[1], number_frames(600, (170, 170, 170))[1]

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i in range(60):
            yield i / fps, (first if i < 30 else second).copy()
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 10.0)[0]
    assert [round(v.t0, 2) for v in views] == [0.0]
