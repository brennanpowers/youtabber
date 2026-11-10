"""Still views from a fake frame stream: a long-held page keeps a bounded number of frames in memory."""
import weakref

import cv2
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


def test_a_box_cursor_riding_lower_just_after_going_back_still_starts_a_new_view(monkeypatch):
    sweep = held_page(180, 30, [], span=(600, 900))

    def frames(path, fps, crop=None, start=0.0, length=None):
        # The frame after each jump back shows the cursor only from y 120 down, its middle 2.6 line gaps lower
        for i, (t, f) in enumerate(sweep(path, fps)):
            if i % 30 == 1:
                f[:120] = PAGE[:120]
            yield t, f
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)

    def starts():
        views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 30.0)[0]
        return [round(v.t0, 2) for v in views]
    assert starts() == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    monkeypatch.setattr(observe, "WRAP_STEADY_GAPS", 2.0)
    assert starts() == [0.0]


def test_a_cursor_going_back_after_less_than_a_view_does_not_cut(monkeypatch):
    # A beat cursor stepping over one measure every 5 frames, under a second, never sweeps as long as a view lasts
    monkeypatch.setattr(observe, "frames", held_page(60, 5, [], span=(600, 900)))
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 10.0)[0]
    assert [(v.t0, v.wrapped) for v in views] == [(0.0, False)]


def sparse_sweeps(n: int, sweep: int):
    """A fake `frames`: one page held for n frames, its cursor crossing the page every `sweep` frames but drawn in
    only every third frame, as a thin cursor that is often lost."""
    swept = held_page(n, sweep, [])

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i, (t, f) in enumerate(swept(path, fps)):
            yield t, (f if i % 3 == 0 else PAGE.copy())
    return frames


def test_a_still_page_swept_twice_by_a_cursor_seen_now_and_then_is_cut_in_two(monkeypatch):
    monkeypatch.setattr(observe, "frames", sparse_sweeps(120, 60))
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 20.0)[0]
    assert [(round(v.t0, 2), v.wrapped) for v in views] == [(0.0, False), (10.0, True)]


def test_a_still_page_swept_once_by_a_cursor_seen_now_and_then_is_not_cut(monkeypatch):
    monkeypatch.setattr(observe, "frames", sparse_sweeps(60, 60))
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 10.0)[0]
    assert [(v.t0, v.wrapped) for v in views] == [(0.0, False)]


def test_a_pass_too_short_to_keep_after_a_cut_carries_on_into_the_next_view(monkeypatch):
    # The second pass shows for 1 s, then a card covers the page for half a second before the pass goes on
    swept = sparse_sweeps(120, 60)

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i, (t, f) in enumerate(swept(path, fps)):
            if 66 <= i < 69:
                f = PAGE.copy()
                f[:, 300:1000] = 0
            yield t, f
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 20.0)[0]
    assert [(round(v.t0, 2), v.wrapped) for v in views] == [(0.0, False), (11.5, True)]


def test_page_swept_again_needs_a_sweep_to_the_end_and_a_jump_back_to_the_start():
    # A sample every half second sweeps x 100 to 1000 over 5 s, then the playhead goes on from x 100
    sweep = [(i / 2, 100 + 90 * i, 200.0) for i in range(11)]
    again = [(5.5 + i / 2, 100 + 90 * i, 200.0) for i in range(6)]
    assert observe.page_swept_again(sweep + again, 18.0, 1280) == [5.5]
    # Back only to the page's middle
    middle = [(5.5 + i / 2, 550 + 90 * i, 200.0) for i in range(6)]
    assert observe.page_swept_again(sweep + middle, 18.0, 1280) == []
    # A jump forward
    forward = [(i / 2, 100 + 40 * i, 200.0) for i in range(6)] + [(3.0 + i / 2, 900 + 20 * i, 200.0) for i in range(6)]
    assert observe.page_swept_again(forward, 18.0, 1280) == []
    # Down to the start of the next system
    below = [(5.5 + i / 2, 100 + 90 * i, 300.0) for i in range(6)]
    assert observe.page_swept_again(sweep + below, 18.0, 1280) == []
    # One stray sample high above the jump doesn't hide it
    stray = sweep[:-1] + [(5.0, 1000, 60.0)]
    assert observe.page_swept_again(stray + again, 18.0, 1280) == [5.5]


def test_page_swept_again_allows_a_hole_in_the_sweep_of_a_third_of_the_page():
    # Samples every quarter second sweep x 100 to 1000 but miss 400 to 670: 270 px of the 840 px page, 0.32 of it
    sweep = [(i / 4, x, 200.0) for i, x in enumerate([x for x in range(100, 1001, 30) if not 400 < x < 670])]
    again = [(5.75 + i / 4, 100 + 30 * i, 200.0) for i in range(6)]
    assert observe.page_swept_again(sweep + again, 18.0, 1280) == [5.75]


def test_page_swept_again_needs_a_sweep_with_no_hole_wider_than_a_third_of_the_page():
    # The sweep misses 400 to 700: 300 px of the 840 px page, 0.36 of it, as a box cursor stepping over wide measures
    sweep = [(i / 4, x, 200.0) for i, x in enumerate([x for x in range(100, 1001, 30) if not 400 < x < 700])]
    again = [(5.5 + i / 4, 100 + 30 * i, 200.0) for i in range(6)]
    assert observe.page_swept_again(sweep + again, 18.0, 1280) == []


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


def test_dark_paper_rows_below_the_staffs_end_at_a_line_gap_of_rows_that_are_not_flat():
    dark = np.array([19.0, 19.0, 19.0])
    frame = np.full((100, 60, 3), 19, np.uint8)
    frame[60:] = dark_footage(40, 60, 2)
    seed = (10, 30, 50, 50)
    assert observe.panel_rect(frame, dark, seed, 10.0) == (0, 5, 60, 80)
    assert observe.panel_rect(frame, dark, seed, 10.0, polarity="light") == (0, 5, 60, 60)
    # Marks under the tab staff less than a line gap tall don't end the band
    frame[60:] = 19
    frame[55:62, ::3] = 230
    assert observe.panel_rect(frame, dark, seed, 10.0, polarity="light") == (0, 5, 60, 80)


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


def test_a_light_gray_number_changing_is_a_change_and_one_lightening_by_under_half_its_contrast_is_not():
    # A number of gray 182 on white paper differs from it by 73 where it appears
    assert observe.number_changed(*number_frames(300, (182, 182, 182)), PANEL, STAFFS, 18.0)
    # A mark of gray 100, 155 from the paper, turning to 177 differs by 77, under half of 155; to 178, by 78
    a = number_frames(300, (100, 100, 100))[1]
    assert not observe.number_changed(a, number_frames(300, (177, 177, 177))[1], PANEL, STAFFS, 18.0)
    assert observe.number_changed(a, number_frames(300, (178, 178, 178))[1], PANEL, STAFFS, 18.0)


def test_a_pale_colored_mark_in_the_number_row_is_held_to_a_change_of_80():
    # Pale blue differs from white paper by 65 in its red channel: a gray mark would count, a colored one doesn't
    assert not observe.number_changed(*number_frames(300, (250, 230, 190)), PANEL, STAFFS, 18.0)
    assert observe.number_changed(*number_frames(300, (190, 190, 190)), PANEL, STAFFS, 18.0)


def test_a_light_gray_mark_with_the_paper_around_it_changing_is_no_change():
    # A pale gray highlight, 25 levels under the paper, comes in under a light gray mark: the paper around the mark
    # changes too, as a highlight moving under gray text does
    a, b = number_frames(300, (182, 182, 182))
    b = b.copy()
    b[40:66, 280:340] = (230, 230, 230)
    b[45:60, 300:310] = (182, 182, 182)
    assert not observe.number_changed(a, b, PANEL, STAFFS, 18.0)
    # The same mark coming in on still paper is a change
    assert observe.number_changed(*number_frames(300, (182, 182, 182)), PANEL, STAFFS, 18.0)


def test_a_number_row_changing_by_no_more_than_40_is_no_change():
    # Gray 215 differs from white paper by 40, gray 214 by 41
    assert not observe.number_changed(*number_frames(300, (215, 215, 215)), PANEL, STAFFS, 18.0)
    assert observe.number_changed(*number_frames(300, (214, 214, 214)), PANEL, STAFFS, 18.0)


def test_a_line_cursor_or_a_speck_in_the_number_row_is_no_change():
    a = PAGE
    line = PAGE.copy()
    line[40:67, 300:302] = (0, 0, 230)
    speck = PAGE.copy()
    speck[50:53, 300:303] = (0, 0, 230)
    # Dilated, the line is 4 pixels wide against 4.5 for a digit, the speck 5 tall against 5.4
    assert not observe.number_changed(a, line, PANEL, STAFFS, 18.0)
    assert not observe.number_changed(a, speck, PANEL, STAFFS, 18.0)


def test_a_number_left_of_where_the_staffs_were_found_is_compared():
    # Staffs found from column 400; a number over the system's first barline sits at column 380
    assert observe.number_changed(*number_frames(380), PANEL, (400, 70, 1280, 160), 18.0)


def test_a_mark_wider_than_a_number_in_the_number_row_is_no_change():
    # Dilated, a 50 px mark is 52 wide, about 2.9 line gaps of 18, and a 53 px one 55, past NUMBER_MAX_W: a black box
    # cursor's top edge moving along the row
    for width, changed in ((50, True), (53, False)):
        edge = PAGE.copy()
        edge[50:56, 300:300 + width] = (0, 0, 0)
        assert observe.number_changed(PAGE, edge, PANEL, STAFFS, 18.0) == changed


def test_a_digit_sized_mark_reaching_the_bottom_of_the_compared_rows_is_no_change():
    # At an 18-pixel gap the compare looks from the number row down to row 78; dilated, a mark over rows 64-78 is 16
    # tall, the size of a digit, and reaches that row; one over rows 62-74 stops short of it
    low, high = PAGE.copy(), PAGE.copy()
    low[64:79, 300:310] = (0, 0, 230)
    high[62:75, 300:310] = (0, 0, 230)
    assert not observe.number_changed(PAGE, low, PANEL, STAFFS, 18.0)
    assert observe.number_changed(PAGE, high, PANEL, STAFFS, 18.0)


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


def test_a_number_row_change_in_skipped_pixels_or_outside_kept_ones_is_no_change():
    a, b = number_frames(300)
    marked = np.zeros(PAGE.shape[:2], bool)
    marked[40:65, 295:315] = True
    assert not observe.number_changed(a, b, PANEL, STAFFS, 18.0, marked)
    assert observe.number_changed(a, b, PANEL, STAFFS, 18.0, keep=marked)
    assert not observe.number_changed(a, b, PANEL, STAFFS, 18.0, keep=~marked)


def thin_digits(text: str, dx: int = 0, dy: int = 0) -> np.ndarray:
    """PAGE with black digits about 13 pixels tall and 1-2 pixels thick in the row of measure numbers."""
    f = PAGE.copy()
    cv2.putText(f, text, (300 + dx, 60 + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return f


def test_a_light_number_changing_beside_a_box_cursor_over_the_number_row_is_a_change():
    # A pale box cursor from the staff up into the number row moves; a light gray number elsewhere in the row changes
    a, b = PAGE.copy(), PAGE.copy()
    a[50:120, 100:140] = (150, 230, 250)
    b[50:120, 160:200] = (150, 230, 250)
    tint = np.zeros(PAGE.shape[:2], bool)
    tint[48:122, 98:202] = True
    assert not observe.number_changed(a, b, PANEL, STAFFS, 18.0, tint=tint)
    b[45:60, 600:610] = (182, 182, 182)
    assert observe.number_changed(a, b, PANEL, STAFFS, 18.0, tint=tint)


def test_a_colored_box_coming_up_from_the_staff_over_the_number_row_hides_the_number_under_it():
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
    # A colored line cursor 17 pixels wide, under the 18 of a line gap, leaves the number row under it compared
    tint[48:122, 278:502] = False
    tint[30:122, 296:313] = True
    assert observe.number_changed(*number_frames(300), PANEL, STAFFS, 18.0, tint=tint)


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


def two_pages(monkeypatch, first: np.ndarray, second: np.ndarray) -> list:
    """Views of two pages held 5 s each, in a segment that prints measure numbers."""
    def frames(path, fps, crop=None, start=0.0, length=None):
        for i in range(60):
            yield i / fps, (first if i < 30 else second).copy()
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    return observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 10.0,
                            prints_numbers=lambda *a: True)[0]


def black_mark(mask: np.ndarray, x: int = 300, y: int = 45) -> np.ndarray:
    """PAGE with the 15x10 mask drawn black at (x, y), in the row of measure numbers."""
    f = PAGE.copy()
    f[y:y + 15, x:x + 10][mask] = (0, 0, 0)
    return f


def test_a_black_number_changing_shape_in_place_tells_the_pages_apart(monkeypatch):
    # Two glyphs in one 10x15 box: bars across the top, middle and left, against bars across the top, bottom and right
    a = np.zeros((15, 10), bool)
    a[:, :3] = a[:3, :] = a[6:9, :] = True
    b = np.zeros((15, 10), bool)
    b[:, 7:] = b[:3, :] = b[12:, :] = True
    views = two_pages(monkeypatch, black_mark(a), black_mark(b))
    assert [round(v.t0, 2) for v in views] == [0.0, 5.0]


def test_a_solid_black_mark_shifted_one_pixel_is_no_new_page(monkeypatch):
    solid = np.ones((15, 10), bool)
    for x, y in ((301, 45), (300, 46)):
        views = two_pages(monkeypatch, black_mark(solid), black_mark(solid, x, y))
        assert [round(v.t0, 2) for v in views] == [0.0]


def test_count_in_screens_without_measures_leave_the_numbers_check_to_the_pages_after_them(monkeypatch):
    # Three count-in screens told apart by a black block in the staff, each with a light-gray count in the number row
    # that changes halfway, then two pages differing only in their light-gray measure numbers. A count-in screen is
    # marked by its top-left pixel, outside the panel
    def screen(k: int, second_half: bool) -> np.ndarray:
        f = number_frames(600 if second_half else 300, (170, 170, 170))[1].copy()
        f[100:120, 100 + 300 * k:140 + 300 * k] = 0
        f[0, 0] = (1, 2, 3)
        return f
    shown = [screen(k, half) for k in range(3) for half in (False, True)]
    pages = [number_frames(300, (170, 170, 170))[1], number_frames(600, (170, 170, 170))[1]]

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i in range(150):
            yield i / fps, (shown[i // 15] if i < 90 else pages[(i - 90) // 30]).copy()

    def prints_numbers(frame, *a):
        return None if frame[0, 0, 0] == 1 else True
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 25.0,
                             prints_numbers=prints_numbers)[0]
    assert [round(v.t0, 2) for v in views] == [0.0, 5.0, 10.0, 15.0, 20.0]


def test_a_thin_black_number_moved_up_to_two_pixels_is_no_new_page_and_one_moved_three_is(monkeypatch):
    for dx, dy in ((1, 0), (2, 0), (2, 1), (0, 2)):
        views = two_pages(monkeypatch, thin_digits("68"), thin_digits("68", dx, dy))
        assert [round(v.t0, 2) for v in views] == [0.0]
    views = two_pages(monkeypatch, thin_digits("68"), thin_digits("68", 3))
    assert [round(v.t0, 2) for v in views] == [0.0, 5.0]


def test_a_thin_black_number_changing_in_place_tells_the_pages_apart(monkeypatch):
    for a, b in (("3", "8"), ("12", "13"), ("68", "69"), ("9", "0"), ("1", "7"), ("7", "1"), ("17", "71")):
        views = two_pages(monkeypatch, thin_digits(a), thin_digits(b))
        assert [round(v.t0, 2) for v in views] == [0.0, 5.0]


def test_a_page_drifting_by_part_of_a_pixel_or_a_red_line_cursor_moving_is_no_new_page(monkeypatch):
    page = thin_digits("68")
    drifted = cv2.warpAffine(page, np.float32([[1, 0, 1.5], [0, 1, 0.5]]), (page.shape[1], page.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)
    assert [round(v.t0, 2) for v in two_pages(monkeypatch, page, drifted)] == [0.0]
    # A 3-pixel red cursor line from the number row down into the staff, moving 100 pixels
    first, second = page.copy(), page.copy()
    first[50:120, 500:503] = (40, 40, 200)
    second[50:120, 600:603] = (40, 40, 200)
    assert [round(v.t0, 2) for v in two_pages(monkeypatch, first, second)] == [0.0]


def test_without_a_numbers_check_the_number_row_is_not_compared(monkeypatch):
    first, second = number_frames(300, (170, 170, 170))[1], number_frames(600, (170, 170, 170))[1]

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i in range(60):
            yield i / fps, (first if i < 30 else second).copy()
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 10.0)[0]
    assert [round(v.t0, 2) for v in views] == [0.0]


def blotches(i: int, shape: tuple) -> np.ndarray:
    """Paper texture in blobs about 16 pixels across, different in every frame, at one gray level of spread."""
    small = np.random.default_rng(i).normal(0, 1, (shape[0] // 16 + 1, shape[1] // 16 + 1))
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)[:, :, None]


def dimmed_frames(pages: list[tuple[np.ndarray, int]]):
    """A fake `frames` showing each (page, paper level) in turn: the page scaled to that level, with textured paper."""
    def frames(path, fps, crop=None, start=0.0, length=None):
        for i, (page, level) in enumerate(pages):
            f = page.astype(np.float32) * level / 255 + 8 * blotches(i, page.shape)
            yield i / fps, np.clip(f, 0, 255).astype(np.uint8)
    return frames


def test_a_page_shown_during_a_fade_out_is_compared_at_the_paper_s_brightness(monkeypatch):
    # The third page comes in with its paper near the ink threshold and keeps darkening
    second = draw_measures(Song(notation=False, measures=8), 4, 4, 1280)
    pages = [(PAGE, 255)] * 12 + [(second, 255)] * 12 + [(PAGE, 145 - 2 * k) for k in range(12)]
    monkeypatch.setattr(observe, "frames", dimmed_frames(pages))
    observe.set_paper(PAPER)

    def spans():
        views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 6.0,
                                 thresholds=(140.0, 230.0))[0]
        return [(round(v.t0, 2), round(v.t1, 2)) for v in views]
    assert spans() == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
    monkeypatch.setattr(observe, "DIM_PAPER_DROP", 255)
    assert spans() == [(0.0, 2.0), (2.0, 4.0)]


def test_dim_frames_before_the_first_view_are_not_brightened(monkeypatch):
    # A fade-in: the first page shows dim for a second before it is shown in full
    second = draw_measures(Song(notation=False, measures=8), 4, 4, 1280)
    pages = [(PAGE, 145)] * 6 + [(PAGE, 255)] * 12 + [(second, 255)] * 12
    monkeypatch.setattr(observe, "frames", dimmed_frames(pages))
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 5.0,
                             thresholds=(140.0, 230.0))[0]
    assert [(round(v.t0, 2), round(v.t1, 2)) for v in views] == [(1.0, 3.0), (3.0, 5.0)]


def test_fading_frames_of_a_page_shown_bright_make_no_view_of_their_own(monkeypatch):
    # The second page darkens from paper level 200: brightened, its frames match the page, so they are no new page
    second = draw_measures(Song(notation=False, measures=8), 4, 4, 1280)
    pages = [(PAGE, 255)] * 12 + [(second, 255)] * 6 + [(second, 200 - 8 * k) for k in range(8)]
    monkeypatch.setattr(observe, "frames", dimmed_frames(pages))
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 26 / 6,
                             thresholds=(140.0, 230.0))[0]
    assert [(round(v.t0, 2), round(v.t1, 2)) for v in views] == [(0.0, 2.0), (2.0, 3.83)]


def test_a_fade_to_black_that_holds_briefly_is_no_screen(monkeypatch):
    # The last page fades to near black, where three frames of noise hold still for half a second
    second = draw_measures(Song(notation=False, measures=8), 4, 4, 1280)
    black = np.zeros_like(PAGE)
    pages = [(PAGE, 255)] * 12 + [(second, 255)] * 12 + [(black, 255)] * 3
    monkeypatch.setattr(observe, "frames", dimmed_frames(pages))
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 27 / 6,
                             thresholds=(140.0, 230.0), last_screen=True)[0]
    assert [(round(v.t0, 2), round(v.t1, 2)) for v in views] == [(0.0, 2.0), (2.0, 4.0)]


def test_a_page_faded_to_near_black_and_held_still_is_no_view(monkeypatch):
    # The last page fades to a paper level of 30 and holds for two seconds: brightened that far, it is noise
    second = draw_measures(Song(notation=False, measures=8), 4, 4, 1280)
    pages = [(PAGE, 255)] * 12 + [(second, 255)] * 12 + [(PAGE, 30)] * 12
    monkeypatch.setattr(observe, "frames", dimmed_frames(pages))
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 6.0,
                             thresholds=(140.0, 230.0), last_screen=True)[0]
    assert [(round(v.t0, 2), round(v.t1, 2)) for v in views] == [(0.0, 2.0), (2.0, 4.0)]


def test_a_page_shown_briefly_while_fading_out_is_the_brief_last_screen(monkeypatch):
    # The last page comes in with its paper at 150 and holds for half a second
    second = draw_measures(Song(notation=False, measures=8), 4, 4, 1280)
    pages = [(PAGE, 255)] * 12 + [(second, 255)] * 12 + [(PAGE, 150)] * 3
    monkeypatch.setattr(observe, "frames", dimmed_frames(pages))
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 27 / 6,
                             thresholds=(140.0, 230.0), last_screen=True)[0]
    assert [(round(v.t0, 2), round(v.t1, 2)) for v in views] == [(0.0, 2.0), (2.0, 4.0), (4.0, 4.5)]
    assert [v.brief for v in views] == [False, False, True]


def test_frames_whose_staffs_lie_outside_the_crop_are_skipped(monkeypatch):
    # The last half second's staffs are below the crop's 270 rows, as the next segment's staffs can be
    second = draw_measures(Song(notation=False, measures=8), 4, 4, 1280)
    shown = [PAGE] * 12 + [second] * 12 + [second] * 3

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i, f in enumerate(shown):
            yield i / fps, f.copy()
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER,
                             lambda t: STAFFS if t < 3.9 else (0, 300, 1280, 390), 0.0, 4.5)[0]
    assert [(round(v.t0, 2), round(v.t1, 2)) for v in views] == [(0.0, 2.0), (2.0, 4.0)]


def test_black_first_frames_are_not_a_screen(monkeypatch):
    # Half a second of black opens the segment: it holds as long as a first screen may, but shows no page
    second = draw_measures(Song(notation=False, measures=8), 4, 4, 1280)
    shown = [np.zeros_like(PAGE)] * 3 + [PAGE] * 12 + [second] * 12

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i, f in enumerate(shown):
            yield i / fps, f.copy()
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 27 / 6)[0]
    assert [(round(v.t0, 2), round(v.t1, 2)) for v in views] == [(0.5, 2.5), (2.5, 4.5)]


def test_the_ink_threshold_is_read_from_the_first_frame_that_shows_a_page(monkeypatch):
    # Gray ink: read from the black frames, the threshold would sit at 0.55 of the way from black to the paper
    gray = np.maximum(PAGE, 60)
    shown = [np.zeros_like(PAGE)] * 3 + [gray] * 12

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i, f in enumerate(shown):
            yield i / fps, f.copy()
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    views, thr, fthr = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 2.5)
    assert (round(thr, 2), round(fthr, 2)) == (167.25, 235.5)
    assert [(round(v.t0, 2), round(v.t1, 2)) for v in views] == [(0.5, 2.5)]


def test_a_page_fading_in_from_near_black_sets_the_ink_threshold_once_it_is_bright(monkeypatch):
    # The first three frames show the page at a paper level of 30: too dark to read ink from or to be a screen
    gray = np.maximum(PAGE, 60)
    monkeypatch.setattr(observe, "frames", dimmed_frames([(gray, 30)] * 3 + [(gray, 255)] * 12))
    observe.set_paper(PAPER)
    views, thr, fthr = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 2.5)
    assert (round(thr, 2), round(fthr, 2)) == (162.75, 234.5)
    assert [(round(v.t0, 2), round(v.t1, 2)) for v in views] == [(0.5, 2.5)]


def cut_line_cursor() -> tuple[np.ndarray, np.ndarray]:
    """Ink maps of a still page at a 20-pixel line gap where a 6-pixel line cursor moves 100 pixels: staff lines every
    20 rows, 2 rows thick, cut each sighting into pieces 18 rows tall."""
    a, b = np.zeros((200, 400), bool), np.zeros((200, 400), bool)
    for top in range(20, 180, 20):
        a[top + 2:top + 20, 100:106] = True
        b[top + 2:top + 20, 200:206] = True
    return a, b


def test_a_line_cursor_cut_up_by_staff_lines_is_no_change():
    a, b = cut_line_cursor()
    assert observe.significant_change(a, b, 20.0, 2)
    assert not observe.significant_change(a, b, 20.0, 2, cursor=True)


def test_staff_line_pieces_a_box_cursor_breaks_off_are_no_change():
    # Pieces 4 rows tall and 30 wide come and go where the box sat
    a, b = np.zeros((200, 400), bool), np.zeros((200, 400), bool)
    for y in (40, 60, 80, 100):
        a[y:y + 4, 100:130] = True
        b[y:y + 4, 200:230] = True
    assert observe.significant_change(a, b, 20.0, 2)
    assert not observe.significant_change(a, b, 20.0, 2, cursor=True)
    # 5 rows tall, as thick as a beam, is a mark
    a[120:125, 100:130] = True
    assert observe.significant_change(a, b, 20.0, 2, cursor=True)


def test_a_line_cursor_cut_where_it_passes_over_a_digit_is_no_change():
    a, b = cut_line_cursor()
    # A digit shown on both pages hides 12 rows of the cursor's new sighting
    a[60:74, 196:210] = b[60:74, 196:210] = True
    b[58:76, 200:206] = False
    assert not observe.significant_change(a, b, 20.0, 2, cursor=True)


def test_a_chord_of_narrow_digits_stacked_on_neighbouring_strings_is_a_change():
    # Three 1s, 7 pixels wide and 14 tall, on lines 20 rows apart, with blank paper between them
    a, b = np.zeros((200, 400), bool), np.zeros((200, 400), bool)
    for k in range(3):
        b[33 + 20 * k:47 + 20 * k, 200:207] = True
    assert observe.significant_change(a, b, 20.0, 2, cursor=True)
    b[73:87] = False             # two of them
    assert observe.significant_change(a, b, 20.0, 2, cursor=True)


def test_a_digit_changing_beside_a_moving_line_cursor_is_a_change():
    a, b = cut_line_cursor()
    b[60:74, 300:310] = True
    assert observe.significant_change(a, b, 20.0, 2, cursor=True)


def footage_tail(paper_to: int) -> tuple[int, int, int, int]:
    """panel_rect of a white page whose paper ends at x `paper_to`, with dark footage after it, for staffs whose box
    runs on to x 95 (line gap 10)."""
    frame = np.full((100, 100, 3), 255, np.uint8)
    frame[:, paper_to:] = 30
    return observe.panel_rect(frame, PAPER.astype(np.float64), (10, 40, 95, 60), 10.0)


def test_a_panel_ending_one_to_two_line_gaps_into_footage_is_cut_back_to_the_paper():
    # 15 px of footage, 1.5 gaps
    assert footage_tail(80) == (0, 15, 80, 90)


def test_a_panel_ending_under_a_line_gap_or_over_two_into_footage_keeps_its_edge():
    # 5 px and 25 px of footage
    assert footage_tail(90)[2] == 95
    assert footage_tail(70)[2] == 95


def swept_frames(n: int, mark):
    """A fake `frames`: the page held for n frames with the cursor sweeping x 20 to 1100 once, and `mark(i, f)`
    drawing on frame i."""
    def frames(path, fps, crop=None, start=0.0, length=None):
        for i in range(n):
            f = PAGE.copy()
            mark(i, f)
            x = 20 + int(1080 * i / n)
            f[:, x:x + 8] = (f[:, x:x + 8] * 0.5 + np.array(SONG.cursor) * 0.5).astype(np.uint8)
            yield i / fps, f
    return frames


def cursor_xs(monkeypatch, mark) -> list[int]:
    """Playhead x of each frame of the sweep, against where the cursor is drawn."""
    monkeypatch.setattr(observe, "frames", swept_frames(90, mark))
    observe.set_paper(PAPER)
    views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 15.0)[0]
    assert len(views) == 1
    return [round(x - (24 + 1080 * i / 90)) for i, (_, x, _) in enumerate(views[0].playhead)]


def test_the_playhead_is_not_looked_for_on_a_colored_strip_that_changes_while_the_page_holds(monkeypatch):
    def strip(i, f):
        # A blue strip whose dark rows move down 3 px halfway through, as when it starts showing the next page
        f[:, 1150:1270] = (200, 90, 20)
        for y in range(70 + (3 if i >= 45 else 0), 160, 6):
            f[y:y + 3, 1160:1260] = 30
    off = cursor_xs(monkeypatch, strip)
    assert len(off) == 90 and max(abs(d) for d in off) <= 2


def test_the_playhead_is_found_crossing_a_thin_colored_line_on_the_page(monkeypatch):
    def line(i, f):
        f[:, 600:603] = (200, 90, 20)
    off = cursor_xs(monkeypatch, line)
    assert len(off) == 90 and max(abs(d) for d in off) <= 2


def strip_ink(gray_digit: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tint, area, ink) for a 250x600 strip at line gap 50: a gray barline at x 300 cut into pieces by the staff
    lines, each with a stub of staff line across its middle, gray staff-line pieces, and five colored digits drawn as
    24x36 outlines 7 pixels thick, the last one gray with `gray_digit`."""
    area = np.ones((250, 600), bool)
    ink = np.zeros_like(area)
    tint = np.zeros_like(area)
    for y in (50, 100, 150, 200):
        ink[y + 3:y + 47, 300:304] = True
        ink[y + 20:y + 22, 292:312] = True
        ink[y:y + 2, 200:280] = True
    for k, x in enumerate((60, 160, 400, 480, 540)):
        ink[110:146, x:x + 24] = True
        ink[117:139, x + 7:x + 17] = False
        if not (gray_digit and k == 4):
            tint[107:149, x - 3:x + 27] = True
    return tint, area, ink


def test_colored_digits_on_a_strip_whose_gray_ink_is_only_lines_are_the_ink_s_own_color():
    tint, area, ink = strip_ink(gray_digit=False)
    assert (tint & ink).sum() < observe.TINT_INK * ink.sum()
    assert not observe.tinted_paper(tint, area, ink)
    assert observe.tinted_paper(tint, area, ink, 50.0)


def test_a_gray_digit_among_colored_ones_keeps_them_a_highlight():
    tint, area, ink = strip_ink(gray_digit=True)
    assert not observe.tinted_paper(tint, area, ink, 50.0)


def test_colored_digits_in_the_frame_count_when_the_reference_shows_only_lines():
    tint, area, ink = strip_ink(gray_digit=False)
    lines = ink.copy()
    lines[110:146] = False
    assert not observe.tinted_paper(tint, area, lines, 50.0)
    assert observe.tinted_paper(tint, area, lines, 50.0, ink)
    tint, area, ink = strip_ink(gray_digit=True)
    assert not observe.tinted_paper(tint, area, lines, 50.0, ink)


def test_a_colored_line_cursor_over_an_empty_staff_is_no_colored_ink():
    # A cursor 8 wide at x 100, cut by the staff lines every 50 rows, over a staff of gray line pieces only
    area = np.ones((250, 600), bool)
    ink = np.zeros_like(area)
    tint = np.zeros_like(area)
    for y in (50, 100, 150):
        ink[y:y + 2, 200:280] = True
        ink[y + 2:y + 50, 100:108] = True
    tint[45:205, 95:113] = True
    assert not observe.tinted_paper(tint, area, ink, 50.0)


STRIP_STAFFS = (0, 60, 1280, 212)
RED = (0, 0, 230)


def strip_page(digits: list[tuple[int, int, int]], color=RED, mark: bool = False) -> np.ndarray:
    """A 270x1280 page at line gap 50: four black staff lines from row 60, and solid digits (x, width, height) in
    `color` between the second and third lines. With `mark`, a black 30x40 block at x 1200 is gray ink beside them."""
    f = np.full((270, 1280, 3), 255, np.uint8)
    for y in (60, 110, 160, 210):
        f[y:y + 2] = 0
    for x, w, h in digits:
        f[120:120 + h, x:x + w] = color
    if mark:
        f[140:180, 1200:1230] = 0
    return f


def strip_views(monkeypatch, pages: list[np.ndarray], strip: bool | None = None) -> list:
    """Views of the pages held 3 s each, read as a strip of colored notes or not, or left to views_in to judge."""
    def frames(path, fps, crop=None, start=0.0, length=None):
        for i in range(18 * len(pages)):
            yield i / fps, pages[i // 18].copy()
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    return observe.views_in(None, (0, 0, 1280, 270), "dark", 50.0, PAPER, lambda t: STRIP_STAFFS, 0.0, 3.0 * len(pages),
                            strip=strip)[0]


# Six 16x24 digits, then the same with the fourth moved, then two 10x14 digits added, then the first moved
BIG = [(100, 16, 24), (250, 16, 24), (400, 16, 24), (550, 16, 24), (700, 16, 24), (850, 16, 24)]
MOVED = BIG[:3] + [(600, 16, 24)] + BIG[4:]
SMALL = MOVED + [(950, 10, 14), (1050, 10, 14)]
LAST = [(150, 16, 24)] + SMALL[1:]


def test_a_digit_piece_smaller_than_a_mark_is_a_change_only_at_the_digit_size():
    a, b = np.zeros((250, 600), bool), np.zeros((250, 600), bool)
    b[100:114, 300:310] = True   # 140 pixels: 0.056 of a 50-pixel line gap squared
    assert not observe.significant_change(a, b, 50.0, 2, cursor=True)
    assert observe.significant_change(a, b, 50.0, 2, cursor=True, piece=observe.DIGIT_PIECE)


def test_views_of_colored_digits_are_a_strip_and_of_black_digits_are_not(monkeypatch):
    red = strip_views(monkeypatch, [strip_page(d) for d in (BIG, MOVED, LAST)], strip=False)
    assert len(red) == 3 and observe.colored_strip(red, "dark", 128.0, 50.0)
    black = strip_views(monkeypatch, [strip_page(d, (0, 0, 0)) for d in (BIG, MOVED, LAST)], strip=False)
    assert len(black) == 3 and not observe.colored_strip(black, "dark", 128.0, 50.0)
    # A cursor box colored over black digits is no strip either
    boxed = [strip_page(d, (0, 0, 0)) for d in (BIG, MOVED, LAST)]
    for f in boxed:
        f[95:130, 90:130] = (230, 180, 0)
    assert not observe.colored_strip(strip_views(monkeypatch, boxed, strip=False), "dark", 128.0, 50.0)
    # Two views are too few to judge
    assert not observe.colored_strip(red[:2], "dark", 128.0, 50.0)


def test_a_strip_page_flip_adding_only_small_digits_starts_a_new_view(monkeypatch):
    pages = [strip_page(d) for d in (BIG, MOVED, SMALL, LAST)]
    assert [round(v.t0, 1) for v in strip_views(monkeypatch, pages, strip=False)] == [0.0, 3.0, 9.0]
    # Read as no strip first, the segment's views show colored digits, so it is read again as a strip
    assert [round(v.t0, 1) for v in strip_views(monkeypatch, pages)] == [0.0, 3.0, 6.0, 9.0]


def test_small_black_digits_added_off_a_strip_are_no_change(monkeypatch):
    pages = [strip_page(d, (0, 0, 0)) for d in (BIG, MOVED, SMALL, LAST)]
    assert [round(v.t0, 1) for v in strip_views(monkeypatch, pages)] == [0.0, 3.0, 9.0]


def test_colored_digits_flipping_beside_gray_ink_are_compared_on_a_strip(monkeypatch):
    # Two digits and a black block: under TINT_INK of the marks are colored, so off a strip the digits are masked
    two = [(100, 16, 24), (250, 16, 24)]
    pages = [strip_page(two, mark=True), strip_page([(400, 16, 24), (550, 16, 24)], mark=True)]
    assert [round(v.t0, 1) for v in strip_views(monkeypatch, pages, strip=False)] == [0.0]
    assert [round(v.t0, 1) for v in strip_views(monkeypatch, pages, strip=True)] == [0.0, 3.0]


def mid_jump_sweeps(n: int, sweep: int, mid: int):
    """A fake `frames`: one page held for n frames, its cursor crossing x 20 to 1220 every `sweep` frames, and the
    first frame of each later pass showing the cursor at x `mid`, caught partway through its jump back."""
    swept = held_page(n, sweep, [])

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i, (t, f) in enumerate(swept(path, fps)):
            if i and i % sweep == 0:
                f = PAGE.copy()
                f[:, mid:mid + 8] = (f[:, mid:mid + 8] * 0.5 + np.array(SONG.cursor) * 0.5).astype(np.uint8)
            yield t, f
    return frames


def test_a_cursor_caught_partway_through_its_jump_back_still_starts_a_new_view(monkeypatch):
    monkeypatch.setattr(observe, "frames", mid_jump_sweeps(120, 60, 800))
    observe.set_paper(PAPER)

    def starts():
        views = observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS, 0.0, 20.0)[0]
        return [(round(v.t0, 2), v.wrapped) for v in views]
    # The page starts over at the first sample past the one caught mid-jump
    assert starts() == [(0.0, False), (10.17, True)]
    monkeypatch.setattr(observe, "JUMP_PART_GAPS", 50.0)
    assert starts() == [(0.0, False)]


def test_only_a_lone_sample_far_from_both_sides_of_a_jump_is_left_out():
    ph = [(0.0, 1200.0, 100.0), (1 / 6, 800.0, 100.0), (2 / 6, 40.0, 100.0)]
    assert observe.without_jump_parts(ph, 18.0) == [ph[0], ph[2]]
    # Under 3 line gaps from the sample before it
    near = [(0.0, 1200.0, 100.0), (1 / 6, 1160.0, 100.0), (2 / 6, 40.0, 100.0)]
    assert observe.without_jump_parts(near, 18.0) == near
    # Half a second after the sample before it
    late = [(0.0, 1200.0, 100.0), (0.5, 800.0, 100.0), (4 / 6, 40.0, 100.0)]
    assert observe.without_jump_parts(late, 18.0) == late


def wrap_pair(second_plate: np.ndarray, xs: list[float], gap_s: float = 0.0):
    """Two views of 1280 px wide panels, the second starting `gap_s` after the first ends with playhead samples at
    `xs`, one every sixth of a second."""
    p = observe.View(0.0, 5.0, PAGE.copy(), rect=(0, 0, 1280, 270))
    v = observe.View(5.0 + gap_s, 10.0, second_plate, rect=(0, 0, 1280, 270))
    v.playhead = [(v.t0 + k / 6, x, 100.0) for k, x in enumerate(xs)]
    v.times = [t for t, _, _ in v.playhead]
    return p, v


def test_a_view_of_the_same_picture_opening_with_a_jump_back_starts_at_the_jump(monkeypatch):
    p, v = wrap_pair(PAGE.copy(), [1180, 1220, 60, 90, 120])
    observe.start_at_wraps([p, v], 18.0)
    assert (round(v.t0, 2), v.wrapped, len(v.playhead), len(v.times)) == (5.33, True, 3, 3)
    p, v = wrap_pair(PAGE.copy(), [1180, 1220, 60, 90, 120])
    monkeypatch.setattr(observe, "WRAP_LEAD_SAMPLES", 1)
    observe.start_at_wraps([p, v], 18.0)
    assert (v.t0, v.wrapped) == (5.0, False)


def test_a_view_opening_with_a_jump_back_keeps_its_start_after_another_picture_a_break_or_a_late_jump():
    other = PAGE.copy()
    other[:, :100] = 0
    for p, v in (wrap_pair(other, [1180, 1220, 60, 90]), wrap_pair(PAGE.copy(), [1180, 1220, 60, 90], gap_s=0.5),
                 wrap_pair(PAGE.copy(), [1100, 1150, 1220, 60, 90]), wrap_pair(PAGE.copy(), [1180, 1220, 600, 650])):
        observe.start_at_wraps([p, v], 18.0)
        assert not v.wrapped and v.t0 in (5.0, 5.5)

