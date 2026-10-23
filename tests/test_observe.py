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
