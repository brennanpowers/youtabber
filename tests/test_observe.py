"""Still views from a fake frame stream: a long-held page keeps a bounded number of frames in memory."""
import weakref

import numpy as np

from synth import Song, draw_measures
from youtabber import observe

PAPER = np.array([255, 255, 255], np.uint8)
SONG = Song(notation=False, measures=4)
PAGE = draw_measures(SONG, 0, 4, 1280)
STAFFS = (0, 70, 1280, 160)


def held_page(n: int, sweep: int, alive: list[int]):
    """A fake `frames`: one page held for n frames, with a cursor crossing it every `sweep` frames.

    At each new frame, appends to `alive` how many of the frames handed out so far are still in memory.
    """
    handed = []

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i in range(n):
            f = PAGE.copy()
            x = 20 + int(1200 * (i % sweep) / sweep)
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
