import numpy as np

from youtabber import track
from youtabber.track import Move, pairs, transition

CROP = (0, 0, 1000, 120)


def strip(seed: int = 0, bar0: int = 0) -> np.ndarray:
    """A 3000 px white strip with 2 px barlines every 250 px and 250 random 7 x 9 px blobs."""
    s = np.full((120, 3000, 3), 255, np.uint8)
    for x in range(bar0, 3000, 250):
        s[20:100, x:x + 2] = 0
    rng = np.random.default_rng(seed)
    for _ in range(250):
        x, y = rng.integers(0, 3000 - 7), rng.integers(25, 95 - 9)
        s[y:y + 9, x:x + 7] = 0
    return s


def fake_frames(monkeypatch, offsets, strips=None):
    """Make track.frames yield one 1000 px window per offset, from strips[i] (default: one strip throughout)."""
    a = strip()
    strips = strips or [a] * len(offsets)

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i, (x, s) in enumerate(zip(offsets, strips)):
            yield start + i / fps, s[:, x:x + 1000].copy()
    monkeypatch.setattr(track, "frames", frames)


def run(t1=1.2):
    return transition(None, CROP, (0, 120), (0, 1000), "dark", 140.0, 14.0, 1.0, t1)


def test_still(monkeypatch):
    fake_frames(monkeypatch, [0] * 6)
    assert run() == Move("still", steps=5)


def test_drawn_slide(monkeypatch):
    fake_frames(monkeypatch, [0, 0, 100, 250, 400, 400])
    m = run()
    assert m.kind == "slide" and m.shifts[0] == 400 and m.drawn[0] and m.steps == 5


def test_one_frame_jump(monkeypatch):
    fake_frames(monkeypatch, [0, 0, 300, 300])
    m = run()
    assert m.shifts[0] == 300 and not m.drawn[0]


def test_wrap_round(monkeypatch):
    # The 600 px step is longer than half the window, so phase correlation reads it as a 400 px step back
    fake_frames(monkeypatch, [0, 0, 100, 700, 700])
    m = run()
    assert m.shifts == [-300, 700] and m.drawn == [False, True]


def flip(monkeypatch, bar0):
    """Two frames of one page, then two of another page whose barlines start at bar0. Both pages carry the same
    clef-like mark at the system start: barlines alone repeat every 250 px and match at any multiple of it."""
    a, b = strip(), strip(seed=1, bar0=bar0)
    for s in (a, b):
        s[30:90, 10:16] = 0
        s[40:80, 16:24] = 0
    fake_frames(monkeypatch, [0] * 4, [a, a, b, b])


def test_flip_same_grid_is_still(monkeypatch):
    # The fixed marks register at no shift, so the motion alone can't see this flip
    flip(monkeypatch, 0)
    assert run().kind == "still"


def test_flip_shifted_grid_is_cut(monkeypatch):
    flip(monkeypatch, 125)
    assert run().kind == "cut"


def test_long_gap_or_no_frames_is_unknown(monkeypatch):
    fake_frames(monkeypatch, [0] * 4)
    assert run(t1=4.5).kind == "unknown"
    fake_frames(monkeypatch, [])
    assert run().kind == "unknown"


def test_pairs_by_key():
    v = [object() for _ in range(4)]
    keys = {id(x): k for x, k in zip(v, [0, 1, 0, 1])}
    assert pairs(v, lambda x: keys[id(x)]) == [(v[0], v[2]), (v[1], v[3])]
