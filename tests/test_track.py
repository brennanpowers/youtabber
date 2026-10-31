from types import SimpleNamespace

import numpy as np

from youtabber import pipeline, track
from youtabber.observe import View
from youtabber.scan import Segment
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
    # The 600 px step is longer than half the window, so phase correlation reads it as a 400 px step back; the
    # reading with both steps one way is drawn and comes first, though the raw sum of the steps is -300
    fake_frames(monkeypatch, [0, 0, 100, 700, 700])
    m = run()
    assert m.shifts == [700, -300] and m.drawn == [True, False]


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


def test_a_view_move_is_measured_on_rows_near_the_staffs(monkeypatch):
    # A band reaching 7 line gaps above the staff (into a title banner) is measured from 3.5 gaps above it
    seen = []
    monkeypatch.setattr(track, "transition", lambda path, crop, rows, cols, *a: seen.append((rows, cols)))
    tab = SimpleNamespace(gap=20.0)

    def view(band_top, top=200.0):
        return SimpleNamespace(systems=[SimpleNamespace(band=(band_top, 330), system=SimpleNamespace(top=top, tab=tab))],
                               rect=(0, 40, 1000, 340), t0=1.0, t1=1.5)
    seg = SimpleNamespace(polarity="dark", gap=20.0)
    pipeline.view_move(None, CROP, seg, view(60), view(140), 140.0, (0, 400))
    pipeline.view_move(None, CROP, seg, view(160), view(170), 140.0, (0, 400))
    pipeline.view_move(None, CROP, seg, view(60), view(60), 140.0, (140, 400))
    assert seen == [((130, 330), (0, 1000)), ((160, 330), (0, 1000)), ((140, 330), (0, 1000))]


def test_registration_leaves_out_marks_high_above_the_staffs(monkeypatch):
    # The staff's top line is at row 200 with a 20 px gap: marks at rows 100-120 are a banner, 140-160 a chord name
    system = SimpleNamespace(top=200.0, tab=SimpleNamespace(gap=20.0, top=200.0, bottom=300.0))
    monkeypatch.setattr(pipeline, "parse_view", lambda *a, **kw: [SimpleNamespace(
        system=system, cells=[], repeat_end=False, repeat_bars=[])])
    monkeypatch.setattr(pipeline, "line_darkness", lambda *a: (100.0, 250.0))
    plate = np.full((360, 640, 3), 255, np.uint8)
    plate[100:120, 100:110] = 0
    plate[140:160, 300:310] = 0
    seg = Segment(t0=0, t1=10, gap=20, polarity="dark", y0=0, y1=360, x0=0, x1=640)
    rect = (0, 0, 640, 360)
    o = pipeline.plate_obs(View(0.0, 5.0, plate, rect=rect, print_rect=rect), seg, 0, 128.0, 230.0, 0)
    assert o.band[100:120, 100:110].all() and not o.tall[100:120, 100:110].any()
    assert o.tall[140:160, 300:310].all()


def test_registration_leaves_out_a_caption_under_the_staff_but_keeps_it_in_the_whole_marks(monkeypatch):
    # The staff's bottom line is at row 300 with a 20 px gap: marks at rows 300-310 are digits on it, 330-345 a
    # caption
    system = SimpleNamespace(top=200.0, tab=SimpleNamespace(gap=20.0, top=200.0, bottom=300.0))
    monkeypatch.setattr(pipeline, "parse_view", lambda *a, **kw: [SimpleNamespace(
        system=system, cells=[], repeat_end=False, repeat_bars=[])])
    monkeypatch.setattr(pipeline, "line_darkness", lambda *a: (100.0, 250.0))
    plate = np.full((360, 640, 3), 255, np.uint8)
    plate[300:310, 100:110] = 0
    for x in range(300, 400, 12):
        plate[330:345, x:x + 6] = 0
    seg = Segment(t0=0, t1=10, gap=20, polarity="dark", y0=0, y1=360, x0=0, x1=640)
    rect = (0, 0, 640, 360)
    o = pipeline.plate_obs(View(0.0, 5.0, plate, rect=rect, print_rect=rect), seg, 0, 128.0, 230.0, 0)
    assert o.tall[300:310, 100:110].all() and not o.tall[330:, :].any()
    assert o.whole_tall[330:345, 300:306].all() and o.whole_tall[330:345, 396:402].all()


def test_a_view_with_no_staff_found_registers_on_all_its_marks(monkeypatch):
    monkeypatch.setattr(pipeline, "parse_view", lambda *a, **kw: [])
    plate = np.full((360, 640, 3), 255, np.uint8)
    for x in range(300, 400, 12):
        plate[330:345, x:x + 6] = 0
    seg = Segment(t0=0, t1=10, gap=20, polarity="dark", y0=0, y1=360, x0=0, x1=640)
    rect = (0, 0, 640, 360)
    o = pipeline.plate_obs(View(0.0, 5.0, plate, rect=rect, print_rect=rect), seg, 0, 128.0, 230.0, 0)
    assert o.tall[330:345, 300:306].all() and o.whole_tall[330:345, 300:306].all()
