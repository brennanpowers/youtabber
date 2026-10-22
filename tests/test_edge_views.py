"""A segment's first and last screens may be shown only briefly, and a brief screen that is a page fading in or
out gives no measures."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_numbering import music, run, view

from synth import Song, draw_measures
from youtabber import glide, observe, pipeline
from youtabber.assemble import Obs, assemble
from youtabber.observe import View
from youtabber.parse import Cell
from youtabber.pipeline import drop_fades, lead_before
from youtabber.scan import Segment
from youtabber.video import VideoInfo

PAPER = np.array([255, 255, 255], np.uint8)
SONG = Song(notation=False, measures=20)
PAGES = [draw_measures(SONG, 4 * k, 4, 1280) for k in range(5)]
STAFFS = (0, 70, 1280, 160)


def shown(runs):
    """A fake `frames` that honors start and length: `runs` lists (page, frames) shown one after another at
    observe.OBS_FPS from time 0."""
    timeline = [PAGES[page] for page, count in runs for _ in range(count)]

    def frames(path, fps, crop=None, start=0.0, length=None):
        for i, page in enumerate(timeline):
            t = i / fps
            if t >= start - 1e-9 and (length is None or t < start + length - 1e-9):
                yield t, page.copy()
    return frames, len(timeline) / observe.OBS_FPS


def read(monkeypatch, runs, t_start=0.0, lead=0.0):
    frames, end = shown(runs)
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    return observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS,
                            t_start, end, lead=lead)[0]


def test_first_and_last_screens_need_half_a_second_and_others_the_usual_hold(monkeypatch):
    # 4 frames (0.67 s) first, 12, then 6 frames (1.0 s) in the middle, 12, and 4 frames last
    views = read(monkeypatch, [(0, 4), (1, 12), (2, 6), (3, 12), (4, 4)])
    assert [(round(v.t0, 2), v.brief) for v in views] == [(0.0, True), (0.67, False), (3.67, False), (5.67, True)]


def test_a_first_screen_of_exactly_half_a_second_is_kept_at_any_start_time(monkeypatch):
    # 3 frames at 6 fps are 0.5 s; frame times from 4.5 s sum to just under 0.5 without rounding
    runs = [(0, 27), (1, 3), (2, 12)]
    views = read(monkeypatch, runs, t_start=4.5)
    assert [(round(v.t0, 2), v.brief) for v in views] == [(4.5, True), (5.0, False)]
    # Two frames are too few
    views = read(monkeypatch, [(0, 27), (1, 2), (2, 12)], t_start=4.5)
    assert [(round(v.t0, 2), v.brief) for v in views] == [(4.83, False)]


def test_a_brief_screen_just_before_the_segment_start_is_read_in_its_own_pass(monkeypatch):
    # The scan first saw the staff at 13.5 s; page 1 showed from 13.0 s, page 2 from 13.5 s
    runs = [(0, 78), (1, 3), (2, 12)]
    views = read(monkeypatch, runs, t_start=13.5, lead=0.5)
    assert [(round(v.t0, 2), round(v.t1, 2), v.brief) for v in views] == [(13.0, 13.5, True), (13.5, 15.5, False)]
    assert np.array_equal(views[0].plate, PAGES[1])
    # With no lead, the brief screen is not looked for
    views = read(monkeypatch, runs, t_start=13.5)
    assert [round(v.t0, 2) for v in views] == [13.5]


def test_a_first_screen_shown_briefly_after_the_segment_start_is_held_the_usual_time(monkeypatch):
    # Page 2 holds 4 frames (0.67 s) from 0.67 s, more than half a second into the segment, so it needs the usual hold
    views = read(monkeypatch, [(0, 2), (1, 2), (2, 4), (3, 12)])
    assert [(round(v.t0, 2), v.brief) for v in views] == [(1.33, False)]


def test_the_lead_is_at_most_one_scan_step_and_never_reaches_into_the_segment_before():
    seg = lambda t0, t1: SimpleNamespace(t0=t0, t1=t1)
    assert lead_before([seg(13.5, 60.0)], 0) == 0.5
    assert lead_before([seg(0.2, 60.0)], 0) == pytest.approx(0.2)
    assert lead_before([seg(0.0, 60.0)], 0) == 0.0
    assert lead_before([seg(0.0, 20.0), seg(30.0, 60.0)], 1) == 0.5
    assert lead_before([seg(0.0, 29.8), seg(30.0, 60.0)], 1) == pytest.approx(0.2)
    assert lead_before([seg(0.0, 30.2), seg(30.0, 60.0)], 1) == 0.0


def screen(index, patterns, brief=False):
    """A still view of one part whose measures hold the music of `patterns`; pattern None is a measure cut by the
    panel's right edge."""
    cells = []
    for i, p in enumerate(patterns):
        ink = music("a", 0 if p is None else p)
        cells.append(Cell(index, 0, i, 200 * i, 200 * i + 200, "tail" if p is None else "full", ink,
                          np.zeros_like(ink), 0, gap=10.0, mid_y=30.0, tint=np.zeros_like(ink)))
    o = Obs(index, 0.0, 1.0, cells, [], 0, (0, 0))
    o.all_cells, o.brief = cells, brief
    return o


def test_a_brief_screen_with_fewer_than_two_whole_measures_gives_none():
    first, second = screen(0, [1, None], brief=True), screen(1, [5, 6, 7])
    drop_fades([first, second])
    assert first.all_cells == [] and len(second.all_cells) == 3
    # The same for a last screen
    before, last = screen(0, [5, 6, 7]), screen(1, [1, None], brief=True)
    drop_fades([before, last])
    assert last.all_cells == [] and len(before.all_cells) == 3
    # Two whole measures of other music are kept
    first, second = screen(0, [1, 2, None], brief=True), screen(1, [5, 6, 7])
    drop_fades([first, second])
    assert len(first.all_cells) == 3


def test_a_brief_screen_its_neighbour_shows_again_in_order_gives_none():
    # A last screen, briefly shown, whose measures 2 and 3 the screen before shows side by side
    before, last = screen(0, [1, 2, 3, 4]), screen(1, [2, 3, None], brief=True)
    drop_fades([before, last])
    assert last.all_cells == [] and len(before.all_cells) == 4
    # A first screen caught while the page draws in, its measures shown again on the screen after it
    first, after = screen(0, [2, 3], brief=True), screen(1, [1, 2, 3, 4])
    drop_fades([first, after])
    assert first.all_cells == [] and len(after.all_cells) == 4
    # The same measures out of order are other music
    before, last = screen(0, [1, 3, 2, 4]), screen(1, [2, 3, None], brief=True)
    drop_fades([before, last])
    assert len(last.all_cells) == 3
    # A screen held long enough keeps its measures, shown again or not
    before, after = screen(0, [1, 2, 3, 4]), screen(1, [2, 3])
    drop_fades([before, after])
    assert len(after.all_cells) == 2


def test_a_brief_screen_that_parses_no_measures_is_not_read(monkeypatch):
    seg = Segment(0.0, 10.0, 10.0, "dark", paper=np.array([255.0, 255.0, 255.0]), x1=640, y1=360)
    plate = np.full((360, 640, 3), 255, np.uint8)
    views = [View(0.0, 0.5, plate, rect=(0, 0, 640, 360), print_rect=(0, 0, 640, 360), brief=True),
             View(0.5, 9.0, plate, rect=(0, 0, 640, 360), print_rect=(0, 0, 640, 360)),
             View(9.0, 9.5, plate, rect=(0, 0, 640, 360), print_rect=(0, 0, 640, 360), brief=True)]
    # The middle screen and the last one show measures; the brief first screen shows none
    measures = {0.0: [], 0.5: [5, 6, 7], 9.0: [8, 9]}

    def plate_obs(v, seg, si, thr, fthr, index):
        o = screen(index, measures[v.t0], brief=v.brief)
        o.t0, o.t1, o.systems = v.t0, v.t1, []
        return o
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(640, 360, 10.0))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None, progress=None: ([seg], {}, []))
    monkeypatch.setattr(glide, "glides", lambda *a: False)
    monkeypatch.setattr(pipeline, "views_in", lambda *a, **kw: (views, 128.0, 230.0))
    monkeypatch.setattr(pipeline, "plate_obs", plate_obs)
    monkeypatch.setattr(pipeline, "view_move", lambda *a: None)
    monkeypatch.setattr(pipeline, "finish", lambda path, info, r, *a: r)
    r = pipeline.read(Path("unused.mp4"))
    assert [(o.index, o.t0) for o in r.views] == [(0, 0.5), (1, 9.0)]


def test_a_brief_screen_left_with_no_measures_does_not_hide_a_break_before_a_restart():
    def restart_fold(emptied_screen):
        obs = [view(0, run("a", 1, 4)), view(1, run("a", 5, 8))]
        if emptied_screen:
            brief = Obs(2, 24.5, 25.0, [], [], 1, (0, 0))
            brief.brief = True
            obs.append(brief)
        again = view(len(obs), run("a", 1, 4), seg=1)
        again.t0, again.t1 = 25.0, 35.0
        obs.append(again)
        # Unnumbered measures, so only the break before the opening measures says the song started over
        for o in obs:
            for c in o.cells:
                c.number = None
        A = assemble(obs, 10.0, False, set(), lambda o: [])
        return [(d.chosen, d.basis) for d in A.decisions if d.kind == "fold"]
    assert restart_fold(False) == [("fold", "restart")]
    # A page fading in just before the restart, emptied of its measures, leaves the break as long as it was
    assert restart_fold(True) == [("fold", "restart")]
