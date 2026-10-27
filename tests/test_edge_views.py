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
from youtabber.observe import View, ink_threshold, to_gray
from youtabber.parse import Cell
from youtabber.pipeline import drop_fades, last_screen_notice, lead_before, mark_fades, same_barlines
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


def read(monkeypatch, runs, t_start=0.0, lead=0.0, last_screen=False):
    frames, end = shown(runs)
    monkeypatch.setattr(observe, "frames", frames)
    observe.set_paper(PAPER)
    return observe.views_in(None, (0, 0, 1280, 270), "dark", 18.0, PAPER, lambda t: STAFFS,
                            t_start, end, lead=lead, last_screen=last_screen)[0]


def test_first_and_last_screens_need_half_a_second_and_others_the_usual_hold(monkeypatch):
    # 4 frames (0.67 s) first, 12, then 6 frames (1.0 s) in the middle, 12, and 4 frames last
    views = read(monkeypatch, [(0, 4), (1, 12), (2, 6), (3, 12), (4, 4)])
    assert [(round(v.t0, 2), v.brief) for v in views] == [(0.0, True), (0.67, False), (3.67, False), (5.67, True)]


def test_a_last_screen_is_kept_when_a_fade_after_it_closes_a_chunk_of_its_own(monkeypatch):
    # 5 frames (0.83 s) of the last page, then 2 frames of another picture as the video fades out
    views = read(monkeypatch, [(0, 12), (1, 5), (2, 2)], last_screen=True)
    assert [(round(v.t0, 2), v.brief) for v in views] == [(0.0, False), (2.0, True)]
    # ...only in the video's last segment
    views = read(monkeypatch, [(0, 12), (1, 5), (2, 2)])
    assert [(round(v.t0, 2), v.brief) for v in views] == [(0.0, False)]
    # A screen held as briefly before another view is a page turn, not the last screen
    views = read(monkeypatch, [(0, 12), (1, 5), (2, 12)], last_screen=True)
    assert [(round(v.t0, 2), v.brief) for v in views] == [(0.0, False), (2.83, False)]


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


def test_a_brief_screen_alone_in_its_segment_whose_cells_are_slivers_gives_none():
    page = screen(0, [1, 2, 3, 4])
    alone = screen(1, [5, 6, 7], brief=True)
    alone.seg = 1
    for i, c in enumerate(alone.all_cells):
        c.x0, c.x1 = 40 * i, 40 * i + 40
    drop_fades([page, alone])
    assert alone.all_cells == [] and len(page.all_cells) == 4
    # Whole measures of the usual width are kept
    page, alone = screen(0, [1, 2, 3, 4]), screen(1, [5, 6, 7], brief=True)
    alone.seg = 1
    drop_fades([page, alone])
    assert len(alone.all_cells) == 3


def barred(index, xs, brief=False):
    """A screen of whole measures between barlines at `xs`."""
    o = screen(index, list(range(10 * index + 1, 10 * index + len(xs))), brief)
    for c, x0, x1 in zip(o.all_cells, xs, xs[1:]):
        c.x0, c.x1 = x0, x1
    return o


def test_same_barlines_needs_a_fifth_of_the_barlines_within_a_quarter_line_gap():
    page = barred(0, [0, 100, 200, 300, 400])
    assert same_barlines(page, barred(1, [2, 101, 202, 298, 400]))
    assert not same_barlines(page, barred(1, [50, 150, 250, 350, 450]))
    assert same_barlines(page, barred(1, [2, 150, 250, 350, 450]))
    assert not same_barlines(page, barred(1, [5, 150, 250, 350, 450]))


def test_a_faded_brief_last_screen_gives_none_when_it_shows_the_barlines_of_the_screen_before():
    before, last = barred(0, [0, 100, 200, 300, 400]), barred(1, [2, 101, 203, 298, 400], brief=True)
    last.faded = True
    drop_fades([before, last])
    assert last.all_cells == [] and len(before.all_cells) == 4
    # A page that moved on has its barlines elsewhere and is kept
    before, last = barred(0, [0, 100, 200, 300, 400]), barred(1, [50, 150, 250, 350, 450], brief=True)
    last.faded = True
    drop_fades([before, last])
    assert len(last.all_cells) == 4
    # A screen before it with no measures left is fading too
    before, last = barred(0, [0, 100, 200, 300, 400]), barred(1, [50, 150, 250, 350, 450], brief=True)
    before.all_cells, last.faded = [], True
    drop_fades([before, last])
    assert last.all_cells == []


def test_a_brief_screen_with_more_than_half_its_cells_slivers_gives_none():
    page, brief = screen(0, [1, 2, 3, 4]), barred(1, [0, 40, 80, 120, 320], brief=True)
    drop_fades([page, brief])
    assert brief.all_cells == [] and len(page.all_cells) == 4
    # Half of them slivers is not more than half
    page, brief = screen(0, [1, 2, 3, 4]), barred(1, [0, 40, 80, 280, 480], brief=True)
    drop_fades([page, brief])
    assert len(brief.all_cells) == 4


def timed(o, t0, t1, seg=0):
    o.t0, o.t1, o.seg = t0, t1, seg
    return o


def held(obs):
    return {id(o) for o in obs if sum(c.kind == "full" for c in o.all_cells) >= 2}


def ending(obs, last_seg=0):
    """The last-screen notices for `obs` after drop_fades has judged its brief screens."""
    kept = held(obs)
    return last_screen_notice(obs, last_seg, kept, drop_fades(obs))


def at(o, xs):
    for c, x0, x1 in zip(o.all_cells, xs, xs[1:]):
        c.x0, c.x1 = x0, x1
    return o


def test_a_video_ending_on_a_brief_screen_with_whole_measures_is_named():
    before, last = timed(screen(0, [1, 2, 3, 4]), 20.0, 30.0), timed(screen(1, [5, 6, 7], brief=True), 30.0, 30.8)
    obs = [timed(screen(2, [8, 9]), 0.0, 20.0), before, last]
    [n] = ending(obs)
    assert (n.t0, n.t1, n.kind, n.doubt) == (20.0, 30.8, "last_screen", True)
    assert len(last.all_cells) == 3
    # A brief last screen with one whole measure is a page going away
    assert ending([timed(screen(0, [1, 2, 3, 4]), 20.0, 30.0), timed(screen(1, [5, None], brief=True), 30.0, 30.8)]) == []
    # A brief first screen is not the last
    assert ending([timed(screen(0, [5, 6, 7], brief=True), 0.0, 0.8), timed(screen(1, [1, 2, 3, 4]), 0.8, 30.0)]) == []
    # A brief last screen of an earlier segment is a transition
    obs = [timed(screen(0, [1, 2, 3, 4]), 20.0, 30.0), timed(screen(1, [5, 6, 7], brief=True), 30.0, 30.8),
           timed(screen(2, [8, 9, 10]), 40.0, 60.0, seg=1)]
    assert ending(obs, 1) == []


def test_a_brief_last_screen_cleared_as_shown_again_is_not_named_but_one_cleared_as_misread_is():
    before, last = timed(screen(0, [1, 2, 3, 4]), 20.0, 30.0), timed(screen(1, [2, 3, 4], brief=True), 30.0, 30.8)
    assert ending([before, last]) == [] and last.all_cells == []
    # Barlines misread into slivers may hide the measures the screen shows
    before = timed(screen(0, [1, 2, 3, 4]), 20.0, 30.0)
    last = timed(barred(1, [0, 40, 80, 120, 320], brief=True), 30.0, 30.8)
    assert [(n.t0, n.t1) for n in ending([before, last])] == [(20.0, 30.8)] and last.all_cells == []


def test_a_faded_brief_last_screen_on_the_barlines_of_the_page_before_is_not_named_even_with_other_measures():
    # The same barlines and the same measures as the screen before: the page itself fading out
    before = timed(barred(0, [0, 100, 200, 300, 400]), 20.0, 30.0)
    last = timed(at(screen(1, [2, 1, 4, 3], brief=True), [2, 101, 202, 298, 400]), 30.0, 30.8)
    last.faded = True
    assert ending([before, last]) == [] and last.all_cells == []
    # The same barlines but other measures is taken as a fade too, though the page may have changed
    before = timed(barred(0, [0, 100, 200, 300, 400]), 20.0, 30.0)
    last = timed(barred(1, [2, 101, 202, 298, 400], brief=True), 30.0, 30.8)
    last.faded = True
    assert ending([before, last]) == [] and last.all_cells == []


def test_a_faded_brief_last_screen_whose_cells_are_slivers_is_a_fade_and_not_named():
    # Its barlines stand away from the screen before's, so only its slivers clear it
    before = timed(screen(0, [1, 2, 3, 4]), 20.0, 30.0)
    last = timed(barred(1, [50, 90, 130, 170, 370], brief=True), 30.0, 30.8)
    assert not same_barlines(last, before)
    last.faded = True
    assert ending([before, last]) == [] and last.all_cells == []
    # The same screen unfaded had its barlines misread and is named
    before = timed(screen(0, [1, 2, 3, 4]), 20.0, 30.0)
    last = timed(barred(1, [50, 90, 130, 170, 370], brief=True), 30.0, 30.8)
    assert [(n.t0, n.t1) for n in ending([before, last])] == [(20.0, 30.8)] and last.all_cells == []


def test_a_brief_last_screen_of_slivers_that_is_the_only_view_of_its_part_is_named_unless_faded():
    # Another part's screen sets the usual measure width; the brief screen has no screen before it of its own part
    for faded in (False, True):
        other = timed(screen(0, [1, 2, 3, 4]), 20.0, 30.0)
        last = timed(barred(1, [50, 90, 130, 170, 370], brief=True), 30.0, 30.8)
        last.part, last.faded = 1, faded
        notices = ending([other, last])
        assert [(n.t0, n.t1, n.kind) for n in notices] == ([] if faded else [(30.0, 30.8, "last_screen")])
        assert last.all_cells == [] and len(other.all_cells) == 4


def test_parts_shown_at_once_judge_each_last_screen_against_the_screen_before_it_of_its_own_part():
    def part(o, k):
        o.part = k
        return o
    # Part 0 ends on a brief screen of new measures; part 1's last screen starts a moment later and is held as long
    # as usual
    obs = [part(timed(screen(0, [1, 2, 3, 4]), 20.0, 30.0), 0), part(timed(screen(1, [11, 12, 13]), 20.02, 30.02), 1),
           part(timed(screen(2, [5, 6, 7], brief=True), 30.0, 30.8), 0),
           part(timed(screen(3, [14, 15, 16]), 30.02, 30.8), 1)]
    assert [(n.t0, n.t1) for n in ending(obs)] == [(20.0, 30.8)]
    # Part 1's brief last screen shows measures its own screen before shows again, so only part 0's is named
    obs = [part(timed(screen(0, [1, 2, 3, 4]), 20.0, 30.0), 0), part(timed(screen(1, [11, 12, 13]), 20.02, 30.02), 1),
           part(timed(screen(2, [5, 6, 7], brief=True), 30.0, 30.8), 0),
           part(timed(screen(3, [12, 13], brief=True), 30.02, 30.8), 1)]
    assert [(n.t0, n.t1) for n in ending(obs)] == [(20.0, 30.8)]
    # Neither part's last screen calls for a notice
    obs = [part(timed(screen(0, [1, 2, 3, 4]), 20.0, 30.0), 0), part(timed(screen(1, [11, 12, 13]), 20.02, 30.02), 1),
           part(timed(screen(2, [3, 4], brief=True), 30.0, 30.8), 0),
           part(timed(screen(3, [12, 13], brief=True), 30.02, 30.8), 1)]
    assert ending(obs) == []


def test_a_brief_screen_that_parses_no_measures_is_not_read(monkeypatch):
    seg = Segment(0.0, 10.0, 10.0, "dark", paper=np.array([255.0, 255.0, 255.0]), x1=640, y1=360)
    plate = np.full((360, 640, 3), 255, np.uint8)
    views = [View(0.0, 0.5, plate, rect=(0, 0, 640, 360), print_rect=(0, 0, 640, 360), brief=True),
             View(0.5, 9.0, plate, rect=(0, 0, 640, 360), print_rect=(0, 0, 640, 360)),
             View(9.0, 9.5, plate, rect=(0, 0, 640, 360), print_rect=(0, 0, 640, 360), brief=True)]
    # The middle screen and the last one show measures; the brief first screen shows none
    measures = {0.0: [], 0.5: [5, 6, 7], 9.0: [8, 9]}

    def plate_obs(v, seg, si, thr, fthr, index, col0=None):
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


def test_a_brief_screen_with_paper_darker_than_the_faint_ink_level_is_read_with_its_own_thresholds(monkeypatch):
    seg = Segment(0.0, 10.0, 10.0, "dark", paper=np.array([255.0, 255.0, 255.0]), x1=640, y1=360)
    plate = np.full((360, 640, 3), 150, np.uint8)
    plate[100:110, 100:300] = 20
    used = []

    def parse_view(index, image, rect, polarity, thr, fthr, *a, **kw):
        used.append((thr, fthr))
        return []
    monkeypatch.setattr(pipeline, "parse_view", parse_view)
    rect = (0, 0, 640, 360)
    own = ink_threshold(to_gray(plate, "dark"), np.median(plate.reshape(-1, 3), axis=0), "dark")
    pipeline.plate_obs(View(9.0, 9.5, plate, rect=rect, print_rect=rect, brief=True), seg, 0, 128.0, 230.0, 0)
    assert used[-1] == pytest.approx(own)
    # A screen held the usual time keeps the segment's thresholds, and so does a brief one on paper lighter than them
    pipeline.plate_obs(View(0.5, 9.0, plate, rect=rect, print_rect=rect), seg, 0, 128.0, 230.0, 0)
    assert used[-1] == (128.0, 230.0)
    pipeline.plate_obs(View(9.0, 9.5, plate, rect=rect, print_rect=rect, brief=True), seg, 0, 100.0, 140.0, 0)
    assert used[-1] == (100.0, 140.0)


def contrasted(index, ink, line, brief=False):
    o = screen(index, [1, 2, 3], brief)
    o.contrast, o.line_contrast = ink, line
    return o


def test_a_faded_brief_screen_after_its_parts_first_is_marked_and_keeps_its_cells():
    parts, segments = [SimpleNamespace(id=0, seg=0)], [SimpleNamespace(polarity="dark")]
    first, last = contrasted(0, 50.0, 20.0, brief=True), contrasted(4, 50.0, 20.0, brief=True)
    obs = [first, contrasted(1, 100.0, 50.0), contrasted(2, 100.0, 50.0), contrasted(3, 100.0, 50.0), last]
    mark_fades(obs, parts, segments, 0)
    assert (last.faded, len(last.all_cells)) == (True, 3)
    # The first screen fading in shows the page the next screen shows clearly, so it gives no measures
    assert (first.faded, first.all_cells) == (False, [])
    # A screen under only one of the two thresholds is not caught in a fade
    for ink, line in ((50.0, 50.0), (100.0, 20.0)):
        last = contrasted(4, ink, line, brief=True)
        mark_fades(obs[:4] + [last], parts, segments, 0)
        assert (last.faded, len(last.all_cells)) == (False, 3)


def test_a_faded_brief_screen_before_the_last_segment_is_cleared():
    # last_screen_notice only covers the video's last segment, so a faded brief screen elsewhere gives no measures
    parts, segments = [SimpleNamespace(id=0, seg=0)], [SimpleNamespace(polarity="dark")] * 2
    last = contrasted(4, 50.0, 20.0, brief=True)
    obs = [contrasted(0, 100.0, 50.0), contrasted(1, 100.0, 50.0), contrasted(2, 100.0, 50.0), last]
    mark_fades(obs, parts, segments, 1)
    assert (last.faded, last.all_cells) == (False, [])
    last = contrasted(4, 50.0, 20.0, brief=True)
    mark_fades(obs[:3] + [last], parts, segments, 0)
    assert (last.faded, len(last.all_cells)) == (True, 3)


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
