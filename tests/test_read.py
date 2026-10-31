import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from synth import Song, views, write_video
from youtabber import glide, pipeline
from youtabber.observe import View
from youtabber.scan import Segment
from youtabber.video import VideoError, VideoInfo, frames, probe

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def test_reads_a_page_flip_video(tmp_path):
    video = tmp_path / "song.mp4"
    write_video(Song(), video)
    r = pipeline.read(video)
    assert r.outcome == "tab", r.message
    assert r.candidates and r.candidates[0].measures
    assert r.every_pass and len(r.every_pass[0].measures) >= len(r.candidates[0].measures)


def test_the_message_counts_measures_in_singular_or_plural(tmp_path):
    # One measure as wide as the page, so no empty staff end follows it
    one = tmp_path / "one.mp4"
    write_video(Song(measures=1, measure_width=1100), one)
    assert pipeline.read(one).message == "Read 1 measure."
    four = tmp_path / "four.mp4"
    write_video(Song(measures=4), four)
    assert pipeline.read(four).message == "Read 4 measures."


def test_staffs_without_measures_fail_as_tab_youtabber_cant_read(tmp_path, monkeypatch):
    video = tmp_path / "song.mp4"
    write_video(Song(measures=4), video)
    # Staffs are found but give no cells, as text tab's dash rows do
    parse_view = pipeline.parse_view
    monkeypatch.setattr(pipeline, "parse_view",
                        lambda *a, **k: [replace(ps, cells=[]) for ps in parse_view(*a, **k)])
    r = pipeline.read(video)
    assert r.outcome == "fail" and r.fail_reason == "no_tab"
    assert "text tab" in r.message


def still_view(t0: float, t1: float) -> View:
    return View(t0, t1, np.full((360, 640, 3), 255, np.uint8), rect=(0, 0, 640, 360), print_rect=(0, 0, 640, 360))


def test_a_long_segment_that_neither_holds_still_nor_glides_fails(monkeypatch):
    seg = Segment(0.0, 30.0, 10.0, "dark", paper=np.array([255.0, 255.0, 255.0]), x1=640, y1=360)
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(640, 360, 30.0))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None, progress=None: ([seg], {}, []))
    monkeypatch.setattr(glide, "glides", lambda *a: False)
    # One still view of 5 s in 30 s of tab
    monkeypatch.setattr(pipeline, "views_in", lambda *a, **kw: ([still_view(0.0, 5.0)], 128.0, 230.0))
    r = pipeline.read(Path("unused.mp4"))
    assert r.outcome == "fail" and r.fail_reason == "check_failed"
    assert "holds still for only 5 of 30 s" in r.message


def brief_tab(monkeypatch, t1: float, duration: float):
    """A video of `duration` s whose only tab is one still segment from 0 to `t1` s."""
    seg = Segment(0.0, t1, 10.0, "dark", paper=np.array([255.0, 255.0, 255.0]), x1=640, y1=360)
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(640, 360, duration))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None, progress=None: ([seg], {}, []))
    monkeypatch.setattr(glide, "glides", lambda *a: False)
    monkeypatch.setattr(pipeline, "text_tab_shown", lambda path, longer_than=0.0: False)
    monkeypatch.setattr(pipeline, "views_in", lambda *a, **kw: ([still_view(0.0, t1)], 128.0, 230.0))
    return pipeline.read(Path("unused.mp4"))


def test_tab_shown_too_briefly_in_a_long_video_is_no_tab(monkeypatch):
    r = brief_tab(monkeypatch, 8.0, 330.0)
    assert r.outcome == "fail" and r.fail_reason == "no_tab"
    assert "only 8 s of 330 s" in r.message


def test_brief_tab_holding_enough_of_a_short_video_is_not_refused_as_brief(monkeypatch):
    # 8 s of 200 s is 4% of the video
    assert "too briefly" not in brief_tab(monkeypatch, 8.0, 200.0).message


def test_tab_shown_ten_seconds_or_more_is_not_refused_as_brief(monkeypatch):
    # 12 s of 400 s is 3% of the video
    assert "too briefly" not in brief_tab(monkeypatch, 12.0, 400.0).message


def read_tab_for(monkeypatch, pages: int):
    """A 60 s segment held as twelve 5 s views, of which only the last `pages` show tab; the others show blank
    paper, so no measure is found on them."""
    song = Song()
    shown = views(song)[:pages]
    blank = np.full_like(shown[0], 255)
    plates = [blank] * (12 - pages) + shown
    h, w = blank.shape[:2]
    held = [View(5.0 * i, 5.0 * i + 5.0, plate, rect=(0, 0, w, h), print_rect=(0, 0, w, h))
            for i, plate in enumerate(plates)]
    seg = Segment(0.0, 60.0, float(song.line_spacing), "dark", paper=np.array([255.0, 255.0, 255.0]), x1=w, y1=h)
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(w, h, 60.0))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None, progress=None: ([seg], {}, []))
    monkeypatch.setattr(glide, "glides", lambda *a: False)
    monkeypatch.setattr(pipeline, "views_in", lambda *a, **kw: (held, 128.0, 230.0))
    monkeypatch.setattr(pipeline, "view_move", lambda *a: None)
    monkeypatch.setattr(pipeline, "text_tab_shown", lambda path, longer_than=0.0: False)
    return pipeline.read(Path("unused.mp4"))


def test_lines_read_as_measures_for_little_of_their_time_are_no_tab(monkeypatch):
    r = read_tab_for(monkeypatch, 1)
    assert r.outcome == "fail" and r.fail_reason == "no_tab"
    assert "found on them for only 5 s" in r.message


def test_lines_read_as_measures_for_a_sixth_of_their_time_are_not_refused_as_no_tab(monkeypatch):
    r = read_tab_for(monkeypatch, 2)
    assert "found on them for only" not in r.message


def two_bands(monkeypatch, seen: tuple[float, float]):
    """A 30 s segment showing two stacked bands seen for `seen` seconds, the upper one holding still throughout and
    the lower one for 5 s."""
    from youtabber.parts import Part
    seg = Segment(0.0, 30.0, 10.0, "dark", paper=np.array([255.0, 255.0, 255.0]), x1=640, y1=360)
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(640, 360, 30.0))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None, progress=None: ([seg], {}, []))
    monkeypatch.setattr(glide, "glides", lambda *a: False)
    monkeypatch.setattr(pipeline, "text_tab_shown", lambda path, longer_than=0.0: False)
    monkeypatch.setattr(pipeline, "find_parts", lambda *a: [Part(0, 0, (20, 160), "band", {}, seen[0]),
                                                           Part(1, 0, (200, 340), "band", {}, seen[1])])
    calls = iter([[still_view(0.0, 30.0)], [still_view(0.0, 5.0)]])
    monkeypatch.setattr(pipeline, "views_in", lambda *a, **kw: (next(calls), 128.0, 230.0))
    return pipeline.read(Path("unused.mp4"))


def test_a_second_part_that_seldom_holds_still_is_left_out_without_refusing_the_video(monkeypatch):
    r = two_bands(monkeypatch, (30.0, 29.0))
    assert r.fail_reason != "check_failed"
    assert r.views and all(o.part == 0 and o.stream == 0 for o in r.views)
    assert [n.kind for n in r.notices if n.kind == "parts"] == ["parts"]


def test_a_second_part_that_glides_but_cant_be_put_together_is_left_out_without_refusing_the_video(monkeypatch):
    from youtabber.parts import Part
    seg = Segment(0.0, 30.0, 10.0, "dark", paper=np.array([255.0, 255.0, 255.0]), x1=640, y1=360)
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(640, 360, 30.0))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None, progress=None: ([seg], {}, []))
    monkeypatch.setattr(pipeline, "text_tab_shown", lambda path, longer_than=0.0: False)
    monkeypatch.setattr(pipeline, "find_parts", lambda *a: [Part(0, 0, (20, 160), "band", {}, 30.0),
                                                           Part(1, 0, (200, 340), "band", {}, 29.0)])
    # The upper band holds still; the lower one glides, and no mosaic can be built from it
    gliding = iter([False, True])
    monkeypatch.setattr(glide, "glides", lambda *a: next(gliding))
    monkeypatch.setattr(pipeline, "first_thresholds", lambda *a: (128.0, 230.0))
    monkeypatch.setattr(pipeline, "glide_pieces", lambda *a: ([], None))
    monkeypatch.setattr(pipeline, "views_in", lambda *a, **kw: ([still_view(0.0, 30.0)], 128.0, 230.0))
    r = pipeline.read(Path("unused.mp4"))
    assert r.fail_reason != "check_failed"
    assert r.views and all(o.part == 0 for o in r.views)
    assert [n.text for n in r.notices if n.kind == "parts"] == [
        "Another part of the tab shows at 0:00-0:30, but it scrolls smoothly and its moving band could not be put "
        "together, so it isn't printed."]


def test_the_main_part_holding_still_too_little_still_refuses_the_video(monkeypatch):
    # The lower band is seen longest, so it is the main part
    r = two_bands(monkeypatch, (29.0, 30.0))
    assert r.outcome == "fail" and r.fail_reason == "check_failed"
    assert "holds still for only 5 of 30 s" in r.message


def test_a_gliding_band_is_read_as_a_glide_and_a_still_band_beside_it_as_still_views(monkeypatch):
    from types import SimpleNamespace

    from youtabber.parts import Part

    def staff(top, bottom):
        return SimpleNamespace(top=top, tab=SimpleNamespace(x0=20, x1=620, bottom=bottom))
    upper = {t / 6: [staff(100, 203)] for t in range(180)}
    lower = {t / 6: [staff(600, 699)] for t in range(180)}
    seg = Segment(0.0, 30.0, 10.0, "dark", paper=np.array([255.0, 255.0, 255.0]), x1=640, y1=720)
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(640, 720, 30.0))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None, progress=None: ([seg], {}, []))
    monkeypatch.setattr(pipeline, "find_parts", lambda *a: [Part(0, 0, (100, 204), "band", upper, 30.0),
                                                           Part(1, 0, (600, 700), "band", lower, 30.0)])
    # Only the upper band's rows glide
    monkeypatch.setattr(glide, "glides", lambda *a: all(100 <= y0 and y1 <= 204 for y0, y1 in a[6](15.0)))
    monkeypatch.setattr(pipeline, "first_thresholds", lambda *a: (128.0, 230.0))
    glided, stills = [], []

    def glide_pieces(path, crop, seg, si, found, thr, fthr, index):
        glided.append(found)
        return [SimpleNamespace(t0=0.0, t1=30.0, rows=(100, 204))], None

    def views_in(*a, **kw):
        stills.append(a[5](15.0))
        return [still_view(0.0, 30.0)], 128.0, 230.0
    monkeypatch.setattr(pipeline, "glide_pieces", glide_pieces)
    monkeypatch.setattr(pipeline, "views_in", views_in)
    monkeypatch.setattr(pipeline, "finish", lambda path, info, r, *a: r)
    pipeline.read(Path("unused.mp4"))
    assert glided == [upper]
    # The still views are read from the lower band's staffs
    assert len(stills) == 1 and stills[0][1] == 600


def test_a_missing_video_is_a_video_error(tmp_path):
    with pytest.raises(VideoError, match="Can't read"):
        probe(tmp_path / "missing.mp4")


def test_a_video_whose_only_staffs_scroll_up_fails_as_no_tab(monkeypatch):
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(640, 360, 120.0))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None, progress=None: ([], {}, [(3.0, 98.5)]))
    r = pipeline.read(Path("unused.mp4"))
    assert r.outcome == "fail" and r.fail_reason == "no_tab"
    assert "scrolls up continuously (0:03-1:38)" in r.message


def test_a_file_that_isnt_a_video_names_its_path_once(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a video\n")
    with pytest.raises(VideoError) as e:
        probe(notes)
    said = "\n".join([e.value.heading, *e.value.detail])
    assert said.count(str(notes)) == 1, said
    # ffprobe names the path on a second error line when an mp4 is cut off
    whole = tmp_path / "whole.mp4"
    write_video(Song(measures=4), whole)
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(whole.read_bytes()[:3000])
    with pytest.raises(VideoError) as e:
        probe(cut)
    said = "\n".join([e.value.heading, *e.value.detail])
    assert said.count(str(cut)) == 1 and " @ 0x" not in said, said
    assert e.value.detail[-1] == "Check it plays in a video player."


def test_ffmpegs_own_error_text_is_dropped(tmp_path, monkeypatch):
    video_file = tmp_path / "song.mp4"
    write_video(Song(measures=4), video_file)
    made = []
    real = subprocess.Popen

    def popen(*a, **kw):
        made.append(kw)
        return real(*a, **kw)

    monkeypatch.setattr(subprocess, "Popen", popen)
    assert len(list(frames(video_file, 1.0))) > 0
    # The last process started is the frame reader; probing the file starts one before it
    assert made[-1]["stderr"] == subprocess.DEVNULL


def test_a_video_showing_text_tab_fails_as_text_tab(monkeypatch):
    from test_staff import text_tab
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(800, 360, 60.0))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None, progress=None: ([], {}, []))
    page = np.dstack([text_tab()] * 3)
    monkeypatch.setattr(pipeline, "frames", lambda path, fps, **kw: ((5.0 * k, page) for k in range(12)))
    r = pipeline.read(Path("unused.mp4"))
    assert r.outcome == "fail" and r.fail_reason == "text_tab"
    assert "typed as text" in r.message


def test_typed_tab_shown_longer_than_lines_of_one_measure_fails_as_text_tab(tmp_path, monkeypatch):
    video = tmp_path / "song.mp4"
    write_video(Song(measures=2, measure_width=1100), video)
    # Each line holds one measure, and typed tab shows for 60 s, longer than the drawn-looking pages hold
    monkeypatch.setattr(pipeline, "text_tab_shown", lambda path, longer_than=0.0: 60.0 > longer_than)
    r = pipeline.read(video)
    assert r.outcome == "fail" and r.fail_reason == "text_tab"


def test_typed_tab_shown_briefly_beside_lines_of_one_measure_is_not_text_tab(tmp_path, monkeypatch):
    video = tmp_path / "song.mp4"
    write_video(Song(measures=2, measure_width=1100), video)
    monkeypatch.setattr(pipeline, "text_tab_shown", lambda path, longer_than=0.0: 2.0 > longer_than)
    assert pipeline.read(video).outcome == "tab"


def test_typed_tab_beside_lines_of_several_measures_is_not_text_tab(tmp_path, monkeypatch):
    video = tmp_path / "song.mp4"
    write_video(Song(measures=4), video)
    monkeypatch.setattr(pipeline, "text_tab_shown", lambda path, longer_than=0.0: 60.0 > longer_than)
    assert pipeline.read(video).outcome == "tab"


def test_text_tab_needs_typed_frames_spanning_longer_than_asked(monkeypatch):
    from test_staff import text_tab
    page = np.dstack([text_tab()] * 3)
    # Twelve frames 5 s apart stand for 60 s of typed tab
    monkeypatch.setattr(pipeline, "frames", lambda path, fps, **kw: ((5.0 * k, page) for k in range(12)))
    assert pipeline.text_tab_shown(Path("unused.mp4"), longer_than=55.0)
    assert not pipeline.text_tab_shown(Path("unused.mp4"), longer_than=60.0)


def test_staffs_without_measures_beside_text_tab_fail_as_text_tab(tmp_path, monkeypatch):
    # Some lines on screen pass for a staff but give no measures, and the video's tab is typed as text
    video = tmp_path / "song.mp4"
    write_video(Song(measures=4), video)
    parse_view = pipeline.parse_view
    monkeypatch.setattr(pipeline, "parse_view", lambda *a, **k: [replace(ps, cells=[]) for ps in parse_view(*a, **k)])
    monkeypatch.setattr(pipeline, "text_tab_shown", lambda path, longer_than=0.0: True)
    r = pipeline.read(video)
    assert r.outcome == "fail" and r.fail_reason == "text_tab"
