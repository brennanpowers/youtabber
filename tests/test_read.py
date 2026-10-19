import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from synth import Song, write_video
from youtabber import glide, pipeline
from youtabber.observe import View
from youtabber.scan import Segment
from youtabber.video import VideoError, VideoInfo, probe

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def test_reads_a_page_flip_video(tmp_path):
    video = tmp_path / "song.mp4"
    write_video(Song(), video)
    r = pipeline.read(video)
    assert r.outcome == "tab", r.message
    assert r.candidates and r.candidates[0].measures
    assert r.every_pass and len(r.every_pass[0].measures) >= len(r.candidates[0].measures)


def test_staffs_without_measures_fail_as_tab_youtabber_cant_read(tmp_path, monkeypatch):
    video = tmp_path / "song.mp4"
    write_video(Song(measures=4), video)
    # Staffs are found but give no cells, as text tab's dash rows do
    parse_view = pipeline.parse_view
    monkeypatch.setattr(pipeline, "parse_view",
                        lambda *a: [replace(ps, cells=[]) for ps in parse_view(*a)])
    r = pipeline.read(video)
    assert r.outcome == "fail" and r.fail_reason == "no_tab"
    assert "text tab" in r.message



def still_view(t0: float, t1: float) -> View:
    return View(t0, t1, np.full((360, 640, 3), 255, np.uint8), rect=(0, 0, 640, 360), print_rect=(0, 0, 640, 360))


def test_a_long_segment_that_neither_holds_still_nor_glides_fails(monkeypatch):
    seg = Segment(0.0, 30.0, 10.0, "dark", paper=np.array([255.0, 255.0, 255.0]), x1=640, y1=360)
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(640, 360, 30.0))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None: ([seg], {}, []))
    monkeypatch.setattr(glide, "glides", lambda *a: False)
    # One still view of 5 s in 30 s of tab
    monkeypatch.setattr(pipeline, "views_in", lambda *a: ([still_view(0.0, 5.0)], 128.0, 230.0))
    r = pipeline.read(Path("unused.mp4"))
    assert r.outcome == "fail" and r.fail_reason == "check_failed"
    assert "holds still for only 5 of 30 s" in r.message


def test_a_missing_video_is_a_video_error(tmp_path):
    with pytest.raises(VideoError, match="Can't read"):
        probe(tmp_path / "missing.mp4")


def test_a_video_whose_only_staffs_scroll_up_fails_as_no_tab(monkeypatch):
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(640, 360, 120.0))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None: ([], {}, [(3.0, 98.5)]))
    r = pipeline.read(Path("unused.mp4"))
    assert r.outcome == "fail" and r.fail_reason == "no_tab"
    assert "scrolls up continuously (0:03-1:38)" in r.message
