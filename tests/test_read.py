import shutil
from dataclasses import replace

import pytest

from synth import Song, write_video
from youtabber import pipeline

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
