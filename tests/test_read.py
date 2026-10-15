import shutil

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
