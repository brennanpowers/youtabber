import shutil

import numpy as np
import pytest

from synth import Song, write_video
from youtabber import pipeline
from youtabber.glide import Mosaic

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def test_time_map_counts_the_lead_columns():
    m = Mosaic(np.zeros((10, 1000, 3), np.uint8), 0, np.array([0.0, 100.0, 200.0]), np.array([0.0, 1.0, 2.0]),
               300.0, 600, 50)
    assert m.time_at(50) == 0.0          # under the playhead in the first frame
    assert m.time_at(150) == 1.0
    assert m.time_at(350) == 3.0         # never reached it: extrapolated at 100 px/s


def test_a_glide_prints_its_first_and_last_measures(tmp_path):
    song = Song(glide=True, notation=False, tab_height=200)
    video = tmp_path / "glide.mp4"
    write_video(song, video)
    r = pipeline.read(video)
    assert r.outcome == "tab", r.message
    assert any(o.mosaic is not None for o in r.views)
    assert len(r.candidates[0].measures) == song.measures
