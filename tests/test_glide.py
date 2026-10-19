import shutil

import cv2
import numpy as np
import pytest

from synth import Song, write_video
from youtabber import glide, pipeline
from youtabber.glide import Mosaic

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def test_time_map_counts_the_lead_columns():
    m = Mosaic(np.zeros((10, 1000, 3), np.uint8), 0, np.array([0.0, 100.0, 200.0]), np.array([0.0, 1.0, 2.0]),
               300.0, 50)
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
    # The first measures were on screen before the scroll moved, the last still ahead when it stopped
    assert {"glide_start", "glide_end"} <= {n.kind for n in r.notices}


GAP = 14
STAFF = (30, 101)


def sheet(width: int, notes_from: int = 0) -> np.ndarray:
    """A long strip of tab: six lines 14 px apart, a barline every 150 px, and fret digits in each measure from
    x `notes_from` on."""
    img = np.full((130, width, 3), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (0, 30 + i * GAP), (width - 1, 30 + i * GAP), (60, 60, 60), 1)
    for x in range(100, width, 150):
        cv2.line(img, (x, 30), (x, 100), (0, 0, 0), 2)
        if x >= notes_from:
            for dx, string in ((30, 1), (60, 3), (95, 4)):
                cv2.putText(img, "7", (x + dx, 30 + string * GAP + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def fake_frames(strip: np.ndarray, count: int, step: int = 20, faded: int = 0, playhead: int | None = None):
    """frames() for a band gliding left `step` px per frame over an 800 px panel; the first `faded` frames are
    washed out, and `playhead` draws a fixed gray line across the staff."""
    def frames(path, fps, crop=None, start=0.0, length=None):
        for k in range(count):
            f = strip[:, k * step:k * step + 800].copy()
            if k < faded:
                f = (255 - (255 - f.astype(np.int16)) // 4).astype(np.uint8)
            if playhead is not None:
                f[STAFF[0]:STAFF[1], playhead:playhead + 3] = 150
            yield start + k / fps, f
    return frames


def mosaic_of(monkeypatch, frames) -> Mosaic:
    monkeypatch.setattr(glide, "frames", frames)
    return glide.build(None, (0, 0, 800, 130), (0, 130), STAFF, "dark", 140.0, GAP, 0.0, 6.0, (0, 130), (0, 800))


def test_no_playhead_reads_the_whole_first_screen(monkeypatch):
    m = mosaic_of(monkeypatch, fake_frames(sheet(3000), 60))
    # Measures are said to be played at the panel's centre, and the canvas starts at the panel's left edge
    assert m.ref_x == 400.0 and m.lead == 400
    assert m.image.shape[1] == 800 + round(m.offsets[-1])
    assert m.offsets[-1] == pytest.approx(59 * 20, abs=5)


def test_a_gray_playhead_line_is_found(monkeypatch):
    m = mosaic_of(monkeypatch, fake_frames(sheet(3000), 60, playhead=250))
    assert 250 <= m.ref_x <= 252
    assert m.lead == int(m.ref_x) - int(m.ref_x - glide.SEED_BACK_GAPS * GAP)


def test_the_mosaic_starts_after_a_fade_in(monkeypatch):
    m = mosaic_of(monkeypatch, fake_frames(sheet(3000), 60, faded=5))
    assert m.times[0] == 0.5 and m.offsets[0] == 0.0


def test_a_sparse_intro_still_moves(monkeypatch):
    # Right of the reading column only barlines pass for the first 30 frames: an intro of rests
    m = mosaic_of(monkeypatch, fake_frames(sheet(3000, notes_from=1400), 60))
    assert m.offsets[29] >= 400


def test_measures_that_never_reach_the_play_column_are_placed_on_their_screen():
    m = Mosaic(np.zeros((10, 1000, 3), np.uint8), 0, np.array([0.0, 100.0, 200.0, 200.0]),
               np.array([0.0, 1.0, 2.0, 3.0]), 300.0, 50)
    assert m.seen_at(20) == (0.0, 270.0)     # left of the play column on the first screen
    assert m.seen_at(150) is None             # crosses it while the band moves
    assert m.seen_at(300) == (2.5, 350.0)     # still right of it when the band stopped at 2 s


def test_a_moving_band_glides_and_a_still_page_does_not(monkeypatch):
    def bands(t):
        return [STAFF]
    monkeypatch.setattr(glide, "frames", fake_frames(sheet(30000), 10))
    assert glide.glides(None, (0, 0, 800, 130), "dark", np.array([255.0, 255.0, 255.0]), 0.0, 60.0, bands, GAP)
    monkeypatch.setattr(glide, "frames", fake_frames(sheet(30000), 10, step=0))
    assert not glide.glides(None, (0, 0, 800, 130), "dark", np.array([255.0, 255.0, 255.0]), 0.0, 60.0, bands, GAP)


def test_a_measure_still_ahead_when_the_band_stops_at_the_end_keeps_the_glide_time():
    m = Mosaic(np.zeros((10, 1000, 3), np.uint8), 0, np.array([0.0, 100.0, 200.0, 300.0]),
               np.array([0.0, 1.0, 2.0, 3.0]), 300.0, 50)
    # The band last moves in the last frame, so half a second later would be past the glide
    assert m.seen_at(500) == (3.0, 450.0)


def test_the_band_slows_down_during_bars_of_rest(monkeypatch):
    strip = sheet(3000)
    # Bars of rest from x 900 on: staff lines and a barline every 400 px, too little ink to trust one frame
    strip[:, 900:] = 255
    for i in range(6):
        cv2.line(strip, (900, 30 + i * GAP), (2999, 30 + i * GAP), (60, 60, 60), 1)
    for x in range(1000, 3000, 400):
        cv2.line(strip, (x, 30), (x, 100), (0, 0, 0), 2)
    # 20 px per frame, then half speed from frame 20 on
    xs = [0]
    for k in range(1, 60):
        xs.append(xs[-1] + (20 if k < 20 else 10))

    def frames(path, fps, crop=None, start=0.0, length=None):
        for k, x in enumerate(xs):
            yield start + k / fps, strip[:, x:x + 800].copy()
    m = mosaic_of(monkeypatch, frames)
    # The new speed is taken after CHANGE_FRAMES frames agree; keeping the old speed would give 1180
    assert 780 <= m.offsets[-1] <= 840


def test_a_long_mosaic_parsed_in_pieces_gives_the_cells_of_one_parse(monkeypatch):
    image = sheet(2400)
    rect = (0, 0, 2400, 130)
    whole = pipeline.parse_mosaic(0, image, rect, "dark", 140.0, 229.0, GAP)
    monkeypatch.setattr(pipeline, "MOSAIC_PIECE", 700)
    monkeypatch.setattr(pipeline, "MOSAIC_OVERLAP", 300)
    pieces = pipeline.parse_mosaic(0, image, rect, "dark", 140.0, 229.0, GAP)
    assert len(pieces) == len(whole) == 1
    assert [(c.x0, c.x1, c.kind) for c in pieces[0].cells] == [(c.x0, c.x1, c.kind) for c in whole[0].cells]
    assert [c.index for c in pieces[0].cells] == list(range(len(whole[0].cells)))
