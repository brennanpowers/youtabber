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


def test_a_glide_behind_ink_fixed_to_the_screen_still_glides(monkeypatch):
    # Bars of rest glide behind a dark column that stays at one screen place, as an uncolored playhead does
    moving = fake_frames(sheet(4000, notes_from=10**6), glide.MOTION_FRAMES)

    def frames(path, fps, crop=None, start=0.0, length=None):
        for t, f in moving(path, fps, crop, start, length):
            f[30:101, 400:420] = 40
            yield t, f
    monkeypatch.setattr(glide, "frames", frames)
    paper = np.array([255.0, 255.0, 255.0])
    assert glide.glides(None, (0, 0, 800, 130), "dark", paper, 0.0, 30.0, lambda t: [(20, 110)], GAP)
    monkeypatch.setattr(glide, "STILL_SHARE", 2.0)
    assert not glide.glides(None, (0, 0, 800, 130), "dark", paper, 0.0, 30.0, lambda t: [(20, 110)], GAP)


def test_a_measure_still_ahead_when_the_band_stops_at_the_end_keeps_the_glide_time():
    m = Mosaic(np.zeros((10, 1000, 3), np.uint8), 0, np.array([0.0, 100.0, 200.0, 300.0]),
               np.array([0.0, 1.0, 2.0, 3.0]), 300.0, 50)
    # The band last moves in the last frame, so half a second later would be past the glide
    assert m.seen_at(500) == (3.0, 450.0)


def test_a_measure_ahead_when_the_picture_fades_out_crosses_at_the_last_speed():
    # The band speeds up to 200 px/s over the last two seconds, and the glide's frames go on to 5 s
    m = Mosaic(np.zeros((10, 1000, 3), np.uint8), 0, np.array([0.0, 100.0, 300.0, 500.0]),
               np.array([0.0, 1.0, 2.0, 3.0]), 300.0, 50, until=5.0, until_offset=620.0)
    assert m.last_speed() == 200.0
    assert m.seen_at(650) is None
    assert m.time_at(650) == 3.5
    # One the band would bring to the play column only after the frames end stays on the last screen
    assert m.seen_at(1000) == (3.0, 750.0)


def test_a_measure_ahead_when_the_picture_fades_out_stays_on_screen_past_how_far_the_band_moved():
    # The last speed would bring canvas x 700 to the play column at 3.75 s, but the band moved only 620 px by 5 s
    m = Mosaic(np.zeros((10, 1000, 3), np.uint8), 0, np.array([0.0, 100.0, 300.0, 500.0]),
               np.array([0.0, 1.0, 2.0, 3.0]), 300.0, 50, until=5.0, until_offset=620.0)
    assert m.time_at(700) == 3.75
    assert m.seen_at(650) is None
    assert m.seen_at(700) == (3.0, 450.0)


def test_a_measure_ahead_when_the_band_stopped_stays_on_its_screen_whatever_the_frames_end():
    stopped = Mosaic(np.zeros((10, 1000, 3), np.uint8), 0, np.array([0.0, 100.0, 300.0, 500.0, 500.0]),
                     np.array([0.0, 1.0, 2.0, 3.0, 4.0]), 300.0, 50, until=5.0)
    assert stopped.seen_at(650) == (3.5, 400.0)
    # Without the glide's end time, a band moving in the last frame shows its last screen
    moving = Mosaic(np.zeros((10, 1000, 3), np.uint8), 0, np.array([0.0, 100.0, 300.0, 500.0]),
                    np.array([0.0, 1.0, 2.0, 3.0]), 300.0, 50)
    assert moving.seen_at(650) == (3.0, 400.0)


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
    # The band's true travel is 780 px; keeping the old speed would give 1180
    assert 775 <= m.offsets[-1] <= 785


def test_vertical_marks_are_thin_tall_lines_not_fixed_to_the_screen():
    staff = np.full((71, 800), 255, np.uint8)
    staff[0:71:GAP, :] = 60
    staff[:, 100:102] = 0
    staff[:, 250:252] = 0
    staff[:, 400:420] = 0         # a block too wide to be a line
    staff[20:30, 600:602] = 0     # a tick shorter than three line gaps
    assert glide.vertical_marks(staff, 140.0, GAP, None).tolist() == [100.5, 250.5]
    fixed = np.zeros(800, bool)
    fixed[95:110] = True
    assert glide.vertical_marks(staff, 140.0, GAP, fixed).tolist() == [250.5]


def test_line_step_takes_the_shift_most_lines_agree_on_near_the_step_before():
    a = np.array
    assert glide.line_step(a([100., 300.]), a([0., 200.]), 0.0, 200.0) == (2, 100.0)
    assert glide.line_step(a([100.]), a([0.]), 0.0, 200.0) == (1, 100.0)
    # Beyond MARK_CHANGE of the step before
    assert glide.line_step(a([100., 300.]), a([0., 200.]), 50.0, 200.0) is None
    assert glide.line_step(a([100., 300.]), a([20., 220.]), 75.0, 200.0) == (2, 80.0)
    assert glide.line_step(a([]), a([0.]), 0.0, 200.0) is None


def test_a_still_band_starts_moving_on_two_lines_or_two_frames_that_agree():
    assert glide.start_guard((1, 100.0), 0.0, None) == (None, 100.0)
    assert glide.start_guard((1, 105.0), 0.0, 100.0) == ((1, 105.0), 105.0)
    assert glide.start_guard((1, 130.0), 0.0, 100.0) == (None, 130.0)
    assert glide.start_guard((2, 100.0), 0.0, None) == ((2, 100.0), None)
    # A moving band, or a step under a pixel, needs no second look
    assert glide.start_guard((1, 100.0), 20.0, None) == ((1, 100.0), None)
    assert glide.start_guard((1, 0.5), 0.0, None) == ((1, 0.5), None)
    assert glide.start_guard(None, 0.0, 100.0) == (None, None)


def test_a_line_fixed_to_the_screen_is_found_in_the_median_of_the_glide(monkeypatch):
    strip = sheet(3000)

    def frames(path, fps, crop=None, start=0.0, length=None):
        x = int(round(start * 10)) * 20
        f = strip[:, x:x + 800].copy()
        cv2.line(f, (300, 30), (300, 100), (90, 90, 90), 2)
        yield start, f
    monkeypatch.setattr(glide, "frames", frames)
    fixed = glide.fixed_columns(None, None, STAFF, "dark", 140.0, 0.0, 10.0)
    assert np.flatnonzero(fixed).tolist() == list(range(296, 305))


def test_a_smoothly_eased_glide_over_bars_of_rest_keeps_its_travel(monkeypatch):
    strip = sheet(5000)
    strip[:, 900:] = 255
    for i in range(6):
        cv2.line(strip, (900, 30 + i * GAP), (4999, 30 + i * GAP), (60, 60, 60), 1)
    for x in range(1000, 5000, 400):
        cv2.line(strip, (x, 30), (x, 100), (0, 0, 0), 2)
    # 20 px a frame, then a speed that eases up and down within each bar
    xs = [0]
    for k in range(1, 100):
        xs.append(xs[-1] + (20 if k < 20 else [20, 23, 26, 29, 26, 23][k % 6]))

    def frames(path, fps, crop=None, start=0.0, length=None):
        for k, x in enumerate(xs):
            yield start + k / fps, strip[:, x:x + 800].copy()
    m = mosaic_of(monkeypatch, frames)
    assert xs[-1] == 2346
    assert abs(m.offsets[-1] - 2346) <= 5



def test_staff_lines_are_thin_dark_rows_and_a_redraw_shares_few_of_them():
    plate = np.full((60, 400), 250, np.uint8)
    plate[[10, 24, 38]] = 120
    plate[45:56] = 30
    assert glide.line_rows(plate, 140.0) == [10, 24, 38]
    assert glide.shared([10, 24, 39], [10, 24, 38, 52]) == 0.75
    assert glide.shared([13, 27], [10, 24]) == 0.0


def lowered_frames(strip: np.ndarray, count: int, drop):
    """frames() for a band gliding left 20 px per frame over an 800 px panel, frame k drawn drop(k) px lower."""
    def frames(path, fps, crop=None, start=0.0, length=None):
        for k in range(count):
            f = strip[:, k * 20:k * 20 + 800].copy()
            dy = drop(k)
            if dy:
                f[dy:] = f[:-dy].copy()
                f[:dy] = 255
            yield start + k / fps, f
    return frames


def test_a_glide_piece_ends_where_the_panel_is_redrawn_lower(monkeypatch):
    m = mosaic_of(monkeypatch, lowered_frames(sheet(3000), 60, lambda k: 20 if k >= 30 else 0))
    assert m.moved_at == 3.0
    assert len(m.offsets) == 30
    assert m.until == 2.9


def test_a_lone_lowered_frame_long_before_the_redraw_is_forgotten(monkeypatch):
    # One lowered frame, then 69 blank frames, then the panel redrawn lower from frame 100
    lowered = lowered_frames(sheet(4000), 120, lambda k: 20 if k == 30 or k >= 100 else 0)

    def frames(path, fps, crop=None, start=0.0, length=None):
        for k, (t, f) in enumerate(lowered(path, fps, crop, start, length)):
            if 30 < k < 100:
                f[:] = 255
            yield t, f
    m = mosaic_of(monkeypatch, frames)
    assert m.moved_at == 10.0
    assert len(m.offsets) == 100


@pytest.mark.parametrize("redraw", [87, 88, 89])
def test_a_redraw_soon_after_a_lone_lowered_frame_ends_the_piece_at_the_redraw(monkeypatch, redraw):
    # One lowered frame at 30, blank frames, then the panel redrawn lower from `redraw`
    lowered = lowered_frames(sheet(4000), redraw + 20, lambda k: 20 if k == 30 or k >= redraw else 0)

    def frames(path, fps, crop=None, start=0.0, length=None):
        for k, (t, f) in enumerate(lowered(path, fps, crop, start, length)):
            if 30 < k < redraw:
                f[:] = 255
            yield t, f
    m = mosaic_of(monkeypatch, frames)
    assert m.moved_at == redraw / 10
    assert m.until < redraw / 10


def test_a_staff_sinking_a_pixel_at_a_time_is_followed(monkeypatch):
    # A slow zoom: the staff sinks 1 px every 10 frames
    m = mosaic_of(monkeypatch, lowered_frames(sheet(3000), 60, lambda k: k // 10))
    assert m.moved_at is None
    assert len(m.offsets) == 60

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


def test_a_mosaic_whose_staff_sits_lower_in_two_stretches_keeps_its_staff():
    image = sheet(2400)
    # The staff drawn 3 px lower from x 600 to 1050 and from 1500 to 1950, as when the band moved between frames
    for a, b in ((600, 1050), (1500, 1950)):
        image[3:, a:b] = image[:-3, a:b].copy()
        image[:3, a:b] = 255
    (ps,) = pipeline.parse_mosaic(0, image, (0, 0, 2400, 130), "dark", 140.0, 229.0, GAP)
    assert len(ps.system.tab.lines) == 6
