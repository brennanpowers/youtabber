"""Run the whole pipeline on made-up videos whose tab we know, one layout per case."""
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from synth import Song, write_video
from youtabber import ink, layout, region, stitch, views
from youtabber.source import Source

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")

CASES = {
    "pages at the bottom": Song(),
    "pages at the top": Song(place="top"),
    "pages in a box": Song(place="inset"),
    "tab only, no notation": Song(notation=False, tab_height=200),
    "bass, four strings": Song(strings=4, notation=False, tab_height=180),
    "jump scroll": Song(scroll=True, measures=20),
    "faint lines on yellow": Song(paper=(170, 245, 255), line_gray=190, cursor=(230, 120, 30)),
    "staffs move between pages": Song(page_offsets=[0, 9, -7, 4]),
}


@dataclass
class Result:
    song: Song
    shown: list[np.ndarray]
    region: region.Region
    views: list[views.View]
    strip: np.ndarray
    placed: list[stitch.Placed]
    pdf_rows: list[np.ndarray]  # each row as it went into the PDF, after edge cleanup


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def result(request, tmp_path_factory) -> Result:
    song = CASES[request.param]
    tmp = tmp_path_factory.mktemp("video")
    video = tmp / "song.mp4"
    shown = write_video(song, video)
    found_region = region.detect(video).region
    assert found_region is not None, "no tab found"
    found = list(views.find_views(video, found_region))
    strip, placed = stitch.stitch(found)
    scale = layout.scale_for(strip, found_region.w)
    rows = layout.rows(strip, placed, int(layout.USABLE_WIDTH / scale))
    pdf_rows: list[np.ndarray] = []

    def record(row: np.ndarray) -> np.ndarray:
        pdf_rows.append(row)
        return row

    source = Source(video, "song", "Test Song", None, None)
    layout.write_pdf(strip, rows, scale, tmp / "song.pdf", source, placed, record)
    return Result(song, shown, found_region, found, strip, placed, pdf_rows)


def test_region_covers_the_tab(result: Result):
    x, y, w, h = result.song.tab_box
    r = result.region
    # The region may stop short of the paper's edge, but must hold every staff line
    lines_top = y + (30 if result.song.notation else result.song.tab_top) - 10
    lines_bottom = y + result.song.tab_top + (result.song.strings - 1) * result.song.line_spacing + 10
    assert r.x >= x - 2 and r.x + r.w <= x + w + 2
    assert r.y <= lines_top and r.y + r.h >= lines_bottom
    assert r.y >= y - 2 and r.y + r.h <= y + h + 2


def test_one_view_per_page_or_scroll_position(result: Result):
    assert len(result.views) == len(result.shown)


def test_scroll_joins_every_view(result: Result):
    if not result.song.scroll:
        pytest.skip("pages don't overlap")
    assert all(p.new_from for p in result.placed[1:]), "a view wasn't joined to the one before"
    step = int(result.song.tab_box[2] * result.song.scroll_step)
    shifts = np.diff([p.x for p in result.placed])
    assert np.all(np.abs(shifts - step) <= 2), shifts


def test_tab_staff_found_with_every_string(result: Result):
    tab = ink.tab_staff(ink.staff_lines(result.strip))
    assert len(tab) == result.song.strings
    assert abs(np.median(np.diff(tab)) - result.song.line_spacing) <= 1


def test_every_barline_found(result: Result):
    bars = layout.barlines(result.strip, ink.staff_lines(result.strip))
    assert len(bars) == result.song.measures


def test_pdf_rows_keep_their_staff_lines(result: Result):
    for i, row in enumerate(result.pdf_rows):
        tab = ink.tab_staff(ink.staff_lines(row))
        assert len(tab) == result.song.strings, f"row {i} lost tab lines"
