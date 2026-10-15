"""Read made-up videos whose tab we know, one layout per case, and check what would print."""
import shutil

import pytest

from synth import Song, write_video
from youtabber import pipeline, printcell
from youtabber.staff import find_staffs

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
    "dark theme": Song(theme="dark", notation=False, tab_height=200),
    "smooth scroll": Song(glide=True, notation=False, tab_height=200),
}


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def case(request, tmp_path_factory):
    song = CASES[request.param]
    video = tmp_path_factory.mktemp("video") / "song.mp4"
    write_video(song, video)
    return song, pipeline.read(video)


def test_writes_a_tab(case):
    song, r = case
    assert r.outcome == "tab", r.message


# Known defects and what goes wrong. Strict: once a defect is fixed, its test fails until the entry is removed.
DEFECTS = {
    "pages in a box": "an empty staff end at the panel's right edge prints as a measure",
    "smooth scroll": "measures on screen when the glide starts or ends are not read",
}


def test_prints_every_measure_once(case, request):
    song, r = case
    reason = DEFECTS.get(request.node.callspec.id)
    if reason:
        request.applymarker(pytest.mark.xfail(strict=True, reason=reason))
    assert len(r.candidates) == 1
    assert len(r.candidates[0].measures) == song.measures


def test_numbers_count_up_when_read(case):
    song, r = case
    if r.numbering == "numbered":
        for i, m in enumerate(r.candidates[0].measures):
            assert m.label in (None, i + 1), f"measure {i + 1} labelled {m.label}"


def test_nothing_to_check(case):
    _, r = case
    assert [n.text for n in r.notices] == []


def test_every_pass_is_the_same_without_replays(case):
    _, r = case
    assert [m.slot for m in r.every_pass[0].measures] == [m.slot for m in r.candidates[0].measures]


def test_printed_measures_keep_every_string(case):
    song, r = case
    for m in r.candidates[0].measures:
        o = r.views[m.view]
        pc = printcell.print_cell(o, m.cell, r.segments[o.seg])
        assert any(len(s.lines) == song.strings for s in find_staffs(pc.image)), f"measure at {m.t} lost a string"
