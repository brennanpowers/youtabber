"""pipeline.read's progress callback: stages in order, shares that only rise, and the same reading either way."""
import argparse
import json
import shutil
from dataclasses import replace

import pytest

from synth import Song, write_video
from youtabber import cli, pipeline

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")

STAGES = ["find", "read", "assemble"]


@pytest.fixture(scope="module", params=["pages", "smooth scroll"])
def video(request, tmp_path_factory):
    song = Song() if request.param == "pages" else Song(glide=True, notation=False, tab_height=200)
    path = tmp_path_factory.mktemp("video") / "song.mp4"
    write_video(song, path)
    return path


def check_stages(calls: list[tuple[str, float]], stages: list[str]) -> None:
    """Each stage's calls come together, in order, starting at 0.0 and never going down."""
    names = [s for s, _ in calls]
    assert [s for i, s in enumerate(names) if i == 0 or names[i - 1] != s] == stages
    for stage in stages:
        shares = [x for s, x in calls if s == stage]
        assert shares[0] == 0.0
        assert all(0.0 <= a <= b <= 1.0 for a, b in zip(shares, shares[1:])), stage


def test_stages_come_in_order_and_each_runs_from_0_to_1(video, monkeypatch):
    # Start the tab a few seconds in, so a share of the whole video differs from a share of the tab's span
    real = pipeline.scan

    def scan(*a, **kw):
        segments, found, scrolls = real(*a, **kw)
        segments[0] = replace(segments[0], t0=segments[0].t0 + 3.0)
        return segments, found, scrolls

    monkeypatch.setattr(pipeline, "scan", scan)
    calls = []
    r = pipeline.read(video, progress=lambda stage, share: calls.append((stage, share)))
    assert r.outcome == "tab", r.message
    check_stages(calls, STAGES)
    for stage in STAGES:
        assert [x for s, x in calls if s == stage][-1] == 1.0
    # The frame loops report along the way, not just at the ends
    assert len({x for s, x in calls if s == "find"}) > 3
    # Reading's share is the video time reached, so before its closing 1.0 it gets about as far as the tab's end
    read = [x for s, x in calls if s == "read"]
    assert read[1] == pytest.approx(r.segments[0].t0 / r.duration, abs=0.01)
    assert max(read[:-1]) == pytest.approx(r.segments[-1].t1 / r.duration, abs=0.05)


def test_a_failing_reading_still_sends_find(video):
    calls = []
    r = pipeline.read(video, region=(0, 0, 1280, 300), progress=lambda stage, share: calls.append((stage, share)))
    assert r.outcome == "fail"
    check_stages(calls, ["find"])
    assert calls[-1][1] > 0.5


def run(video, out, monkeypatch, progress) -> None:
    """cli.run with pipeline.read given `progress`, whatever the CLI passes it."""
    read = pipeline.read
    monkeypatch.setattr(pipeline, "read", lambda path, region=None, **kw: read(path, region, progress=progress))
    args = argparse.Namespace(source=str(video), out=out, title=None, region=None, enhance=False, model=None,
                              passes=None)
    cli.run(args, {})


def test_listening_doesnt_change_what_is_read(video, tmp_path, monkeypatch):
    run(video, tmp_path / "quiet", monkeypatch, None)
    run(video, tmp_path / "heard", monkeypatch, lambda stage, share: None)
    metas = []
    for name in ("quiet", "heard"):
        meta = json.loads((tmp_path / name / "meta.json").read_text())
        meta.pop("runtime_s")
        metas.append(meta)
    assert metas[0] == metas[1]
    images = [sorted((tmp_path / name / "measures").glob("*.png")) for name in ("quiet", "heard")]
    assert len(images[0]) > 0
    assert [p.read_bytes() for p in images[0]] == [p.read_bytes() for p in images[1]]
