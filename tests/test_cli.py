import argparse
import json
import shutil

import pytest

from synth import Song, write_video
from youtabber import cli

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def args(video, out, **kw) -> argparse.Namespace:
    base = dict(source=str(video), out=out, title=None, region=None, enhance=False, model=None, passes=None)
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "song.mp4"
    write_video(Song(), path)
    return path


def test_writes_pdf_meta_and_measure_images(video, tmp_path):
    code = cli.run(args(video, tmp_path / "out"), {})
    assert code in (cli.EXIT_CLEAN, cli.EXIT_CHECK)
    meta = json.loads((tmp_path / "out" / "meta.json").read_text())
    assert meta["version"] == 2
    assert meta["outcome"]["status"] in ("ok", "warn")
    assert (tmp_path / "out" / "song.pdf").exists()
    assert len(list((tmp_path / "out" / "measures").glob("*.png"))) == len(meta["measures"]) > 0
    assert len(list((tmp_path / "out" / "views").glob("*.png"))) == len(meta["views"])


def test_passes_comes_from_the_config(video, tmp_path):
    cli.run(args(video, tmp_path / "out"), {"passes": "all"})
    assert json.loads((tmp_path / "out" / "meta.json").read_text())["passes"] == "all"


def test_bad_passes_in_the_config_fails_before_reading(video, tmp_path):
    with pytest.raises(SystemExit, match="passes"):
        cli.run(args(video, tmp_path / "out"), {"passes": "twice"})


def test_region_without_tab_fails(video, tmp_path):
    with pytest.raises(SystemExit, match="--region"):
        cli.run(args(video, tmp_path / "out", region=(0, 0, 1280, 300)), {})


def test_region_around_the_tab_reads_it(video, tmp_path):
    assert cli.run(args(video, tmp_path / "out", region=(0, 440, 1280, 280)), {}) in (cli.EXIT_CLEAN, cli.EXIT_CHECK)


def test_pdfs_are_copied_to_pdf_dir(video, tmp_path):
    cli.run(args(video, tmp_path / "out"), {"pdf_dir": str(tmp_path / "copies")})
    assert (tmp_path / "copies" / "song.pdf").exists()
