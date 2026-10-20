import argparse
import io
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from synth import Song, write_video
from youtabber import cli, enhance, layout, pipeline, source
from youtabber.ui import Ui

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")

NO_TAB = ("No tab on screen: no steady set of evenly spaced staff lines on paper in 9 s of video. youtabber can't "
          "read text tab (e|--3--) or tab drawn see-through over the video either.")


def args(video, out, **kw) -> argparse.Namespace:
    base = dict(source=str(video), out=out, title=None, region=None, enhance=False, model=None, passes=None)
    base.update(kw)
    return argparse.Namespace(**base)


def main(monkeypatch, *argv) -> int:
    """cli.main with these arguments and no config file, returning its exit code."""
    monkeypatch.setattr(cli, "load_config", lambda: {})
    monkeypatch.setattr(sys, "argv", ["youtabber", *map(str, argv)])
    with pytest.raises(SystemExit) as e:
        cli.main()
    return e.value.code


def lines(text: str, key: str) -> list[str]:
    return [line[len(key) + 2:] for line in text.splitlines() if line.startswith(key + ": ")]


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "song.mp4"
    write_video(Song(), path)
    return path


@pytest.fixture(scope="module")
def reading(video):
    return pipeline.read(video)


@pytest.fixture
def reads(monkeypatch):
    """Replace pipeline.read with one that returns the reading put in the list, and records each call. It reports
    each stage's start and end to progress, as pipeline.read does. A failed reading reports only finding, as one
    that finds no tab does."""
    calls, result = [], []

    def read(path, region=None, progress=None):
        calls.append(path)
        if isinstance(result[0], BaseException):
            raise result[0]
        if progress:
            for stage in ("find",) if result[0].outcome == "fail" else ("find", "read", "assemble"):
                progress(stage, 0.0)
                progress(stage, 1.0)
        return result[0]

    monkeypatch.setattr(pipeline, "read", read)
    return calls, result


def test_writes_pdf_meta_and_measure_images(video, tmp_path):
    code = cli.run(args(video, tmp_path / "out"), {})
    assert code in (cli.EXIT_CLEAN, cli.EXIT_CHECK)
    meta = json.loads((tmp_path / "out" / "meta.json").read_text())
    assert meta["version"] == 2
    assert meta["outcome"]["status"] in ("ok", "warn")
    assert meta["runtime_s"] > 0
    assert (tmp_path / "out" / "song.pdf").exists()
    assert len(list((tmp_path / "out" / "measures").glob("*.png"))) == len(meta["measures"]) > 0
    assert len(list((tmp_path / "out" / "views").glob("*.png"))) == len(meta["views"])


def test_passes_comes_from_the_config(video, tmp_path):
    cli.run(args(video, tmp_path / "out"), {"passes": "all"})
    assert json.loads((tmp_path / "out" / "meta.json").read_text())["passes"] == "all"


def test_bad_passes_in_the_config_fails_before_reading(video, tmp_path, reads, capsys):
    assert cli.run(args(video, tmp_path / "out"), {"passes": "twice"}) == cli.EXIT_FAILED
    assert "passes" in capsys.readouterr().err
    assert reads[0] == []


def test_a_broken_config_file_says_how_to_fix_it_and_exits_1(tmp_path, monkeypatch, reads, capsys):
    config = tmp_path / "config.toml"
    config.write_text('passes = "all\n')
    monkeypatch.setattr(cli, "CONFIG", config)
    monkeypatch.setattr(sys, "argv", ["youtabber", "song.mp4"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == cli.EXIT_FAILED
    err = capsys.readouterr().err.splitlines()
    assert err[0] == f"Can't read {config}"
    assert err[1].startswith("  ") and err[-1].endswith("Fix the file or move it aside.")
    assert reads[0] == []


def test_region_without_tab_fails(video, tmp_path, capsys):
    assert cli.run(args(video, tmp_path / "out", region=(0, 0, 1280, 300)), {}) == cli.EXIT_FAILED
    assert "No tab staff inside --region 0,0,1280,300" in capsys.readouterr().err


def test_region_around_the_tab_reads_it(video, tmp_path):
    assert cli.run(args(video, tmp_path / "out", region=(0, 440, 1280, 280)), {}) in (cli.EXIT_CLEAN, cli.EXIT_CHECK)


def test_pdfs_are_copied_to_pdf_dir(video, tmp_path, capsys):
    cli.run(args(video, tmp_path / "out"), {"pdf_dir": str(tmp_path / "copies")})
    assert (tmp_path / "copies" / "song.pdf").exists()
    assert lines(capsys.readouterr().out, "copied") == [str(tmp_path / "copies" / "song.pdf")]


def test_pdf_dir_same_as_out_skips_the_copy(video, tmp_path, capsys):
    code = cli.run(args(video, tmp_path / "out"), {"pdf_dir": str(tmp_path / "out")})
    assert code in (cli.EXIT_CLEAN, cli.EXIT_CHECK)
    assert (tmp_path / "out" / "song.pdf").exists()
    assert "copied:" not in capsys.readouterr().out


def test_pdf_dir_spelled_in_other_case_skips_the_copy(video, tmp_path, capsys):
    (tmp_path / "out").mkdir()
    if not (tmp_path / "OUT").exists():
        pytest.skip("filesystem is case-sensitive")
    code = cli.run(args(video, tmp_path / "out"), {"pdf_dir": str(tmp_path / "OUT")})
    assert code in (cli.EXIT_CLEAN, cli.EXIT_CHECK)
    assert (tmp_path / "out" / "song.pdf").exists()
    assert "copied:" not in capsys.readouterr().out


@pytest.fixture(scope="module")
def folded_reading(video):
    """The video's reading with a second, folded reading added, as when a replay can't be told from a repeat."""
    r = pipeline.read(video)
    r.candidates.append(pipeline.Candidate("folded", "once", list(r.candidates[0].measures)))
    r.notices.append(pipeline.Notice(1.0, 2.0, "unresolved_fold", "These measures repeat.", False))
    r.message = "Wrote two PDFs."
    return r


def test_all_mode_writes_one_pdf_and_says_so(folded_reading, video, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.pipeline, "read", lambda path, region=None, progress=None: folded_reading)
    out = tmp_path / "out"
    assert cli.run(args(video, out), {}) == cli.EXIT_CHECK
    assert sorted(p.name for p in out.glob("*.pdf")) == ["song-folded.pdf", "song.pdf"]
    # Rerun into the same folder: the folded PDF from the first run is gone, and the fold choice alone is no
    # reason to check
    assert cli.run(args(video, out, passes="all"), {}) == cli.EXIT_CLEAN
    meta = json.loads((out / "meta.json").read_text())
    assert [p.name for p in out.glob("*.pdf")] == meta["outcome"]["pdfs"] == ["song.pdf"]
    assert meta["outcome"]["status"] == "ok"
    assert "two PDFs" not in meta["outcome"]["message"]


def test_a_clean_run_prints_its_pdf_and_exits_0(reading, video, tmp_path, monkeypatch, reads, capsys):
    reads[1].append(replace(reading, notices=[]))
    assert main(monkeypatch, video, "-o", tmp_path / "out", "--no-enhance") == cli.EXIT_CLEAN
    out = capsys.readouterr().out
    assert lines(out, "source") == [str(video)]
    assert lines(out, "stage") == ["finding the tab", "reading the screens", "piecing it together", "printing"]
    assert lines(out, "pdf") == [str(tmp_path / "out" / "song.pdf")]
    assert lines(out, "check") == []
    assert out.splitlines()[-1] == "status: ok"


def test_spots_print_as_check_lines_and_exit_3(reading, video, tmp_path, monkeypatch, reads, capsys):
    spots = [pipeline.Notice(4.0, 6.0, "wide_measure", "A measure is wide.", True),
             pipeline.Notice(1.0, 2.0, "unresolved_join", "Couldn't tell.", True)]
    reads[1].append(replace(reading, notices=spots))
    assert main(monkeypatch, video, "-o", tmp_path / "out", "--no-enhance") == cli.EXIT_CHECK
    out = capsys.readouterr().out
    assert lines(out, "check") == ["0:01-0:02 Couldn't tell.", "0:04-0:06 A measure is wide."]
    assert out.splitlines()[-1] == "status: check"


def test_a_refusal_says_why_and_exits_1(video, tmp_path, monkeypatch, reads, capsys):
    reads[1].append(pipeline.Reading(9.0).fail("no_tab", NO_TAB))
    assert main(monkeypatch, video, "-o", tmp_path / "out") == cli.EXIT_FAILED
    out, err = capsys.readouterr()
    assert lines(out, "status") == ["failed"] and lines(out, "reason") == [NO_TAB]
    assert lines(out, "pdf") == []
    assert err.startswith("No tab here\n\n  No steady set of evenly spaced staff lines")
    # No screens were written, so there's nothing in views/ to point at
    assert f"Details: {tmp_path / 'out' / 'meta.json'}" in err and "screens it looked at" not in err


def test_ctrl_c_while_reading_exits_130(video, tmp_path, monkeypatch, reads, capsys):
    reads[1].append(KeyboardInterrupt())
    assert main(monkeypatch, video, "-o", tmp_path / "out") == cli.EXIT_STOPPED
    assert capsys.readouterr().err.strip() == "Stopped. Nothing was written."
    assert not (tmp_path / "out").exists()


def test_ctrl_c_while_writing_says_the_folder_may_be_incomplete(reading, video, tmp_path, monkeypatch, reads,
                                                                capsys):
    reads[1].append(reading)

    def stop(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(layout, "write_pdf", stop)
    assert main(monkeypatch, video, "-o", tmp_path / "out", "--no-enhance") == cli.EXIT_STOPPED
    assert capsys.readouterr().err.strip() == f"Stopped while writing; {tmp_path / 'out'}/ may be incomplete."


def test_an_error_while_writing_is_shown_without_a_traceback(reading, video, tmp_path, monkeypatch, reads, capsys):
    reads[1].append(reading)

    def full(*a, **kw):
        raise OSError(28, "No space left on device", "song.pdf")

    monkeypatch.setattr(layout, "write_pdf", full)
    assert main(monkeypatch, video, "-o", tmp_path / "out", "--no-enhance") == cli.EXIT_FAILED
    assert capsys.readouterr().err == "Can't write to song.pdf\n  No space left on device. Pick another folder with -o.\n"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_a_read_only_folder_fails_before_reading(video, tmp_path, monkeypatch, reads, capsys):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        assert main(monkeypatch, video, "-o", locked / "out") == cli.EXIT_FAILED
    finally:
        locked.chmod(0o700)
    assert capsys.readouterr().err == f"Can't write to {locked / 'out'}\n  Permission denied. Pick another folder with -o.\n"
    assert reads[0] == []


def test_out_naming_a_file_fails_before_reading(video, tmp_path, monkeypatch, reads, capsys):
    (tmp_path / "notes.txt").write_text("")
    assert main(monkeypatch, video, "-o", tmp_path / "notes.txt") == cli.EXIT_FAILED
    assert "That's a file, not a folder" in capsys.readouterr().err
    assert reads[0] == []


def test_a_region_outside_the_video_fails_before_reading(video, tmp_path, monkeypatch, reads, capsys):
    assert main(monkeypatch, video, "-o", tmp_path / "out", "--region", "5000,5000,100,100") == cli.EXIT_USAGE
    assert capsys.readouterr().err.startswith("--region 5000,5000,100,100 is outside the video, which is 1280x720\n")
    assert reads[0] == []


def test_an_unknown_model_fails_before_loading_it_or_reading(video, tmp_path, monkeypatch, reads, capsys):
    loaded = []
    monkeypatch.setattr(enhance, "model_upscaler", loaded.append)
    assert main(monkeypatch, video, "-o", tmp_path / "out", "--model", "realesrgan-anim") == cli.EXIT_USAGE
    assert capsys.readouterr().err.startswith("Unknown model 'realesrgan-anim'\n  Use classic, realesrgan-anime,")
    assert loaded == [] and reads[0] == []


def test_help_lists_the_exit_codes(monkeypatch, capsys):
    # argparse wraps its help to the terminal's width, so pin the width to the 80 columns being checked
    monkeypatch.setenv("COLUMNS", "80")
    assert main(monkeypatch, "--help") == 0
    out = capsys.readouterr().out
    assert "--region X,Y,W,H" in out
    assert "exit codes: 0 clean PDF, 1 no PDF, 2 bad arguments, 3 PDF with spots to check,\n  130 stopped" in out
    assert max(len(line) for line in out.splitlines()) < 80


class StageUi:
    """Records each stage call, and ignores everything else the run shows."""

    def __init__(self):
        self.stages = []

    def stage(self, *a):
        self.stages.append(a)

    def stage_done(self, *a):
        pass


def test_reading_shows_the_video_time_it_has_reached(reading, monkeypatch):
    def read(path, region=None, progress=None):
        progress("find", 0.0)
        progress("read", 0.5)
        return reading

    monkeypatch.setattr(pipeline, "read", read)
    ui = StageUi()
    cli.read_with_progress("song.mp4", None, 40.0, ui)
    assert ("read", 0.5, 20.0) in ui.stages


def test_the_read_message_counts_one_measure_in_the_singular(reading):
    one = replace(reading, candidates=[replace(reading.candidates[0], measures=reading.candidates[0].measures[:1])])
    assert cli.run_message(one, "once") == "Read 1 measure."


class Tty(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


def terminal() -> Ui:
    return Ui(Tty(), Tty(), {"NO_COLOR": "1"}, width=80)


class BrokenUi(StageUi):
    """A ui whose second stage call raises."""

    def stage(self, *a):
        super().stage(*a)
        if len(self.stages) == 2:
            raise ValueError("bad draw")


def test_an_error_showing_progress_is_raised_after_the_reading_finishes(reading, monkeypatch):
    heard = []

    def read(path, region=None, progress=None):
        for share in (0.0, 0.5, 1.0):
            progress("find", share)
            heard.append(share)
        return reading

    monkeypatch.setattr(pipeline, "read", read)
    ui = BrokenUi()
    with pytest.raises(ValueError, match="bad draw"):
        cli.read_with_progress("song.mp4", None, 40.0, ui)
    assert heard == [0.0, 0.5, 1.0]
    assert len(ui.stages) == 2


def test_ctrl_c_while_showing_progress_stops_the_reading_at_once(reading, monkeypatch):
    heard = []

    def read(path, region=None, progress=None):
        for share in (0.0, 0.5, 1.0):
            progress("find", share)
            heard.append(share)
        return reading

    class StopUi(StageUi):
        def stage(self, *a):
            super().stage(*a)
            if len(self.stages) == 2:
                raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "read", read)
    with pytest.raises(KeyboardInterrupt):
        cli.read_with_progress("song.mp4", None, 40.0, StopUi())
    assert heard == []


def test_an_unexpected_error_clears_the_live_line_first(video, tmp_path, reads):
    reads[1].append(RuntimeError("a bug"))
    ui = terminal()
    with pytest.raises(RuntimeError):
        cli.run(args(video, tmp_path / "out"), {}, ui)
    assert ui.err.getvalue().endswith("\r\x1b[K") and not ui.drawn


def test_one_screen_is_counted_in_the_singular(reading, video, tmp_path, monkeypatch, reads):
    reads[1].append(replace(reading, views=reading.views[:1]))

    def stop(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "write_outputs", stop)
    ui = terminal()
    assert cli.run(args(video, tmp_path / "out"), {}, ui) == cli.EXIT_STOPPED
    assert "✓ Read 1 screen " in ui.out.getvalue()


def test_a_download_of_unknown_length_prints_the_header_first(reading, video, tmp_path, monkeypatch, reads):
    reads[1].append(replace(reading, notices=[]))

    def fetch(text, on_progress=None, on_start=None):
        found = source.Source(video, "abcdefghijk", "Song", "Chan", "https://www.youtube.com/watch?v=abcdefghijk")
        on_start(found)
        on_progress(1000, None, None)
        return replace(found, downloaded_bytes=1_234_567, download_s=0.4)

    monkeypatch.setattr(source, "fetch", fetch)
    ui = terminal()
    cli.run(args("https://www.youtube.com/watch?v=abcdefghijk", tmp_path / "out"), {}, ui)
    out = ui.out.getvalue().splitlines()
    assert out[:3] == ["♪ Song", "  tab by Chan · youtube.com/watch?v=abcdefghijk",
                       "✓ Downloaded 1.2 MB in 0 s"]
    assert sum(line.startswith("♪") for line in out) == 1


def test_megabytes_keep_a_decimal_under_10():
    assert [cli.megabytes(n) for n in (400_000, 9_940_000, 10_000_000, 78_400_000)] == ["0.4", "9.9", "10", "78"]


def test_ctrl_c_while_copying_names_the_copy_not_the_output_folder(reading, video, tmp_path, monkeypatch, reads,
                                                                   capsys):
    reads[1].append(reading)

    def stop(*a):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "copy_pdf", stop)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert cli.run(args(video, tmp_path / "out"), {"pdf_dir": str(tmp_path / "tabs")}) == cli.EXIT_STOPPED
    assert capsys.readouterr().err.strip() == (
        f"Stopped while copying to ~/tabs/; the copy there may be incomplete. "
        f"Everything was written to {tmp_path / 'out'}/.")


def test_a_file_that_isnt_a_video_says_what_to_check(tmp_path, monkeypatch, capsys):
    notes = tmp_path / "notes.mp4"
    notes.write_text("not a video\n")
    assert main(monkeypatch, notes, "-o", tmp_path / "out") == cli.EXIT_FAILED
    err = capsys.readouterr().err
    assert err.startswith(f"Can't read {notes} as a video\n  ")
    assert err.endswith("\n  Check it plays in a video player.\n")


def test_a_model_without_pytorch_says_how_to_install_it(video, tmp_path, monkeypatch, reads, capsys):
    # A None entry makes the import fail, whether or not PyTorch is installed
    monkeypatch.setitem(sys.modules, "torch", None)
    assert main(monkeypatch, video, "-o", tmp_path / "out", "--model", "realesrgan-anime") == cli.EXIT_FAILED
    err = capsys.readouterr().err.splitlines()
    assert err[:2] == ["Running a model needs PyTorch and spandrel", "  Reinstall with the ai extra:"]
    assert err[2].startswith("  uv tool install --editable '") and err[2].endswith("[ai]'")
    assert reads[0] == []


def test_downloading_a_model_says_so_on_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(enhance, "CACHE", tmp_path)
    monkeypatch.setattr(enhance.urllib.request, "urlretrieve", lambda url, path: Path(path).write_bytes(b"m"))
    assert enhance._model_file("realesrgan-anime").exists()
    assert capsys.readouterr() == ("", "Downloading model realesrgan-anime\n")


def test_a_quoted_tilde_in_out_is_the_home_folder():
    assert cli.parser().parse_args(["song.mp4", "-o", "~/tabs/x"]).out == Path.home() / "tabs" / "x"


def test_an_image_that_cant_be_saved_is_an_error_naming_it(reading, video, tmp_path, monkeypatch, reads, capsys):
    reads[1].append(reading)
    monkeypatch.setattr(cli.cv2, "imwrite", lambda *a: False)
    assert main(monkeypatch, video, "-o", tmp_path / "out", "--no-enhance") == cli.EXIT_FAILED
    assert capsys.readouterr().err == (f"Can't write to {tmp_path / 'out' / 'measures' / '0000.png'}\n"
                                       f"  Couldn't save the image. Pick another folder with -o.\n")
