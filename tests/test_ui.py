import io
import os
import json
import re
from pathlib import Path

from youtabber.ui import spot_lines
from youtabber.pipeline import Notice
from youtabber.ui import LABELS, Ui

DATA = Path(__file__).parent / "data"
PDF = "out/slipknot-before-i-forget-pov-screen/slipknot-before-i-forget-pov-screen"


class Tty(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


class AsciiTty(Tty):
    encoding = "ascii"


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def notices(name):
    return [Notice(**n) for n in json.loads((DATA / f"{name}_notices.json").read_text())]


def result(out, width=80, env=None):
    """The comparison run's PDF paths and spots, as printed in one mode."""
    ui = Ui(out, Tty(), {"NO_COLOR": "1"} if env is None else env, width=width)
    ui.wrote(PDF + ".pdf")
    ui.wrote(PDF + "-folded.pdf")
    ui.spots(notices("glide"))
    ui.finish("check", "Check those 21 spots against the video, starting with the first PDF.")
    return out.getvalue()


def test_terminal_transcript_of_the_comparison_run():
    assert result(Tty()) == f"""
→ {PDF}.pdf
→ {PDF}-folded.pdf
  The folded one prints 3:11-3:13 once; the video doesn't say if it's played
  twice.

! 21 spots to check, listed on the last page of the PDF:
  19 wide measures, a barline may be missing
     0:04 0:19 0:22 0:25 0:27 0:30 0:32 1:40 1:42 1:45 3:09 3:12 3:15 4:08 …
   2 screens that may not join up         0:32 4:21

! Check those 21 spots against the video, starting with the first PDF.
"""


def test_ascii_transcript_of_the_comparison_run():
    assert result(AsciiTty()) == f"""
-> {PDF}.pdf
-> {PDF}-folded.pdf
  The folded one prints 3:11-3:13 once; the video doesn't say if it's played
  twice.

! 21 spots to check, listed on the last page of the PDF:
  19 wide measures, a barline may be missing
     0:04 0:19 0:22 0:25 0:27 0:30 0:32 1:40 1:42 1:45 3:09 3:12 3:15 4:08 ...
   2 screens that may not join up         0:32 4:21

! Check those 21 spots against the video, starting with the first PDF.
"""


def test_narrow_transcript_of_the_comparison_run():
    assert result(Tty(), width=50) == f"""
→ {PDF}.pdf
→ {PDF}-folded.pdf
  The folded one prints 3:11-3:13 once; the video
  doesn't say if it's played twice.

! 21 spots to check (last page of the PDF):
  19 wide measures, a barline may be missing
     0:04 0:19 0:22 0:25 0:27 0:30 0:32 1:40 …
   2 screens that may not join up
     0:32 4:21

! Check those 21 spots against the video, starting with the first PDF.
"""


def test_plain_transcript_lists_every_spot_of_the_pdf():
    lines = result(io.StringIO()).splitlines()
    checks = [line for line in lines if line.startswith("check: ")]
    assert lines[:2] == [f"pdf: {PDF}.pdf", f"pdf: {PDF}-folded.pdf"]
    assert checks == ["check: " + s.replace("  ", " ", 1) for s in spot_lines(notices("glide"))]
    assert len(checks) == 21
    assert checks[0] == ("check: 0:02-0:06 A measure around 0:04 is much wider than the others; a barline may have "
                         "been missed there.")
    assert lines[-2:] == ["note: 3:11-3:13 The measures at 3:11-3:13 repeat ones first shown at 3:04-3:08, with no "
                          "sign whether the song plays them again or the video replays them.", "status: check"]


def test_busy_run_groups_by_kind_largest_first():
    out = Tty()
    Ui(out, Tty(), {"NO_COLOR": "1"}, width=80).spots(notices("busy"))
    text = out.getvalue()
    assert "and 4 more once" in text
    assert "! 42 spots to check" in text and len(spot_lines(notices("busy"))) == 42
    heads = [line[:5] for line in text.splitlines() if line[:5].strip().isdigit()]
    assert heads == ["  12 ", "  11 ", "   9 ", "   5 ", "   3 ", "   1 ", "   1 "]
    assert "   1 more measures than numbers read      0:00-6:46" in text
    assert "   1 measure number skipped               2:42" in text


def test_every_doubtful_kind_made_has_a_label():
    src = Path(__file__).parent.parent / "src" / "youtabber"
    text = (src / "pipeline.py").read_text() + (src / "check.py").read_text()
    made = set(re.findall(r'"kind": "(\w+)"|Notice\([^"]*"(\w+)",', text))
    kinds = {a or b for a, b in made} - {"unresolved_fold"}   # the one notice made without doubt
    assert len(kinds) == 33 and kinds <= set(LABELS)


def test_a_kind_without_a_label_shows_its_first_spots_text_and_duplicate_times_print_once():
    out = Tty()
    spots = [Notice(10.0, 12.0, "brand_new", "Something odd here.", True),
             Notice(10.4, 11.8, "brand_new", "Something odd there.", True)]
    Ui(out, Tty(), {"NO_COLOR": "1"}, width=80).spots(spots)
    assert out.getvalue().splitlines()[2:] == ["  2 Something odd here.                   0:11"]


def test_colour_marks_symbols_and_no_color_or_dumb_turns_it_off():
    assert "\x1b[33m!\x1b[0m" in result(Tty(), env={})
    assert "\x1b[" not in result(Tty(), env={"NO_COLOR": ""})
    assert "\x1b[" not in result(Tty(), env={"TERM": "dumb"})


def test_modes_follow_the_streams_and_env():
    assert Ui(io.StringIO(), Tty(), {}, width=80).plain
    piped_err = Ui(Tty(), io.StringIO(), {}, width=80)
    assert not piped_err.plain and not piped_err.live
    dumb = Ui(Tty(), Tty(), {"TERM": "dumb"}, width=80)
    assert not dumb.live and not dumb.colour and dumb.sym["ok"] == "✓"
    assert Ui(Tty(), Tty(), {}, width=50).narrow
    assert Ui(AsciiTty(), Tty(), {}, width=80).sym["ok"] == "ok"


def test_start_block_in_each_mode():
    out = Tty()
    Ui(out, Tty(), {"NO_COLOR": "1"}, width=80).start(
        "Slipknot – Before I Forget PoV - SCREEN", "Sightline Guitar", 273,
        "https://www.youtube.com/watch?v=wmboCKWIRhQ")
    assert out.getvalue() == ("♪ Slipknot – Before I Forget PoV - SCREEN\n"
                              "  tab by Sightline Guitar · 4:33 · youtube.com/watch?v=wmboCKWIRhQ\n")
    out = AsciiTty()
    Ui(out, Tty(), {"NO_COLOR": "1"}, width=80).start("A – B", None, 142, "/cache/GlGd8_eaPog.mp4")
    assert out.getvalue() == "* A - B\n  2:22 - GlGd8_eaPog.mp4\n"
    out = io.StringIO()
    Ui(out, Tty(), {}).start("A – B", None, 142, "/cache/song.mp4")
    assert out.getvalue() == "source: /cache/song.mp4\ntitle: A – B\nlength: 2:22\n"


def test_live_line_is_cleared_before_anything_else_prints():
    out, err = Tty(), Tty()
    ui = Ui(out, err, {"NO_COLOR": "1"}, width=80)
    ui.stage("find")
    ui.stage("find", 0.5, 30.0)
    assert err.getvalue() == "› Finding the tab\r\x1b[K› Finding the tab  ━━━━━━━━━━──────────  50%  at 0:30"
    ui.stage_done("Found the tab", 14)
    assert err.getvalue().endswith("\r\x1b[K")
    assert out.getvalue() == "✓ Found the tab" + " " * 52 + "14 s\n"
    ui.stage("read", 0.1)
    ui.error("Stopped. Nothing was written.")
    assert err.getvalue().endswith("10%\r\x1b[K✗ Stopped. Nothing was written.\n")


def test_narrow_live_line_uses_a_short_name_and_bar():
    err = Tty()
    ui = Ui(Tty(), err, {"NO_COLOR": "1"}, width=50)
    ui.stage("read", 0.63, 172.0)
    assert err.getvalue() == "› Reading  ━━━━━━────  63%  at 2:52"


def test_a_live_line_at_40_columns_drops_the_size_rather_than_wrap():
    err = Tty()
    Ui(Tty(), err, {"NO_COLOR": "1"}, width=40).download(41_000_000, 78_000_000, 30.0)
    assert err.getvalue() == "› Downloading  ━━━━━─────  53%"


def test_live_line_rounds_the_video_time_like_the_header():
    err = Tty()
    Ui(Tty(), err, {"NO_COLOR": "1"}, width=50).stage("read", 0.5, 29.6)
    assert err.getvalue().endswith("at 0:30")


def test_stage_lines_print_once_in_plain_mode_and_not_at_all_without_a_live_line():
    out, err = io.StringIO(), Tty()
    ui = Ui(out, err, {})
    for name in ("find", "find", "read", "assemble", "print"):
        ui.stage(name, 0.5)
    ui.fact("read", "151 measures in 53 s")
    ui.stage_done("Read 61 screens", 29)
    assert out.getvalue() == ("stage: finding the tab\nstage: reading the screens\nstage: piecing it together\n"
                              "stage: printing\nread: 151 measures in 53 s\n")
    err = Tty()
    ui = Ui(Tty(), err, {"TERM": "dumb"}, width=80)
    ui.stage("find", 0.2)
    ui.stage("read", 0.4)
    ui.stage_done("Read 61 screens", 29)
    assert err.getvalue() == ""


def test_time_left_waits_rounds_and_does_not_creep_up():
    clock = Clock()
    err = Tty()
    ui = Ui(Tty(), err, {"NO_COLOR": "1"}, width=80, clock=clock)

    def left(name, done, t):
        clock.t = t
        ui.stage(name, done, 60.0)
        return err.getvalue().rpartition("\r\x1b[K")[2].partition(" · ")[2] or None

    ui.stage("find", 0.0)
    assert left("find", 0.9, 9) is None          # under 10 s
    assert left("find", 0.4, 12) is None         # 10% of the run done
    assert left("read", 0.0, 12) == "about 35 s left"   # 25% done in 12 s: 36 s to go
    assert left("read", 0.0, 13) == "about 35 s left"   # 39 s is less than 5 s above what's shown
    assert left("read", 0.0, 14) == "about 40 s left"   # 42 s
    assert left("read", 0.5, 14) == "about 10 s left"   # 57.5% done: 10.3 s
    assert left("read", 0.0, 100) == "about 5 min left"  # 300 s


def test_time_left_rises_past_a_minute_and_rounds_halves_up():
    clock = Clock()
    err = Tty()
    ui = Ui(Tty(), err, {"NO_COLOR": "1"}, width=80, clock=clock)

    def left(done, t):
        clock.t = t
        ui.stage("find", done, 60.0)
        return err.getvalue().rpartition("\r\x1b[K")[2].partition(" · ")[2] or None

    ui.stage("find", 0.0)
    assert left(0.8, 11) == "about 45 s left"    # 20% done in 11 s: 44 s to go
    assert left(0.8, 13.5) == "about 55 s left"  # 54 s
    assert left(0.8, 16) == "about 1 min left"   # 64 s is 5 s or more above what's shown
    assert left(0.8, 30) == "about 2 min left"   # 120 s
    clock.t = 0.0
    ui = Ui(Tty(), err, {"NO_COLOR": "1"}, width=80, clock=clock)
    ui.stage("find", 0.0)
    assert left(0.8, 10.625) == "about 45 s left"  # 42.5 s rounds up, not to the even 40


def test_download_line():
    err = Tty()
    ui = Ui(Tty(), err, {"NO_COLOR": "1"}, width=80)
    ui.download(41_000_000, 78_000_000, 9)
    assert err.getvalue() == "› Downloading  ━━━━━━━━━━━─────────  53%  41 of 78 MB · 9 s left"
    ui.download(41_000_000, None, None)
    assert err.getvalue().endswith("\r\x1b[K› Downloading  41 MB so far")


def test_failure_clears_the_live_line():
    out, err = Tty(), Tty()
    ui = Ui(out, err, {"NO_COLOR": "1"}, width=80)
    ui.stage("read", 0.4)
    ui.finish("failed", "Stopped")
    assert err.getvalue().endswith("40%\r\x1b[K")
    assert out.getvalue() == ""


def test_ascii_streams_print_any_title_credit_and_file_name():
    out = AsciiTty()
    ui = Ui(out, Tty(), {"NO_COLOR": "1"}, width=80)
    ui.start("Beyoncé – Halo", "Jean·Marc", 142, "/cache/café.mp4")
    ui.wrote("out/café/café.pdf")
    assert out.getvalue() == "* Beyonc? - Halo\n  tab by Jean-Marc - 2:22 - caf?.mp4\n\n-> out/caf?/caf?.pdf\n"
    out = AsciiTty()
    Ui(out, Tty(), {}).start("Beyoncé – Halo", None, 142, "/cache/song.mp4")
    assert out.getvalue() == "\x1b[1m*\x1b[0m \x1b[1mBeyonc? - Halo\x1b[0m\n  2:22 - \x1b[2msong.mp4\x1b[0m\n"


def test_colour_only_goes_to_a_terminal():
    err = io.StringIO()
    ui = Ui(Tty(), err, {}, width=80)
    ui.stage("find")
    ui.error("No tab here", "Try another video.")
    assert err.getvalue() == "✗ No tab here\n  Try another video.\n"


def test_default_streams_are_the_current_ones(capsys):
    ui = Ui(env={})
    ui.fact("read", "3 measures in 2 s")
    ui.error("Stopped")
    assert capsys.readouterr() == ("read: 3 measures in 2 s\n", "Stopped\n")


def test_errors_and_failure_in_each_mode():
    out, err = Tty(), Tty()
    ui = Ui(out, err, {"NO_COLOR": "1"}, width=80)
    ui.error("No tab here", "", "No steady set of evenly spaced staff lines on paper in 142 s of video.")
    ui.finish("failed", "No tab on screen")
    assert err.getvalue() == ("✗ No tab here\n\n  No steady set of evenly spaced staff lines on paper in 142 s of "
                              "video.\n")
    assert out.getvalue() == ""
    out, err = io.StringIO(), io.StringIO()
    ui = Ui(out, err, {})
    ui.finish("failed", "No tab on screen")
    ui.error("No file at song.mp4", "", "Check the path, or pass a YouTube URL.")
    assert out.getvalue() == "status: failed\nreason: No tab on screen\n"
    assert err.getvalue() == "No file at song.mp4\n\n  Check the path, or pass a YouTube URL.\n"


def test_clean_result_and_copy():
    out = Tty()
    ui = Ui(out, Tty(), {"NO_COLOR": "1"}, width=80)
    ui.wrote("out/a/a.pdf", Path.home() / "Documents" / "tabs" / "a.pdf")
    ui.finish("ok", "All clean: 30 measures, nothing to check. Happy playing.")
    assert out.getvalue() == ("\n→ out/a/a.pdf\n→ copied to ~/Documents/tabs/a.pdf\n\n"
                              "✓ All clean: 30 measures, nothing to check. Happy playing.\n")


class Closed(Tty):
    """A terminal stream whose reader has gone away, counting the writes tried on it."""

    def __init__(self):
        super().__init__()
        self.tries = 0

    def write(self, text):
        self.tries += 1
        raise BrokenPipeError(32, "Broken pipe")


def test_a_closed_stream_is_skipped_and_the_other_one_still_prints():
    out, err = Tty(), Closed()
    ui = Ui(out, err, {"NO_COLOR": "1"}, width=80)
    ui.stage("find", 0.5)
    ui.stage("find", 0.6)
    ui.error("No tab here")
    ui.stage_done("Found the tab", 3)
    assert err.tries == 1
    assert out.getvalue().startswith("✓ Found the tab")


def test_a_closed_pipe_is_pointed_at_dev_null_so_the_run_and_its_exit_flush_go_on():
    r, w = os.pipe()
    os.close(r)
    out = open(w, "w")
    try:
        ui = Ui(out, io.StringIO(), {})
        ui.fact("read", "3 measures in 2 s")
        ui.fact("read", "3 measures in 2 s")
        ui.finish("ok", "All clean.")
        assert ui.closed == [out]
        # What Python does at exit: flush whatever is left in the stream's buffer
        out.write("left over\n")
        out.flush()
    finally:
        out.close()


def test_start_without_a_length_leaves_it_out():
    out = Tty()
    Ui(out, Tty(), {"NO_COLOR": "1"}, width=80).start("Song", "Chan", None, "https://www.youtube.com/watch?v=abc")
    assert out.getvalue() == "♪ Song\n  tab by Chan · youtube.com/watch?v=abc\n"
    out = io.StringIO()
    Ui(out, Tty(), {}).start("Song", None, None, "https://www.youtube.com/watch?v=abc")
    assert out.getvalue() == "source: https://www.youtube.com/watch?v=abc\ntitle: Song\n"


def test_under_40_columns_there_is_no_live_line():
    assert not Ui(Tty(), Tty(), {}, width=39).live
    assert Ui(Tty(), Tty(), {}, width=40).live


def test_a_group_of_one_uses_the_singular_label():
    out = Tty()
    Ui(out, Tty(), {"NO_COLOR": "1"}, width=80).spots([
        Notice(10.0, 12.0, "wide_measure", "A wide one.", True),
        Notice(20.0, 22.0, "glide_gap", "A gap.", True), Notice(40.0, 42.0, "glide_gap", "Another gap.", True)])
    text = out.getvalue()
    assert "  2 long gaps between measures, some may be missing" in text
    assert "  1 wide measure, a barline may be missing" in text


def test_every_label_has_a_singular_and_a_plural():
    assert all(len(pair) == 2 and all(pair) for pair in LABELS.values())
