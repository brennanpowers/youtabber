"""What the command prints: a terminal view with symbols, colour and one live progress line on stderr, or plain
key: value lines when stdout is piped."""
import os
import re
import shutil
import sys
import textwrap
import time
from dataclasses import replace
from pathlib import Path

from youtabber.check import fmt
from youtabber.pipeline import Notice

# Display name of each stage, and the share of the whole run that each reading stage takes
STAGES = {"find": ("Finding the tab", 0.25), "read": ("Reading the screens", 0.65),
          "assemble": ("Piecing it together", 0.10), "print": ("Printing", 0.0), "download": ("Downloading", 0.0)}
ORDER = ["find", "read", "assemble"]
# Label of each notice kind's line in the terminal's spot list, for a count of one and for more
LABELS = {
    "wide_measure": ("wide measure, a barline may be missing", "wide measures, a barline may be missing"),
    "long_measure": ("measure played as long as two, a barline may be missing",
                     "measures played as long as two, a barline may be missing"),
    "header": ("first cell left out as a clef and key, a measure may be missing",
               "first cells left out as a clef and key, measures may be missing"),
    "colored_bar": ("colored line read as a barline, a measure may be split",
                    "colored lines read as barlines, measures may be split"),
    "faint_bar": ("faint line not read as a barline, two measures may print as one",
                  "faint lines not read as barlines, two measures may print as one"),
    "header_kept": ("first cell that may be a clef and key printed as a measure",
                    "first cells that may be a clef and key printed as measures"),
    "unresolved_join": ("screen that may not join up", "screens that may not join up"),
    "edge_join": ("page edge joined by the playhead, a measure or part of one may be missing",
                  "page edges joined by the playhead, a measure or part of one may be missing"),
    "part_switch": ("possible part switch or jump", "possible part switches or jumps"),
    "shared_barlines": ("two staves that may be two parts", "places with two staves that may be two parts"),
    "intro": ("opening in another layout left out", "openings in another layout left out"),
    "version_change": ("possible other version printed over earlier measures",
                       "possible other versions printed over earlier measures"),
    "parts": ("place where parts show at once", "places where parts show at once"),
    "possible_duplicate": ("measure that may print twice", "measures that may print twice"),
    "number_again": ("measure number printed again", "measure numbers printed again"),
    "few_measures": ("screen with fewer measures than expected", "screens with fewer measures than expected"),
    "numbers_skipped": ("measure number skipped", "measure numbers skipped"),
    "numbers_start": ("first number read is high, opening measures may be missing",
                      "first number read is high, opening measures may be missing"),
    "count_over": ("more measures than numbers read", "more measures than numbers read"),
    "glide_repeat": ("measure that may repeat an earlier one", "measures that may repeat earlier ones"),
    "unsure_repeat": ("measure first shown where the video may have gone back, which may repeat",
                      "measures first shown where the video may have gone back, which may repeat"),
    "glide_gap": ("long gap between measures, some may be missing", "long gaps between measures, some may be missing"),
    "glide_number_gap": ("measure number that skips, a barline may be missing",
                         "measure numbers that skip, a barline may be missing"),
    "glide_step": ("sharp speed change in a scroll, a measure may be cut or doubled",
                   "sharp speed changes in a scroll, measures may be cut or doubled"),
    "glide_rows": ("stretch of a scroll with no tab staff found", "stretches of a scroll with no tab staff found"),
    "glide_piece": ("scroll redrawn mid-measure, a measure may be split",
                    "scrolls redrawn mid-measure, measures may be split"),
    "glide_start": ("scroll's first measures, may be cut off or misread",
                    "first measures of a scroll, may be cut off or misread"),
    "glide_end": ("scroll end read from one frame or not at all, measures may be missing",
                  "scroll ends read from one frame or not at all, measures may be missing"),
    "no_measures": ("stretch of tab with no measures printed", "stretches of tab with no measures printed"),
    "unread_tab": ("stretch of tab with no screen read, measures may be missing",
                   "stretches of tab with no screen read, measures may be missing"),
    "last_screen": ("brief last screen, its measures may be missing or wrong",
                    "brief last screens, their measures may be missing or wrong"),
    "numbers_disagree": ("measure number that doesn't fit the barlines", "measure numbers that don't fit the barlines"),
    "dropped_opening": ("brief page drawn at another size left out, its measures may be missing",
                        "brief pages drawn at another size left out, their measures may be missing"),
    "vertical_scroll": ("page scrolling up, its measures are missing", "pages scrolling up, their measures are missing"),
    "short_tab": ("stretch was the only tab read, check it's the right music",
                  "stretch was the only tab read, check it's the right music"),
}
UNICODE = {"mark": "♪", "ok": "✓", "check": "!", "failed": "✗", "stage": "›", "file": "→", "fill": "━",
           "empty": "─", "more": "…", "sep": "·", "dash": "–"}
ASCII = {"mark": "*", "ok": "ok", "check": "!", "failed": "x", "stage": ">", "file": "->", "fill": "#",
         "empty": "-", "more": "...", "sep": "-", "dash": "-"}
SGR = {"bold": 1, "dim": 2, "red": 31, "green": 32, "yellow": 33, "cyan": 36}
COLOURS = {"mark": "bold", "ok": "green", "check": "yellow", "failed": "red", "stage": "cyan"}
# Column where a spot group's times start, and the right edge of a finished stage's seconds
TIMES_COL, DONE_COL = 42, 71
# A spot longer than this shows its range; a shorter one shows its midpoint
RANGE_S = 30
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Notices reported in pieces of one long stretch
STRETCHES = ("no_measures", "vertical_scroll")


class Ui:
    def __init__(self, out=None, err=None, env=os.environ, width=None, clock=time.monotonic):
        # Look up the streams when the Ui is made, so a stream swapped in later is the one written to
        out = sys.stdout if out is None else out
        err = sys.stderr if err is None else err
        self.out, self.err, self.clock = out, err, clock
        dumb = env.get("TERM") == "dumb"
        self.width = width or shutil.get_terminal_size().columns
        self.narrow = self.width < 60
        self.plain = not out.isatty()
        # Under 40 columns even the short live line wraps, and a wrapped line can't be redrawn in place
        self.live = not self.plain and err.isatty() and not dumb and self.width >= 40
        self.colour = not self.plain and "NO_COLOR" not in env and not dumb
        self.err_colour = self.colour and err.isatty()
        try:
            "✓♪━".encode(getattr(out, "encoding", None) or "ascii")
            self.sym = UNICODE
        except UnicodeEncodeError:
            self.sym = ASCII
        self.drawn = False        # the live line is on screen
        self.stage_name = None    # last stage announced
        self.started = None       # clock time of the first reading stage
        self.left_shown = None    # last time-left estimate shown, in seconds
        self.wrote_any = False
        self.closed: list = []     # streams whose reader has gone away, as when piped into head

    # Writing

    def clear(self) -> None:
        """Wipe the live line if it's on screen."""
        if self.drawn:
            self._put(self.err, "\r\x1b[K")
            self.drawn = False

    def _write(self, stream, text: str, live: bool = False) -> None:
        """Every write goes through here so the live line is wiped before anything else prints on either stream."""
        self.clear()
        if stream is self.err and not self.err_colour:
            text = ANSI.sub("", text)
        enc = getattr(stream, "encoding", None)
        if enc:
            # Titles, credits, file names and symbols can hold letters the stream can't encode; they print as ?
            text = text.encode(enc, "replace").decode(enc)
        self.drawn = self._put(stream, text if live else text + "\n") and live

    def _put(self, stream, text: str) -> bool:
        """Write and flush text, and return whether it went out. When the reader of the stream has closed it, the
        stream is pointed at /dev/null and skipped from then on, so the run finishes and Python's own flush at exit
        has nowhere to fail."""
        if any(stream is s for s in self.closed):
            return False
        try:
            stream.write(text)
            stream.flush()
            return True
        except BrokenPipeError:
            self.closed.append(stream)
            try:
                fd = stream.fileno()
            except (AttributeError, OSError):
                return False
            os.dup2(os.open(os.devnull, os.O_WRONLY), fd)
            return False

    def _paint(self, text: str, style: str | None) -> str:
        return f"\x1b[{SGR[style]}m{text}\x1b[0m" if self.colour and style else text

    def _sym(self, name: str) -> str:
        return self._paint(self.sym[name], COLOURS.get(name))

    def _text(self, text: str) -> str:
        """Text from the video or notices, with the en dash and middle dot spelled as the current symbol set
        allows."""
        return text.replace("–", self.sym["dash"]).replace("·", self.sym["sep"])

    def _wrap(self, text: str, style: str | None = None) -> list[str]:
        """Text indented 2 spaces and wrapped to the terminal's width, at most 80 columns. Paths stay whole on one
        line, even a long one, so they can be copied."""
        lines = textwrap.wrap(self._text(text), min(self.width, 80) - 1, initial_indent="  ", subsequent_indent="  ",
                              break_long_words=False, break_on_hyphens=False)
        return [self._paint(line, style) for line in lines]

    def _plain(self, key: str, value) -> None:
        self._write(self.out, f"{key}: {value}")

    # Start and progress

    def start(self, title: str, credit: str | None, length_s: float | None, where: str) -> None:
        """The header: the video's title, then who made it, its length and where it came from. A length of None
        leaves it out."""
        length = fmt(round(length_s)) if length_s is not None else None
        if self.plain:
            for key, value in (("source", where), ("title", title), ("credit", credit), ("length", length)):
                if value:
                    self._plain(key, value)
            return
        shown = re.sub(r"^\w+://(www\.)?", "", where) if "://" in where else Path(where).name
        parts = ([f"tab by {self._text(credit)}"] if credit else []) + ([length] if length else []) + [
            self._paint(shown, "dim")]
        self._write(self.out, f"{self._sym('mark')} {self._paint(self._text(title), 'bold')}")
        self._write(self.out, "  " + f" {self.sym['sep']} ".join(parts))

    def stage(self, name: str, done: float | None = None, video_t: float | None = None,
              detail: str | None = None) -> None:
        """Show a stage in progress. done is the stage's share finished (None draws no bar), video_t the video time
        it has reached, and detail any short text to add, such as a count of PDFs."""
        label = STAGES.get(name, (name, 0))[0]
        new = name != self.stage_name
        self.stage_name = name
        if name in ORDER and self.started is None:
            self.started = self.clock()
        if self.plain:
            if new:
                self._plain("stage", label.lower())
            return
        # Without a live line the finished stages' lines tell the story
        if not self.live:
            return
        if self.narrow:
            label = label.split()[0]
        # Rounded as the header's length is, so the two agree at the video's end
        extra = [] if video_t is None else [f"at {fmt(round(video_t))}"]
        if detail:
            extra.append(detail)
        left = self._time_left(name, done)
        self._live_line(label, done, extra, [left] if left else [])

    def download(self, done_bytes: int, total_bytes: int | None, eta_s: float | None) -> None:
        """Show download progress; with no total known it shows the amount so far and no bar."""
        if self.plain or not self.live:
            self.stage("download")
            return
        self.stage_name = "download"
        mb = round(done_bytes / 1e6)
        if not total_bytes:
            self._live_line(STAGES["download"][0], None, [f"{mb} MB so far"], [])
            return
        left = [f"{round(eta_s)} s left"] if eta_s is not None else []
        size = f"{mb} of {round(total_bytes / 1e6)} MB"
        self._live_line(STAGES["download"][0], done_bytes / total_bytes, [size], left)

    def _live_line(self, label: str, done: float | None, extra: list[str], left: list[str]) -> None:
        """Draw the live line. The time left, and then the extra text, are dropped when the line would pass the
        terminal's width, since a line that wraps can't be redrawn in place."""
        line = f"{self._sym('stage')} {label}"
        if done is not None:
            cells = 10 if self.narrow else 20
            filled = round(min(max(done, 0.0), 1.0) * cells)
            fill, empty = self.sym["fill"] * filled, self.sym["empty"] * (cells - filled)
            line += f"  {self._paint(fill, 'cyan')}{self._paint(empty, 'dim')}  {round(done * 100)}%"
        sep = f" {self.sym['sep']} "
        width = len(ANSI.sub("", line))
        for tail in (extra + ([] if self.narrow else left), extra, []):
            if not tail or width + 2 + len(sep.join(tail)) <= self.width - 1:
                break
        if tail:
            line += "  " + sep.join(tail)
        self._write(self.err, line, live=True)

    def _time_left(self, name: str, done: float | None) -> str | None:
        """A rounded estimate of the whole run's time left, from the time so far and the share of work done. It
        waits for enough to go on, and rises only once the estimate is a full rounding step above the time shown, so
        it doesn't flicker."""
        if name not in ORDER or done is None or self.started is None:
            return None
        k = ORDER.index(name)
        share = sum(STAGES[s][1] for s in ORDER[:k]) + STAGES[name][1] * done
        elapsed = self.clock() - self.started
        if elapsed < 10 or share < 0.15:
            return None
        raw = elapsed * (1 - share) / share
        step = 5 if raw < 60 else 60
        left = max(step, int(raw / step + 0.5) * step)
        shown = self.left_shown
        if shown is not None and shown < left and raw - shown < (5 if shown < 60 else 60):
            left = shown
        self.left_shown = left
        return f"about {left} s left" if left < 60 else f"about {left // 60} min left"

    def stage_done(self, text: str, seconds: float | None = None) -> None:
        """A finished stage, with its time on the right when seconds is given."""
        if self.plain:
            return
        if seconds is None:
            self._write(self.out, f"{self._sym('ok')} {text}")
            return
        took = f"{round(seconds)} s"
        head = f"{self.sym['ok']} {text}"
        pad = max(1, min(DONE_COL, self.width - 1) - len(head) - len(took))
        self._write(self.out, f"{self._sym('ok')} {text}" + " " * pad + self._paint(took, "dim"))

    def fact(self, key: str, value) -> None:
        """A plain-mode only line, such as read: 151 measures in 53 s; the terminal shows it in the stage lines."""
        if self.plain:
            self._plain(key, value)

    # Results

    def wrote(self, path, copied_to=None) -> None:
        """A PDF written, and where it was copied to."""
        if self.plain:
            self._plain("pdf", path)
            if copied_to:
                self._plain("copied", copied_to)
            return
        if not self.wrote_any:
            self._write(self.out, "")
            self.wrote_any = True
        self._write(self.out, f"{self.sym['file']} {path}")
        if copied_to:
            self._write(self.out, self._paint(f"{self.sym['file']} copied to {tilde(copied_to)}", "dim"))

    def spots(self, notices) -> None:
        """The spots to check, the same set as the PDF's last page. The terminal groups them by kind; plain mode
        prints each one as a check: line and each notice without doubt as a note: line."""
        merged = merged_spots(notices)
        quiet = list(dict.fromkeys((n.t0, n.t1, n.text) for n in notices if not n.doubt))
        if self.plain:
            for n in merged:
                self._plain("check", f"{fmt(n.t0)}-{fmt(n.t1)} {n.text}")
            for t0, t1, text in quiet:
                self._plain("note", f"{fmt(t0)}-{fmt(t1)} {text}")
            return
        if quiet:
            spans = [f"{fmt(t0)}-{fmt(t1)}" for t0, t1, _ in quiet]
            said = ", ".join(spans[:3]) + (f" and {len(spans) - 3} more" if len(spans) > 3 else "")
            self._write(self.out, "\n".join(self._wrap(
                f"The folded one prints {said} once; the video doesn't say if it's played twice.", "dim")))
        groups = _groups(merged)
        total = sum(len(times) for _, times in groups)
        if not total:
            return
        where = "(last page of the PDF):" if self.narrow else "listed on the last page of the PDF:"
        noun = "spot" if total == 1 else "spots"
        self._write(self.out, f"\n{self._sym('check')} {total} {noun} to check{' ' if self.narrow else ', '}{where}")
        cw = max(len(str(len(times))) for _, times in groups)
        for label, times in groups:
            head = f"  {str(len(times)).rjust(cw)} {self._text(label)}"
            shown = list(dict.fromkeys(times))
            if not self.narrow and len(head) + 2 <= TIMES_COL:
                self._write(self.out, head.ljust(TIMES_COL) + self._fit(shown, self.width - 1 - TIMES_COL))
            else:
                indent = 3 + cw
                self._write(self.out, head)
                self._write(self.out, " " * indent + self._fit(shown, self.width - 1 - indent))

    def _fit(self, times: list[str], room: int) -> str:
        """As many times as fit in room characters, ending in the more symbol when some are left out."""
        more, line = self.sym["more"], ""
        for i, t in enumerate(times):
            piece = (" " if line else "") + t
            reserve = 0 if i == len(times) - 1 else len(more) + 1
            if len(line) + len(piece) + reserve > room:
                return line + (" " if line else "") + more
            line += piece
        return line

    def finish(self, status: str, text: str) -> None:
        """The closing line. status is ok, check or failed; a failure's heading and reason come from error, so the
        terminal adds nothing for it."""
        if self.plain:
            self._plain("status", status)
            if status == "failed":
                self._plain("reason", text)
        elif status == "failed":
            self.clear()
        else:
            self._write(self.out, f"\n{self._sym(status)} {text}")

    def error(self, heading: str, *detail: str) -> None:
        """An error on stderr: a heading, then indented lines that say what to do. An empty detail is a blank
        line."""
        head = heading if self.plain else f"{self._sym('failed')} {heading}"
        lines = [head]
        for d in detail:
            lines += [f"  {d}" if d else ""] if self.plain else (self._wrap(d) or [""])
        self._write(self.err, "\n".join(lines))


def tilde(path) -> str:
    """A path under the home folder written with ~, as the user would type it."""
    home = str(Path.home())
    return "~" + str(path)[len(home):] if str(path).startswith(home + os.sep) else str(path)


def merged_spots(notices: list[Notice]) -> list[Notice]:
    """The PDF's spots to check: each doubtful notice once, in time order. no_measures and vertical_scroll notices
    come in short pieces, so pieces of one kind that touch end to end become one notice over the whole stretch."""
    distinct = sorted({(n.t0, n.t1, n.text): n for n in notices if n.doubt}.values(), key=lambda n: (n.t0, n.t1))
    spots = [n for n in distinct if n.kind not in STRETCHES]
    stretches: list[Notice] = []
    for n in distinct:
        if n.kind not in STRETCHES:
            continue
        if stretches and n.kind == stretches[-1].kind and n.t0 <= stretches[-1].t1:
            last = stretches[-1]
            t1 = max(last.t1, n.t1)
            # The text names the piece's own times; give it the whole stretch's
            text = re.sub(r"\d+:\d\d-\d+:\d\d", f"{fmt(last.t0)}-{fmt(t1)}", last.text, count=1)
            stretches[-1] = replace(last, t1=t1, text=text)
        else:
            stretches.append(n)
    return sorted(spots + stretches, key=lambda n: (n.t0, n.t1))


def spot_lines(notices: list[Notice]) -> list[str]:
    """The PDF's "Check these spots" lines, one for each of merged_spots."""
    return [f"{fmt(n.t0)}-{fmt(n.t1)}  {n.text}" for n in merged_spots(notices)]


def _groups(spots: list[Notice]) -> list[tuple[str, list[str]]]:
    """Spots from merged_spots grouped by kind, as (label, time of each spot), largest group first. A kind with no
    label uses its first spot's text."""
    groups = []
    for kind in dict.fromkeys(n.kind for n in spots):
        mine = [n for n in spots if n.kind == kind]
        times = [f"{fmt(n.t0)}-{fmt(n.t1)}" if n.t1 - n.t0 > RANGE_S else fmt((n.t0 + n.t1) / 2) for n in mine]
        one, more = LABELS.get(kind, (mine[0].text, mine[0].text))
        groups.append((kind, one if len(mine) == 1 else more, times))
    order = list(LABELS)
    groups.sort(key=lambda g: (-len(g[2]), order.index(g[0]) if g[0] in order else len(order)))
    return [(label, times) for _, label, times in groups]
