import argparse
import errno
import json
import os
import re
import shutil
import sys
import time
import tomllib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from youtabber import assemble, enhance, layout, pipeline, printcell, source, video
from youtabber.ui import Ui, spot_lines, tilde
from youtabber.video import VideoError

CONFIG = Path.home() / ".config" / "youtabber" / "config.toml"
# Smallest tab area, in video pixels, that can hold a readable staff
MIN_REGION = 32
PASSES = ("once", "all")
# Exit codes: a PDF with nothing to check, no PDF, bad arguments, a PDF with spots to check or a second reading,
# and stopped with Ctrl-C (the shell's usual 128 + SIGINT)
EXIT_CLEAN, EXIT_FAILED, EXIT_USAGE, EXIT_CHECK, EXIT_STOPPED = 0, 1, 2, 3, 130
EPILOG = """\
examples:
  youtabber "https://www.youtube.com/watch?v=..."
  youtabber song.mp4 --title "Song - Artist"
  youtabber URL | grep '^pdf:'    piped output is plain key: value lines

the output folder holds the PDF, meta.json (what was read, with times),
measures/ and views/.
defaults can be changed in ~/.config/youtabber/config.toml
(pdf_dir, model, enhance, passes).
NO_COLOR turns off colour.

exit codes: 0 clean PDF, 1 no PDF, 2 bad arguments, 3 PDF with spots to check,
  130 stopped
"""


class Failure(Exception):
    """A run that stops early: the error's heading, lines that say what to do, and the exit code."""

    def __init__(self, code: int, heading: str, *detail: str):
        super().__init__(heading)
        self.code, self.heading, self.detail = code, heading, detail


def load_config() -> dict:
    if not CONFIG.exists():
        return {}
    try:
        return tomllib.loads(CONFIG.read_text())
    except tomllib.TOMLDecodeError as e:
        raise Failure(EXIT_FAILED, f"Can't read {tilde(CONFIG)}", f"{e}. Fix the file or move it aside.")


def parse_region(text: str) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected x,y,w,h in pixels, for example 0,800,1920,280")
    if x < 0 or y < 0 or w < MIN_REGION or h < MIN_REGION:
        raise argparse.ArgumentTypeError(f"x and y can't be negative, and w and h must be at least {MIN_REGION}")
    return x, y, w, h


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="youtabber", description="Turn a play-along video into a printable PDF of its tab.",
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="YouTube URL or local video file")
    # ~ is expanded here so a quoted "~/x" is checked and written as the same folder
    p.add_argument("-o", "--out", type=lambda text: Path(text).expanduser(), help="output folder (default: out/<song name>)")
    p.add_argument("--title", help="song name for the PDF, when the one taken from the video title is wrong")
    p.add_argument("--region", type=parse_region, metavar="X,Y,W,H",
                   help="only read tab inside this box, in video pixels, when it picks up something else; the box it "
                        "used is in meta.json under segments[].crop")
    p.add_argument("--passes", choices=PASSES,
                   help="once: a replayed section prints once (default); all: every pass prints in play order")
    p.add_argument("--enhance", action=argparse.BooleanOptionalAction, default=None,
                   help="upscale and sharpen the tab images (on unless the config says otherwise)")
    p.add_argument("--model", help=f"how to enhance: classic, {', '.join(enhance.MODELS)}, or a path to a model file "
                                   "(models need the ai extra)")
    return p


def main() -> None:
    # Each line goes out as it's printed, even into a pipe or a log file
    sys.stdout.reconfigure(line_buffering=True)
    args = parser().parse_args()
    ui = Ui()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            ui.error(f"youtabber needs {tool} on your PATH", "On macOS: brew install ffmpeg")
            sys.exit(EXIT_FAILED)
    try:
        config = load_config()
    except Failure as e:
        ui.error(e.heading, *e.detail)
        sys.exit(e.code)
    sys.exit(run(args, config, ui))


def run(args: argparse.Namespace, config: dict, ui: Ui | None = None) -> int:
    """Read the video, write the outputs, show what happened, and return the exit code."""
    ui = ui or Ui()
    # Holds "writing" (the output folder) while files are being written, "written" once they all are, and
    # "copying" (pdf_dir) while the PDFs are copied there, so a Ctrl-C can say what it left behind
    state: dict = {}
    try:
        return _run(args, config, ui, state)
    except Failure as e:
        ui.error(e.heading, *e.detail)
        return e.code
    except (source.SourceError, VideoError, enhance.ModelError) as e:
        ui.error(e.heading, *e.detail)
        return EXIT_FAILED
    except KeyboardInterrupt:
        if "copying" in state:
            ui.error(f"Stopped while copying to {tilde(state['copying'])}/; the copy there may be incomplete. "
                     f"Everything was written to {state['written']}/.")
        elif "writing" in state:
            ui.error(f"Stopped while writing; {state['writing']}/ may be incomplete.")
        elif "written" in state:
            ui.error(f"Stopped. Everything was written to {state['written']}/.")
        else:
            ui.error("Stopped. Nothing was written.")
        return EXIT_STOPPED
    finally:
        # Wipe the live line, so an unexpected error's traceback starts on a line of its own
        ui.clear()


def _run(args: argparse.Namespace, config: dict, ui: Ui, state: dict) -> int:
    pdf_dir = Path(config["pdf_dir"]).expanduser() if config.get("pdf_dir") else None
    model = args.model or config.get("model")
    passes = args.passes or config.get("passes", "once")
    if passes not in PASSES:
        raise Failure(EXIT_FAILED, f'passes must be "once" or "all", not {passes!r}', f"Check {tilde(CONFIG)}.")
    # The command line wins over the config file. Naming a model asks for enhancing, and with
    # neither saying, enhancing is on.
    enhanced = args.enhance if args.enhance is not None else bool(args.model) or config.get("enhance", True)
    # Load the model before the slow video work so a bad name or missing extra fails right away. The name is
    # checked first, since loading a model imports PyTorch, which takes seconds.
    if not enhanced:
        upscale = None
    elif model and model != "classic":
        if model not in enhance.MODELS and not Path(model).expanduser().exists():
            raise Failure(EXIT_USAGE, f"Unknown model {model!r}",
                          f"Use classic, {', '.join(enhance.MODELS)}, or a path to a model file.")
        upscale = enhance.model_upscaler(model)
    else:
        upscale = enhance.classic
    if pdf_dir:
        check_writable(pdf_dir, f"Can't write to {tilde(pdf_dir)} (pdf_dir in {tilde(CONFIG)})",
                       "Change pdf_dir in the config.")
    if args.out:
        check_writable(args.out, f"Can't write to {args.out}", "Pick another folder with -o.")

    def looked_up(found: source.Source) -> None:
        ui.start(args.title or found.title, found.credit, found.duration_s, found.url or args.source)

    src = source.fetch(args.source, on_progress=ui.download, on_start=looked_up)
    if src.downloaded_bytes is not None:
        ui.stage_done(f"Downloaded {megabytes(src.downloaded_bytes)} MB in {round(src.download_s)} s")
    info = video.probe(src.path)
    if args.region:
        x, y, w, h = args.region
        if x + w > info.width or y + h > info.height:
            raise Failure(EXIT_USAGE, f"--region {x},{y},{w},{h} is outside the video, which is "
                                      f"{info.width}x{info.height}",
                          f"x+w must be at most {info.width} and y+h at most {info.height}. The box youtabber found "
                          f"last time is in meta.json under segments[].crop.")
    if args.title:
        src = replace(src, title=args.title)
    if src.url is None:
        # A local file's header is shown here, once its length is known
        ui.start(src.title, src.credit, info.duration, str(src.path))
    name = source.slug(src.title) or source.slug(src.id) or "tab"
    out = args.out or Path("out") / name
    if not args.out:
        check_writable(out, f"Can't write to {out}", "Pick another folder with -o.")

    started = time.time()
    reading, took = read_with_progress(src.path, args.region, info.duration, ui)
    cands = printed_candidates(reading, passes) if reading.outcome == "tab" else []
    measures = len(cands[0].measures) if cands else 0
    if "read" in took:
        ui.stage_done(f"Read {len(reading.views)} {plural(len(reading.views), 'screen')}", took["read"])
    if "assemble" in took:
        ui.stage_done(f"Pieced together {measures} {plural(measures, 'measure')}", took["assemble"])

    pages: list[int] = []

    def printed_one(path: Path, count: int) -> None:
        pages.append(count)
        if len(pages) < len(cands):
            ui.stage("print", detail=f"{len(pages) + 1} of {len(cands)} PDFs")

    if cands:
        ui.stage("print", detail=f"1 of {len(cands)} PDFs" if len(cands) > 1 else None)
    began = time.monotonic()
    state["writing"] = out
    try:
        pdfs = write_outputs(reading, src, out, name, passes, upscale, started=started, on_pdf=printed_one)
    except OSError as e:
        raise Failure(EXIT_FAILED, f"Can't write to {e.filename or out}",
                      f"{e.strerror or e}. Pick another folder with -o.")
    del state["writing"]
    state["written"] = out
    copies = []
    if pdf_dir:
        state["copying"] = pdf_dir
    for p in pdfs:
        try:
            copies.append(copy_pdf(p, pdf_dir))
        except OSError as e:
            raise Failure(EXIT_FAILED, f"Can't write to {tilde(pdf_dir)} (pdf_dir in {tilde(CONFIG)})",
                          f"{e.strerror or e}. Change pdf_dir in the config.")
    state.pop("copying", None)

    if reading.outcome == "fail":
        ui.finish("failed", reading.message)
        heading, reason = refusal(reading)
        pointer = f"Details: {out / 'meta.json'}"
        if any((out / "views").glob("*.png")):
            pointer += f" · screens it looked at: {out / 'views'}/"
        ui.error(heading, "", reason, "", pointer)
        return EXIT_FAILED

    if len(pdfs) == 1:
        ui.stage_done(f"Printed the PDF, {pages[0]} {plural(pages[0], 'page')}", time.monotonic() - began)
    elif len(set(pages)) == 1:
        ui.stage_done(f"Printed {len(pdfs)} PDFs, {pages[0]} {plural(pages[0], 'page')} each", time.monotonic() - began)
    else:
        ui.stage_done(f"Printed {len(pdfs)} PDFs, {' and '.join(map(str, pages))} pages", time.monotonic() - began)
    ui.fact("read", f"{measures} {plural(measures, 'measure')} in {round(time.time() - started)} s")
    for p, copy in zip(pdfs, copies):
        ui.wrote(p, copy)
    notices = to_check(reading, passes)
    ui.spots(notices)
    spots = len(spot_lines(notices))
    code = EXIT_CHECK if notices or len(pdfs) > 1 else EXIT_CLEAN
    if spots:
        those = "that spot" if spots == 1 else f"those {spots} spots"
        then = ", starting with the first PDF." if len(pdfs) > 1 else " before you rely on the PDF."
        ui.finish("check", f"Check {those} against the video{then}")
    elif len(pdfs) > 1:
        ui.finish("check", "Two takes on the repeats: play the first PDF, or the folded one if the video replays.")
    else:
        ui.finish("ok", f"All clean: {measures} {plural(measures, 'measure')}, nothing to check. Happy playing.")
    return code


def read_with_progress(path: Path, region, duration: float, ui: Ui) -> tuple[pipeline.Reading, dict[str, float]]:
    """pipeline.read with its progress on the live line. Also returns the seconds each finished stage took; a
    stage counts as finished when the next one starts, or for the last one, when a reading with tab returns."""
    began: dict[str, float] = {}
    took: dict[str, float] = {}
    # An error from showing progress is kept here and raised once the reading is done, so it can't stop the reader
    # halfway through
    failed: list[Exception] = []

    def heard(stage: str, share: float) -> None:
        if failed:
            return
        try:
            if stage not in began:
                now = time.monotonic()
                if began:
                    last = list(began)[-1]
                    took[last] = now - began[last]
                    if last == "find":
                        ui.stage_done("Found the tab", took[last])
                began[stage] = now
            # Finding and reading report how far through the whole video they are, so share * duration is the
            # video time reached. Piecing together reports only its start and end, so it shows no bar.
            if stage in ("find", "read"):
                ui.stage(stage, share, share * duration)
            else:
                ui.stage(stage)
        except Exception as e:
            failed.append(e)

    ui.stage("find")
    reading = pipeline.read(path, region, progress=heard)
    if failed:
        raise failed[0]
    if reading.outcome == "tab" and "assemble" in began:
        took["assemble"] = time.monotonic() - began["assemble"]
    return reading, took


def check_writable(folder: Path, heading: str, fix: str) -> None:
    """Fail before the slow work when folder can't be written: it, or the nearest folder above it that exists,
    must be a folder this user can write to."""
    p = folder
    while not p.exists():
        p = p.parent
    if not p.is_dir():
        what = "That's a file" if p == folder else f"{p} is a file"
        raise Failure(EXIT_FAILED, heading, f"{what}, not a folder. {fix}")
    if not os.access(p, os.W_OK | os.X_OK):
        raise Failure(EXIT_FAILED, heading, f"Permission denied. {fix}")


def copy_pdf(pdf: Path, pdf_dir: Path | None) -> Path | None:
    """Copy a PDF into pdf_dir and return where it went, or None when there's no pdf_dir or the PDF is already
    there."""
    if not pdf_dir:
        return None
    target = pdf_dir / pdf.name
    # When -o is already pdf_dir, however it is spelled, the PDF is in place and copying it onto itself fails
    if target.exists() and target.samefile(pdf):
        return None
    pdf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, target)
    return target


def refusal(reading: pipeline.Reading) -> tuple[str, str]:
    """The heading and reason shown when no PDF was written. A message opening "No tab ...: why" shows the why
    under "No tab here"; one opening with a sentence of its own, as about --region, keeps that sentence as the
    heading."""
    m = re.match(r"(No tab [^:]*?)(: |\. )(.*)", reading.message, re.DOTALL)
    if m and m.group(2) == ". ":
        return m.group(1), m.group(3)
    if m:
        why = m.group(3)
        return "No tab here", why[:1].upper() + why[1:]
    return ("No tab here" if reading.fail_reason == "no_tab" else "Couldn't read this tab"), reading.message


def plural(n: int, word: str) -> str:
    return word if n == 1 else word + "s"


def megabytes(n: int) -> str:
    """A size in MB, with one decimal under 10 MB so a small download doesn't show as 0."""
    return f"{n / 1e6:.1f}" if n < 10e6 else str(round(n / 1e6))


def write_image(path: Path, image: np.ndarray) -> None:
    # cv2.imwrite returns False rather than raising when it can't write the file
    if not cv2.imwrite(str(path), image):
        raise OSError(errno.EIO, "Couldn't save the image", str(path))


def printed_candidates(reading: pipeline.Reading, passes: str) -> list[pipeline.Candidate]:
    """The readings that get a PDF: the every-pass one in all mode, else the main one and any folded one."""
    return reading.every_pass if passes == "all" else reading.candidates


def run_message(reading: pipeline.Reading, passes: str) -> str:
    """The run's summary line: pipeline's message for a failed run or when two PDFs were written, else the
    measure count of the one PDF written."""
    cands = printed_candidates(reading, passes)
    if reading.outcome == "tab" and len(cands) == 1:
        n = len(cands[0].measures)
        return f"Read {n} {plural(n, 'measure')}."
    return reading.message


def to_check(reading: pipeline.Reading, passes: str) -> list[pipeline.Notice]:
    """Notices that mean the run needs checking. A notice without doubt is about the folded PDF, which all mode
    doesn't write."""
    return [n for n in reading.notices if n.doubt or passes != "all"]


def write_outputs(reading: pipeline.Reading, src: source.Source, out: Path, name: str, passes: str,
                  upscale: enhance.Upscaler | None, started: float | None = None,
                  on_pdf: Callable[[Path, int], None] | None = None) -> list[Path]:
    """Write the PDFs, meta.json, measures/ and views/ for one reading. Returns the PDFs written. on_pdf hears
    each PDF's path and page count once it's written."""
    out.mkdir(parents=True, exist_ok=True)
    # A PDF left from an earlier run of the same video would sit beside a meta.json that doesn't list it
    for old in (f"{name}.pdf", f"{name}-folded.pdf"):
        (out / old).unlink(missing_ok=True)
    for folder in ("measures", "views"):
        (out / folder).mkdir(exist_ok=True)
        for old in (out / folder).glob("*.png"):
            old.unlink()
    pdfs: list[Path] = []
    printed: list[printcell.PrintCell] = []
    if reading.outcome == "tab":
        cands = printed_candidates(reading, passes)
        spots = spot_lines(reading.notices)
        for k, cand in enumerate(cands):
            cells = [printcell.print_cell(reading.views[m.view], m.cell, reading.segments[reading.views[m.view].seg])
                     for m in cand.measures]
            path = out / (f"{name}.pdf" if k == 0 else f"{name}-{cand.name}.pdf")
            count = layout.write_pdf(cells, row_breaks(reading, cand), path, src, upscale,
                                     note_for(len(cands), k, name), spots)
            pdfs.append(path)
            if on_pdf:
                on_pdf(path, count)
            if k == 0:
                printed = cells
                for i, pc in enumerate(cells):
                    write_image(out / "measures" / f"{i:04d}.png", pc.image)
    for o in reading.views:
        write_image(out / "views" / f"{o.index:03d}.png", view_image(reading, o))
    (out / "meta.json").write_text(json.dumps(meta(reading, src, passes, pdfs, printed, started), indent=1, default=_plain))
    return pdfs


def note_for(count: int, k: int, name: str) -> str | None:
    """The page 1 note of each PDF when a second, folded reading was written."""
    if count < 2:
        return None
    if k == 0:
        return (f"Some screens repeat with no sign whether the song plays them again. This PDF prints them every "
                f"time; {name}-folded.pdf prints them once.")
    return f"This PDF prints repeated screens once; {name}.pdf prints them every time."


def row_breaks(reading: pipeline.Reading, cand: pipeline.Candidate) -> set[int]:
    """Measures that start a new row: the first one printed from each later segment (a new song or layout)."""
    out, seen = set(), -1
    for i, m in enumerate(cand.measures):
        seg = reading.views[m.view].seg
        if seg > seen:
            if i > 0:
                out.add(i)
            seen = seg
    return out


def slot_ids(reading: pipeline.Reading, o) -> dict[int, int]:
    """Slot of each of a view's cells, keyed by the cell's id()."""
    A = reading.assembly
    if A is None:
        return {}
    return {id(c): s for c, s in zip(o.cells, A.view_slots.get(o.index, [])) if s is not None}


def view_image(reading: pipeline.Reading, o) -> np.ndarray:
    """A view's plate with each cell boxed (green complete, red cut short) and its slot id."""
    img = o.image.copy() if o.image.ndim == 3 else cv2.cvtColor(o.image, cv2.COLOR_GRAY2BGR)
    slots = slot_ids(reading, o)
    for c in o.all_cells:
        color = (0, 160, 0) if c.complete else (0, 0, 220)
        cv2.rectangle(img, (int(c.x0), int(c.y0)), (int(c.x1), int(c.y0 + c.ink.shape[0] - 1)), color, 2)
        if id(c) in slots:
            cv2.putText(img, str(slots[id(c)]), (int(c.x0) + 4, int(c.y0) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)
    return img


def meta(reading: pipeline.Reading, src: source.Source, passes: str, pdfs: list[Path],
         printed: list[printcell.PrintCell], started: float | None = None) -> dict:
    """Everything read from the video, to check a result without rerunning it and for later tools."""
    A = reading.assembly
    status = "failed" if reading.outcome == "fail" else ("warn" if to_check(reading, passes) or len(pdfs) > 1 else "ok")
    cands = printed_candidates(reading, passes)
    views = []
    for o in reading.views:
        # Still views are in crop coordinates; adding the crop's corner gives source pixels
        ox, oy = (0, 0) if o.mosaic is not None else o.offset
        slots = slot_ids(reading, o)
        views.append({"id": o.index, "t": [round(o.t0, 2), round(o.t1, 2)],
                      "kind": "mosaic" if o.mosaic is not None else "plate", "segment": o.seg,
                      "image": f"views/{o.index:03d}.png",
                      "cells": [{"x": [int(c.x0 + ox), int(c.x1 + ox)], "system": c.system,
                                 "complete": c.complete, "number": c.number, "slot": slots.get(id(c))}
                                for c in o.all_cells],
                      "playhead": [[round(t, 2), round(x + ox), round(y + oy)] for t, x, y in o.playhead],
                      "part": o.part, "stream": o.stream, "move": move_meta(o.move),
                      "track": [[round(t, 2), round(x + ox), round(y + oy)]
                                for t, x, y in assemble.playhead_track(o)]})
        if o.mosaic is not None:
            # The bar counter's (start, end, measure) runs over the glide, empty when it shows none
            views[-1]["counter"] = [[round(t0, 2), round(t1, 2), n] for t0, t1, n in o.counter]
    path, npass, prev = [], 0, None
    for sid, vi, ci, t in reading.path:
        # A step back to an earlier slot starts a new pass
        if prev is not None and sid < prev:
            npass += 1
        prev = sid
        path.append({"slot": sid, "view": vi, "cell": ci, "play_t": round(t, 2), "pass": npass})
    measures = []
    if cands:
        for i, (m, pc) in enumerate(zip(cands[0].measures, printed)):
            measures.append({"slot": m.slot, "view": m.view, "t": m.t, "play_t": m.play_t, "label": m.label,
                             "complete": m.cell.complete, "image": f"measures/{i:04d}.png",
                             "gap": round(pc.gap, 2), "left": int(pc.left), "stream": m.stream})
    return {
        "version": 2,
        "source": {"id": src.id, "url": src.url, "title": src.title, "credit": src.credit},
        "outcome": {"status": status, "reason": reading.fail_reason, "message": run_message(reading, passes),
                    "pdfs": [p.name for p in pdfs]},
        "passes": passes,
        "numbering": reading.numbering,
        "tab_spans": [list(s) for s in reading.tab_spans],
        "panels": reading.panels,
        "segments": [{"t": [round(s.t0, 2), round(s.t1, 2)], "crop": [s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0],
                      "gap": round(s.gap, 2), "polarity": s.polarity} for s in reading.segments],
        "views": views,
        "slots": [{"id": s.id, "number": s.number, "first_view": s.obs, "stream": s.stream} for s in A.slots]
        if A else [],
        "parts": [{"id": p.id, "segment": p.seg, "stream": p.stream, "kind": p.kind,
                   "rows": [int(p.rows[0]), int(p.rows[1])]} for p in reading.parts],
        "streams": [{"stream": k, "order": A.stream_orders[k]} for k in sorted(A.stream_orders)] if A else [],
        "path": path,
        "measures": measures,
        "decisions": [{"t": round(d.t, 2), "kind": d.kind, "chosen": d.chosen, "alternatives": d.alternatives,
                       "basis": d.basis, "resolved": d.resolved, "view": d.view} for d in A.decisions] if A else [],
        "notices": [{"t0": n.t0, "t1": n.t1, "kind": n.kind, "doubt": n.doubt, "text": n.text}
                    for n in reading.notices],
        "runtime_s": round(time.time() - started, 1) if started is not None else None,
    }


def move_meta(m) -> dict | None:
    """How a view's band moved from the part's view before; None for a part's first view and a glide's first piece."""
    if m is None:
        return None
    return {"kind": m.kind, "shift": int(m.shifts[0]) if m.shifts else None, "steps": int(m.steps)}


def _plain(value):
    """numpy numbers as plain JSON numbers."""
    return value.item() if hasattr(value, "item") else str(value)
