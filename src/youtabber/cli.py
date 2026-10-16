import argparse
import json
import re
import shutil
import sys
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from youtabber import enhance, layout, pipeline, printcell, source
from youtabber.check import fmt
from youtabber.video import VideoError

CONFIG = Path.home() / ".config" / "youtabber" / "config.toml"
# Smallest tab area, in video pixels, that can hold a readable staff
MIN_REGION = 32
PASSES = ("once", "all")
# Exit codes: a PDF with nothing to check, no PDF, and a PDF with spots to check or a second reading
EXIT_CLEAN, EXIT_FAILED, EXIT_CHECK = 0, 1, 3


def load_config() -> dict:
    if not CONFIG.exists():
        return {}
    try:
        return tomllib.loads(CONFIG.read_text())
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"Can't read {CONFIG}: {e}")


def parse_region(text: str) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected x,y,w,h in pixels, for example 0,800,1920,280")
    if x < 0 or y < 0 or w < MIN_REGION or h < MIN_REGION:
        raise argparse.ArgumentTypeError(f"x and y can't be negative, and w and h must be at least {MIN_REGION}")
    return x, y, w, h


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn a play-along video into a printable PDF of its tab.")
    parser.add_argument("source", help="YouTube URL or local video file")
    parser.add_argument("-o", "--out", type=Path, help="output folder (default: out/<song name>)")
    parser.add_argument("--title", help="song name for the PDF, when the one taken from the video title is wrong")
    parser.add_argument("--region", type=parse_region,
                        help="only read tab inside x,y,w,h (video pixels), when detection picks up something else")
    parser.add_argument("--passes", choices=PASSES,
                        help="once: a replayed section prints once (default); all: every pass prints in play order")
    parser.add_argument("--enhance", action=argparse.BooleanOptionalAction, default=None,
                        help="upscale and sharpen the tab images in the PDF (on unless the config says otherwise)")
    parser.add_argument("--model", help=f"how to enhance: classic, {', '.join(enhance.MODELS)}, "
                                        "or a path to a model file (models need the ai extra)")
    args = parser.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"youtabber needs {tool} on your PATH. On macOS: brew install ffmpeg")
    try:
        code = run(args, load_config())
    except VideoError as e:
        raise SystemExit(str(e))
    sys.exit(code)


def run(args: argparse.Namespace, config: dict) -> int:
    """Read the video, write the outputs, and return the exit code. A failure raises SystemExit."""
    pdf_dir = Path(config["pdf_dir"]).expanduser() if config.get("pdf_dir") else None
    model = args.model or config.get("model")
    passes = args.passes or config.get("passes", "once")
    if passes not in PASSES:
        raise SystemExit(f'passes must be "once" or "all", not {passes!r}. Check {CONFIG}.')
    # The command line wins over the config file. Naming a model asks for enhancing, and with
    # neither saying, enhancing is on.
    enhanced = args.enhance if args.enhance is not None else bool(args.model) or config.get("enhance", True)
    # Load the model before the slow video work so a bad name or missing extra fails right away
    if not enhanced:
        upscale = None
    elif model and model != "classic":
        upscale = enhance.model_upscaler(model)
    else:
        upscale = enhance.classic

    src = source.fetch(args.source)
    if args.title:
        src = replace(src, title=args.title)
    name = source.slug(src.title) or source.slug(src.id) or "tab"
    out = args.out or Path("out") / name

    print("Reading the video")
    started = time.time()
    reading = pipeline.read(src.path, args.region)
    pdfs = write_outputs(reading, src, out, name, passes, upscale, started=started)
    for n in reading.notices:
        print(f"  {fmt(n.t0)}-{fmt(n.t1)}  {n.text}")
    if reading.outcome == "fail":
        raise SystemExit(f"{reading.message}\nDetails are in {out / 'meta.json'}")
    print(reading.message)
    for p in pdfs:
        print(f"Wrote {p}")
        if pdf_dir:
            pdf_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, pdf_dir / p.name)
            print(f"Copied to {pdf_dir / p.name}")
    return EXIT_CHECK if reading.notices or len(pdfs) > 1 else EXIT_CLEAN


def write_outputs(reading: pipeline.Reading, src: source.Source, out: Path, name: str, passes: str,
                  upscale: enhance.Upscaler | None, started: float | None = None) -> list[Path]:
    """Write the PDFs, meta.json, measures/ and views/ for one reading. Returns the PDFs written."""
    out.mkdir(parents=True, exist_ok=True)
    for folder in ("measures", "views"):
        (out / folder).mkdir(exist_ok=True)
        for old in (out / folder).glob("*.png"):
            old.unlink()
    pdfs: list[Path] = []
    printed: list[printcell.PrintCell] = []
    if reading.outcome == "tab":
        cands = reading.every_pass if passes == "all" else reading.candidates
        spots = spot_lines(reading.notices)
        for k, cand in enumerate(cands):
            cells = [printcell.print_cell(reading.views[m.view], m.cell, reading.segments[reading.views[m.view].seg])
                     for m in cand.measures]
            path = out / (f"{name}.pdf" if k == 0 else f"{name}-{cand.name}.pdf")
            layout.write_pdf(cells, row_breaks(reading, cand), path, src, upscale, note_for(len(cands), k, name), spots)
            pdfs.append(path)
            if k == 0:
                printed = cells
                for i, pc in enumerate(cells):
                    cv2.imwrite(str(out / "measures" / f"{i:04d}.png"), pc.image)
    for o in reading.views:
        cv2.imwrite(str(out / "views" / f"{o.index:03d}.png"), view_image(reading, o))
    (out / "meta.json").write_text(json.dumps(meta(reading, src, passes, pdfs, printed, started), indent=1, default=_plain))
    return pdfs


def spot_lines(notices: list[pipeline.Notice]) -> list[str]:
    """The PDF's "Check these spots" lines: each doubtful notice once, in time order. no_measures notices come
    in short pieces, so pieces that touch end to end print as one line over the whole stretch."""
    distinct = sorted({(n.t0, n.t1, n.text): n for n in notices if n.doubt}.values(), key=lambda n: (n.t0, n.t1))
    spots = [n for n in distinct if n.kind != "no_measures"]
    stretches: list[pipeline.Notice] = []
    for n in distinct:
        if n.kind != "no_measures":
            continue
        if stretches and n.t0 <= stretches[-1].t1:
            last = stretches[-1]
            t1 = max(last.t1, n.t1)
            # The text names the piece's own times; give it the whole stretch's
            text = re.sub(r"\d+:\d\d-\d+:\d\d", f"{fmt(last.t0)}-{fmt(t1)}", last.text, count=1)
            stretches[-1] = replace(last, t1=t1, text=text)
        else:
            stretches.append(n)
    spots = sorted(spots + stretches, key=lambda n: (n.t0, n.t1))
    return [f"{fmt(n.t0)}-{fmt(n.t1)}  {n.text}" for n in spots]


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
    status = "failed" if reading.outcome == "fail" else ("warn" if reading.notices or len(pdfs) > 1 else "ok")
    cands = reading.every_pass if passes == "all" else reading.candidates
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
                      "playhead": [[round(t, 2), round(x + ox), round(y + oy)] for t, x, y in o.playhead]})
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
                             "gap": round(pc.gap, 2)})
    return {
        "version": 2,
        "source": {"id": src.id, "url": src.url, "title": src.title, "credit": src.credit},
        "outcome": {"status": status, "reason": reading.fail_reason, "message": reading.message,
                    "pdfs": [p.name for p in pdfs]},
        "passes": passes,
        "numbering": reading.numbering,
        "tab_spans": [list(s) for s in reading.tab_spans],
        "panels": reading.panels,
        "segments": [{"t": [round(s.t0, 2), round(s.t1, 2)], "crop": [s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0],
                      "gap": round(s.gap, 2), "polarity": s.polarity} for s in reading.segments],
        "views": views,
        "slots": [{"id": s.id, "number": s.number, "first_view": s.obs} for s in A.slots] if A else [],
        "path": path,
        "measures": measures,
        "decisions": [{"t": round(d.t, 2), "kind": d.kind, "chosen": d.chosen, "alternatives": d.alternatives,
                       "basis": d.basis, "resolved": d.resolved} for d in A.decisions] if A else [],
        "notices": [{"t0": n.t0, "t1": n.t1, "kind": n.kind, "doubt": n.doubt, "text": n.text}
                    for n in reading.notices],
        "runtime_s": round(time.time() - started, 1) if started is not None else None,
    }


def _plain(value):
    """numpy numbers as plain JSON numbers."""
    return value.item() if hasattr(value, "item") else str(value)
