"""Run youtabber on videos whose measures are known, and score the results with the frozen scorer.

The truth files, the scorer, the reference scores and the reference outputs live outside this repository: pass
--eval DIR or set YOUTABBER_EVAL to the folder holding them, with mediator/score.py and eval-sets.json. Videos are
read from ~/.cache/youtabber/<id>.mp4. Nothing here reads the youtabber config file.

  uv run python tools/evaluate.py run --sets dev,h1 --out DIR [--jobs 4] [--ids A,B] [--passes once|all]
  uv run python tools/evaluate.py score --sets dev,h1 --out DIR [--gate port|core|pdf|accept]
  uv run python tools/evaluate.py compare --sets dev --out DIR
  uv run python tools/evaluate.py partials --sets dev,confirm --out DIR
  uv run python tools/evaluate.py corpus --corpus DIR --out DIR [--exclude FILE] [--jobs 4] [--ids A,B]

The closed acceptance set lives outside the eval folder; only its one run sets YOUTABBER_ACCEPT.
--eval DIR goes before the subcommand.
"""
import argparse
import importlib.util
import json
import os
import resource
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

CACHE = Path.home() / ".cache" / "youtabber"
# A run longer than this counts as a crash, as the scorer rules
TIMEOUT_S = 600

# The eval folder's eval-sets.json gives each set's paths; accept has none there, see truth_folder
SETS = ("dev", "h1", "h2", "h3", "confirm", "reserve", "accept")
# The videos the earlier tool was built on: they must stay correct (the last at least near correct)
FLOOR_CORRECT = {"HLVzc4nNiY0", "EpQWdwfmQo8", "Jqt1vJEtyik"}
FLOOR_NEAR = {"JzjGuv3lkq4"}
# Smooth-scroll videos and the most errors (missing + extra) each may have; each of their errors must be warned
GLIDE_ERRORS = {"DeDDz4c_ajE": 12, "sRWpd8EcZ3c": 11, "oDGo0LDH9UE": 5, "ksSDjQ8dJUs": 6}
# Videos whose extra measures must all be gone
NO_EXTRAS = {"3yFuYJjO0oM"}
# Silent wrong results the reference already had: the port gate tolerates them; later gates require each to be
# correct or warned next to its errors
MUST_WARN = {"oDGo0LDH9UE", "hOnXJYCePa8"}
# Every smooth-scroll video in the sets: each needs 0 errors without a notice near the time the truth plays them
GLIDES = {"ksSDjQ8dJUs", "oDGo0LDH9UE", "hOnXJYCePa8", "DeDDz4c_ajE", "sRWpd8EcZ3c"}
# Verdicts that count as a good PDF
GOOD = {"correct", "near_correct", "alternate_correct"}
# Largest share of the truth panel's ink a video's print area may leave out
CUT_INK_MAX = 0.02
# Gray levels a cursor box or highlight bar takes once the paper is whitened
TINT_LO, TINT_HI = 170, 250
# A printed measure whose solid light-gray area is above this share keeps cursor tint
RESIDUE_SHARE = 0.15
# A light-gray column over this share of a measure's height is a playhead line left in
LINE_SHARE = 0.8
# A printed measure cut by the panel edge and narrower than this share of the video's median printed measure is a
# sliver (short complete measures, such as 3/8 bars among 6/8 ones, are not)
SLIVER_SHARE = 0.5
# Slivers are gated only on this video; elsewhere they are reported
SLIVER_CHECK = {"lJAnWNgrrTo"}


def eval_dir(arg: str | None) -> Path:
    path = arg or os.environ.get("YOUTABBER_EVAL")
    if not path:
        sys.exit("Set YOUTABBER_EVAL to the folder holding truth/ and mediator/score.py, or pass --eval DIR.")
    p = Path(path).expanduser()
    if not (p / "mediator" / "score.py").exists():
        sys.exit(f"{p} has no mediator/score.py. YOUTABBER_EVAL must name the folder holding truth/ and mediator/.")
    return p


def eval_sets(ev: Path) -> dict:
    """The eval folder's eval-sets.json: per set, the truth folder, reference score file and reference outputs
    (relative to the eval folder), and the reference's key in mediator/pdfpass/cut_ink.json."""
    path = ev / "eval-sets.json"
    if not path.exists():
        sys.exit(f"No {path}. Add it with {{\"sets\": {{\"dev\": {{\"truth\": ..., \"reference_score\": ..., "
                 f"\"reference_outputs\": ...}}, ...}}, \"cut_ink_reference\": ...}}.")
    return json.loads(path.read_text())


def load(path: Path, name: str):
    """Import a Python file by path, so the frozen scorer is used as it is."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def truths(folder: Path) -> list[dict]:
    """Truth records in a folder; index files that sit beside them are skipped."""
    if not folder.is_dir():
        sys.exit(f"No truth folder at {folder}.")
    out = []
    for f in sorted(folder.glob("*.json")):
        data = json.loads(f.read_text())
        if isinstance(data, dict) and "id" in data and "outcome" in data:
            out.append(data)
    return out


def truth_folder(ev: Path, name: str) -> Path:
    """Truth folder of a set. The closed acceptance set is kept outside the eval folder; YOUTABBER_ACCEPT names it."""
    if name != "accept":
        return ev / eval_sets(ev)["sets"][name]["truth"]
    path = os.environ.get("YOUTABBER_ACCEPT")
    if not path:
        sys.exit("Set YOUTABBER_ACCEPT to the folder holding the acceptance truth files.")
    return Path(path).expanduser()


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


# ---------------------------------------------------------------- running

def to_result(video_id: str, reading, passes: str) -> dict:
    """The reading in the format the scorer reads."""
    cands = reading.every_pass if passes == "all" else reading.candidates

    def measure(m):
        return {"t": m.t, "x": m.x, "y": m.y, "t_ref": m.t_ref,
                "label": str(m.label) if m.label is not None else None, "partial": m.partial}

    return {
        "video": video_id, "side": "v1",
        "fills": ["outcome", "fail_reason", "message", "mode", "tab_spans", "panels", "measures", "warnings"],
        "assisted": [], "mode": passes,
        "outcome": reading.outcome, "fail_reason": reading.fail_reason, "message": reading.message,
        "tab_spans": [list(s) for s in reading.tab_spans], "panels": reading.panels,
        "candidates": [{"name": c.name, "mode": c.mode, "measures": [measure(m) for m in c.measures]}
                       for c in cands] if reading.outcome == "tab" else [],
        "warnings": [{"t0": round(n.t0, 2), "t1": round(n.t1, 2), "kind": n.kind, "text": n.text}
                     for n in reading.notices],
    }


def cmd_one(a) -> None:
    """One video in this process: read it, write the outputs and result.json."""
    from youtabber import pipeline
    started = time.time()
    video, out = Path(a.video), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reading = pipeline.read(video)
    from youtabber import cli, enhance
    from youtabber.source import Source
    pdfs = cli.write_outputs(reading, Source(video, video.stem, video.stem, None, None), out, video.stem, a.passes,
                             enhance.classic, started=started)
    result = to_result(video.stem, reading, a.passes)
    result["pdfs"] = [p.name for p in pdfs]
    result["runtime_s"] = round(time.time() - started, 1)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    result["peak_rss_mb"] = round(rss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    (out / "result.json").write_text(json.dumps(result, indent=1))


def run_one(vid: str, dest: Path, passes: str) -> str:
    """Run one video in its own process, so its memory is measured alone and a hang can be stopped."""
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "result.json").unlink(missing_ok=True)
    cmd = ["/usr/bin/time", "-l", sys.executable, str(Path(__file__).resolve()), "one",
           str(CACHE / f"{vid}.mp4"), str(dest), "--passes", passes]
    started = time.time()
    with open(dest / "run.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            # The whole process group, so ffmpeg children stop too
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            log.write(f"\nstopped after {TIMEOUT_S} s\n")
    status = "ok" if (dest / "result.json").exists() else "no result.json, see run.log"
    return f"{vid}: {status} ({time.time() - started:.0f} s)"


def run_all(jobs: list[tuple[str, Path]], workers: int, passes: str) -> None:
    with ThreadPoolExecutor(workers) as pool:
        for line in pool.map(lambda j: run_one(j[0], j[1], passes), jobs):
            print(line, flush=True)


def cmd_run(a) -> None:
    ev = eval_dir(a.eval)
    wanted = set(a.ids.split(",")) if a.ids else None
    jobs = []
    for name in a.sets.split(","):
        for t in truths(truth_folder(ev, name)):
            if wanted is None or t["id"] in wanted:
                jobs.append((t["id"], a.out / name / t["id"]))
    run_all(jobs, a.jobs, a.passes)


# ---------------------------------------------------------------- scoring

def played_at(truth, mid) -> float | None:
    """When the truth first plays measure mid: between a smooth scroll's anchors, or the start of the screen
    showing it. (A smooth scroll is one long truth entry, so its start says nothing about where a measure is.)"""
    for e in truth.entries:
        ids = truth.reading_ids(e)
        if mid not in ids:
            continue
        if e["kind"] != "run":
            return e["t0"]
        pts = sorted((ids.index(a["id"]), a["t"]) for a in e["anchors"] if a["id"] in ids)
        return float(np.interp(ids.index(mid), [k for k, _ in pts], [t for _, t in pts]))
    return None


def unwarned(sc, truth, result, rep) -> int | None:
    """Errors in the primary PDF with no local notice near the time they are played: the silent part of a wrong
    result."""
    p = rep.get("primary")
    if not p:
        return None
    times = [e["t"] for e in p["extra"]] + [played_at(truth, m) for m in p["missing"]]
    return sum(1 for t in times if not sc.warned_near(result, [t], truth.data.get("duration")))


def gate(kind: str, name: str, reports: list[dict], base: dict, styles: dict) -> list[str]:
    """Reasons the named gate fails for one set (empty when it passes)."""
    out = []
    if base:
        total, ref = sum(r["points"] for r in reports), sum(v["points"] for v in base.values())
        if total < ref:
            out.append(f"{name}: total {total:.1f} below the reference {ref:.1f}")
    for r in reports:
        vid, v = r["id"], r["verdict"]
        p = r.get("primary") or {}
        b = base.get(vid)
        if b is not None and r["points"] < b["points"]:
            out.append(f"{name} {vid}: {v} ({r['points']}) below the reference {b['verdict']} ({b['points']})")
        if v == "silent_wrong" and not (kind == "port" and vid in MUST_WARN):
            out.append(f"{name} {vid}: silent_wrong")
        if kind in ("core", "pdf"):
            if vid in FLOOR_CORRECT and v != "correct":
                out.append(f"{name} {vid}: floor video is {v}, must be correct")
            if vid in FLOOR_NEAR and v not in ("correct", "near_correct"):
                out.append(f"{name} {vid}: floor video is {v}, must be near_correct or better")
            if vid in GLIDE_ERRORS:
                errors = p.get("errors")
                if errors is None or errors > GLIDE_ERRORS[vid]:
                    out.append(f"{name} {vid}: {errors} errors, at most {GLIDE_ERRORS[vid]} allowed")
            # Every glide is judged by play time, whatever the scorer's verdict: the frozen scorer puts a glide's
            # missing measures at the glide's start
            if vid in GLIDES and r.get("unwarned") != 0:
                out.append(f"{name} {vid}: {v} with {r.get('unwarned')} errors without a notice near their play time")
            if vid in NO_EXTRAS and p.get("extra"):
                out.append(f"{name} {vid}: {len(p['extra'])} extra measures, none allowed")
        if kind == "pdf" and r.get("pdf"):
            m = r["pdf"]
            if m["cut_ink"] is None or m["cut_ink"] > CUT_INK_MAX:
                out.append(f"{name} {vid}: cut_ink {m['cut_ink']}, at most {CUT_INK_MAX}")
            if m["residue_cells"]:
                out.append(f"{name} {vid}: {len(m['residue_cells'])} measures keep cursor tint")
            if m["slivers"] and vid in SLIVER_CHECK:
                out.append(f"{name} {vid}: {len(m['slivers'])} printed measures are slivers")
            if m["paper"] != 255:
                out.append(f"{name} {vid}: paper prints at {m['paper']}, not white")
            if m["cut_staff"]:
                out.append(f"{name} {vid}: {len(m['cut_staff'])} printed measures lost their staff")
            # Widening the print area lowers cut ink and can raise spill; spill may not grow past the reference
            if m["spill"] is not None and m["reference_spill"] is not None and m["spill"] > m["reference_spill"] + 0.005:
                out.append(f"{name} {vid}: spill {m['spill']} above the reference {m['reference_spill']}")
        if kind == "accept":
            glide = "smooth_scroll" in styles.get(vid, [])
            if glide:
                # The same play-time rule as every other gate: wrong is acceptable only with every error warned
                ok = v in GOOD | {"warned_wrong", "vague_wrong"} and r.get("unwarned") == 0
            else:
                ok = v in {"correct", "near_correct", "correct_fail"}
            if not ok:
                out.append(f"{name} {vid}: {v}, {r.get('unwarned')} unwarned ({'glide' if glide else 'page or band'})")
    return out


def print_table(name: str, reports: list[dict], base: dict) -> None:
    print(f"== {name}")
    print(f"{'video':<12} {'reference':<18} {'v1':<18} {'pts':>5} {'exp':>4} {'prt':>4} {'miss':>4} {'xtra':>4} "
          f"{'unw':>4} {'sec':>5} {'MB':>6}")
    for r in reports:
        b = base.get(r["id"], {})
        p = r.get("primary") or {}

        def n(x):
            return "-" if x is None else str(x)
        print(f"{r['id']:<12} {b.get('verdict', '-'):<18} {r['verdict']:<18} {r['points']:>5.1f} "
              f"{n(p.get('expected')):>4} {n(p.get('printed')):>4} {n(len(p['missing']) if p else None):>4} "
              f"{n(len(p['extra']) if p else None):>4} {n(r.get('unwarned')):>4} {n(r.get('runtime_s')):>5} "
              f"{n(r.get('peak_rss_mb')):>6}")
    total = sum(r["points"] for r in reports)
    ref = sum(v["points"] for v in base.values()) if base else None
    print(f"total {total:.1f}" + (f" (reference {ref:.1f})" if ref is not None else ""))


def tint_share(img: np.ndarray, gap: float, left: int = 0) -> tuple[float, bool]:
    """(share of solid light-gray area, whether a light-gray vertical line runs through) for a printed measure
    whose left barline is image column `left`. Staff lines and the soft edges of digits are thin, so an opening of
    about half a line gap removes them first."""
    light = ((img >= TINT_LO) & (img < TINT_HI)).astype(np.uint8)
    k = max(3, int(0.5 * gap))
    solid = cv2.morphologyEx(light, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    # Barlines sit at a measure's edges and can print light gray, so lines count only half a gap away from them.
    # A measure number can reach left past the left barline, so that barline is not always at the image edge.
    edge = int(0.5 * gap)
    lines = light.mean(axis=0) > LINE_SHARE
    # A light column beside dark ink over most of its rows is the soft edge of a bracket or barline, not a playhead
    lit, dark = light.astype(bool), img < 128
    pad = np.zeros((img.shape[0], 1), bool)
    beside = np.hstack([pad, dark[:, :-1]]) | np.hstack([dark[:, 1:], pad])
    lines &= (lit & beside).sum(axis=0) <= 0.5 * np.maximum(lit.sum(axis=0), 1)
    lines[:edge] = lines[img.shape[1] - edge:] = False
    lines[max(0, left - edge + 1):max(0, left + edge)] = False
    return float(solid.mean()), bool(lines.any())


def keeps_staff(img: np.ndarray, gap: float) -> bool:
    """Whether a printed measure shows 4 or more tab lines with paper above and below them."""
    from youtabber.staff import find_staffs, line_rows
    h = img.shape[0]
    # Both outer lines need half a gap of paper outside them, or the digits on them are cut
    margin = 0.5 * gap
    if any(len(st.lines) >= 4 and st.top >= margin and st.bottom <= h - margin for st in find_staffs(img)):
        return True
    # Stacked digits can break the lines so find_staffs drops the staff; evenly spaced rows still show it.
    # Rows must cover half the width, so a column of digits can't join neighboring lines into one row.
    ys = [y for y, *_ in line_rows(img, "dark", img.shape[1] // 2)]
    runs = [ys[:1]]
    for prev, y in zip(ys, ys[1:]):
        if abs(y - prev - gap) <= 0.25 * gap:
            runs[-1].append(y)
        else:
            runs.append([y])
    return any(len(run) >= 4 and run[0] >= margin and run[-1] <= h - margin for run in runs)


def pdf_metrics(ev: Path, truth, result: dict, folder: Path) -> dict:
    """Cut ink and spill of the print area against the truth panel, and printed measures that keep tint or are
    cut short, for one video."""
    ci = load(ev / "mediator" / "pdfpass" / "cut_ink.py", "frozen_cut_ink")
    times = ci.sample_times(truth.data)
    frames = ci.read_frames(CACHE / f"{truth.id}.mp4", [t for t, _ in times])
    cuts, spills = [], []
    for t, tp in times:
        f = frames.get(t)
        if f is None:
            continue
        cut, _, spill, _, _ = ci.measure(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), tp["box"],
                                         ci.side_box_at(result.get("panels", []), t), tp["polarity"])
        if cut is not None:
            cuts.append(cut)
        if spill is not None:
            spills.append(spill)
    meta = read_json(folder / "meta.json") or {"measures": []}
    residue, slivers = [], []
    images = [(m, cv2.imread(str(folder / m["image"]), cv2.IMREAD_GRAYSCALE)) for m in meta["measures"]]
    # Widths in tab line gaps, so measures from layouts of different sizes compare
    widths = [img.shape[1] / m["gap"] for m, img in images]
    typical = float(np.median(widths)) if widths else 0.0
    for (m, img), w in zip(images, widths):
        share, line = tint_share(img, m["gap"], m.get("left", 0))
        if share > RESIDUE_SHARE or line:
            residue.append(f"{m['image']} (t {m['t']}, label {m['label']})")
        if not m["complete"] and w < SLIVER_SHARE * typical:
            slivers.append(f"{m['image']} (t {m['t']}, {w:.1f} gaps wide)")
    cut_staff = [m["image"] for m, img in images if not keeps_staff(img, m["gap"])]
    reference = read_json(ev / "mediator" / "pdfpass" / "cut_ink.json") or {"videos": {}}
    side = eval_sets(ev)["cut_ink_reference"]
    ref_spill = reference["videos"].get(truth.id, {}).get(side, {}).get("mean_spill_nonpaper")
    return {"cut_ink": round(float(np.mean(cuts)), 4) if cuts else None,
            "spill": round(float(np.mean(spills)), 4) if spills else None,
            "reference_spill": ref_spill,
            "paper": min((int(np.percentile(img, 90)) for _, img in images), default=255),
            "residue_cells": residue, "slivers": slivers, "cut_staff": cut_staff}


def cmd_score(a) -> None:
    ev = eval_dir(a.eval)
    sc = load(ev / "mediator" / "score.py", "frozen_score")
    problems = []
    for name in a.sets.split(","):
        base_file = eval_sets(ev)["sets"].get(name, {}).get("reference_score")
        base = {v["id"]: v for v in json.loads((ev / base_file).read_text())["videos"]} if base_file else {}
        data = truths(truth_folder(ev, name))
        styles = {d["id"]: d.get("style", []) for d in data}
        reports = []
        for d in data:
            truth = sc.Truth(d)
            result = sc.load_result(a.out / name, truth.id)
            rep = sc.score_video(truth, result)
            rep["unwarned"] = unwarned(sc, truth, result, rep)
            if name == "confirm" and result and result.get("outcome") == "tab":
                rep["pdf"] = pdf_metrics(ev, truth, result, a.out / name / truth.id)
            reports.append(rep)
        (a.out / f"score-{name}.json").write_text(json.dumps(
            {"total_points": sum(r["points"] for r in reports), "videos": reports}, indent=1, default=str))
        print_table(name, reports, base)
        if name == "confirm":
            print(f"{'video':<12} {'cut_ink':>7} {'spill':>6} {'ref':>6} {'paper':>5} {'tint':>4} {'sliv':>4} {'staff':>5}")
            for r in reports:
                m = r.get("pdf")
                if m:
                    print(f"{r['id']:<12} {m['cut_ink']!s:>7} {m['spill']!s:>6} {m['reference_spill']!s:>6} "
                          f"{m['paper']:>5} {len(m['residue_cells']):>4} {len(m['slivers']):>4} {len(m['cut_staff']):>5}")
        if a.gate:
            problems += gate(a.gate, name, reports, base, styles)
    if a.gate:
        if problems:
            print(f"\ngate {a.gate} FAILED:")
            for p in problems:
                print(f"  {p}")
            sys.exit(1)
        print(f"\ngate {a.gate} passed")


def cmd_compare(a) -> None:
    """Outcome, measure counts and notice counts against the reference outputs, video by video."""
    ev = eval_dir(a.eval)

    def summary(r):
        if r is None:
            return None
        return (r.get("outcome"), [len(c["measures"]) for c in r.get("candidates", [])], len(r.get("warnings", [])))

    def sequence(r):
        # Times to a tenth of a second: current libraries shift some by 0.01 s with the same measures
        cands = (r or {}).get("candidates") or [{}]
        return [(round(m["t"], 1), m.get("label"), round(m["x"])) for m in cands[0].get("measures", [])]

    for name in a.sets.split(","):
        frozen = ev / eval_sets(ev)["sets"][name]["reference_outputs"]
        for t in truths(truth_folder(ev, name)):
            mine = read_json(a.out / name / t["id"] / "result.json")
            ref = read_json(frozen / t["id"] / "result.json")
            same = "same" if sequence(mine) == sequence(ref) else "DIFFERS"
            print(f"{name:<8} {t['id']:<12} {same:<8} reference {summary(ref)}  v1 {summary(mine)}")


def cmd_partials(a) -> None:
    """Our cell completeness against the truth's partial flags, on screens where the truth gives x ranges."""
    ev = eval_dir(a.eval)
    for name in a.sets.split(","):
        for t in truths(truth_folder(ev, name)):
            meta = read_json(a.out / name / t["id"] / "meta.json")
            if not meta or t["outcome"] != "tab":
                continue
            screens = [e for e in t.get("screens", []) if e.get("kind", "screen") == "screen"]
            agree = said_complete = said_partial = 0
            examples = []
            for v in meta["views"]:
                if v["kind"] != "plate":
                    continue
                mid = (v["t"][0] + v["t"][1]) / 2
                e = next((e for e in screens if e["t0"] - 1.5 <= mid <= e["t1"] + 1.5), None)
                if e is None:
                    continue
                for c in v["cells"]:
                    if c["system"] >= len(e["systems"]):
                        continue
                    for m in e["systems"][c["system"]]["measures"]:
                        if not isinstance(m, dict) or not m.get("x"):
                            continue
                        x0, x1 = m["x"]
                        if min(c["x"][1], x1) - max(c["x"][0], x0) < 0.6 * min(c["x"][1] - c["x"][0], x1 - x0):
                            continue
                        truth_complete = not m.get("partial", False)
                        if c["complete"] == truth_complete:
                            agree += 1
                        elif c["complete"]:
                            said_complete += 1
                            examples.append(f"t {mid:.1f} x {c['x']}: complete here, partial in truth")
                        else:
                            said_partial += 1
                            examples.append(f"t {mid:.1f} x {c['x']}: partial here, complete in truth")
                        break
            print(f"{name:<8} {t['id']:<12} agree {agree:>4}  complete-but-partial {said_complete:>3}  "
                  f"partial-but-complete {said_partial:>3}")
            for ex in examples[:5]:
                print(f"    {ex}")


def cmd_allpass(a) -> None:
    """Runs made with --passes all, against the truth's every-pass sequence (the scorer holds a primary PDF to
    the main sequence, so this compares directly)."""
    ev = eval_dir(a.eval)
    sc = load(ev / "mediator" / "score.py", "frozen_score")
    for name in a.sets.split(","):
        for d in truths(truth_folder(ev, name)):
            truth = sc.Truth(d)
            result = sc.load_result(a.out / name, truth.id)
            if truth.outcome != "tab" or not result or result.get("outcome") != "tab":
                print(f"{name:<8} {truth.id:<12} truth {truth.outcome}, v1 {result and result.get('outcome')}")
                continue
            s = sc.score_sequence(truth, truth.print_all(), result["candidates"][0]["measures"])
            times = [e["t"] for e in s["extra"]] + [played_at(truth, m) for m in s["missing"]]
            silent = sum(1 for t in times if not sc.warned_near(result, [t], truth.data.get("duration")))
            print(f"{name:<8} {truth.id:<12} expected {s['expected']:>4} printed {s['printed']:>4} "
                  f"missing {len(s['missing']):>3} extra {len(s['extra']):>3} unwarned {silent:>3}")


def cmd_corpus(a) -> None:
    """Every cached corpus video, and an index to compare with the earlier tool's outputs."""
    corpus = json.loads((a.corpus / "corpus.json").read_text())
    skip = set()
    if a.exclude:
        ids = json.loads(a.exclude.read_text())["ids"]
        skip = set(ids if isinstance(ids, list) else ids.keys())
    wanted = set(a.ids.split(",")) if a.ids else None
    todo = [v for v in corpus if v["id"] not in skip and (wanted is None or v["id"] in wanted)
            and (CACHE / f"{v['id']}.mp4").exists()]
    print(f"{len(todo)} videos to run, {len(corpus) - len(todo)} skipped (closed or not cached)")
    run_all([(v["id"], a.out / v["id"]) for v in todo], a.jobs, "once")
    lines = ["| id | earlier PDF | v1 outcome | v1 measures | v1 notices | v1 PDFs |", "|---|---|---|---|---|---|"]
    for v in todo:
        r = read_json(a.out / v["id"] / "result.json") or {}
        earlier = sorted((a.corpus / v["id"]).glob("*.pdf"))
        cands = r.get("candidates") or []
        lines.append(f"| {v['id']} | {earlier[0] if earlier else '-'} | {r.get('outcome', 'crash')} "
                     f"{r.get('fail_reason') or ''} | {len(cands[0]['measures']) if cands else 0} | "
                     f"{len(r.get('warnings', []))} | {', '.join(r.get('pdfs', []))} |")
    (a.out / "index.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out / 'index.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run and score youtabber on videos with known measures.")
    ap.add_argument("--eval", help="folder holding the truth, the scorer and eval-sets.json (default: $YOUTABBER_EVAL)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one")
    one.add_argument("video")
    one.add_argument("out")
    one.add_argument("--passes", choices=["once", "all"], default="once")
    for cmd in ("run", "score", "compare", "partials", "allpass"):
        p = sub.add_parser(cmd)
        p.add_argument("--sets", required=True, help="comma-separated: " + ",".join(SETS))
        p.add_argument("--out", type=Path, required=True)
        if cmd == "run":
            p.add_argument("--jobs", type=int, default=4)
            p.add_argument("--ids", help="comma-separated ids to run from those sets")
            p.add_argument("--passes", choices=["once", "all"], default="once")
        if cmd == "score":
            p.add_argument("--gate", choices=["port", "core", "pdf", "accept"])
    corpus = sub.add_parser("corpus")
    corpus.add_argument("--corpus", type=Path, required=True)
    corpus.add_argument("--out", type=Path, required=True)
    corpus.add_argument("--exclude", type=Path, help="JSON file whose 'ids' are skipped without being shown")
    corpus.add_argument("--jobs", type=int, default=4)
    corpus.add_argument("--ids", help="comma-separated corpus ids to run (default: all)")
    a = ap.parse_args()
    {"one": cmd_one, "run": cmd_run, "score": cmd_score, "compare": cmd_compare, "partials": cmd_partials,
     "allpass": cmd_allpass, "corpus": cmd_corpus}[a.cmd](a)


if __name__ == "__main__":
    main()
