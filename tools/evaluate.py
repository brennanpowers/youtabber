"""Run youtabber on videos whose measures are known, and score the results with the frozen scorer.

The truth files, the scorer and the reference scores live outside this repository: pass --eval DIR or set
YOUTABBER_EVAL to the folder holding truth/, mediator/ and opus/. Videos are read from
~/.cache/youtabber/<id>.mp4. Nothing here reads the youtabber config file.

  uv run python tools/evaluate.py run --sets dev,h1 --out DIR [--jobs 4] [--ids A,B] [--passes once|all]
  uv run python tools/evaluate.py score --sets dev,h1 --out DIR [--gate port|core|pdf|accept]
  uv run python tools/evaluate.py compare --sets dev --out DIR
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

import numpy as np

CACHE = Path.home() / ".cache" / "youtabber"
# A run longer than this counts as a crash, as the scorer rules
TIMEOUT_S = 600

# Per set: truth folder, reference score file, reference outputs, relative to the eval folder
SETS = {
    "dev": ("truth", "mediator/scores/opus-3c-dev.json", "opus/out"),
    "h1": ("truth/heldout", "mediator/scores/opus-3c-h1.json", "opus/heldout-3c-h1"),
    "h2": ("truth/heldout2", "mediator/scores/opus-3c-h2.json", "opus/heldout-3c-h2"),
    "h3": ("truth/heldout3", "mediator/scores/opus-3c-h3.json", "opus/heldout3"),
    "confirm": ("truth/confirm", "mediator/scores/opus-confirm.json", "opus/confirm"),
    "reserve": ("truth/reserve-r4", "mediator/scores/opus-r4-reserve.json", "opus/r4-reserve"),
    "accept": (None, None, None),
}
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


def eval_dir(arg: str | None) -> Path:
    path = arg or os.environ.get("YOUTABBER_EVAL")
    if not path:
        sys.exit("Set YOUTABBER_EVAL to the folder holding truth/ and mediator/score.py, or pass --eval DIR.")
    p = Path(path).expanduser()
    if not (p / "mediator" / "score.py").exists():
        sys.exit(f"{p} has no mediator/score.py. YOUTABBER_EVAL must name the folder holding truth/ and mediator/.")
    return p


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
        return ev / SETS[name][0]
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
                             enhance.classic)
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


def cmd_score(a) -> None:
    ev = eval_dir(a.eval)
    sc = load(ev / "mediator" / "score.py", "frozen_score")
    problems = []
    for name in a.sets.split(","):
        _, base_file, _ = SETS[name]
        base = {v["id"]: v for v in json.loads((ev / base_file).read_text())["videos"]} if base_file else {}
        data = truths(truth_folder(ev, name))
        styles = {d["id"]: d.get("style", []) for d in data}
        reports = []
        for d in data:
            truth = sc.Truth(d)
            result = sc.load_result(a.out / name, truth.id)
            rep = sc.score_video(truth, result)
            rep["unwarned"] = unwarned(sc, truth, result, rep)
            reports.append(rep)
        (a.out / f"score-{name}.json").write_text(json.dumps(
            {"total_points": sum(r["points"] for r in reports), "videos": reports}, indent=1, default=str))
        print_table(name, reports, base)
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
        frozen = ev / SETS[name][2]
        for t in truths(truth_folder(ev, name)):
            mine = read_json(a.out / name / t["id"] / "result.json")
            ref = read_json(frozen / t["id"] / "result.json")
            same = "same" if sequence(mine) == sequence(ref) else "DIFFERS"
            print(f"{name:<8} {t['id']:<12} {same:<8} reference {summary(ref)}  v1 {summary(mine)}")


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
    ap.add_argument("--eval", help="folder holding truth/, mediator/ and opus/ (default: $YOUTABBER_EVAL)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one")
    one.add_argument("video")
    one.add_argument("out")
    one.add_argument("--passes", choices=["once", "all"], default="once")
    for cmd in ("run", "score", "compare"):
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
    {"one": cmd_one, "run": cmd_run, "score": cmd_score, "compare": cmd_compare, "corpus": cmd_corpus}[a.cmd](a)


if __name__ == "__main__":
    main()
