import argparse
import json
import shutil
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import cv2

from youtabber import enhance, layout, region, source, stitch, views

CONFIG = Path.home() / ".config" / "youtabber" / "config.toml"


def load_config() -> dict:
    if not CONFIG.exists():
        return {}
    try:
        return tomllib.loads(CONFIG.read_text())
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"Can't read {CONFIG}: {e}")


def parse_region(text: str) -> region.Region:
    try:
        x, y, w, h = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected x,y,w,h in pixels, for example 0,800,1920,280")
    return region.Region(x, y, w, h)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rip the tab from a play-along video into a PDF.")
    parser.add_argument("source", help="YouTube URL or local video file")
    parser.add_argument("-o", "--out", type=Path, help="output folder (default: out/<song name>)")
    parser.add_argument("--title", help="song name for the PDF, when the one taken from the video title is wrong")
    parser.add_argument("--region", type=parse_region,
                        help="tab area as x,y,w,h in video pixels, when detection gets it wrong")
    parser.add_argument("--enhance", action=argparse.BooleanOptionalAction, default=None,
                        help="upscale and sharpen the tab images in the PDF (on unless the config says otherwise)")
    parser.add_argument("--model", help=f"super-resolution model for enhancing: {', '.join(enhance.MODELS)} "
                                        "or a path to a model file (needs the ai extra)")
    args = parser.parse_args()
    config = load_config()
    pdf_dir = Path(config["pdf_dir"]).expanduser() if config.get("pdf_dir") else None
    # The command line wins over the config file, and enhancing is on when neither says
    enhanced = args.enhance if args.enhance is not None else config.get("enhance", True)
    model = args.model or config.get("model")
    # Load the model before the slow video work so a bad name or missing extra fails right away
    upscale = (enhance.model_upscaler(model) if model else enhance.classic) if enhanced else None

    src = source.fetch(args.source)
    if args.title:
        src = replace(src, title=args.title)
    video = src.path
    name = source.slug(src.title)
    out = args.out or Path("out") / name
    pdf_path = out / f"{name}.pdf"
    (out / "views").mkdir(parents=True, exist_ok=True)

    print("Finding the tab area")
    try:
        analysis = region.detect(video)
    except RuntimeError as e:
        if not args.region:
            raise SystemExit(str(e))
        analysis = None
    if analysis:
        region.debug_image(analysis, out / "region.png", args.region)
    tab_region = args.region or analysis.region
    print(f"  {tab_region}, see {out / 'region.png'}")

    print("Finding stable views")
    found = list(views.find_views(video, tab_region))
    if not found:
        raise SystemExit(f"No tab views found; check {out / 'region.png'} and try --region")
    for old in (out / "views").glob("*.png"):
        old.unlink()
    for i, v in enumerate(found):
        cv2.imwrite(str(out / "views" / f"{i:03d}.png"), v.image)
    print(f"  {len(found)} views")

    print("Stitching")
    strip, placed = stitch.stitch(found)
    cv2.imwrite(str(out / "strip.png"), strip)
    scale = layout.scale_for(strip, tab_region.w)
    row_ranges = layout.rows(strip, placed, int(layout.USABLE_WIDTH / scale))
    layout.write_pdf(strip, row_ranges, scale, pdf_path, src, upscale)
    overlaps = sum(1 for p in placed if p.new_from)
    print(f"  {overlaps} overlapping views joined, {len(row_ranges)} rows")
    guessed = [p for p in placed if p.ambiguous]
    if guessed:
        times = ", ".join(f"{p.view.start:.0f}s" for p in guessed)
        print(f"  {len(guessed)} joins had repeated measures and used the usual scroll distance: views at {times}")

    meta = {
        "source": args.source,
        "title": src.title,
        "credit": src.credit,
        "url": src.url,
        "region": asdict(tab_region),
        "views": [{"start": p.view.start, "end": p.view.end, "strip_x": p.x, "new_from": p.new_from,
                   "ambiguous": p.ambiguous} for p in placed],
        "staff_lines": layout.staff_lines(strip),
        "rows": row_ranges,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {pdf_path}")
    if pdf_dir:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, pdf_dir / pdf_path.name)
        print(f"Copied to {pdf_dir / pdf_path.name}")
