import argparse
import json
import shutil
import tomllib
from dataclasses import asdict, replace
from pathlib import Path

import cv2

from youtabber import enhance, layout, region, source, stitch, views
from youtabber.video import VideoError

CONFIG = Path.home() / ".config" / "youtabber" / "config.toml"
# Smallest tab area, in video pixels, that can hold a readable staff
MIN_REGION = 32


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
    if x < 0 or y < 0 or w < MIN_REGION or h < MIN_REGION:
        raise argparse.ArgumentTypeError(f"x and y can't be negative, and w and h must be at least {MIN_REGION}")
    return region.Region(x, y, w, h)


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn a play-along video into a printable PDF of its tab.")
    parser.add_argument("source", help="YouTube URL or local video file")
    parser.add_argument("-o", "--out", type=Path, help="output folder (default: out/<song name>)")
    parser.add_argument("--title", help="song name for the PDF, when the one taken from the video title is wrong")
    parser.add_argument("--region", type=parse_region,
                        help="tab area as x,y,w,h in video pixels, when detection gets it wrong")
    parser.add_argument("--enhance", action=argparse.BooleanOptionalAction, default=None,
                        help="upscale and sharpen the tab images in the PDF (on unless the config says otherwise)")
    parser.add_argument("--model", help=f"how to enhance: classic, {', '.join(enhance.MODELS)}, "
                                        "or a path to a model file (models need the ai extra)")
    args = parser.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"youtabber needs {tool} on your PATH. On macOS: brew install ffmpeg")
    try:
        run(args, load_config())
    except VideoError as e:
        raise SystemExit(str(e))


def run(args: argparse.Namespace, config: dict) -> None:
    pdf_dir = Path(config["pdf_dir"]).expanduser() if config.get("pdf_dir") else None
    model = args.model or config.get("model")
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
    video = src.path
    name = source.slug(src.title) or source.slug(src.id) or "tab"
    out = args.out or Path("out") / name
    pdf_path = out / f"{name}.pdf"
    (out / "views").mkdir(parents=True, exist_ok=True)
    region_png = out / "region.png"

    print("Finding the tab area")
    analysis = region.detect(video)
    region.debug_image(analysis, region_png, args.region)
    tab_region = args.region or analysis.region
    if tab_region is None:
        raise SystemExit(f"No staff lines found. Open {region_png} and pass the tab area with --region x,y,w,h")
    print(f"  {tab_region}, see {region_png}")

    print("Finding stable views")
    found = list(views.find_views(video, tab_region))
    if not found:
        raise SystemExit(f"No tab views found; check {region_png} and try --region")
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
    layout.write_pdf(strip, row_ranges, scale, pdf_path, src, placed, upscale)
    overlaps = sum(1 for p in placed if p.new_from)
    print(f"  {overlaps} overlapping views joined, {len(row_ranges)} rows")
    guessed = [p for p in placed if p.ambiguous]
    if guessed:
        print(f"  {len(guessed)} joins matched in more than one place because of repeated measures; "
              f"check the views at {_times(guessed)}")
    unjoined = [p for p in placed if p.unjoined]
    if unjoined:
        print(f"  {len(unjoined)} views didn't line up with the one before and were added whole, so some "
              f"measures may appear twice; check the views at {_times(unjoined)}")

    meta = {
        "source": args.source,
        "title": src.title,
        "credit": src.credit,
        "url": src.url,
        "region": asdict(tab_region),
        "views": [{"start": p.view.start, "end": p.view.end, "strip_x": p.x, "new_from": p.new_from,
                   "ambiguous": p.ambiguous, "unjoined": p.unjoined} for p in placed],
        "staff_lines": layout.staff_lines(strip),
        "rows": row_ranges,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {pdf_path}")
    if pdf_dir:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, pdf_dir / pdf_path.name)
        print(f"Copied to {pdf_dir / pdf_path.name}")


def _times(placed: list[stitch.Placed]) -> str:
    return ", ".join(f"{p.view.start:.0f}s" for p in placed)
