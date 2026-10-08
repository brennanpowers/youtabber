import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import yt_dlp

from tabrip import layout, region, stitch, views

CACHE = Path.home() / ".cache" / "tabrip"


def fetch(source: str) -> tuple[Path, str, str]:
    """Return (video path, id, title) for a local file or a YouTube URL, downloading into the cache."""
    path = Path(source)
    if path.exists():
        return path, path.stem, path.stem
    opts = {
        "format": "bv*[height<=1080][ext=mp4]/bv*[height<=1080]",
        "outtmpl": str(CACHE / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=False)
        cached = next(CACHE.glob(f"{info['id']}.*"), None)
        if cached is None:
            print(f"Downloading {info['title']}")
            ydl.download([source])
            cached = next(CACHE.glob(f"{info['id']}.*"))
    return cached, info["id"], info["title"]


def parse_region(text: str) -> region.Region:
    try:
        x, y, w, h = (int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected x,y,w,h in pixels, for example 0,800,1920,280")
    return region.Region(x, y, w, h)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rip the tab from a play-along video into a PDF.")
    parser.add_argument("source", help="YouTube URL or local video file")
    parser.add_argument("-o", "--out", type=Path, help="output folder (default: out/<video id>)")
    parser.add_argument("--region", type=parse_region,
                        help="tab area as x,y,w,h in video pixels, when detection gets it wrong")
    args = parser.parse_args()

    video, video_id, title = fetch(args.source)
    out = args.out or Path("out") / video_id
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
    row_ranges = layout.rows(strip, placed, tab_region.w)
    layout.write_pdf(strip, row_ranges, tab_region.w, out / "tab.pdf", title)
    overlaps = sum(1 for p in placed if p.new_from)
    print(f"  {overlaps} overlapping views joined, {len(row_ranges)} rows")
    guessed = [p for p in placed if p.ambiguous]
    if guessed:
        times = ", ".join(f"{p.view.start:.0f}s" for p in guessed)
        print(f"  {len(guessed)} joins had repeated measures and used the usual scroll distance: views at {times}")

    meta = {
        "source": args.source,
        "title": title,
        "region": asdict(tab_region),
        "views": [{"start": p.view.start, "end": p.view.end, "strip_x": p.x, "new_from": p.new_from,
                   "ambiguous": p.ambiguous} for p in placed],
        "staff_lines": layout.staff_lines(strip),
        "rows": row_ranges,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out / 'tab.pdf'}")
