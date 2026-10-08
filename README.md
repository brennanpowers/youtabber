# tabrip

Rips the tab from a YouTube play-along video into a clean PDF.

```sh
uv run tabrip "https://www.youtube.com/watch?v=..."
uv run tabrip video.mp4 -o out/song
uv run tabrip video.mp4 --region 370,712,1546,364   # set the tab area by hand
```

Requires `ffmpeg` on the PATH. Downloads are cached in `~/.cache/tabrip`.

## Output

| File | What it holds |
|---|---|
| `tab.pdf` | The tab, laid out in rows that break at barlines |
| `region.png` | A frame with the tab area outlined; check this first when the PDF looks wrong |
| `views/` | Each distinct tab image the video showed, with the cursor removed |
| `strip.png` | Every view joined into one long line of music |
| `meta.json` | Timestamps of each view, where it sits in the strip, staff line rows, and row cuts |

## How it works

1. **Find the tab area.** Staff lines are thin, dark, horizontal, and never move, so averaging a
   thin-line mask over the whole video finds them. The area then grows over neighboring paper.
2. **Find stable views.** Frames are compared by their ink only (pixels where every color channel is
   dark), which ignores colored cursors and highlights. Each run of matching frames becomes one view,
   and the per-pixel median of its frames removes the cursor.
3. **Stitch.** Players that scroll keep part of the previous view on screen. Each new view is slid
   across the previous one to find where they overlap, and only the new part is appended. When
   repeated measures match in more than one place, the player's usual scroll distance decides.
4. **Lay out.** The strip is cut at barlines into rows that fit the page.
