# tabrip

Rips the tab from a YouTube play-along video into a clean PDF.

```sh
uv run tabrip "https://www.youtube.com/watch?v=..."
uv run tabrip video.mp4 -o out/song
uv run tabrip video.mp4 --region 370,712,1546,364   # set the tab area by hand
uv run tabrip "https://..." --title "Song - Artist" # when the name taken from the video title is wrong
```

Output goes to `out/<song-name>/`, where the song name comes from the video title with words like
"Bass Cover (Play Along Tabs)" dropped.

Requires `ffmpeg` on the PATH. Downloads are cached in `~/.cache/tabrip`.

## Config

To also copy each finished PDF to a folder of your tabs, create `~/.config/tabrip/config.toml`:

```toml
pdf_dir = "~/Documents/tabs"
```

## Output

| File | What it holds |
|---|---|
| `<song-name>.pdf` | The tab on Letter pages, headed with the song name and the channel that made it |
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
4. **Lay out.** Every song is scaled so its tab lines sit the same distance apart on paper, so
   notation prints the same size whatever the video looked like. The strip is then cut at barlines
   into rows of even width, and blank paper above and below the music is trimmed.
