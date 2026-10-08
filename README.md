# youtabber

Turns a YouTube play-along video into a clean, printable PDF of its tab.

Play-along videos usually show someone playing on one part of the screen and the tab on another,
with a cursor moving through it. youtabber finds the tab, removes the cursor, joins the pieces the
video shows over time into one continuous piece of music, and lays it out on Letter pages.

It handles the common layouts:

- Tab along the bottom or top of the frame, or in a box that covers only part of it
- Videos that show one line of tab at a time and swap in the next
- Videos that jump forward while keeping part of the previous view on screen
- Light colored highlight boxes and cursor lines, such as yellow, blue, or red

## Requirements

- [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/) on your `PATH` (`brew install ffmpeg` on macOS)

## Install

```sh
git clone https://github.com/brennanpowers/youtabber.git
uv tool install --editable ./youtabber
```

This puts a `youtabber` command on your `PATH`. `--editable` runs the code straight from the clone,
so changes take effect without reinstalling. To run it without installing, use
`uv run youtabber ...` from inside the clone.

## Usage

```sh
youtabber "https://www.youtube.com/watch?v=..."
```

| Option | What it does |
|---|---|
| `SOURCE` | A YouTube URL or a local video file |
| `--title "Song - Artist"` | Sets the song name, when the one taken from the video title is wrong |
| `--region x,y,w,h` | Sets the tab area by hand, in video pixels, when detection gets it wrong |
| `-o FOLDER` | Writes output somewhere other than `out/<song-name>/` |

The song name comes from the video title with phrases like "Bass Cover (Play Along Tabs)" removed.
Downloads are cached in `~/.cache/youtabber`, so running the same video again doesn't download it again.

### Output

Each run writes a folder, `out/<song-name>/` under the current directory by default:

| File | What it holds |
|---|---|
| `<song-name>.pdf` | The tab, headed with the song name and the channel that made the video |
| `region.png` | A frame with the detected tab area outlined; check this first when the PDF looks wrong |
| `views/` | Each distinct tab image the video showed, with the cursor removed |
| `strip.png` | Every view joined into one long line of music |
| `meta.json` | When each view was on screen, where it sits in the strip, and where rows were cut |

### Config

To also copy every finished PDF into one folder, create `~/.config/youtabber/config.toml`:

```toml
pdf_dir = "~/Documents/tabs"
```

### Fixing a wrong tab area

Run once, open `region.png`, and compare the outline with the tab. The run prints the detected area as
`Region(x=…, y=…, w=…, h=…)`; adjust those numbers and pass them back with `--region x,y,w,h`.

## How it works

1. **Find the tab area.** Staff lines are thin, dark, horizontal, and stay in the same place for the
   whole video. Averaging a thin-line mask over frames sampled across the video finds them, even
   though notes cover parts of them in any single frame. The area then grows over the surrounding
   paper to take in chord names and measure numbers.
2. **Find stable views.** Frames are compared by their ink only: pixels where every color channel is
   dark. Colored cursors and highlights aren't ink, so they don't count as changes. Each run of
   matching frames becomes one view, and taking the median of each pixel across the run erases the
   cursor.
3. **Stitch.** When a video jumps forward but keeps part of the previous view on screen, each new
   view is slid across the previous one to find where they overlap, and only the new part is kept.
   Bass lines often repeat a measure exactly, which can make several overlaps match. Players scroll
   the same distance every time, so the usual distance decides between them.
4. **Lay out.** Every song is scaled so its tab lines sit the same distance apart on paper, which
   makes notation print the same size whatever the video looked like. The music is cut at barlines
   into rows of even width, and blank paper above and below it is trimmed.

## Limitations

- Videos that scroll smoothly instead of jumping aren't supported yet.
- Detection expects dark notation on light paper. Dark-themed tab won't be found.
- A video that shows standard notation above the tab produces taller rows, so its PDF runs longer.

## Keeping downloads working

YouTube changes often enough to break [yt-dlp](https://github.com/yt-dlp/yt-dlp) a few times a year.
If downloads start failing, update it:

```sh
uv tool upgrade youtabber
```

## Use

Tabs belong to the people who transcribe them. youtabber is for practicing with videos you already
follow along with. Please don't redistribute the PDFs, and support the channels whose work you use.
