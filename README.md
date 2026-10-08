# youtabber

Turns a YouTube play-along video into a printable PDF of the tab.

| Video | PDF |
|---|---|
| ![A play-along video frame with a guitarist above and tab below](docs/example-frame.jpg) | ![The PDF youtabber made from it](docs/example-pdf.png) |

```sh
youtabber "https://www.youtube.com/watch?v=..."
```

It finds the tab on screen, wipes out the moving cursor, and pieces the whole song together, whether
the video shows one line at a time or scrolls forward. Tab at the bottom, at the top, or in a box all
work. (The frame above is made up, not from a real video.)

## Install

You'll need [uv](https://docs.astral.sh/uv/) and [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`).

```sh
git clone https://github.com/brennanpowers/youtabber.git
uv tool install --editable ./youtabber
```

## Usage

```sh
youtabber URL                          # writes out/<song-name>/<song-name>.pdf
youtabber song.mp4                     # a local video works too
youtabber URL --title "Song - Artist"  # when the name pulled from the video title is wrong
youtabber URL --region 370,712,1546,364  # when it grabs the wrong part of the screen
youtabber URL --no-enhance             # skip sharpening, for a smaller file
```

If a PDF looks wrong, open `region.png` in the output folder. It shows the area youtabber read the
tab from, and the run prints that area as `Region(x=…, y=…, w=…, h=…)` so you can adjust it and pass
it back with `--region`.

Videos are cached in `~/.cache/youtabber`, so a second run doesn't download again.

## Config

`~/.config/youtabber/config.toml`:

```toml
pdf_dir = "~/Documents/tabs"   # also copy every PDF here
model = "realesrgan-anime"     # sharper output, see below
enhance = false                # skip sharpening unless --enhance or --model is passed
```

## Sharper output

By default the tab is upscaled and sharpened with ordinary image filters. For cleaner notes, it can
use [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)'s line-art model instead. That needs
PyTorch, which is a big download, so it's an optional extra:

```sh
uv tool install --editable './youtabber[ai]'
youtabber URL --model realesrgan-anime
```

It adds a few seconds to a minute per song. The model downloads on first use. `--model` also takes
a path to any model file [spandrel](https://github.com/chaiNNer-org/spandrel) can load, or `classic`
to skip the model your config names. These models can invent detail, so check a new one against the
video before trusting it.

## How it works

Staff lines stay put for the whole video while everything else moves, so averaging frames finds the
tab. Frames are compared by their dark ink only, which ignores colored cursors, and each stretch of
matching frames is merged into one clean image. When a video scrolls forward with some overlap, each
new image is slid across the last one to find where they line up. The result is cut at barlines into
rows and scaled so the notation prints the same size no matter how big it was on screen.

## Known gaps

- Videos that scroll smoothly, instead of jumping, aren't supported yet.
- Dark-themed tab won't be found, and colored notation, like red X notes, prints as blank.
- A video that shows one page of tab the whole time may need `--region`.
- If downloads start failing, YouTube probably changed something. `uv tool upgrade youtabber` pulls
  the latest [yt-dlp](https://github.com/yt-dlp/yt-dlp).

## Please be nice

Use it with videos you have the right to use: your own, Creative Commons ones, or ones whose creator
is fine with it. The tabs belong to whoever transcribed them, so keep the PDFs for your own practice,
and follow YouTube's terms. Support the channels you learn from.

## License

[MIT](LICENSE)
