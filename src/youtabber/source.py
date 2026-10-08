import re
from dataclasses import dataclass
from pathlib import Path

import yt_dlp

CACHE = Path.home() / ".cache" / "youtabber"

# Words that describe the video rather than the song, as in "Bass Guitar Cover (Play Along Tabs)"
_NOISE = {"solo", "bass", "guitar", "cover", "with", "tab", "tabs", "play", "along", "playalong",
          "play-along", "lesson", "tutorial", "how", "to"}
# A run of noise words counts only if it holds one of these, so "Fly Me To The Moon" keeps its "To"
_NOISE_CORE = {"cover", "tab", "tabs", "playalong", "play-along", "lesson", "tutorial"}


@dataclass(frozen=True)
class Source:
    path: Path
    id: str  # YouTube video id, or the file name for a local video
    title: str
    credit: str | None  # channel that made the video, usually the tab's transcriber
    url: str | None


def clean_title(raw: str) -> str:
    """Song name from a video title, dropping words like "Bass Cover (Play Along Tabs)"."""
    # Joined so "play along" counts as noise without making "Along" alone count, as in "Along Came Mary"
    text = re.sub(r"\bplay[\s-]+along\b", "play-along", raw, flags=re.IGNORECASE)
    text = re.sub(r"[(\[]([^)\]]*)[)\]]", lambda m: "" if _is_noise(m.group(1)) else m.group(0), text)
    words = re.findall(r"[^\s|:]+|[|:]", text)
    kept: list[str] = []
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j].lower() in _NOISE:
            j += 1
        if j > i and any(w.lower() in _NOISE_CORE for w in words[i:j]):
            i = j
        else:
            kept.append(words[i])
            i += 1
    # Separators left over from the dropped words become a single " - "
    parts = [p.strip(" -") for p in re.split(r"\s*[|:]\s*|\s+-\s+", " ".join(kept))]
    return " - ".join(p for p in parts if p) or raw


def _is_noise(text: str) -> bool:
    words = text.lower().split()
    return bool(words) and all(w in _NOISE for w in words) and any(w in _NOISE_CORE for w in words)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def fetch(source: str) -> Source:
    """Resolve a local file or a YouTube URL, downloading the video into the cache when needed."""
    path = Path(source).expanduser()
    if path.is_file():
        return Source(path, path.stem, clean_title(path.stem), None, None)
    opts = {
        "format": "bv*[height<=1080][ext=mp4]/bv*[height<=1080]",
        "outtmpl": str(CACHE / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=False)
            # The exact name yt-dlp writes, so a leftover partial download never counts as cached
            cached = Path(ydl.prepare_filename(info))
            if not cached.exists():
                print(f"Downloading {info['title']}")
                ydl.process_ie_result(info, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise SystemExit(f"{source} isn't a file here, and yt-dlp couldn't download it:\n  {e}")
    return Source(cached, info["id"], clean_title(info["title"]), info.get("uploader"), info.get("webpage_url"))
