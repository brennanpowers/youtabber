import dataclasses
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

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
    # Set only when this run downloaded the video, for the "Downloaded N MB in S s" line
    downloaded_bytes: int | None = None
    download_s: float | None = None
    duration_s: float | None = None  # video length in seconds from yt-dlp, None for a local file or when unknown


# done_bytes, total_bytes (None when unknown), eta_s (None when unknown)
Progress = Callable[[int, int | None, float | None], None]

# Hosts a YouTube link can start with when it is typed without "https://"
_SCHEMELESS_HOSTS = ("youtube.com/", "www.youtube.com/", "m.youtube.com/", "music.youtube.com/", "youtu.be/")


class SourceError(Exception):
    """A source that can't be turned into a video, with a one-line heading and lines of detail."""

    def __init__(self, heading: str, *detail: str):
        super().__init__(heading)
        self.heading, self.detail = heading, detail


class _Silent:
    """A yt-dlp logger that drops everything, so only our own messages reach the terminal."""

    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


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


def fetch(source: str, on_progress: Progress | None = None,
          on_start: Callable[[Source], None] | None = None) -> Source:
    """Resolve a local file or a YouTube URL, downloading the video into the cache when needed.

    For a URL, on_start gets called once with the video's title, credit, link and length as soon as
    yt-dlp has looked them up, before any download, and on_progress gets called while a download runs.
    A cached video makes no on_progress calls and leaves downloaded_bytes and download_s as None.
    A local file calls neither.
    """
    path = Path(source).expanduser()
    if path.is_file():
        return Source(path, path.stem, clean_title(path.stem), None, None)
    if "://" not in source:
        if not source.lower().startswith(_SCHEMELESS_HOSTS):
            raise SourceError(f"No file at {source}", "Check the path, or pass a YouTube URL.")
        source = "https://" + source

    def hook(d: dict) -> None:
        if on_progress and d["status"] == "downloading":
            # yt-dlp's estimate can be a float
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            on_progress(d.get("downloaded_bytes") or 0, int(total) if total else None, d.get("eta"))

    opts = {
        "format": "bv*[height<=1080][ext=mp4]/bv*[height<=1080]",
        "outtmpl": str(CACHE / "%(id)s.%(ext)s"),
        "noplaylist": True,
        # List a playlist's entries without looking each one up, so a playlist link is refused right away
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "logger": _Silent(),
        # On a terminal yt-dlp colors the "ERROR:" in its error text, which would end up in our message
        "color": "never",
        "progress_hooks": [hook],
    }
    downloaded_bytes = download_s = None
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=False)
            # A playlist or channel link would download every video in it
            if info.get("_type") in ("playlist", "multi_video"):
                raise SourceError("That link is a playlist", "Pass the link of one video.")
            # The exact name yt-dlp writes, so a leftover partial download never counts as cached
            cached = Path(ydl.prepare_filename(info))
            found = Source(cached, info["id"], clean_title(info["title"]), info.get("uploader"),
                           info.get("webpage_url"), duration_s=info.get("duration"))
            if on_start:
                on_start(found)
            if not cached.exists():
                started = time.monotonic()
                ydl.process_ie_result(info, download=True)
                download_s = time.monotonic() - started
                downloaded_bytes = cached.stat().st_size
    except yt_dlp.utils.DownloadError as e:
        raise _download_error(source, str(e))
    return dataclasses.replace(found, downloaded_bytes=downloaded_bytes, download_s=download_s)


def _download_error(source: str, message: str) -> SourceError:
    # yt-dlp's text starts with "ERROR: " and can end with a long "(caused by ...)" exception dump,
    # sometimes followed by a request to report a bug
    message = re.sub(r"^ERROR:\s*", "", message.strip())
    message = re.sub(r"\s*\(caused by .*", "", message, flags=re.DOTALL)
    host = (urlparse(source).hostname or "").lower()
    if host in ("youtube.com", "youtu.be") or host.endswith(".youtube.com"):
        return SourceError("Couldn't get that video from YouTube", f"yt-dlp says: {message}",
                           "Check the link opens in a browser. If downloads keep failing: uv tool upgrade youtabber")
    return SourceError("Couldn't get that video", f"yt-dlp says: {message}")
