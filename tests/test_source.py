from pathlib import Path

import pytest
import yt_dlp

from youtabber import source

URL = "https://www.youtube.com/watch?v=abcdefghijk"
UPGRADE = "Check the link opens in a browser. If downloads keep failing: uv tool upgrade youtabber"


class FakeYoutubeDL:
    """Stands in for yt_dlp.YoutubeDL: writes a small file and feeds the progress hooks, with no network."""

    made: list["FakeYoutubeDL"] = []
    error: str | None = None  # raise a DownloadError with this text from extract_info
    updates: list[dict] = []  # progress hook payloads sent during the download

    def __init__(self, opts):
        self.opts = opts
        self.downloads = 0
        self.urls: list[str] = []
        FakeYoutubeDL.made.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        self.urls.append(url)
        if self.error:
            raise yt_dlp.utils.DownloadError(self.error)
        return {"id": "abcdefghijk", "title": "Song (Bass Cover)", "uploader": "Chan", "webpage_url": url,
                "duration": 212.5}

    def prepare_filename(self, info):
        return self.opts["outtmpl"].replace("%(id)s", info["id"]).replace("%(ext)s", "mp4")

    def process_ie_result(self, info, download=True):
        self.downloads += 1
        for d in self.updates:
            for hook in self.opts["progress_hooks"]:
                hook(d)
        Path(self.prepare_filename(info)).write_bytes(b"x" * 1234)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    FakeYoutubeDL.made = []
    FakeYoutubeDL.error = None
    FakeYoutubeDL.updates = []
    monkeypatch.setattr(source.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(source, "CACHE", tmp_path)
    return FakeYoutubeDL


def test_missing_path_names_it_without_calling_yt_dlp(fake):
    with pytest.raises(source.SourceError) as e:
        source.fetch("~/nope/song.mp4")
    assert e.value.heading == "No file at ~/nope/song.mp4"
    assert e.value.detail == ("Check the path, or pass a YouTube URL.",)
    assert str(e.value) == "No file at ~/nope/song.mp4"
    assert fake.made == []


def test_schemeless_text_that_is_not_youtube_stays_a_missing_file(fake):
    for text in ("example.com/watch?v=abcdefghijk", "songs/youtube.com/x.mp4", "youtube.com"):
        with pytest.raises(source.SourceError) as e:
            source.fetch(text)
        assert e.value.heading == f"No file at {text}"
    assert fake.made == []


@pytest.mark.parametrize("link", [
    "youtube.com/watch?v=abcdefghijk",
    "www.youtube.com/watch?v=abcdefghijk",
    "m.youtube.com/watch?v=abcdefghijk",
    "music.youtube.com/watch?v=abcdefghijk",
    "youtu.be/abcdefghijk",
])
def test_youtube_link_without_scheme_is_fetched_over_https(fake, link):
    src = source.fetch(link)
    assert fake.made[0].urls == [f"https://{link}"]
    assert src.url == f"https://{link}"


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=zzzzzzzzzzz",
    "https://m.youtube.com/watch?v=zzzzzzzzzzz",
    "https://youtube.com/watch?v=zzzzzzzzzzz",
])
def test_unavailable_youtube_video_is_trimmed_with_upgrade_hint(fake, url):
    fake.error = "ERROR: [youtube] zzzzzzzzzzz: This video is unavailable"
    with pytest.raises(source.SourceError) as e:
        source.fetch(url)
    assert e.value.heading == "Couldn't get that video from YouTube"
    assert e.value.detail == ("yt-dlp says: [youtube] zzzzzzzzzzz: This video is unavailable", UPGRADE)


def test_caused_by_tail_is_cut(fake):
    fake.error = ("ERROR: [youtube] GlGd8_eaPog: Unable to download API page: Unable to connect to proxy "
                  "(caused by ProxyError('Unable to connect to proxy', OSError('Tunnel connection failed')))"
                  "; please report this issue on  https://github.com/yt-dlp/yt-dlp/issues?q= , filling out the "
                  "appropriate issue template. Confirm you are on the latest version using  yt-dlp -U")
    with pytest.raises(source.SourceError) as e:
        source.fetch("https://youtu.be/GlGd8_eaPog")
    assert e.value.detail == (
        "yt-dlp says: [youtube] GlGd8_eaPog: Unable to download API page: Unable to connect to proxy", UPGRADE)


def test_other_site_gets_no_upgrade_hint(fake):
    fake.error = "ERROR: [generic] notavideo: Unable to download webpage: HTTP Error 404: Not Found"
    with pytest.raises(source.SourceError) as e:
        source.fetch("https://example.com/notavideo")
    assert e.value.heading == "Couldn't get that video"
    assert e.value.detail == ("yt-dlp says: [generic] notavideo: Unable to download webpage: HTTP Error 404: Not Found",)


def test_yt_dlp_is_kept_quiet(fake, capsys):
    source.fetch(URL)
    opts = fake.made[0].opts
    assert opts["quiet"] and opts["no_warnings"] and opts["color"] == "never"
    # A playlist's entries are listed, not each looked up
    assert opts["extract_flat"] == "in_playlist"
    capsys.readouterr()
    for level in ("debug", "info", "warning", "error"):
        getattr(opts["logger"], level)("text")
    assert capsys.readouterr() == ("", "")


def test_progress_passes_bytes_through(fake):
    fake.updates = [
        {"status": "downloading", "downloaded_bytes": 100, "total_bytes": 1000, "eta": 9},
        {"status": "downloading", "downloaded_bytes": 300, "total_bytes": None, "total_bytes_estimate": 1100,
         "eta": 7.5},
        {"status": "downloading", "downloaded_bytes": 500, "eta": None},
        {"status": "finished", "downloaded_bytes": 1234, "total_bytes": 1234},
    ]
    calls = []
    src = source.fetch(URL, on_progress=lambda *a: calls.append(a))
    assert calls == [(100, 1000, 9), (300, 1100, 7.5), (500, None, None)]
    assert src.downloaded_bytes == 1234
    assert src.download_s is not None and src.download_s >= 0
    assert (src.id, src.title, src.credit, src.url, src.duration_s) == ("abcdefghijk", "Song", "Chan", URL, 212.5)


def test_float_size_estimate_is_passed_as_whole_bytes(fake):
    fake.updates = [{"status": "downloading", "downloaded_bytes": 300, "total_bytes_estimate": 1100.7, "eta": 7}]
    calls = []
    source.fetch(URL, on_progress=lambda *a: calls.append(a))
    assert calls == [(300, 1100, 7)]
    assert type(calls[0][1]) is int


def test_on_start_comes_once_before_any_progress(fake, tmp_path):
    fake.updates = [{"status": "downloading", "downloaded_bytes": 1, "total_bytes": 2, "eta": 1}] * 2
    events = []
    source.fetch(URL, on_progress=lambda *a: events.append(("progress", a)),
                 on_start=lambda s: events.append(("start", s)))
    assert [kind for kind, _ in events] == ["start", "progress", "progress"]
    started = events[0][1]
    assert (started.path, started.id, started.title, started.credit, started.url, started.duration_s) == (
        tmp_path / "abcdefghijk.mp4", "abcdefghijk", "Song", "Chan", URL, 212.5)
    assert (started.downloaded_bytes, started.download_s) == (None, None)


def test_on_start_is_called_for_a_cached_video(fake, tmp_path):
    (tmp_path / "abcdefghijk.mp4").write_bytes(b"cached")
    started = []
    src = source.fetch(URL, on_start=started.append)
    assert started == [src]


def test_unknown_duration_is_none(fake, monkeypatch):
    info = FakeYoutubeDL.extract_info
    monkeypatch.setattr(FakeYoutubeDL, "extract_info",
                        lambda self, url, download=False: {**info(self, url), "duration": None})
    started = []
    src = source.fetch(URL, on_start=started.append)
    assert started[0].duration_s is None and src.duration_s is None


def test_cached_video_makes_no_progress_calls_or_download_facts(fake, tmp_path):
    (tmp_path / "abcdefghijk.mp4").write_bytes(b"cached")
    fake.updates = [{"status": "downloading", "downloaded_bytes": 1, "total_bytes": 2, "eta": 1}]
    calls = []
    src = source.fetch(URL, on_progress=lambda *a: calls.append(a))
    assert calls == []
    assert fake.made[0].downloads == 0
    assert src.path == tmp_path / "abcdefghijk.mp4"
    assert (src.downloaded_bytes, src.download_s) == (None, None)


def test_local_file_needs_no_yt_dlp(fake, tmp_path):
    video = tmp_path / "song.mp4"
    video.write_bytes(b"v")
    started = []
    src = source.fetch(str(video), on_start=started.append)
    assert (src.path, src.downloaded_bytes, src.duration_s) == (video, None, None)
    assert started == []
    assert fake.made == []


@pytest.mark.parametrize("kind", ["playlist", "multi_video"])
def test_a_playlist_link_is_refused_before_any_download(fake, monkeypatch, kind):
    monkeypatch.setattr(FakeYoutubeDL, "extract_info", lambda self, url, download=False: {
        "_type": kind, "id": "PLxyz", "title": "Mix", "entries": [{"id": "abcdefghijk"}]})
    with pytest.raises(source.SourceError) as e:
        source.fetch("https://www.youtube.com/playlist?list=PLxyz")
    assert (e.value.heading, e.value.detail) == ("That link is a playlist", ("Pass the link of one video.",))
    assert fake.made[0].downloads == 0
