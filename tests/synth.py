"""Made-up play-along videos: drawn tab over a moving stand-in for the player, encoded with ffmpeg."""
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

VIDEO_W, VIDEO_H = 1280, 720
FPS = 8
WHITE = (255, 255, 255)
# BGR, the pale yellow some channels use for paper
YELLOW = (170, 245, 255)


@dataclass
class Song:
    """What to draw and how the video shows it. Lengths are in video pixels."""
    measures: int = 16
    notes: int = 8  # per measure
    measure_width: int = 290
    strings: int = 6
    line_spacing: int = 18
    notation: bool = True  # a five-line staff above the tab
    paper: tuple[int, int, int] = WHITE
    line_gray: int = 60  # brightness of staff lines and barlines
    tab_height: int = 270
    place: str = "bottom"  # bottom, top, or inset (a box with video around it)
    scroll: bool = False  # jump-scroll with overlap instead of turning pages
    scroll_step: float = 0.6  # share of the width each jump moves, when scrolling
    page_offsets: list[int] = field(default_factory=list)  # vertical shift of each page's staffs
    seconds_per_view: float = 3.0
    cursor: tuple[int, int, int] = (40, 40, 230)
    seed: int = 1

    @property
    def tab_box(self) -> tuple[int, int, int, int]:
        """x, y, w, h of the tab paper in the frame."""
        if self.place == "top":
            return 0, 0, VIDEO_W, self.tab_height
        if self.place == "inset":
            return 120, VIDEO_H - self.tab_height - 60, VIDEO_W - 240, self.tab_height
        return 0, VIDEO_H - self.tab_height, VIDEO_W, self.tab_height

    @property
    def measures_per_page(self) -> int:
        return (self.tab_box[2] - 60) // self.measure_width

    @property
    def tab_top(self) -> int:
        return 140 if self.notation else 70


def draw_measures(song: Song, first: int, count: int, width: int, offset: int = 0, lead: int = 50) -> np.ndarray:
    """A strip of paper holding measures first..first+count-1, starting `lead` pixels in."""
    rng = np.random.default_rng(song.seed * 1000 + first)
    paper = np.full((song.tab_height, width, 3), song.paper, np.uint8)
    line = (song.line_gray,) * 3
    tab_top = song.tab_top + offset
    tab_bottom = tab_top + (song.strings - 1) * song.line_spacing
    staffs = [(tab_top, tab_bottom, song.line_spacing, song.strings)]
    if song.notation:
        staffs.insert(0, (30 + offset, 30 + offset + 4 * 12, 12, 5))
    for top, _, spacing, count_lines in staffs:
        for i in range(count_lines):
            cv2.line(paper, (0, top + i * spacing), (width - 1, top + i * spacing), line, 1)
    for m in range(count):
        x0 = lead + m * song.measure_width
        x1 = x0 + song.measure_width
        for top, bottom, _, _ in staffs:
            cv2.line(paper, (x1, top), (x1, bottom), line, 2)
        # Measure numbers are gray and sit above the top staff, just after the barline
        cv2.putText(paper, str(first + m + 1), (x0 + 4, staffs[0][0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        for k in range(song.notes):
            # Notes land at different places on each page, as rhythms do
            slot = (song.measure_width - 50) // song.notes
            x = x0 + 30 + k * slot + int(rng.integers(0, max(slot - 16, 1)))
            y = tab_top + int(rng.integers(song.strings)) * song.line_spacing
            fret = str(int(rng.integers(0, 13)))
            (tw, th), _ = cv2.getTextSize(fret, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(paper, (x - 2, y - th // 2 - 2), (x + tw + 2, y + th // 2 + 2), song.paper, -1)
            cv2.putText(paper, fret, (x, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if song.notation:
                note_y = staffs[0][0] + int(rng.integers(9)) * 6
                cv2.ellipse(paper, (x + 4, note_y), (6, 4), -20, 0, 360, (0, 0, 0), -1)
                cv2.line(paper, (x + 9, note_y), (x + 9, note_y - 30), (0, 0, 0), 2)
    return paper


def views(song: Song) -> list[np.ndarray]:
    """The tab images the video shows, in order."""
    w = song.tab_box[2]
    if song.scroll:
        sheet = draw_measures(song, 0, song.measures, 50 + song.measures * song.measure_width + 40)
        step = int(w * song.scroll_step)
        return [sheet[:, x:x + w] for x in range(0, sheet.shape[1] - w + step, step) if x < sheet.shape[1] - w // 2]
    per = song.measures_per_page
    pages = []
    for p, first in enumerate(range(0, song.measures, per)):
        offset = song.page_offsets[p % len(song.page_offsets)] if song.page_offsets else 0
        pages.append(draw_measures(song, first, min(per, song.measures - first), w, offset))
    return pages


def write_video(song: Song, path: Path) -> list[np.ndarray]:
    """Encode the song's video to `path` and return the views it shows."""
    shown = views(song)
    x, y, w, h = song.tab_box
    frames_per_view = int(song.seconds_per_view * FPS)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{VIDEO_W}x{VIDEO_H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         str(path)], stdin=subprocess.PIPE)
    n = 0
    for view in shown:
        for i in range(frames_per_view):
            frame = np.full((VIDEO_H, VIDEO_W, 3), 45, np.uint8)
            # A bright blob wandering around stands in for the player, so only the tab stays still
            cx, cy = int(VIDEO_W / 2 + 400 * np.sin(n / 7)), int(VIDEO_H / 2 + 200 * np.cos(n / 5))
            cv2.circle(frame, (cx, cy), 90, (90, 150, 200), -1)
            frame[y:y + h, x:x + w] = view
            # A translucent cursor sweeping across the tab once per view
            cursor_x = x + int(w * i / frames_per_view)
            frame[y:y + h, cursor_x:cursor_x + 8] = (
                frame[y:y + h, cursor_x:cursor_x + 8] * 0.5 + np.array(song.cursor) * 0.5).astype(np.uint8)
            ffmpeg.stdin.write(frame.tobytes())
            n += 1
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the test video")
    return shown
