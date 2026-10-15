import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float


class VideoError(Exception):
    pass


def probe(path: Path) -> VideoInfo:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        info = VideoInfo(stream["width"], stream["height"], float(data["format"]["duration"]))
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        raise VideoError(f"Can't read {path} as a video: {result.stderr.strip() or 'no video stream'}")
    if info.duration <= 0:
        raise VideoError(f"{path} has no length")
    return info


def frames(path: Path, fps: float, scale: float = 1.0,
           crop: tuple[int, int, int, int] | None = None, gray: bool = False,
           start: float = 0.0, length: float | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, BGR frame) pairs sampled at `fps`.

    `crop` is (x, y, w, h) in source pixels and is applied before resizing by `scale`.
    """
    info = probe(path)
    filters = [f"fps={fps}"]
    src_w, src_h = info.width, info.height
    if crop:
        # Even sizes and offsets: ffmpeg rounds odd crops for chroma subsampling, which would put every
        # frame out of step with the bytes read
        x, y, src_w, src_h = (v - v % 2 for v in crop)
        filters.append(f"crop={src_w}:{src_h}:{x}:{y}")
    w = int(src_w * scale)
    h = int(src_h * scale)
    if scale != 1.0:
        filters.append(f"scale={w}:{h}")
    channels = 1 if gray else 3
    span = ["-ss", f"{start:.3f}"] if start else []
    if length is not None:
        span += ["-t", f"{length:.3f}"]
    cmd = ["ffmpeg", "-v", "error", *span, "-i", str(path),
           "-vf", ",".join(filters), "-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24", "-"]
    frame_bytes = w * h * channels
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        assert proc.stdout is not None
        try:
            i = 0
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                shape = (h, w) if gray else (h, w, 3)
                yield start + i / fps, np.frombuffer(buf, np.uint8).reshape(shape)
                i += 1
            # A cut-off file still reports its full length, and ffmpeg stops at the cut without
            # an error, so compare the frames read with the length the file claims
            whole = not start and length is None
            if proc.wait() != 0 or (whole and i < info.duration * fps - max(2, 2 * fps)):
                raise VideoError(f"ffmpeg couldn't read all of {path}. If it's an incomplete download, "
                                 f"delete it and run again.")
        finally:
            # Stop ffmpeg quietly when the caller stops reading early
            proc.kill()
