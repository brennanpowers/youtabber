"""Read short digit strings (measure numbers) and navigation words with RapidOCR's ONNX models."""
import re

import cv2
import numpy as np

# Digit crops are scaled so the digits are this many pixels tall before recognition
TARGET_HEIGHT = 64
# Width of the tiles a wide image is read in, in pixels
WORD_TILE = 2000

_engine = None


def engine():
    """The recognizer, loaded on first use. Its models ship inside the rapidocr wheel, so nothing downloads."""
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def prepare(gray: np.ndarray) -> np.ndarray:
    """Stretch the crop's contrast to full black on white, scale the digits to TARGET_HEIGHT, and pad.

    The crop stays gray: thresholding it breaks light-gray digits into pieces the recognizer misreads."""
    g = gray.astype(np.float32)
    paper = float(np.percentile(g, 90))
    ink = float(g.min())
    n = np.clip((g - ink) / max(paper - ink, 1.0) * 255, 0, 255).astype(np.uint8)
    scale = TARGET_HEIGHT / max(n.shape[0], 1)
    big = cv2.resize(n, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    return cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def read_line(gray: np.ndarray) -> tuple[str, float]:
    """(text, score) for a crop holding one short line of digits, dark on light."""
    img = cv2.cvtColor(prepare(gray), cv2.COLOR_GRAY2BGR)
    rec = engine().text_rec
    # The crop is scaled to the model's height at its own width. The engine's own call pads a short crop out to
    # a long text line first, and then misreads lone digits.
    ratio = img.shape[1] / img.shape[0]
    batch = rec.resize_norm_img(img, ratio)[np.newaxis].astype(np.float32)
    text, score = rec.postprocess_op(rec.session(batch)[0], False, wh_ratio_list=[ratio], max_wh_ratio=ratio)[0]
    return text.strip(), float(score)


def read_number(gray: np.ndarray) -> int | None:
    text, score = read_line(gray)
    if score < 0.5 or not re.fullmatch(r"\d{1,3}", text):
        return None
    return int(text)


def find_words(image: np.ndarray) -> list[str]:
    """All text lines found anywhere in a plate (detection plus recognition)."""
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    words = []
    # Wide images (a glide mosaic) are read in overlapping tiles; the detector's memory grows with width
    step = WORD_TILE - WORD_TILE // 8
    for x in range(0, max(bgr.shape[1] - WORD_TILE // 8, 1), step):
        result, _ = engine()(np.ascontiguousarray(bgr[:, x:x + WORD_TILE]))
        words += [r[1] for r in result or [] if float(r[2]) >= 0.5]
    return words


def numbers_in_line(image: np.ndarray) -> list[tuple[float, float, float, int]]:
    """(left, right, middle row, number) of each digit-only text the detector finds in a strip, 1 to 3 digits."""
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image
    # The detector misses text touching the image edge
    bgr = cv2.copyMakeBorder(np.ascontiguousarray(bgr), 20, 20, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    result, _ = engine()(bgr)
    out = []
    for box, text, score in result or []:
        text = text.strip()
        if float(score) >= 0.5 and re.fullmatch(r"\d{1,3}", text):
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            out.append((float(min(xs)), float(max(xs)), (min(ys) + max(ys)) / 2 - 20, int(text)))
    return out
