import cv2
import numpy as np

from youtabber import ocr


def drawn(text: str) -> np.ndarray:
    img = np.full((40, 30 * len(text) + 20), 255, np.uint8)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    return img


def test_reads_a_measure_number():
    assert ocr.read_number(drawn("17")) == 17
    assert ocr.read_number(drawn("128")) == 128


def test_rejects_text_that_is_not_a_number():
    assert ocr.read_number(drawn("Am")) is None


def test_reads_without_the_network(monkeypatch):
    import socket

    def refuse(*args, **kwargs):
        raise OSError("no network in this test")
    monkeypatch.setattr(socket, "socket", refuse)
    ocr._engine = None
    assert ocr.read_number(drawn("17")) == 17


def soft(text: str) -> np.ndarray:
    """A light-gray number 17 px tall with blurred strokes, as small measure numbers come out of a video."""
    big = np.full((80, 45 * len(text) + 20), 245, np.uint8)
    cv2.putText(big, text, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 110, 2, cv2.LINE_AA)
    big = cv2.GaussianBlur(big, (0, 0), 5)
    return cv2.resize(big, None, fx=17 / 80, fy=17 / 80, interpolation=cv2.INTER_AREA)


def test_reads_soft_light_gray_numbers():
    assert [ocr.read_number(soft(str(n))) for n in range(1, 30)] == list(range(1, 30))
