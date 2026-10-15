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
