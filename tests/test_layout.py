import re
import shutil
import subprocess
import zlib

import numpy as np

from youtabber.layout import TAB_LINE_SPACING, compose, rows, write_pdf
from youtabber.printcell import PrintCell, whiten
from youtabber.source import Source


def cell(width: int, gap: float = 10.0, height: int = 120, top: int = 40) -> PrintCell:
    """A blank measure: a left barline and six staff lines `gap` apart from row `top`."""
    img = np.full((height, width), 255, np.uint8)
    img[:, 0] = 0
    for k in range(6):
        img[top + k * int(gap), :] = 0
    return PrintCell(img, gap, top, 0, width)


def test_tab_lines_print_at_the_current_size():
    assert TAB_LINE_SPACING == 6.5


def test_rows_fill_the_width_and_never_split_a_measure():
    # 100 px at 10 px line spacing is 65 pt on paper: 8 fit in 540 pt, the 9th starts a new row
    assert [len(r) for r in rows([cell(100) for _ in range(20)], set())] == [8, 8, 4]


def test_a_break_starts_a_new_row():
    assert [len(r) for r in rows([cell(100) for _ in range(10)], {3})] == [3, 7]


def test_mixed_spacing_prints_at_one_size():
    img, g = compose([cell(100, gap=10.0), cell(200, gap=20.0, height=240, top=80)])
    assert g == 20.0
    assert img.shape[1] == 400


def test_measures_line_up_on_the_tab_staff():
    img, _ = compose([cell(100, top=40), cell(100, top=70, height=150)])
    assert (img[70, :200] == 0).all()


def test_whiten_makes_paper_white():
    gray = np.full((50, 50), 235, np.uint8)
    gray[20:30, 20:30] = 40
    out = whiten(gray)
    assert out[0, 0] == 255 and out[25, 25] < 60


def test_spots_get_a_last_page(tmp_path):
    src = Source(tmp_path / "x.mp4", "x", "Test Song", None, None)
    plain = write_pdf([cell(100) for _ in range(10)], set(), tmp_path / "a.pdf", src)
    checked = write_pdf([cell(100) for _ in range(10)], set(), tmp_path / "b.pdf", src,
                        note="A note", spots=["0:10-0:20  check this"])
    assert checked == plain + 1


def pdf_text(path) -> str:
    """The PDF's text: pdftotext's when it's installed, else the strings in its page drawing commands."""
    if shutil.which("pdftotext"):
        return subprocess.run(["pdftotext", str(path), "-"], capture_output=True, check=True).stdout.decode()
    data = path.read_bytes()
    streams = re.findall(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S)
    drawn = b"".join(zlib.decompress(s) for s in streams if s.startswith(b"x"))
    return b"".join(re.findall(rb"\((.*?)\) Tj", drawn, re.S)).decode("cp1252")


def test_an_en_dash_in_the_title_prints_as_a_dash(tmp_path):
    src = Source(tmp_path / "x.mp4", "x", "Slipknot \u2013 Before I Forget", "Sightline \u2013 Guitar", None)
    out = tmp_path / "a.pdf"
    write_pdf([cell(100) for _ in range(3)], set(), out, src, note="A \u2013 note", spots=["0:10-0:20 \u2013 check"])
    text = pdf_text(out)
    for line in ("Slipknot \u2013 Before I Forget", "Tab by Sightline \u2013 Guitar", "A \u2013 note",
                 "0:10-0:20 \u2013 check"):
        assert line in text


def test_the_pdf_metadata_title_keeps_any_character(tmp_path):
    title = "君が代 – Live"
    out = tmp_path / "a.pdf"
    write_pdf([cell(100) for _ in range(3)], set(), out, Source(tmp_path / "x.mp4", "x", title, None, None))
    # fpdf writes a non-Latin title as UTF-16 hex in the Info dictionary
    (raw,) = re.findall(rb"/Title <([0-9a-fA-F]+)>", out.read_bytes())
    assert bytes.fromhex(raw.decode()).decode("utf-16") == title
