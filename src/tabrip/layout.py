import math
from pathlib import Path

import numpy as np
from fpdf import FPDF
from PIL import Image

from tabrip.source import Source
from tabrip.stitch import Placed

# Page measurements in points (1/72 inch), on US Letter
MARGIN = 36
USABLE_WIDTH = 612 - 2 * MARGIN
FOOTER = 14
ROW_GAP = 14
# Distance between tab lines on paper; every song is scaled to match so notation prints the same size
TAB_LINE_SPACING = 6.5
# Blank strip pixels kept above and below the music
TRIM_PAD = 8
# Darker than paper; loose enough to catch light gray staff lines
INK = 200


def staff_lines(strip: np.ndarray) -> list[int]:
    """Rows of the strip holding staff lines, which run nearly its whole length."""
    rows = np.flatnonzero((strip < INK).mean(axis=1) > 0.5)
    lines: list[int] = []
    for r in rows:
        if not lines or r > lines[-1] + 2:
            lines.append(int(r))
    return lines


def tab_staff(lines: list[int]) -> list[int]:
    """Lines of the bottom staff, which is the tab: the last run of evenly spaced lines."""
    if len(lines) < 2:
        return []
    gaps = np.diff(lines)
    spacing = np.median(gaps)
    top = len(lines) - 1
    while top > 0 and gaps[top - 1] <= spacing * 1.5:
        top -= 1
    return lines[top:]


def scale_for(strip: np.ndarray, region_width: int) -> float:
    """Points per strip pixel that give every song the same tab line spacing on paper."""
    tab = tab_staff(staff_lines(strip))
    if len(tab) < 2:
        # Without a staff to measure, fit the video's tab area to the page width
        return USABLE_WIDTH / region_width
    return TAB_LINE_SPACING / float(np.median(np.diff(tab)))


def content_rows(strip: np.ndarray) -> tuple[int, int]:
    """First and last strip rows holding any ink, padded, so blank paper above and below is dropped."""
    ink_rows = np.flatnonzero((strip < INK).sum(axis=1) > 20)
    if ink_rows.size == 0:
        return 0, strip.shape[0]
    return max(ink_rows[0] - TRIM_PAD, 0), min(ink_rows[-1] + 1 + TRIM_PAD, strip.shape[0])


def barlines(strip: np.ndarray, lines: list[int]) -> list[tuple[int, int]]:
    """(first, last) columns of each barline, found where ink spans the tab staff top to bottom."""
    tab = tab_staff(lines)
    if not tab:
        return []
    y0, y1 = tab[0], tab[-1]
    full = (strip[y0:y1 + 1] < INK).mean(axis=0) > 0.95
    bars: list[tuple[int, int]] = []
    for x in np.flatnonzero(full):
        # Columns close together belong to one thick or double barline
        if bars and x <= bars[-1][1] + 12:
            bars[-1] = (bars[-1][0], int(x))
        else:
            bars.append((int(x), int(x)))
    return bars


def rows(strip: np.ndarray, placed: list[Placed], width: int) -> list[tuple[int, int]]:
    """Split the strip into (start, end) column ranges no wider than `width`, as even in width as barlines allow."""
    bars = barlines(strip, staff_lines(strip))
    # Each page turn starts a new line of music with its own clef, so always cut there
    turns = [p.x for p in placed if p.new_from == 0] + [strip.shape[1]]
    out = []
    for seg_start, seg_end in zip(turns, turns[1:]):
        start = seg_start
        rows_left = math.ceil((seg_end - seg_start) / width)
        while start < seg_end:
            if seg_end - start <= width:
                out.append((start, seg_end))
                break
            # Aim for an equal share of what's left, so rows come out alike instead of full rows and a stub
            rows_left = max(rows_left, math.ceil((seg_end - start) / width))
            target = start + (seg_end - start) / rows_left
            fits = [b for b in bars if start < b[0] and b[1] < start + width]
            if fits:
                # End the row after the barline and start the next one on it, so measure numbers stay whole
                first, last = min(fits, key=lambda b: abs(b[1] - target))
                out.append((start, last + 1))
                start = first
            else:
                out.append((start, start + width))
                start += width
            rows_left -= 1
    return out


def whiten(strip: np.ndarray) -> np.ndarray:
    """Scale brightness so the video's off-white paper prints as white."""
    paper = float(np.median(strip))
    out = np.clip(strip.astype(np.float32) * (255 / paper), 0, 255)
    out[out > 240] = 255
    return out.astype(np.uint8)


class _TabPDF(FPDF):
    def __init__(self, source: Source):
        super().__init__(orientation="portrait", unit="pt", format="letter")
        self.source = source
        self.set_margins(MARGIN, MARGIN)
        self.set_auto_page_break(False)

    def header(self) -> None:
        if self.page_no() == 1:
            self.set_font("Helvetica", "B", 20)
            self.cell(0, 26, _latin1(self.source.title), new_x="LMARGIN", new_y="NEXT")
            if self.source.credit or self.source.url:
                self.set_font("Helvetica", "", 9)
                self.set_text_color(110)
                if self.source.credit:
                    self.cell(self.get_string_width(f"Tab by {self.source.credit}   "), 14,
                              _latin1(f"Tab by {self.source.credit}"))
                if self.source.url:
                    self.cell(0, 14, self.source.url, link=self.source.url)
                self.ln(14)
                self.set_text_color(0)
        else:
            self.set_font("Helvetica", "", 9)
            self.set_text_color(110)
            self.cell(0, 14, _latin1(self.source.title), new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0)
        self.set_draw_color(190)
        self.line(MARGIN, self.get_y() + 4, self.w - MARGIN, self.get_y() + 4)
        self.set_y(self.get_y() + 4 + ROW_GAP)

    def footer(self) -> None:
        self.set_y(-MARGIN + 8)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(110)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")
        self.set_text_color(0)


def _latin1(text: str) -> str:
    """The built-in PDF fonts only cover Latin-1, so swap anything else for "?"."""
    return text.encode("latin-1", "replace").decode("latin-1")


def write_pdf(strip: np.ndarray, row_ranges: list[tuple[int, int]], scale: float, out: Path, source: Source) -> None:
    """Lay the rows out on Letter pages at `scale` points per strip pixel."""
    y0, y1 = content_rows(strip)
    strip = whiten(strip[y0:y1])
    pdf = _TabPDF(source)
    pdf.set_title(_latin1(source.title))
    row_h = strip.shape[0] * scale
    pdf.add_page()
    for start, end in row_ranges:
        if pdf.get_y() + row_h > pdf.h - MARGIN - FOOTER:
            pdf.add_page()
        pdf.image(Image.fromarray(strip[:, start:end]), x=MARGIN, y=pdf.get_y(), w=(end - start) * scale)
        pdf.set_y(pdf.get_y() + row_h + ROW_GAP)
    pdf.output(str(out))
