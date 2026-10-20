"""Lay measure images out in rows on Letter pages and write the PDF."""
from pathlib import Path

import cv2
import numpy as np
from fpdf import FPDF
from PIL import Image

from youtabber.enhance import Upscaler
from youtabber.printcell import PrintCell
from youtabber.source import Source

# Page measurements in points (1/72 inch), on US Letter
MARGIN = 36
USABLE_WIDTH = 612 - 2 * MARGIN
FOOTER = 14
ROW_GAP = 14
# Spreading leftover space never makes a gap more than this many times ROW_GAP
MAX_GAP_STRETCH = 4
# Distance between tab lines on paper; every measure is scaled to match so notation prints the same size
TAB_LINE_SPACING = 6.5
# Largest and smallest title size, in points; long titles shrink to fit the page
TITLE_SIZE = 20
MIN_TITLE_SIZE = 10
# Size of the notes on page 1 and the check list, in points
NOTE_SIZE = 9


def rows(cells: list[PrintCell], breaks: set[int], width: float = USABLE_WIDTH) -> list[list[int]]:
    """Indexes of the measures on each row: consecutive measures while they fit `width` points, and a new
    row at every index in `breaks`. A measure too wide for the page gets a row to itself."""
    out: list[list[int]] = []
    cur: list[int] = []
    lead = used = 0.0
    for i, pc in enumerate(cells):
        k = TAB_LINE_SPACING / pc.gap
        adv = pc.advance * k
        right = (pc.image.shape[1] - pc.left - pc.advance) * k
        if cur and (i in breaks or lead + used + adv + right > width):
            out.append(cur)
            cur, used = [], 0.0
        if not cur:
            lead = pc.left * k
        cur.append(i)
        used += adv
    if cur:
        out.append(cur)
    return out


def compose(row: list[PrintCell]) -> tuple[np.ndarray, float]:
    """One image of a row's measures side by side, lined up on the tab staff's top line, at the row's widest
    line spacing. Returns the image and that spacing in pixels."""
    g = max(pc.gap for pc in row)
    parts = []
    for pc in row:
        f = g / pc.gap
        img = pc.image if f == 1 else cv2.resize(pc.image, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        parts.append((img, pc.top * f, pc.left * f, pc.advance * f))
    starts, x = [], 0.0
    for _, _, left, adv in parts:
        starts.append(x - left)
        x += adv
    shift = -min(starts)
    above = max(top for _, top, _, _ in parts)
    below = max(img.shape[0] - top for img, top, _, _ in parts)
    width = max(s + shift + img.shape[1] for s, (img, _, _, _) in zip(starts, parts))
    canvas = np.full((int(np.ceil(above + below)), int(np.ceil(width))), 255, np.uint8)
    for s, (img, top, _, _) in zip(starts, parts):
        y0, x0 = int(round(above - top)), int(round(s + shift))
        region = canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]]
        # Darker wins where neighbors overlap, so marks hanging over a barline join the next measure's ink
        np.minimum(region, img[:region.shape[0], :region.shape[1]], out=region)
    return canvas, g


class _TabPDF(FPDF):
    def __init__(self, source: Source, note: str | None = None):
        super().__init__(orientation="portrait", unit="pt", format="letter")
        self.source = source
        self.note = note
        # The built-in fonts are written with the Windows-1252 code page, which has dashes and curly quotes
        self.core_fonts_encoding = "windows-1252"
        self.set_margins(MARGIN, MARGIN)
        self.set_auto_page_break(False)

    def header(self) -> None:
        if self.page_no() == 1:
            title = _cp1252(self.source.title)
            size = TITLE_SIZE
            self.set_font("Helvetica", "B", size)
            while size > MIN_TITLE_SIZE and self.get_string_width(title) > USABLE_WIDTH:
                size -= 1
                self.set_font("Helvetica", "B", size)
            self.cell(0, 26, title, new_x="LMARGIN", new_y="NEXT")
            if self.source.credit or self.source.url:
                self.set_font("Helvetica", "", 9)
                self.set_text_color(110)
                if self.source.credit:
                    self.cell(self.get_string_width(f"Tab by {self.source.credit}   "), 14,
                              _cp1252(f"Tab by {self.source.credit}"))
                if self.source.url:
                    self.cell(0, 14, self.source.url, link=self.source.url)
                self.ln(14)
                self.set_text_color(0)
            if self.note:
                self.set_font("Helvetica", "", NOTE_SIZE)
                self.set_text_color(110)
                self.multi_cell(0, 11, _cp1252(self.note), new_x="LMARGIN", new_y="NEXT")
                self.set_text_color(0)
        else:
            self.set_font("Helvetica", "", 9)
            self.set_text_color(110)
            self.cell(0, 14, _cp1252(self.source.title), new_x="LMARGIN", new_y="NEXT")
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


def _cp1252(text: str) -> str:
    """The built-in PDF fonts only cover Windows-1252, so swap anything else for "?"."""
    return text.encode("windows-1252", "replace").decode("windows-1252")


def _draw_page(pdf: FPDF, page: list[tuple[np.ndarray, float, float]], top: float, bottom: float,
               spread: bool, upscale: Upscaler | None = None) -> None:
    gap = ROW_GAP
    if spread and len(page) > 1:
        # Share the leftover space between rows so full pages end at the same place
        spare = bottom - top - sum(h for _, _, h in page) - (len(page) - 1) * ROW_GAP
        gap = min(ROW_GAP + spare / (len(page) - 1), ROW_GAP * MAX_GAP_STRETCH)
    y = top
    for img, w, h in page:
        # Rows are upscaled only as they are drawn, so a long song never holds every upscaled row at once
        pdf.image(Image.fromarray(upscale(img) if upscale else img), x=MARGIN, y=y, w=w, h=h)
        y += h + gap


def write_pdf(cells: list[PrintCell], breaks: set[int], out: Path, source: Source,
              upscale: Upscaler | None = None, note: str | None = None, spots: list[str] | None = None) -> int:
    """Write the measures in rows on Letter pages, with `spots` listed on a last page. Returns the page count."""
    pdf = _TabPDF(source, note)
    pdf.set_title(source.title)
    pdf.add_page()
    top = y = pdf.get_y()
    bottom = pdf.h - MARGIN - FOOTER
    placed = []
    for row in rows(cells, breaks):
        img, g = compose([cells[i] for i in row])
        w = img.shape[1] * TAB_LINE_SPACING / g
        h = img.shape[0] * TAB_LINE_SPACING / g
        # A row too wide or too tall for the page shrinks to fit
        fit = min(1.0, USABLE_WIDTH / w, (bottom - top) / h)
        placed.append((img, w * fit, h * fit))
    page: list = []
    for item in placed:
        if page and y + item[2] > bottom:
            _draw_page(pdf, page, top, bottom, spread=True, upscale=upscale)
            pdf.add_page()
            top = y = pdf.get_y()
            page = []
        page.append(item)
        y += item[2] + ROW_GAP
    _draw_page(pdf, page, top, bottom, spread=False, upscale=upscale)
    if spots:
        pdf.add_page()
        pdf.set_auto_page_break(True, MARGIN + FOOTER)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 18, "Check these spots", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", NOTE_SIZE)
        for line in spots:
            pdf.multi_cell(0, 12, _cp1252(line), new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(out))
    return pdf.page_no()
