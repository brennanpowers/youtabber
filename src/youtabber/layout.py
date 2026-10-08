import math
from pathlib import Path

import cv2
import numpy as np
from fpdf import FPDF
from PIL import Image

from youtabber import ink
from youtabber.enhance import Upscaler
from youtabber.source import Source
from youtabber.stitch import Placed

# Page measurements in points (1/72 inch), on US Letter
MARGIN = 36
USABLE_WIDTH = 612 - 2 * MARGIN
FOOTER = 14
ROW_GAP = 14
# Spreading leftover space never makes a gap more than this many times ROW_GAP
MAX_GAP_STRETCH = 4
# Distance between tab lines on paper; every song is scaled to match so notation prints the same size
TAB_LINE_SPACING = 6.5
# Measure numbers are drawn in gray; music in black prints darker than this
GRAY_INK = 100
# Lightest mark that still counts as ink when looking for cut-off fragments
FAINT_INK = 235
# Blank strip pixels kept above and below the music
TRIM_PAD = 8
# Largest and smallest title size, in points; long titles shrink to fit the page
TITLE_SIZE = 20
MIN_TITLE_SIZE = 10


def scale_for(strip: np.ndarray, region_width: int) -> float:
    """Points per strip pixel that give every song the same tab line spacing on paper."""
    tab = ink.tab_staff(ink.staff_lines(strip))
    if len(tab) < 2:
        # Without a staff to measure, fit the video's tab area to the page width
        return USABLE_WIDTH / region_width
    return TAB_LINE_SPACING / float(np.median(np.diff(tab)))


def content_rows(strip: np.ndarray) -> tuple[int, int]:
    """First and last strip rows holding any ink, padded, so blank paper above and below is dropped."""
    ink_rows = np.flatnonzero((strip < ink.FAINT).sum(axis=1) > 20)
    if ink_rows.size == 0:
        return 0, strip.shape[0]
    return max(ink_rows[0] - TRIM_PAD, 0), min(ink_rows[-1] + 1 + TRIM_PAD, strip.shape[0])


def barlines(strip: np.ndarray, lines: list[int]) -> list[tuple[int, int]]:
    """(first, last) columns of each barline, found where ink spans the tab staff top to bottom."""
    tab = ink.tab_staff(lines)
    if not tab:
        return []
    y0, y1 = tab[0], tab[-1]
    full = (strip[y0:y1 + 1] < ink.FAINT).mean(axis=0) > 0.95
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
    bars = barlines(strip, ink.staff_lines(strip))
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
                # End the row after the barline and start the next one on it, so both rows show it
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
            title = _latin1(self.source.title)
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


def clear_edge_fragments(strip: np.ndarray, start: int, end: int, view_edges: set[int],
                         top_line: int, bottom_line: int, near: int) -> np.ndarray:
    """The row strip[:, start:end] with marks its edges cut through erased, staves left alone.

    A row edge is either a cut at a barline or the edge of what the video showed (`view_edges`).
    At a barline, a black mark such as a boxed section label is kept by whichever row holds most of
    it. At the video's edge, anything cut is erased, such as the ends of a system bracket. Measure
    numbers sit over or just after barlines and are drawn in gray, so gray marks above the staff
    within `near` pixels of an edge are erased: a cut leaves part of a number, and part of "17"
    reads as "7".
    """
    row = strip[:, start:end].copy()
    w = end - start
    # Look a little past each barline cut to see how much of a cut mark lies outside the row
    left = start if start in view_edges else max(start - near, 0)
    right = end if end in view_edges else min(end + near, strip.shape[1])
    window = strip[:, left:right]
    staves = slice(max(top_line - 1, 0), bottom_line + 4)

    # A loose threshold so faint marks come out as whole shapes, not scattered specks.
    # Blanking the staves keeps a mark that touches a barline from joining it.
    marks = (window < FAINT_INK).astype(np.uint8)
    marks[staves] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(marks, connectivity=8)
    inside = labels[:, start - left:end - left]
    for i in range(1, count):
        x, _, cw, _, total = stats[i]
        x -= start - left
        in_row = inside == i
        cut_left, cut_right = x <= 0 < x + cw, x < w <= x + cw
        if not in_row.any() or not (cut_left or cut_right):
            continue
        at_video_edge = (cut_left and start in view_edges) or (cut_right and end in view_edges)
        if at_video_edge or in_row.sum() < total / 2 or row[in_row].min() > GRAY_INK:
            row[in_row] = 255

    # Gray pixels above the staff, minus the soft edges of black marks such as section labels
    black = cv2.dilate((row <= GRAY_INK).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    gray = (row < FAINT_INK) & ~black
    gray[max(top_line - 1, 0):] = False
    # Join the digits of one measure number so the whole number goes, not just the digit nearest the edge
    joined = cv2.dilate(gray.astype(np.uint8), np.ones((1, near // 3 + 1), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    for i in range(1, count):
        x, _, cw, _, _ = stats[i]
        if x < near or x + cw > w - near:
            row[gray & (labels == i)] = 255
    return row


def write_pdf(strip: np.ndarray, row_ranges: list[tuple[int, int]], scale: float, out: Path, source: Source,
              placed: list[Placed], upscale: Upscaler | None = None) -> None:
    """Lay the rows out on Letter pages at `scale` points per strip pixel, running `upscale` on each row if given."""
    lines = ink.staff_lines(strip)
    y0, y1 = content_rows(strip)
    trimmed = strip[y0:y1]
    strip = whiten(trimmed)

    def staves(start: int, end: int) -> tuple[int, int]:
        """Top and bottom staff lines of one row. Pages of a video can place the staffs differently."""
        found = ink.staff_lines(trimmed[:, start:end]) or [line - y0 for line in lines]
        return (found[0], found[-1]) if found else (0, strip.shape[0])

    # About the width of a two-digit measure number
    tab = ink.tab_staff(lines)
    near = int(2 * np.median(np.diff(tab))) if len(tab) > 1 else 0
    view_edges = {p.x for p in placed if p.new_from == 0} | {strip.shape[1]}
    pdf = _TabPDF(source)
    pdf.set_title(_latin1(source.title))
    row_h = strip.shape[0] * scale
    bottom = pdf.h - MARGIN - FOOTER

    def draw_page(rows_on_page: list[tuple[int, int]], top: float, spread: bool) -> None:
        gap = ROW_GAP
        if spread and len(rows_on_page) > 1:
            # Share the leftover space between rows so full pages end at the same place
            spare = bottom - top - len(rows_on_page) * row_h - (len(rows_on_page) - 1) * ROW_GAP
            gap = min(ROW_GAP + spare / (len(rows_on_page) - 1), ROW_GAP * MAX_GAP_STRETCH)
        y = top
        for start, end in rows_on_page:
            row = clear_edge_fragments(strip, start, end, view_edges, *staves(start, end), near)
            if upscale:
                row = upscale(row)
            pdf.image(Image.fromarray(row), x=MARGIN, y=y, w=(end - start) * scale)
            y += row_h + gap

    pdf.add_page()
    top = y = pdf.get_y()
    page_rows: list[tuple[int, int]] = []
    for r in row_ranges:
        if page_rows and y + row_h > bottom:
            draw_page(page_rows, top, spread=True)
            pdf.add_page()
            top = y = pdf.get_y()
            page_rows = []
        page_rows.append(r)
        y += row_h + ROW_GAP
    draw_page(page_rows, top, spread=False)
    pdf.output(str(out))
