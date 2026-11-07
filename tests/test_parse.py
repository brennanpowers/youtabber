from types import SimpleNamespace

import cv2
import numpy as np

from youtabber import ocr, parse
from youtabber.parse import end_repeats, pair_far_notation, parse_view
from youtabber.parts import Part
from youtabber.pipeline import part_notation
from youtabber.scan import System
from youtabber.staff import Staff


def plate(light_line: bool) -> np.ndarray:
    """A tab-only staff with black barlines at x 100, 400 and 700, and optionally a light vertical line at 250."""
    img = np.full((200, 800, 3), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (20, 60 + i * 14), (780, 60 + i * 14), (60, 60, 60), 1)
    for x in (100, 400, 700):
        cv2.line(img, (x, 60), (x, 130), (0, 0, 0), 2)
    if light_line:
        # Only across the staff, as a playhead line is; a line running on below it is dropped as a stem anyway
        cv2.line(img, (250, 60), (250, 130), (190, 190, 190), 3)
    return img


def full_cells(img: np.ndarray) -> list[int]:
    # Ink threshold and faint threshold as observe.ink_threshold gives them for white paper and black ink
    systems = parse_view(0, img, (0, 0, 800, 200), "dark", 140.0, 229.0, 14.0)
    return [c.x0 for ps in systems for c in ps.cells if c.kind == "full"]


def test_barlines_make_measures():
    xs = full_cells(plate(light_line=False))
    assert len(xs) == 2 and abs(xs[0] - 100) <= 2 and abs(xs[1] - 400) <= 2


def test_a_light_line_is_not_a_barline():
    assert full_cells(plate(light_line=True)) == full_cells(plate(light_line=False))


def test_a_panel_border_at_the_panel_edge_is_no_staff_line():
    # A 4-line bass staff whose panel ends at a light border one line gap below it, with footage past it
    img = np.full((200, 800, 3), 255, np.uint8)
    for i in range(4):
        cv2.line(img, (20, 60 + i * 14), (780, 60 + i * 14), (60, 60, 60), 1)
    for x in (100, 400, 700):
        cv2.line(img, (x, 60), (x, 102), (0, 0, 0), 2)
    img[116] = 120
    img[117:] = 70
    systems = parse_view(0, img, (0, 0, 800, 117), "dark", 140.0, 229.0, 14.0)
    assert [len(ps.system.tab.lines) for ps in systems] == [4]
    xs = [c.x0 for ps in systems for c in ps.cells if c.kind == "full"]
    assert len(xs) == 2 and abs(xs[0] - 100) <= 2 and abs(xs[1] - 400) <= 2


def test_a_soft_panel_top_one_step_above_the_staff_is_no_seventh_line():
    # Dark footage fades into the paper over eight rows; the panel starts partway down the fade, one line gap above
    # a 6-line staff
    img = np.full((300, 800, 3), 55, np.uint8)
    for r, v in zip(range(18, 26), (177, 196, 210, 223, 234, 241, 245, 248)):
        img[r] = v
    img[26:] = 249
    for i in range(6):
        cv2.line(img, (20, 62 + i * 37), (780, 62 + i * 37), (60, 60, 60), 2)
    for x in (100, 400, 700):
        cv2.line(img, (x, 62), (x, 247), (0, 0, 0), 2)
    systems = parse_view(0, img, (0, 21, 800, 300), "dark", 140.0, 229.0, 37.0)
    assert [len(ps.system.tab.lines) for ps in systems] == [6]
    xs = [c.x0 for ps in systems for c in ps.cells if c.kind == "full"]
    assert len(xs) == 2 and abs(xs[0] - 100) <= 2 and abs(xs[1] - 400) <= 2


def test_staff_lines_cut_by_the_panels_top_and_bottom_rows_are_kept():
    # Lines five rows thick with paper between; the panel's top and bottom rows run through the outer lines
    img = np.full((300, 800, 3), 238, np.uint8)
    for i in range(4):
        img[66 + i * 37:71 + i * 37, 20:780] = 189
    for x in (100, 400, 700):
        cv2.line(img, (x, 68), (x, 179), (0, 0, 0), 2)
    systems = parse_view(0, img, (0, 67, 800, 179), "dark", 140.0, 229.0, 37.0)
    assert [len(ps.system.tab.lines) for ps in systems] == [4]


def staff_from(x_start: int, opening: bool) -> np.ndarray:
    """A tab staff starting at x_start with a fret digit in its first measure, barlines at 100, 400 and 700."""
    img = np.full((200, 800, 3), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (x_start, 60 + i * 14), (780, 60 + i * 14), (60, 60, 60), 1)
    for x in (100, 400, 700):
        cv2.line(img, (x, 60), (x, 130), (0, 0, 0), 2)
    cv2.putText(img, "5", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if opening:
        cv2.line(img, (x_start + 2, 60), (x_start + 2, 130), (0, 0, 0), 2)
    return img


def first_cell(img: np.ndarray):
    return parse_view(0, img, (0, 0, 800, 200), "dark", 140.0, 229.0, 14.0)[0].cells[0]


def test_a_staff_cut_by_the_panel_edge_has_an_incomplete_first_cell():
    c = first_cell(staff_from(0, opening=False))
    assert c.kind == "lead" and not c.complete


def test_a_staff_starting_inside_the_panel_has_a_complete_first_measure():
    c = first_cell(staff_from(40, opening=False))
    assert c.kind == "lead" and c.complete


def test_an_opening_line_at_the_panel_edge_makes_the_first_measure_complete():
    assert first_cell(staff_from(0, opening=True)).complete


def test_band_reaches_chord_names_but_stops_at_a_blank_stretch():
    marks = np.zeros((400, 100), bool)
    marks[150:160, 10:40] = True     # a chord name 2 gaps above the top line at row 180
    marks[95:105, 10:40] = True      # footage 6 gaps above, inside the reach but past a blank stretch
    from youtabber.parse import print_band
    y0, y1 = print_band(marks, top=180.0, bottom=250.0, gap=14.0, lo=0, hi=400)
    assert 140 <= y0 <= 150          # the chord name and a little padding
    assert y1 >= 250


def test_band_keeps_a_gap_of_paper_past_the_outer_lines():
    from youtabber.parse import print_band
    y0, y1 = print_band(np.zeros((400, 100), bool), top=180.0, bottom=250.0, gap=14.0, lo=0, hi=400)
    assert y0 <= 180 - 14 and y1 >= 250 + 14


def test_a_staff_outside_the_print_area_is_no_system():
    # The panel reaches the staff at rows 60 to 130, the print area stops above it
    img = plate(light_line=False)
    assert parse_view(0, img, (0, 0, 800, 200), "dark", 140.0, 229.0, 14.0, (0, 0, 800, 40)) == []
    assert len(parse_view(0, img, (0, 0, 800, 200), "dark", 140.0, 229.0, 14.0, (0, 40, 800, 160))) == 1


def test_a_measure_number_with_a_word_beside_it_is_still_read():
    from youtabber.parse import number_crop
    # Staff top at row 100, gap 14, barline at x 200: a five-letter word ends just left of the barline and a
    # one-digit number starts just right of it, both at the same height
    g = np.full((200, 400), 255, np.uint8)
    for x in (160, 168, 176, 184, 192):
        g[80:90, x:x + 6] = 0
    g[80:90, 207:213] = 0
    crop = number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=200)
    assert crop is not None and crop.shape[1] == 6 + 2 * 2


def test_a_barline_beside_a_number_is_not_taken_for_a_digit():
    from youtabber.parse import number_crop
    # Staff top at row 100, gap 14, a barline at x 200 from the top line down, and one digit just above the line
    # right of it
    g = np.full((200, 400), 255, np.uint8)
    g[100:170, 200:202] = 0
    g[86:98, 204:211] = 0
    crop = number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=200)
    assert crop is not None and crop.shape == (12 + 2 * 2, 7 + 2 * 2) and crop.min() == 0


def test_a_number_whose_last_digit_joins_a_stem_is_not_cropped():
    from youtabber.parse import number_crop
    # A "1" just right of the barline at x 200, and its second digit merged with a note stem rising past the staff
    g = np.full((200, 400), 255, np.uint8)
    g[86:98, 204:206] = 0
    g[86:98, 209:215] = 0
    g[60:98, 214:216] = 0
    assert number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=200) is None


def test_a_number_cut_by_the_panel_top_is_not_cropped():
    from youtabber.parse import number_crop
    # The panel starts at row 88, so only the lower part of a number above the barline at x 200 shows
    g = np.full((200, 400), 255, np.uint8)
    g[88:98, 204:211] = 0
    assert number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=200, panel_top=88) is None
    assert number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=200) is not None


def test_a_digit_touching_the_barline_is_cropped_without_it():
    from youtabber.parse import number_crop
    # A barline at x 200 from the top line at row 100 down, and a digit right of it that runs down into the line and
    # touches the barline there
    g = np.full((200, 400), 255, np.uint8)
    g[100:170, 200:202] = 0
    g[88:101, 202:209] = 0
    crop = number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=200)
    assert crop is not None and crop.shape == (13 + 2 * 2, 7 + 2 * 2)


def test_a_thin_mark_running_down_past_the_top_line_is_no_digit():
    from youtabber.parse import number_crop
    # Gap 20: the crop window ends at row 108, and STUB_GAPS 0.3 allows marks up to 6 columns wide
    def crop(x1: int, y1: int):
        g = np.full((200, 400), 255, np.uint8)
        g[100:170, 200:202] = 0
        g[90:y1, 206:x1] = 0
        return number_crop(g, g < 128, top=100.0, top_gap=20.0, bar_x=200)

    assert crop(212, 140) is None
    assert crop(213, 140) is not None
    # The same 6 columns ending one row above the window's bottom
    assert crop(212, 108) is None
    assert crop(212, 107) is not None


def test_digits_up_to_two_line_gaps_tall_are_cropped():
    from youtabber.parse import number_crop
    # Gap 14: NUMBER_DIGIT_GAPS 2.0 allows digits up to 28 rows tall
    def crop(y0: int):
        g = np.full((200, 400), 255, np.uint8)
        g[y0:98, 204:211] = 0
        return number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=200)

    tall = crop(70)
    assert tall is not None and tall.shape == (28 + 2 * 2, 7 + 2 * 2)
    assert crop(69) is None


def test_a_mark_close_beside_a_number_and_level_with_it_stops_the_crop():
    from youtabber.parse import number_crop
    # A digit 10 rows tall at columns 204-209: MERGED_DIGIT_GAP 0.6 reaches 6 columns past either side
    def crop(rows: tuple[int, int], cols: tuple[int, int]):
        g = np.full((200, 400), 255, np.uint8)
        g[86:96, 204:210] = 0
        g[rows[0]:rows[1], cols[0]:cols[1]] = 0
        return number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=200)

    # A stem too tall for a digit, right then left of the number
    assert crop((60, 96), (216, 218)) is None
    assert crop((60, 96), (217, 219)) is not None
    assert crop((60, 96), (196, 198)) is None
    assert crop((60, 96), (195, 197)) is not None
    # A mark too wide for a digit, ending just above the number or reaching one row into it
    assert crop((70, 86), (211, 240)) is not None
    assert crop((70, 87), (211, 240)) is None
    # A speck of 5 pixels is ignored, one of 6 is not
    assert crop((90, 91), (213, 218)) is not None
    assert crop((90, 91), (213, 219)) is None


def test_a_mark_starting_on_a_numbers_bottom_row_is_level_with_it():
    from youtabber.parse import number_crop
    # A digit on rows 86-95, and a mark too wide for a digit just right of it, starting on row 95 or row 96
    def crop(y0: int):
        g = np.full((200, 400), 255, np.uint8)
        g[86:96, 204:210] = 0
        g[y0:100, 211:240] = 0
        return number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=200)

    assert crop(95) is None
    assert crop(96) is not None


def test_a_digit_touching_a_thick_barline_is_cropped_without_it():
    from youtabber.parse import number_crop
    # A barline 6 columns wide (200-205) found at its middle, and a digit touching its right side down into the line
    g = np.full((200, 400), 255, np.uint8)
    g[100:170, 200:206] = 0
    g[88:101, 206:213] = 0
    crop = number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=203)
    assert crop is not None and crop.shape == (13 + 2 * 2, 7 + 2 * 2)


def signature_staff(x0: int, x1: int, y0: int = 100, y1: int = 157) -> np.ndarray:
    """Dark pixels of a five-line staff, lines at rows 100-156 (gap 14), with a barline at columns 200-201 and a
    solid mark at rows y0..y1, columns x0..x1."""
    dark = np.zeros((200, 400), bool)
    for y in (100, 114, 128, 142, 156):
        dark[y, :] = True
    dark[100:157, 200:202] = True
    dark[y0:y1, x0:x1] = True
    return dark


def test_a_time_signature_is_a_solid_mark_from_the_top_line_to_the_bottom_line_just_past_the_barline():
    from youtabber.parse import time_signature_end
    staff = Staff([100.0, 114.0, 128.0, 142.0, 156.0], 0, 400, 14.0, "dark")
    assert time_signature_end(signature_staff(210, 226), staff, 201) == 226
    # TIME_SIG_START 1.0 lets it start up to 14 columns past the barline's last column
    assert time_signature_end(signature_staff(216, 230), staff, 201) == 230
    assert time_signature_end(signature_staff(217, 231), staff, 201) is None
    # A stem running 4 rows above the top line ends past TIME_SIG_ENDS 0.25 line gaps (3.5 rows); 3 rows don't
    assert time_signature_end(signature_staff(210, 226, y0=96), staff, 201) is None
    assert time_signature_end(signature_staff(210, 226, y0=97), staff, 201) == 226
    # TIME_SIG_WIDTH starts at 0.8 line gaps (11.2 columns)
    assert time_signature_end(signature_staff(210, 222), staff, 201) == 222
    assert time_signature_end(signature_staff(210, 221), staff, 201) is None
    # A note head just under the top line, its stem down to the bottom line, fills too little of its box
    note = signature_staff(224, 226)
    note[100:112, 210:226] = True
    assert time_signature_end(note, staff, 201) is None


def test_a_number_printed_past_a_time_signature_is_cropped_without_the_signature():
    from youtabber.parse import number_crop
    # The barline at x 200, a time signature at columns 210-225 from the top line down, and a digit at columns
    # 230-236 above it, 30 columns past the barline: within 2.5 line gaps of the signature only. Without the
    # signature's place, the top of the signature is taken for the number
    g = np.full((200, 400), 255, np.uint8)
    g[100:170, 200:202] = 0
    g[100:157, 210:226] = 0
    g[86:96, 230:237] = 0
    assert number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=200).shape == (5 + 2 * 2, 16 + 2 * 2)
    crop = number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=200, opening_x=226)
    assert crop is not None and crop.shape == (10 + 2 * 2, 7 + 2 * 2)


def test_a_number_starting_one_row_below_the_panel_top_is_cropped():
    from youtabber.parse import number_crop
    g = np.full((200, 400), 255, np.uint8)
    g[89:98, 204:211] = 0
    assert number_crop(g, g < 128, top=100.0, top_gap=14.0, bar_x=200, panel_top=88) is not None


def test_parse_view_leaves_out_a_number_cut_by_the_panel_top():
    # The number 12 above the barline at 400 starts at row 43
    def crop(panel_top: int):
        cells = [c for ps in parse_view(0, numbered((0, 0, 0)), (0, panel_top, 800, 200), "dark", 140.0, 229.0, 14.0)
                 for c in ps.cells]
        return next(c for c in cells if abs(c.x0 - 400) <= 2).number_crop

    assert crop(42) is not None
    assert crop(43) is None


def closing_staff(after: str) -> np.ndarray:
    """Barlines at 100 and 400 and a closing barline (thin at 700, thick 708-714), then paper or dark footage."""
    img = np.full((200, 800, 3), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (20, 60 + i * 14), (714, 60 + i * 14), (60, 60, 60), 1)
    for x in (100, 400, 700):
        cv2.line(img, (x, 60), (x, 130), (0, 0, 0), 2)
    img[60:131, 708:715] = 0
    if after == "footage":
        img[:, 718:] = 40
    return img


def closing_cells(after: str):
    # The panel stops just before the thin line, as a panel's edge search does at a closing barline
    return parse_view(0, closing_staff(after), (0, 0, 697, 200), "dark", 140.0, 229.0, 14.0)[0]


def test_a_closing_barline_just_past_the_panel_edge_ends_the_last_measure():
    ps = closing_cells("paper")
    last = ps.cells[-1]
    assert last.kind == "full" and last.complete and abs(last.x0 - 400) <= 2 and abs(last.x1 - 700) <= 2
    assert ps.bars[-1][1] >= 714


def test_a_line_past_the_panel_edge_with_footage_after_it_is_no_barline():
    last = closing_cells("footage").cells[-1]
    assert last.kind == "tail" and not last.complete


def edge_staff(lines_past: bool = False, border: bool = False) -> np.ndarray:
    """Barlines at 100 and 400 and a thin closing barline at 708-710 with a faint column each side, six columns of
    paper before the frame's faint last two columns. The staff lines may run on past the barline, and the barline
    may run about a gap past the staff above and below, as a panel border does."""
    img = np.full((200, 720, 3), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (20, 60 + i * 14), (717 if lines_past else 710, 60 + i * 14), (60, 60, 60), 1)
    for x in (100, 400):
        cv2.line(img, (x, 60), (x, 130), (0, 0, 0), 2)
    rows = slice(46, 145) if border else slice(60, 131)
    img[rows, 708:711] = 0
    img[rows, 707] = img[rows, 711] = 200
    img[:, 718:] = 200
    return img


def edge_last_cell(img: np.ndarray):
    # The panel stops at the barline's dark columns, as a panel's edge search does at a closing barline
    return parse_view(0, img, (0, 0, 708, 200), "dark", 140.0, 229.0, 14.0)[0].cells[-1]


def test_a_thin_closing_barline_at_the_frames_edge_ends_the_last_measure():
    last = edge_last_cell(edge_staff())
    assert last.kind == "full" and last.complete and abs(last.x0 - 400) <= 2 and abs(last.x1 - 707) <= 2


def test_a_thin_barline_at_the_frames_edge_with_the_staff_lines_running_on_is_no_closing_barline():
    last = edge_last_cell(edge_staff(lines_past=True))
    assert last.kind == "tail" and not last.complete


def test_a_thin_line_at_the_frames_edge_running_past_the_staff_is_a_panel_border(monkeypatch):
    last = edge_last_cell(edge_staff(border=True))
    assert last.kind == "tail" and not last.complete
    # With the test above and below the staff turned off, the same line closes the staff
    monkeypatch.setattr(parse, "EDGE_OUTSIDE_INK", 1.01)
    last = edge_last_cell(edge_staff(border=True))
    assert last.kind == "full" and last.complete


def test_a_horizontal_edge_above_a_thin_closing_barline_is_no_panel_border():
    img = edge_staff()
    # The bottom edge of footage drawn across the frame's right side, just above the staff
    img[47:50, 600:] = 0
    last = edge_last_cell(img)
    assert last.kind == "full" and last.complete


def stroke_staff() -> np.ndarray:
    """A gray tab staff, lines at y 60 to 160 with gap 20, and a straight barline at x 100."""
    gray = np.full((240, 400), 255, np.uint8)
    gray[60:161:20, 20:380] = 60
    gray[60:161, 100:102] = 0
    return gray


def staff_bars(gray: np.ndarray) -> list[tuple[int, int]]:
    return parse.barlines(gray, 140.0, 229.0, 60.0, 160.0, 20.0, 20, 380, None)


def test_a_stroke_whose_edges_wave_in_step_is_a_brace_not_a_barline(monkeypatch):
    gray = stroke_staff()
    # A rolled-chord brace at x 250: 4 columns wide, stepping a column left and right every few rows
    for y in range(60, 161):
        s = int(round(0.8 * np.sin(2 * np.pi * y / 10)))
        gray[y, 250 + s:254 + s] = 0
    assert [p for p, _ in staff_bars(gray)] == [100]
    monkeypatch.setattr(parse, "BAR_WAVY_SWING", 9.0)
    assert [p for p, _ in staff_bars(gray)] == [100, 251]


def test_a_stroke_leaving_paper_under_the_top_line_is_an_arrow_shaft(monkeypatch):
    gray = stroke_staff()
    # A strum arrow's shaft at x 250, starting 0.45 gaps under the top line
    gray[69:161, 250:252] = 0
    assert [p for p, _ in staff_bars(gray)] == [100]
    monkeypatch.setattr(parse, "BAR_PAPER_RUN", 1.0)
    assert [p for p, _ in staff_bars(gray)] == [100, 250]


def test_a_stroke_with_a_head_over_the_staff_and_a_shaft_past_it_is_a_strum_arrow(monkeypatch):
    gray = stroke_staff()
    # An up arrow at x 250: its head over y 32-51, its shaft on to y 175, 0.75 gaps under the staff
    for y in range(32, 52):
        w = (y - 32) // 4
        gray[y, 250 - w:252 + w] = 0
    gray[52:176, 250:252] = 0
    # A barline at x 300 under the left side of a box drawn from y 30 to 180, with its top edge running right
    gray[30:181, 300:302] = 0
    gray[30:32, 300:360] = 0
    assert [p for p, _ in staff_bars(gray)] == [100, 300]
    monkeypatch.setattr(parse, "ARROW_WIDE", 0.0)
    assert [p for p, _ in staff_bars(gray)] == [100]
    monkeypatch.setattr(parse, "ARROW_WIDE", 0.1)
    monkeypatch.setattr(parse, "ARROW_TAIL", 0.9)
    assert [p for p, _ in staff_bars(gray)] == [100, 250, 300]


def test_a_strum_arrow_needs_its_head_rows_filled_and_little_ink_above_them(monkeypatch):
    gray = stroke_staff()
    # An up arrow at x 200 with rows 40-41 of its head blank, so its head fills 0.9 of the rows from y 36 to 56
    for y in range(32, 52):
        w = (y - 32) // 4
        gray[y, 200 - w:202 + w] = 0
    gray[40:42, 190:212] = 255
    gray[52:176, 200:202] = 0
    # An arrow at x 300 whose stroke runs on up to y 18, inking 0.75 of the rows from y 12 to 36
    for y in range(32, 52):
        w = (y - 32) // 4
        gray[y, 300 - w:302 + w] = 0
    gray[18:176, 300:302] = 0
    assert [p for p, _ in staff_bars(gray)] == [100, 200, 300]
    monkeypatch.setattr(parse, "ARROW_HEAD", 0.85)
    assert [p for p, _ in staff_bars(gray)] == [100, 300]
    monkeypatch.setattr(parse, "ARROW_ABOVE", 0.8)
    assert [p for p, _ in staff_bars(gray)] == [100]


def test_a_stroke_with_ink_off_one_side_above_the_staff_is_no_strum_arrow():
    gray = stroke_staff()
    # A box side over a barline at x 300 from y 30 to 175, with a line running right from it at y 45
    gray[30:176, 300:302] = 0
    gray[45:47, 300:360] = 0
    assert [p for p, _ in staff_bars(gray)] == [100, 300]


def test_a_colored_head_over_a_barline_is_no_strum_arrow():
    gray = stroke_staff()
    # A gray stroke at x 250 from y 52 to 175 under a red head over y 32-51, such as a playhead's marker
    gray[52:176, 250:252] = 0
    plate = np.repeat(gray[:, :, None], 3, axis=2)
    for y in range(32, 52):
        w = (y - 32) // 4
        gray[y, 250 - w:252 + w] = 60
        plate[y, 250 - w:252 + w] = (0, 0, 220)
    assert [p for p, _ in staff_bars(gray)] == [100]
    bars = parse.barlines(gray, 140.0, 229.0, 60.0, 160.0, 20.0, 20, 380, None, plate=plate)
    assert [p for p, _ in bars] == [100, 250]


def test_a_line_over_part_of_the_panel_width_is_a_line_at_its_edge():
    g = np.full((300, 800), 238, np.uint8)
    # A staff line over 40% of the width on the panel's top row, and a line over 10% on its bottom row
    g[50:52, 0:320] = 60
    g[250:252, 0:80] = 60
    assert parse.line_at_edge(g, 50, slice(0, 800))
    assert not parse.line_at_edge(g, 250, slice(0, 800))


def test_a_faint_barline_broken_for_three_rows_at_a_small_line_gap_is_kept(monkeypatch):
    # Lines 8 apart from y 60 to 100; the barline at x 250 is one pixel wide and missing rows 80-82, 0.375 line gaps
    gray = np.full((160, 400), 255, np.uint8)
    gray[60:101:8, 20:380] = 60
    gray[60:101, 100:102] = 0
    gray[60:101, 250] = 120
    gray[80:83, 250] = 255
    bars = parse.barlines(gray, 140.0, 229.0, 60.0, 100.0, 8.0, 20, 380, None)
    assert [p for p, _ in bars] == [100, 250]
    monkeypatch.setattr(parse, "BAR_PAPER_PX", 0)
    assert [p for p, _ in parse.barlines(gray, 140.0, 229.0, 60.0, 100.0, 8.0, 20, 380, None)] == [100]


def cursor_below(rgb: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """stroke_staff with a barline at x 250 and a see-through cursor line of color `rgb` running on below it, as
    the gray plate and the color plate."""
    gray = stroke_staff()
    gray[60:161, 250:252] = 0
    gray[161:200, 250:252] = 196
    plate = np.repeat(gray[:, :, None], 3, axis=2)
    plate[161:200, 250:252] = rgb
    return gray, plate


def test_a_barline_with_a_tinted_cursor_running_on_below_it_is_no_stem():
    gray, plate = cursor_below((230, 180, 180))
    bars = parse.barlines(gray, 140.0, 229.0, 60.0, 160.0, 20.0, 20, 380, None, plate=plate)
    assert [p for p, _ in bars] == [100, 250]
    assert [p for p, _ in staff_bars(gray)] == [100]


def test_a_stroke_running_on_below_the_staff_in_gray_is_a_stem():
    gray, plate = cursor_below((196, 196, 196))
    bars = parse.barlines(gray, 140.0, 229.0, 60.0, 160.0, 20.0, 20, 380, None, plate=plate)
    assert [p for p, _ in bars] == [100]


def faint_line_over_footage(level: int) -> np.ndarray:
    """stroke_staff with a line of gray `level` at x 250-252 and footage of gray 220 seen through the panel below
    the staff."""
    gray = stroke_staff()
    gray[60:161, 250:253] = level
    gray[161:200, 200:300] = 220
    return gray


def test_a_faint_barline_over_footage_below_the_staff_is_no_stem():
    assert [p for p, _ in staff_bars(faint_line_over_footage(183))] == [100, 251]
    # A line darker than the ink threshold keeps the plain test, and the footage under it reads as a stem
    assert [p for p, _ in staff_bars(faint_line_over_footage(0))] == [100]


def test_a_box_edge_beside_a_barline_makes_no_squiggle():
    gray = stroke_staff()
    gray[60:161, 250:252] = 0
    # The sides of a box cursor drawn along the barline, running a gap past the staff above and below
    gray[30:191, 248] = gray[30:191, 253] = 200
    assert [p for p, _ in staff_bars(gray)] == [100, 250]
    # Faint columns that start on the staff are a squiggle's or a stem's ink
    gray[30:60, 248] = gray[30:60, 253] = 255
    assert [p for p, _ in staff_bars(gray)] == [100]


def test_a_barline_partly_behind_a_note_tile_counts_in_a_glide():
    img = plate(light_line=False)
    # A note tile, paper-colored inside, drawn over the middle barline on one string
    cv2.rectangle(img, (392, 98), (408, 104), (255, 255, 255), -1)

    def bars(hidden):
        systems = parse_view(0, img, (0, 0, 800, 200), "dark", 140.0, 229.0, 14.0, hidden_bars=hidden)
        return [c.x0 for ps in systems for c in ps.cells if c.kind == "full"]
    assert len(bars(False)) == 1
    xs = bars(True)
    assert len(xs) == 2 and abs(xs[1] - 400) <= 2


def test_an_arpeggio_squiggle_is_not_a_barline():
    img = plate(light_line=False)
    # A wavy vertical stroke across the staff, as an arpeggio sign is drawn
    pts = np.array([(250 + (3 if (y // 4) % 2 else -3), y) for y in range(56, 136, 4)], np.int32)
    cv2.polylines(img, [pts], False, (0, 0, 0), 3)
    assert full_cells(img) == full_cells(plate(light_line=False))


def test_blurred_barlines_still_make_measures():
    img = cv2.GaussianBlur(plate(light_line=False), (0, 0), 2.0)
    assert full_cells(img) == [97, 397]


def test_a_glide_barline_behind_a_tile_with_ink_beside_it_stays():
    # Gap 30: the columns checked for paper beside a barline lie 4 px out, inside the note tiles of a glide
    img = np.full((260, 800, 3), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (20, 60 + i * 30), (780, 60 + i * 30), (60, 60, 60), 1)
    for x in (100, 400, 700):
        cv2.line(img, (x, 60), (x, 210), (0, 0, 0), 2)
    # A tile hides part of the middle barline, and its edges run down both sides of it
    img[130:148, 395:406] = 255
    for x in (396, 404):
        cv2.line(img, (x, 60), (x, 160), (0, 0, 0), 1)
    systems = parse_view(0, img, (0, 0, 800, 260), "dark", 140.0, 229.0, 30.0, hidden_bars=True)
    assert [c.x0 for ps in systems for c in ps.cells if c.kind == "full"] == [99, 400]


def notation_page(lone: bool) -> np.ndarray:
    """A system of notation (gap 10) over tab (gap 14), and optionally the next system's notation staff alone at
    the bottom, its tab below the panel."""
    img = np.full((400, 800, 3), 255, np.uint8)
    for i in range(5):
        cv2.line(img, (20, 40 + i * 10), (780, 40 + i * 10), (60, 60, 60), 1)
    for i in range(6):
        cv2.line(img, (20, 140 + i * 14), (780, 140 + i * 14), (60, 60, 60), 1)
    for x in (100, 400, 700):
        cv2.line(img, (x, 40), (x, 210), (0, 0, 0), 2)
    if lone:
        for i in range(5):
            cv2.line(img, (20, 320 + i * 10), (780, 320 + i * 10), (60, 60, 60), 1)
        for x in (100, 400, 700):
            cv2.line(img, (x, 320), (x, 360), (0, 0, 0), 2)
    return img


def test_a_notation_staff_without_its_tab_gives_notation_only_cells():
    top, bottom = parse_view(0, notation_page(True), (0, 0, 800, 400), "dark", 140.0, 229.0, 14.0)
    assert top.system.notation is not None and not any(c.notation_only for c in top.cells)
    assert bottom.system.notation is None and len(bottom.cells) == 4 and all(c.notation_only for c in bottom.cells)


def numbered(color: tuple[int, int, int]) -> np.ndarray:
    """The plate of plate(False) with the number 12 printed in the given color just above the barline at 400."""
    img = plate(light_line=False)
    cv2.putText(img, "12", (395, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img


def test_a_measure_number_printed_in_red_is_cropped_for_reading():
    for color in ((0, 0, 0), (0, 0, 230)):
        cells = [c for ps in parse_view(0, numbered(color), (0, 0, 800, 200), "dark", 140.0, 229.0, 14.0)
                 for c in ps.cells]
        crop = next(c for c in cells if abs(c.x0 - 400) <= 2).number_crop
        assert crop is not None and ocr.read_number(crop) == 12


def test_a_colored_cursor_box_is_not_taken_for_a_number():
    img = plate(light_line=False)
    cv2.rectangle(img, (380, 20), (470, 56), (0, 0, 230), -1)
    cells = [c for ps in parse_view(0, img, (0, 0, 800, 200), "dark", 140.0, 229.0, 14.0) for c in ps.cells]
    assert next(c for c in cells if abs(c.x0 - 400) <= 2).number_crop is None


def two_staffs(joined: bool) -> np.ndarray:
    """Two tab staffs at y 60 and 200 with barlines at x 100, 400 and 700, drawn through the space between the
    staffs or on each staff only."""
    img = np.full((360, 800, 3), 255, np.uint8)
    for top in (60, 200):
        for i in range(6):
            cv2.line(img, (20, top + i * 14), (780, top + i * 14), (60, 60, 60), 1)
    for x in (100, 400, 700):
        if joined:
            cv2.line(img, (x, 60), (x, 270), (0, 0, 0), 2)
        else:
            cv2.line(img, (x, 60), (x, 130), (0, 0, 0), 2)
            cv2.line(img, (x, 200), (x, 270), (0, 0, 0), 2)
    return img


def two_staff_systems(joined: bool):
    return parse_view(0, two_staffs(joined), (0, 0, 800, 360), "dark", 140.0, 229.0, 14.0)


def test_barlines_running_through_two_staffs_make_one_system():
    systems = two_staff_systems(joined=True)
    assert len(systems) == 1
    full = [c for c in systems[0].cells if c.kind == "full"]
    assert [c.x0 for c in full] == [99, 399]
    assert all(c.ink.shape[0] == 294 for c in full)
    assert systems[0].band == (46, 285)


def test_barlines_on_each_staff_only_keep_two_systems():
    systems = two_staff_systems(joined=False)
    assert len(systems) == 2
    assert [[c.x0 for c in ps.cells if c.kind == "full"] for ps in systems] == [[99, 399], [99, 399]]
    assert [ps.band for ps in systems] == [(46, 145), (186, 285)]


def dotted(gap: float, w: int, h: int, stacked: bool, bar: int = 200, before: int | None = None,
           top: int = 40) -> list[int]:
    """end_repeats on a 6-line staff (top line at row `top`) with a barline at x `bar` and two w x h dots left of
    it: one above the other two line gaps apart, or side by side at one height, as two fret digits stand. With
    `before`, a one-pixel barline at that x comes before it."""
    dark = np.zeros((200, bar + 100), bool)
    for i in range(6):
        dark[int(round(top + i * gap))] = True
    y = int(top + 1.5 * gap - h / 2)
    x = int(bar - 0.4 * gap - w)
    places = [(x, y), (x, int(y + 2 * gap))] if stacked else [(x - int(0.5 * gap), y), (x, y)]
    for px, py in places:
        dark[py:py + h, px:px + w] = True
    st = SimpleNamespace(top=float(top), bottom=top + 5 * gap, gap=gap, lines=[top + i * gap for i in range(6)])
    bars = ([(before, before)] if before is not None else []) + [(bar, bar + 2)]
    return end_repeats(dark, st, bars)


def test_an_end_repeat_sign_is_two_dots_one_above_the_other_at_any_dot_size():
    assert dotted(20.0, 8, 8, stacked=True) == [200]
    assert dotted(15.5, 3, 4, stacked=True) == [200]
    # Two fret digits side by side next to a barline are no sign
    assert dotted(20.0, 8, 8, stacked=False) == []
    assert dotted(23.5, 5, 8, stacked=False) == []
    assert dotted(15.5, 3, 4, stacked=False) == []


def test_dots_just_after_the_barline_before_are_no_end_repeat():
    # Line gap 10, staff rows 0-50, the barline before at x 100. Dots before a barline 2.5 gaps on are taken for the
    # digits of a time signature after a system's opening barline; more than 3 gaps on, they end a measure
    assert dotted(10.0, 4, 4, stacked=True, bar=300, before=100, top=0) == [300]
    assert dotted(10.0, 4, 4, stacked=True, bar=125, before=100, top=0) == []
    assert dotted(10.0, 4, 4, stacked=True, bar=131, before=100, top=0) == [131]


def test_end_repeat_dots_are_round():
    # The smallest real dots seen, 3 px wide and 5 tall at line gap 15.5, still count
    assert dotted(15.5, 3, 5, stacked=True) == [200]
    # Two fret digits stacked in a chord, 7 px wide and 13 tall at line gap 22.9, are no sign
    assert dotted(22.9, 7, 13, stacked=True) == []


def test_an_end_repeat_sign_may_have_both_dots_in_one_line_space():
    # 6-line staff with line gap 27 and lines at rows 40 to 175; a barline at x 200. Two 10 px wide dots, 9 and 10 rows
    # tall, sit in the space between rows 94 and 121, the upper touching the line above and the lower the line below
    dark = np.zeros((220, 300), bool)
    for i in range(6):
        dark[40 + i * 27] = True
    dark[94:103, 186:196] = True
    dark[112:122, 186:196] = True
    st = SimpleNamespace(top=40.0, bottom=175.0, gap=27.0, lines=[40.0 + i * 27 for i in range(6)])
    assert end_repeats(dark, st, [(200, 222)]) == [200]
    # The same pair one space lower is off the staff's middle, as pieces of a time signature's digit can be
    dark[95:121, 186:196] = False
    dark[121:130, 186:196] = True
    dark[139:149, 186:196] = True
    assert end_repeats(dark, st, [(200, 222)]) == []


def test_a_fret_digit_split_by_the_middle_line_is_no_end_repeat():
    # 5-line staff with line gap 14.7 and its middle line at row 69; a barline at x 200. A 4 px wide digit about 20 px
    # left of it is cut by the middle line into an 8-row piece above (rows 60-67) and a 3-row piece on it (rows 69-71)
    dark = np.zeros((150, 300), bool)
    for i in range(5):
        dark[int(round(40 + i * 14.7))] = True
    dark[60:68, 181:185] = True
    dark[69:72, 181:185] = True
    st = SimpleNamespace(top=40.0, bottom=40 + 4 * 14.7, gap=14.7, lines=[40 + i * 14.7 for i in range(5)])
    assert end_repeats(dark, st, [(200, 202)]) == []


def guitar_over_cut_staff(below: int, lower: int = 5) -> np.ndarray:
    """A 6-line staff at y 60 over one of `lower` lines whose bottom line is at y 256, line gap 14, barlines at x 100,
    400 and 700 on each staff, with `below` rows of plate under the lower staff's bottom line."""
    img = np.full((257 + below, 800, 3), 255, np.uint8)
    for top, n in ((60, 6), (256 - (lower - 1) * 14, lower)):
        for i in range(n):
            cv2.line(img, (20, top + i * 14), (780, top + i * 14), (60, 60, 60), 1)
        for x in (100, 400, 700):
            cv2.line(img, (x, top), (x, top + (n - 1) * 14), (0, 0, 0), 2)
    return img


def completeness(img: np.ndarray, panel_bottom: int) -> list[list[bool]]:
    systems = parse_view(0, img, (0, 0, 800, panel_bottom), "dark", 140.0, 229.0, 14.0)
    return [[c.complete for c in ps.cells if c.kind == "full"] for ps in systems]


def test_a_staff_of_fewer_lines_at_the_plate_s_bottom_edge_is_cut_by_it():
    img = guitar_over_cut_staff(below=14)
    assert completeness(img, img.shape[0]) == [[True, True], [False, False]]


def test_a_staff_of_fewer_lines_well_above_the_plate_s_bottom_edge_is_whole():
    img = guitar_over_cut_staff(below=42)
    assert completeness(img, img.shape[0]) == [[True, True], [True, True]]


def test_a_panel_ending_just_below_a_staff_of_fewer_lines_does_not_cut_it():
    img = guitar_over_cut_staff(below=42)
    assert completeness(img, 257 + 14) == [[True, True], [True, True]]


def test_a_staff_of_as_many_lines_as_the_others_at_the_plate_s_bottom_edge_is_whole():
    img = guitar_over_cut_staff(below=14, lower=6)
    assert completeness(img, img.shape[0]) == [[True, True], [True, True]]


def test_numbers_are_printed_when_neighboring_measures_of_one_staff_read_one_apart(monkeypatch):
    def system(*reads):
        return SimpleNamespace(cells=[SimpleNamespace(number_crop=None if r is None else np.full((1, 1), r))
                                      for r in reads])
    pages = {"apart": [system(7, None, 9)], "next": [system(None, 7, 8)], "two staffs": [system(3), system(4)]}
    monkeypatch.setattr(ocr, "read_number", lambda crop: int(crop[0, 0]))
    # Two staffs of one measure each have no neighboring measures to judge
    for name, want in (("apart", False), ("next", True), ("two staffs", None)):
        monkeypatch.setattr(parse, "parse_view", lambda *a, systems=pages[name]: systems)
        frame = np.full((10, 10, 3), 255, np.uint8)
        assert parse.numbers_printed(frame, (0, 0, 10, 10), 120.0, 230.0, "dark", 18.0) is want


def far_pair(up_gap: float = 14.0, apart_gaps: float = 11.3, up_lines: int = 5, tab_lines: int = 6):
    """An unpaired staff with lines `up_gap` apart, `apart_gaps` tab gaps above a tab staff spaced 21."""
    tab = Staff([400.0 + 21 * i for i in range(tab_lines)], 0, 1800, 21.0, "dark")
    bottom = 400.0 - apart_gaps * 21
    up = Staff([bottom - up_gap * (up_lines - 1 - i) for i in range(up_lines)], 0, 1800, up_gap, "dark")
    return [System(up, None), System(tab, None)]


def test_an_unpaired_staff_spaced_much_closer_far_above_tab_is_its_notation():
    (s,) = pair_far_notation(far_pair())
    assert s.tab.gap == 21.0 and s.notation.gap == 14.0


def test_a_far_unpaired_staff_stays_apart_when_spaced_near_alike_too_far_or_with_more_lines():
    # 15.75 is 0.75 of 21; 12 gaps is NOTATION_FAR_GAPS
    for syss in (far_pair(up_gap=15.75), far_pair(apart_gaps=12.0), far_pair(up_lines=7)):
        assert [s.notation for s in pair_far_notation(syss)] == [None, None]
    (s,) = pair_far_notation(far_pair(up_gap=15.7, apart_gaps=11.9, up_lines=6))
    assert s.notation is not None


def test_five_line_notation_far_above_four_line_bass_tab_is_its_notation():
    (s,) = pair_far_notation(far_pair(tab_lines=4))
    assert len(s.notation.lines) == 5 and len(s.tab.lines) == 4
    assert [s.notation for s in pair_far_notation(far_pair(up_lines=6, tab_lines=4))] == [None, None]


def test_close_lines_above_far_notation_leave_it_with_its_tab():
    up, tab = far_pair()
    # Lines spaced 8 sit 10 notation gaps (140 px) above the notation staff, whose top is at 106.7: past the 9 gaps
    # the scan pairs staffs within, but inside the 12 a staff spaced much closer may sit
    stray = Staff([-33.3 - 8 * (4 - i) for i in range(5)], 0, 1800, 8.0, "dark")
    first, second = pair_far_notation([System(stray, None), up, tab])
    assert first.tab is stray and first.notation is None
    assert second.tab is tab.tab and second.notation is up.tab


def test_a_staff_close_above_tab_with_a_line_lost_is_its_notation():
    # 17.9 is 0.852 of 21: four notation lines left, 4.4 tab gaps above six-line tab, with notation spaced 17.4 seen
    # on other screens
    (s,) = pair_far_notation(far_pair(up_gap=17.9, apart_gaps=4.4, up_lines=4), [17.4])
    assert s.tab.gap == 21.0 and s.notation.gap == 17.9
    # Its lines found over only the last 1400 of the tab's 1800 columns
    up, tab = far_pair(up_gap=17.9, apart_gaps=4.4, up_lines=4)
    up.tab = Staff(up.tab.lines, 400, 1800, 17.9, "dark")
    (s,) = pair_far_notation([up, tab], [17.4])
    assert s.notation is up.tab
    # 17.9 is 0.16 wider than notation spaced 15.4, past NEAR_MATCH
    assert [s.notation for s in pair_far_notation(far_pair(up_gap=17.9, apart_gaps=4.4, up_lines=4), [15.4])] == [
        None, None]


def test_a_lone_four_line_staff_close_above_tab_stays_apart_unless_notation_elsewhere_is_spaced_like_it():
    # 17.85 is 0.85 of 21: a 4-line bass tab over guitar tab, with no notation anywhere, is spaced like this
    assert [s.notation for s in pair_far_notation(far_pair(up_gap=17.85, apart_gaps=4.4, up_lines=4))] == [None, None]
    # Five-line notation spaced 17.85 over another tab staff further down the screen
    notation = Staff([1000.0 + 17.85 * i for i in range(5)], 0, 1800, 17.85, "dark")
    lower = System(Staff([1200.0 + 21 * i for i in range(6)], 0, 1800, 21.0, "dark"), notation)
    first, second = pair_far_notation(far_pair(up_gap=17.85, apart_gaps=4.4, up_lines=4) + [lower])
    assert first.notation.gap == 17.85 and len(first.notation.lines) == 4 and second is lower


def test_a_four_line_bass_staff_spaced_a_little_wider_than_notation_elsewhere_stays_apart():
    # 18.9 is 0.90 of 21 and 0.084 wider than notation spaced 17.43 (0.83 of 21), past NEAR_MATCH
    assert [s.notation for s in pair_far_notation(far_pair(up_gap=18.9, apart_gaps=4.4, up_lines=4), [17.43])] == [
        None, None]


def test_a_close_staff_stays_apart_when_spaced_near_alike_too_far_or_with_as_many_lines():
    # 19.33 is just over 0.92 of 21 (NEAR_SHARE); 9 gaps is NOTATION_GAPS
    for syss in (far_pair(up_gap=19.33, apart_gaps=4.4, up_lines=4), far_pair(up_gap=17.9, apart_gaps=9.0, up_lines=4),
                 far_pair(up_gap=17.9, apart_gaps=4.4, up_lines=6)):
        assert [s.notation for s in pair_far_notation(syss)] == [None, None]
    (s,) = pair_far_notation(far_pair(up_gap=19.3, apart_gaps=8.9, up_lines=5))
    assert s.notation is not None


def test_a_staff_already_paired_takes_no_far_notation():
    up, tab = far_pair()
    tab.notation = Staff([300.0 + 14 * i for i in range(5)], 0, 1800, 14.0, "dark")
    assert pair_far_notation([up, tab]) == [up, tab]


def test_notation_seen_in_one_part_is_not_counted_for_another_part():
    seg = SimpleNamespace(t0=0.0, t1=10.0)
    notation = Staff([100.0 + 17 * i for i in range(5)], 0, 1800, 17.0, "dark")
    upper = System(Staff([200.0 + 21 * i for i in range(6)], 0, 1800, 21.0, "dark"), notation)
    lower = System(Staff([600.0 + 21 * i for i in range(6)], 0, 1800, 21.0, "dark"), None)
    found = {1.0: [upper, lower]}
    top = Part(0, 0, (100, 305), "band", {1.0: [upper]})
    bottom = Part(1, 0, (600, 705), "band", {1.0: [lower]})
    assert part_notation(found, seg, top) == [17.0] and part_notation(found, seg, bottom) == []
    # A page part takes the segment's staffs within its rows
    assert part_notation(found, seg, Part(0, 0, (0, 1000), "page", found)) == [17.0]
    assert part_notation(found, seg, Part(0, 0, (500, 1000), "page", found)) == []


def system_page(top: int, bottom: int) -> tuple[np.ndarray, Staff]:
    """Gray paper (240) with a 5-line notation staff at rows 20-60 over a 4-line tab at rows 120-150, line gap 10,
    lines at 100, and a 1-pixel line at x 200 of level 180 from row `top` to row `bottom`."""
    g = np.full((200, 400), 240, np.uint8)
    for y in (20, 30, 40, 50, 60, 120, 130, 140, 150):
        g[y, 10:390] = 100
    g[top:bottom + 1, 200] = 180
    return g, Staff([120.0, 130.0, 140.0, 150.0], 10, 390, 10.0, "dark")


def test_a_thin_light_line_through_the_whole_system_is_a_barline():
    g, st = system_page(20, 150)
    # The notation staff's second and next-to-last lines
    assert parse.system_bar(g, st, (30.0, 50.0), (200, 200))


def test_a_thin_light_line_running_on_above_and_below_the_system_is_no_barline():
    # A cursor line from the top of the plate to below the tab
    g, st = system_page(0, 190)
    assert not parse.system_bar(g, st, (30.0, 50.0), (200, 200))
    # A stroke on the tab alone, with paper between the staffs
    g, st = system_page(120, 150)
    assert not parse.system_bar(g, st, (30.0, 50.0), (200, 200))


def edge_page(edge_level: int, edge: tuple[int, int]) -> tuple[np.ndarray, Staff, list[tuple[int, int]]]:
    """Gray paper (240) with a 4-line tab at rows 40-70, barlines of level 40 at x 200 and 400, and a line of
    `edge_level` over columns `edge`."""
    g = np.full((120, 500), 240, np.uint8)
    for y in (40, 50, 60, 70):
        g[y] = 100
    for x in (200, 400):
        g[40:71, x:x + 2] = 40
    g[40:71, edge[0]:edge[1] + 1] = edge_level
    return g, Staff([40.0, 50.0, 60.0, 70.0], 0, 500, 10.0, "dark"), [edge, (200, 201), (400, 401)]


def test_a_light_wide_line_at_the_plates_left_edge_is_the_papers_edge():
    # (240 - 160) / (240 - 40) = 0.4 of the other barlines' contrast
    assert parse.paper_edge_line(*edge_page(160, (1, 4)))


def test_a_dark_or_one_pixel_line_at_the_left_edge_stays_a_barline():
    # As dark as the other barlines
    assert not parse.paper_edge_line(*edge_page(40, (1, 4)))
    # 1 pixel wide
    assert not parse.paper_edge_line(*edge_page(160, (2, 2)))
    # 0.4 of the contrast, but 10 pixels in from the edge
    assert not parse.paper_edge_line(*edge_page(160, (10, 13)))


def four_line_dots(centre: int) -> list[int]:
    """end_repeats on a 4-line staff (lines at rows 40-100, line gap 20, middle at 70) with a barline at x 200 and
    two 6 x 6 dots left of it, 20 rows apart, centred on row `centre`."""
    dark = np.zeros((150, 300), bool)
    for y in (40, 60, 80, 100):
        dark[y] = True
    for y in (centre - 13, centre + 7):
        dark[y:y + 6, 186:192] = True
    st = SimpleNamespace(top=40.0, bottom=100.0, gap=20.0, lines=[40.0, 60.0, 80.0, 100.0])
    return end_repeats(dark, st, [(200, 202)])


def test_end_repeat_dots_on_a_4_line_staff_may_centre_on_its_second_line():
    # Row 60 is 0.5 gaps from the middle, so the middle alone wouldn't take them
    assert four_line_dots(60) == [200]
    assert four_line_dots(70) == [200]


def test_end_repeat_dots_centred_between_lines_off_the_middle_are_no_sign():
    # Row 50 is a whole gap from the middle and half a gap from the nearest line
    assert four_line_dots(50) == []
