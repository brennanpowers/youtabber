from types import SimpleNamespace

import cv2
import numpy as np

from youtabber import ocr
from youtabber.parse import end_repeats, parse_view


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
    st = SimpleNamespace(top=float(top), bottom=top + 5 * gap, gap=gap)
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
    st = SimpleNamespace(top=40.0, bottom=175.0, gap=27.0)
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
    st = SimpleNamespace(top=40.0, bottom=40 + 4 * 14.7, gap=14.7)
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
