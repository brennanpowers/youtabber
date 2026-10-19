import cv2
import numpy as np

from youtabber import ocr
from youtabber.parse import parse_view


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
