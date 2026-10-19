import cv2
import numpy as np

from youtabber import printcell
from youtabber.assemble import Obs
from youtabber.observe import panel_rect
from youtabber.parse import Cell, parse_view
from youtabber.printcell import clean, cleaned, drop_widgets, joined_cells, mark_groups, owner, print_cell
from youtabber.scan import Segment


def test_a_mark_belongs_to_the_measure_its_left_edge_is_in():
    assert owner(150, 190, [0, 100, 200], 10.0) == 1


def test_a_number_centered_on_a_barline_belongs_to_the_measure_starting_there():
    assert owner(95, 110, [0, 100, 200], 10.0) == 1


def test_a_long_chord_name_crossing_far_from_its_start_stays_with_its_measure():
    assert owner(150, 260, [0, 100, 200], 10.0) == 1


def test_a_mark_before_the_first_barline_belongs_to_the_first_measure():
    assert owner(-5, 8, [0, 100], 10.0) == 0


def test_both_digits_of_a_number_centered_on_a_barline_print_in_the_measure_starting_there():
    # "12" drawn across the barline at x = 200, its "1" ending before the barline
    marks = np.zeros((30, 400), np.uint8)
    cv2.putText(marks, "12", (188, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 2)
    groups = mark_groups(marks, 0, [0, 200, 400], 20.0)
    assert [k for *_, k in groups] == [1]
    assert groups[0][0].sum() == marks.sum()


def test_a_yellow_box_is_lifted_but_its_ink_stays():
    band = np.full((100, 200, 3), 250, np.uint8)
    band[20:80, 50:150] = (120, 230, 250)      # BGR pale yellow highlight
    band[45:55, 90:100] = (20, 20, 20)         # a digit under it
    seg = Segment(0.0, 1.0, 10.0, "dark", paper=np.array([250.0, 250.0, 250.0]))
    out = clean(band, seg, 10.0)
    assert out[30, 60] == 255                  # the box is paper
    assert out[50, 95] < 60                    # the digit is ink


def test_a_small_red_number_stays():
    band = np.full((100, 200, 3), 250, np.uint8)
    band[10:18, 10:16] = (40, 40, 220)         # a red digit, smaller than two square gaps
    seg = Segment(0.0, 1.0, 10.0, "dark", paper=np.array([250.0, 250.0, 250.0]))
    assert clean(band, seg, 10.0)[14, 12] < 200


def test_a_dark_barline_tinted_by_yellow_paper_stays():
    band = np.full((100, 200, 3), (172, 255, 253), np.uint8)   # BGR yellow paper
    band[10:90, 100:102] = (12, 69, 68)        # a barline that compression tinted yellow
    seg = Segment(0.0, 1.0, 10.0, "dark", paper=np.array([172.0, 255.0, 253.0]))
    assert clean(band, seg, 10.0)[50, 100] < 100


def cell(x0: int, x1: int, kind: str = "full", complete: bool = True, print_x0: int | None = None) -> Cell:
    return Cell(0, 0, 0, x0, x1, kind, np.zeros((1, 1), bool), np.zeros((1, 1), bool), 0, gap=22.0,
                complete=complete, print_x0=x0 if print_x0 is None else print_x0)


def test_a_time_signature_cell_left_out_prints_with_the_measure_after_it():
    sig, first, mid, second = cell(45, 79, print_x0=12), cell(79, 516), cell(516, 550), cell(550, 900)
    parts = joined_cells([sig, first, mid, second], [first, second])
    assert [(x is first or x is second, at, col) for x, at, col in parts] == [(True, 45, 12), (True, 516, 516)]


def test_a_sliver_cut_by_the_panel_edge_stays_out_of_the_next_measure():
    sliver, first = cell(0, 20, "lead", complete=False), cell(20, 400)
    parts = joined_cells([sliver, first], [first])
    assert [(at, col) for _, at, col in parts] == [(0, 0), (20, 20)]


def test_dropped_cells_side_by_side_or_wide_print_on_their_own():
    # Two dropped cells in a row, then one wider than a clef or time signature (110 px is 5 line gaps)
    a, b, first, wide, second = cell(0, 40), cell(40, 80), cell(80, 500), cell(500, 610), cell(610, 1000)
    parts = joined_cells([a, b, first, wide, second], [first, second])
    assert [(at, col) for x, at, col in parts if x is first or x is second] == [(80, 80), (610, 610)]


def test_an_empty_column_range_prints_white():
    band = np.full((40, 100, 3), 30, np.uint8)
    seg = Segment(0.0, 1.0, 10.0, "dark", paper=np.array([250.0, 250.0, 250.0]))
    assert (clean(band, seg, 10.0, (60, 60)) == 255).all()


def test_a_measure_with_an_empty_band_prints_as_blank_paper():
    img = np.full((200, 800, 3), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (20, 60 + i * 14), (780, 60 + i * 14), (60, 60, 60), 1)
    for x in (100, 400, 700):
        cv2.line(img, (x, 60), (x, 130), (0, 0, 0), 2)
    systems = parse_view(0, img, (0, 0, 800, 200), "dark", 140.0, 229.0, 14.0)
    c = next(x for x in systems[0].cells if x.kind == "full")
    c.band = (140, 40)
    o = Obs(0, 0.0, 1.0, systems[0].cells, [], 0, (0, 0), image=img, systems=systems, print_rect=(0, 0, 800, 200))
    seg = Segment(0.0, 1.0, 14.0, "dark", paper=np.array([255.0, 255.0, 255.0]))
    pc = print_cell(o, c, seg)
    assert pc.image.shape[0] >= 70 and (pc.image == 255).all()


def strip_frame(left_strip: bool) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Paper with staffs between x 26 and 274, line gap 14, and a 5 px dark strip just left or right of them."""
    frame = np.full((200, 300, 3), 250, np.uint8)
    if left_strip:
        frame[40:160, 16:21] = 0
    else:
        frame[40:160, 279:284] = 0
    return frame, (26, 50, 274, 150)


def test_the_print_area_steps_over_a_bracket_left_of_the_staffs():
    frame, seed = strip_frame(True)
    paper = np.array([250, 250, 250], np.uint8)
    assert panel_rect(frame, paper, seed, 14.0)[0] == 21
    assert panel_rect(frame, paper, seed, 14.0, for_print=True)[0] == 0


def test_the_print_area_stops_at_a_dark_strip_right_of_the_staffs():
    frame, seed = strip_frame(False)
    paper = np.array([250, 250, 250], np.uint8)
    assert panel_rect(frame, paper, seed, 14.0, for_print=True)[2] == 279


def test_a_closing_barline_past_the_print_area_prints_with_the_last_measure():
    img = np.full((200, 800, 3), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (20, 60 + i * 14), (714, 60 + i * 14), (60, 60, 60), 1)
    for x in (100, 400, 700):
        cv2.line(img, (x, 60), (x, 130), (0, 0, 0), 2)
    img[60:131, 708:715] = 0
    systems = parse_view(0, img, (0, 0, 700, 200), "dark", 140.0, 229.0, 14.0, (0, 0, 700, 200))
    c = systems[0].cells[-1]
    o = Obs(0, 0.0, 1.0, systems[0].cells, [], 0, (0, 0), image=img, systems=systems, print_rect=(0, 0, 700, 200))
    seg = Segment(0.0, 1.0, 14.0, "dark", paper=np.array([255.0, 255.0, 255.0]))
    pc = print_cell(o, c, seg)
    # The thick line's columns, counted from the measure's left barline
    thick = pc.image[pc.top + 5:pc.top + 65, pc.left + 708 - c.x0:pc.left + 715 - c.x0]
    assert thick.shape[1] == 7 and (thick < 60).all()


def test_a_band_is_cleaned_once_for_all_the_measures_printed_from_it():
    image = np.full((100, 400, 3), 250, np.uint8)
    for i in range(6):
        cv2.line(image, (20, 30 + i * 10), (380, 30 + i * 10), (60, 60, 60), 1)
    o = Obs(0, 0.0, 1.0, [], [], 0, (0, 0), image)
    o.image = image
    seg = Segment(0.0, 1.0, 10.0, "dark", paper=np.array([250.0, 250.0, 250.0]))
    ps = parse_view(0, image, (0, 0, 400, 100), "dark", 140.0, 229.0, 10.0)[0]
    first = cleaned(o, ps, (10, 90), seg, 10.0, (0, 400))
    assert cleaned(o, ps, (10, 90), seg, 10.0, (0, 400)) is first
    other = cleaned(o, ps, (20, 90), seg, 10.0, (0, 400))
    assert other is not first and other.shape == (70, 400)


def test_a_long_mosaic_prints_each_measure_from_a_piece_of_its_band(monkeypatch):
    image = np.full((130, 3000, 3), 255, np.uint8)
    for i in range(6):
        cv2.line(image, (0, 30 + i * 14), (2999, 30 + i * 14), (60, 60, 60), 1)
    for x in range(100, 3000, 150):
        cv2.line(image, (x, 30), (x, 100), (0, 0, 0), 2)
        cv2.putText(image, "7", (x + 60, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    systems = parse_view(0, image, (0, 0, 3000, 130), "dark", 140.0, 229.0, 14.0, (0, 0, 3000, 130))
    cells = systems[0].cells
    o = Obs(0, 0.0, 1.0, cells, [], 0, (0, 0), image)
    o.image, o.systems, o.print_rect, o.mosaic = image, systems, (0, 0, 3000, 130), object()
    seg = Segment(0.0, 1.0, 14.0, "dark", paper=np.array([255.0, 255.0, 255.0]))
    whole = [print_cell(o, c, seg) for c in cells[5:8]]
    monkeypatch.setattr(printcell, "PRINT_PIECE", 1000)
    pieces = [print_cell(o, c, seg) for c in cells[5:8]]
    for a, b in zip(whole, pieces):
        assert np.array_equal(a.image, b.image) and (a.left, a.advance, a.top) == (b.left, b.advance, b.top)


def test_a_solid_screen_widget_off_the_staffs_is_left_out_of_the_print():
    band = np.full((120, 300), 255, np.uint8)
    for i in range(6):
        cv2.line(band, (60, 40 + i * 14), (299, 40 + i * 14), 60, 1)
    # A timer box with light text, and below it a play button that touches a bracket line, left of the staff
    cv2.rectangle(band, (5, 2), (50, 32), 30, -1)
    cv2.putText(band, "0:07", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    cv2.rectangle(band, (5, 38), (50, 80), 30, -1)
    cv2.rectangle(band, (18, 50), (37, 68), 160, -1)
    cv2.line(band, (28, 80), (28, 119), 0, 3)
    # A chord of solid note heads on the staff, which stays
    for y in (54, 68, 82):
        cv2.ellipse(band, (150, y), (9, 7), 0, 0, 360, 0, -1)
    out = drop_widgets(band, 40, 110, 60, 300, 14.0)
    assert out[2:81, 5:51].min() == 255
    assert out[50:90, 140:160].min() == 0


def test_a_play_button_over_the_staffs_left_end_is_left_out_of_the_print():
    band = np.full((120, 300), 255, np.uint8)
    for i in range(6):
        cv2.line(band, (56, 40 + i * 14), (299, 40 + i * 14), 60, 1)
    # A system bracket where the staff starts, and a play button with a gray square inside it, drawn over the
    # bracket's top and the start of the staff lines
    cv2.line(band, (56, 70), (56, 119), 0, 3)
    cv2.rectangle(band, (20, 30), (64, 76), 30, -1)
    cv2.rectangle(band, (30, 42), (50, 62), 150, -1)
    # A clef's stroke just right of the button and a chord of solid note heads further on, which stay
    cv2.line(band, (70, 36), (70, 104), 0, 2)
    for y in (54, 68, 82):
        cv2.ellipse(band, (150, y), (9, 7), 0, 0, 360, 0, -1)
    out = drop_widgets(band, 40, 110, 54, 300, 14.0)
    assert out[30:77, 20:65].min() == 255
    assert np.array_equal(out[:, 69:72], band[:, 69:72])
    assert out[50:90, 140:160].min() == 0


def test_a_chord_grid_and_an_outlined_box_off_the_staffs_stay():
    band = np.full((120, 300), 255, np.uint8)
    for i in range(6):
        cv2.line(band, (60, 40 + i * 14), (299, 40 + i * 14), 60, 1)
    # A chord diagram's grid and a section box outlined around its word, above the staff
    for k in range(6):
        cv2.line(band, (70 + k * 8, 2), (70 + k * 8, 34), 0, 1)
    for k in range(5):
        cv2.line(band, (70, 2 + k * 8), (110, 2 + k * 8), 0, 1)
    cv2.rectangle(band, (150, 4), (230, 30), 0, 2)
    cv2.putText(band, "Chorus", (155, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    out = drop_widgets(band, 40, 110, 60, 300, 14.0)
    assert np.array_equal(out[:36], band[:36])
