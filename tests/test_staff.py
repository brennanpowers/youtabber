import cv2
import numpy as np

from youtabber.scan import systems_in
from youtabber.staff import find_staffs, split_rows


def staff_image(dark: bool = True, notation: bool = False) -> np.ndarray:
    """A white page with a six-line tab staff (spacing 14 px), optionally a five-line staff (10 px) above it."""
    img = np.full((360, 800), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (50, 200 + i * 14), (750, 200 + i * 14), 60, 1)
    if notation:
        for i in range(5):
            cv2.line(img, (50, 80 + i * 10), (750, 80 + i * 10), 60, 1)
    return img if dark else 255 - img


def test_finds_six_evenly_spaced_lines():
    (st,) = find_staffs(staff_image())
    assert len(st.lines) == 6
    assert abs(st.gap - 14) < 0.6
    assert st.polarity == "dark"


def test_finds_light_lines_on_dark_paper():
    (st,) = find_staffs(staff_image(dark=False))
    assert len(st.lines) == 6
    assert st.polarity == "light"


def test_pairs_notation_over_tab():
    (system,) = systems_in(staff_image(notation=True))
    assert len(system.tab.lines) == 6
    assert system.notation is not None and len(system.notation.lines) == 5


def test_a_close_line_group_above_the_notation_does_not_steal_it():
    img = staff_image(notation=True)
    for i in range(4):
        cv2.line(img, (300, 30 + i * 7), (700, 30 + i * 7), 60, 1)
    systems = systems_in(img)
    tab = [s for s in systems if len(s.tab.lines) == 6]
    assert len(tab) == 1 and tab[0].notation is not None and len(tab[0].notation.lines) == 5


def test_wide_lines_below_a_tab_staff_do_not_take_it_as_notation():
    img = staff_image()
    for i in range(4):
        cv2.line(img, (50, 280 + i * 25), (750, 280 + i * 25), 60, 1)
    tab = [s for s in systems_in(img) if len(s.tab.lines) == 6]
    assert len(tab) == 1 and tab[0].notation is None


def lined(rows: list[tuple[int, int, int]], height: int = 360) -> np.ndarray:
    """A white page with horizontal lines (y, x0, x1), each 1 px thick."""
    img = np.full((height, 800), 255, np.uint8)
    for y, x0, x1 in rows:
        cv2.line(img, (x0, y), (x1, y), 60, 1)
    return img


def test_a_line_half_hidden_by_stacked_digits_stays_in_the_staff():
    rows = [(200 + i * 14, 50, 750) for i in range(6)]
    # The third line shows only between stacks of digits: well under half the others' coverage
    rows[2] = (228, 50, 300)
    (st,) = find_staffs(lined(rows))
    assert len(st.lines) == 6


def test_ledger_rows_that_carry_on_the_spacing_do_not_hide_the_staff():
    rows = [(100 + i * 13, 50, 750) for i in range(5)]
    # Four evenly spaced shorter rows below it, as dense ledger lines and beams make
    rows += [(165 + i * 13, 150, 600) for i in range(4)]
    (st,) = find_staffs(lined(rows))
    assert [round(y) for y in st.lines] == [100, 113, 126, 139, 152]


def test_a_staff_with_lines_thickened_by_notes_is_found():
    img = lined([(100 + i * 10, 50, 750) for i in range(5)])
    for i in (0, 2, 3, 4):
        # Ties drawn just above and below these lines all along them make the lines' rows 5 px tall
        for dy in (-2, 2):
            cv2.line(img, (50, 100 + i * 10 + dy), (750, 100 + i * 10 + dy), 160, 1)
    staffs = [s for s in find_staffs(img) if s.polarity == "dark"]
    assert len(staffs) == 1 and len(staffs[0].lines) == 5


def test_a_staff_whose_spacing_drifts_a_little_is_found():
    # Line centers 9 to 11 px apart, as notes on the lines pull them
    (st,) = find_staffs(lined([(y, 50, 750) for y in (66, 75, 85, 96, 107)]))
    assert len(st.lines) == 5


def test_notation_far_above_its_tab_pairs_when_only_paper_lies_between():
    rows = [(60 + i * 10, 50, 750) for i in range(5)] + [(254 + i * 16, 50, 750) for i in range(6)]
    img = lined(rows, 400)
    (system,) = systems_in(img)
    assert system.notation is not None and len(system.notation.lines) == 5


def test_notation_far_above_its_tab_does_not_pair_across_a_ruled_row():
    rows = [(60 + i * 10, 50, 750) for i in range(5)] + [(254 + i * 16, 50, 750) for i in range(6)]
    img = lined(rows, 400)
    # A thick bar across the page between the staffs
    cv2.rectangle(img, (50, 170), (750, 174), 0, -1)
    tab = [s for s in systems_in(img) if len(s.tab.lines) == 6]
    assert len(tab) == 1 and tab[0].notation is None


def diagram_strip(step: int = 150) -> np.ndarray:
    """A six-line tab staff (gap 14) under a strip of chord diagrams: grids 50 px wide every `step` px (gap 11)."""
    img = lined([(250 + i * 14, 50, 750) for i in range(6)])
    for x in range(60, 740, step):
        for i in range(6):
            cv2.line(img, (x, 80 + i * 11), (x + 50, 80 + i * 11), 60, 1)
        for k in range(6):
            cv2.line(img, (x + k * 10, 80), (x + k * 10, 135), 60, 1)
    return img


def test_a_strip_of_chord_diagrams_is_left_out_when_whole_lines_are_asked_for():
    # Without the check, the strip passes for the notation staff of the tab below it
    assert systems_in(diagram_strip())[0].notation is not None
    (system,) = systems_in(diagram_strip(), whole_lines=True)
    assert len(system.tab.lines) == 6 and system.notation is None


def test_a_strip_of_chord_diagrams_set_close_together_is_left_out_when_whole_lines_are_asked_for():
    # 15 px between the grids, under two line gaps: the lines run on in pieces one diagram long
    assert systems_in(diagram_strip(65))[0].notation is not None
    (system,) = systems_in(diagram_strip(65), whole_lines=True)
    assert len(system.tab.lines) == 6 and system.notation is None


def test_a_tab_staff_with_two_digit_chords_close_together_stays_a_staff():
    # "12" on every string every 7 line gaps, each digit on a white box that clears the line under it
    img = lined([(200 + i * 14, 50, 750) for i in range(6)], 360)
    for x in range(90, 740, 98):
        for i in range(6):
            cv2.rectangle(img, (x - 1, 195 + i * 14), (x + 15, 205 + i * 14), 255, -1)
            cv2.putText(img, "12", (x, 205 + i * 14), cv2.FONT_HERSHEY_PLAIN, 0.9, 60, 1)
    (system,) = systems_in(img, whole_lines=True)
    assert len(system.tab.lines) == 6


def test_two_lines_run_together_by_notes_are_cut_apart():
    counts = np.zeros(140, int)
    # Two line peaks 11 rows apart with the rows between them filled by notes: one row group 18 rows tall
    counts[100:118] = [300, 700, 900, 700, 300, 200, 200, 200, 200, 200, 200, 300, 700, 900, 700, 300, 200, 200]
    assert split_rows(counts, 100, 118, 12.0) == [(100, 107), (107, 118)]
    # A group no taller than a thick line stays whole
    assert split_rows(counts, 100, 108, 12.0) == [(100, 108)]


def test_a_short_last_system_under_a_full_one_is_found():
    img = np.vstack([staff_image(), np.full((100, 800), 255, np.uint8)])
    for i in range(6):
        cv2.line(img, (50, 300 + i * 14), (200, 300 + i * 14), 60, 1)
    tops = [round(s.tab.top) for s in systems_in(img)]
    assert tops == [200, 300]


def test_a_short_line_group_away_from_the_staffs_left_end_is_not_a_staff():
    img = np.vstack([staff_image(), np.full((100, 800), 255, np.uint8)])
    for i in range(6):
        cv2.line(img, (500, 300 + i * 14), (650, 300 + i * 14), 60, 1)
    assert [round(s.tab.top) for s in systems_in(img)] == [200]


def test_two_systems_of_5_string_tab_are_not_notation_and_tab():
    for apart in (8, 10.5, 11.5):
        # Two 5-line staffs 14 px apart, with fret digits on the lines of each
        top2 = int(60 + 4 * 14 + apart * 14)
        rows = [(60 + i * 14, 50, 750) for i in range(5)] + [(top2 + i * 14, 50, 750) for i in range(5)]
        img = lined(rows, top2 + 120)
        for y0 in (60, top2):
            for k, x in enumerate(range(100, 700, 60)):
                cv2.putText(img, str(k % 10), (x, y0 + (k % 5) * 14 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
        systems = systems_in(img)
        assert len(systems) == 2 and all(s.notation is None for s in systems), apart


def test_tab_drawn_with_dashed_lines_is_kept_when_whole_lines_are_asked_for():
    # Dashes 13 px long every 28 px: each line covers under half the staff's width
    img = np.full((360, 800), 255, np.uint8)
    for i in range(6):
        for x in range(50, 750, 28):
            cv2.line(img, (x, 200 + i * 14), (x + 12, 200 + i * 14), 60, 1)
    (system,) = systems_in(img, whole_lines=True)
    assert len(system.tab.lines) == 6


def test_a_line_thickened_by_notes_on_a_wide_staff_is_not_split():
    # Gap 46: notes along the third line draw strokes 3 px apart over 12 rows, full at both ends and half drawn
    # between: one row group 15 rows tall with two peaks. At this spacing it is one line
    rows = [(60 + i * 46, 50, 750) for i in range(6) if i != 2]
    img = lined(rows, 400)
    y = 60 + 2 * 46 - 6
    for dy in (0, 12):
        cv2.line(img, (50, y + dy), (750, y + dy), 60, 1)
    for dy in (3, 6, 9):
        img[y + dy, 50:750:2] = 60
    (st,) = [s for s in find_staffs(img) if s.polarity == "dark"]
    assert len(st.lines) == 6


def test_outer_lines_showing_only_between_rows_of_digits_stay_in_the_staff():
    rows = [(200 + i * 14, 50, 750) for i in range(4)]
    # The two bottom lines are hidden under a two-digit number every 44 px along the whole staff: a patch of paper
    # with two dark digit bodies on it
    img = lined(rows + [(256, 50, 750), (270, 50, 750)])
    for y in (256, 270):
        for x in range(60, 740, 44):
            cv2.rectangle(img, (x, y - 5), (x + 26, y + 5), 255, -1)
            for d in (2, 14):
                cv2.rectangle(img, (x + d, y - 4), (x + d + 9, y + 4), 0, -1)
    (st,) = find_staffs(img)
    assert len(st.lines) == 6
    # A short row one step below a staff, as a ledger line or beam makes, is still left out
    (st,) = find_staffs(lined(rows + [(256, 300, 450)]))
    assert len(st.lines) == 4
    # So are ledger lines across the staff, each under a notehead: the row shows where the ink is
    img = lined(rows)
    for x in range(60, 740, 44):
        cv2.line(img, (x, 256), (x + 22, 256), 60, 1)
        cv2.ellipse(img, (x + 11, 256), (5, 4), -20, 0, 360, 0, -1)
    (st,) = find_staffs(img)
    assert len(st.lines) == 4


def test_a_band_ending_a_third_of_a_gap_below_its_staff_keeps_its_bottom_line():
    img = lined([(200 + i * 14, 50, 750) for i in range(6)])
    img[275:] = 70   # footage from 5 px below the bottom line
    (st,) = find_staffs(img)
    assert len(st.lines) == 6


def test_a_panel_border_one_step_below_the_staff_is_no_staff_line():
    for count in (6, 4):
        img = lined([(200 + i * 14, 50, 750) for i in range(count)])
        # The panel ends one line gap below the bottom line, at a dark border with footage below it
        border = 200 + count * 14
        img[border] = 20
        img[border + 1:] = 70
        (st,) = find_staffs(img)
        assert len(st.lines) == count, count


def text_tab() -> np.ndarray:
    """Six rows of typed dashes 22 px apart, each dash 11 px long every 14 px, with fret numbers typed among them."""
    img = np.full((360, 800), 255, np.uint8)
    for i in range(6):
        for x in range(50, 750, 14):
            cv2.line(img, (x, 100 + i * 22), (x + 10, 100 + i * 22), 60, 1)
        cv2.putText(img, "3", (200 + 60 * i, 106 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    return img


def test_text_tab_typed_as_dashes_is_no_staff():
    assert find_staffs(text_tab()) == []
    (st,) = find_staffs(text_tab(), keep_dashed=True)
    assert st.dashed and len(st.lines) == 6
