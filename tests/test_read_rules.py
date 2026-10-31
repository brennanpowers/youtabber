from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from youtabber import glide, pipeline
from youtabber.assemble import Assembly, Obs, Slot
from youtabber.check import empty, screen_breaks
from youtabber.observe import View
from youtabber.parse import Cell, parse_view
from youtabber.pipeline import (Measure, detected_numbers, drawn_full_size, end_card, intro, one_layout,
                                overlap_share, placed_notation, scattered, without_lead_in)
from youtabber.scan import Segment
from youtabber.video import VideoInfo
from test_edge_views import screen
from test_empty_bar import cell


def segment(t0: float, t1: float, gap: float) -> Segment:
    return Segment(t0, t1, gap, "dark")


def test_short_pieces_of_tab_at_different_sizes_are_scattered():
    pieces = [segment(3, 7, 8.0), segment(34, 36.5, 9.5), segment(137.5, 139.5, 4.0), segment(174, 180.5, 6.0)]
    assert scattered(pieces, 260.0)


def test_one_steady_stretch_of_tab_is_not_scattered():
    assert not scattered([segment(0, 9, 14.0), segment(20, 60, 14.0)], 120.0)


def test_short_pieces_all_drawn_alike_are_not_scattered():
    # A lesson that shows the same tab layout for a few seconds now and then
    assert not scattered([segment(10, 16, 14.0), segment(50, 56, 14.2), segment(90, 96, 13.9)], 300.0)


def test_a_short_small_picture_of_the_page_is_left_out():
    song, thumbnail, insert = segment(0, 39, 16.0), segment(52.5, 57.5, 7.5), segment(80, 86, 15.5)
    assert drawn_full_size([thumbnail, song, insert]) == [song, insert]
    # A long stretch drawn small is still read
    assert drawn_full_size([segment(0, 20, 7.5), song]) == [segment(0, 20, 7.5), song]


def still(t0: float, t1: float, playhead: bool = False) -> View:
    plate = np.full((360, 640, 3), 255, np.uint8)
    return View(t0, t1, plate, [(t0 + 0.5, 100.0, 50.0)] if playhead else [], rect=(0, 0, 640, 360),
                print_rect=(0, 0, 640, 360))


def test_a_last_segment_drawn_small_with_no_playhead_is_an_end_card():
    song, card = segment(3.5, 362.0, 30.15), segment(364.0, 382.5, 14.44)
    assert end_card([song, card], 1, [still(364.0, 370.0), still(370.0, 376.0)], True)
    # A playhead seen on any of its views makes it music
    assert not end_card([song, card], 1, [still(364.0, 370.0), still(370.0, 376.0, playhead=True)], True)
    # Spaced at 0.6 of the song's lines or wider, it is tab drawn at another size
    assert not end_card([song, segment(364.0, 382.5, 18.09)], 1, [still(364.0, 370.0)], True)
    # Only the last segment can be one
    assert not end_card([card, song], 0, [still(364.0, 370.0)], True)


def test_a_last_segment_drawn_small_is_read_when_the_song_shows_no_playhead_either():
    song, coda = segment(0.0, 60.0, 21.49), segment(60.5, 90.0, 9.49)
    assert not end_card([song, coda], 1, [still(60.5, 90.0)], False)


def test_an_end_card_is_left_out_of_the_reading(monkeypatch):
    paper = np.array([255.0, 255.0, 255.0])
    song = Segment(0.0, 60.0, 30.0, "dark", paper=paper, x1=640, y1=360)
    card = Segment(62.0, 80.0, 14.0, "dark", paper=paper, x1=640, y1=360)
    views = {0.0: [still(0.0, 30.0, playhead=True), still(30.0, 60.0)], 62.0: [still(62.0, 70.0), still(70.0, 80.0)]}

    def plate_obs(v, seg, si, thr, fthr, index, col0=None, fading=False):
        o = screen(index, [1, 2, 3])
        o.t0, o.t1, o.seg, o.systems, o.playhead = v.t0, v.t1, si, [], v.playhead
        return o
    monkeypatch.setattr(pipeline, "probe", lambda path: VideoInfo(640, 360, 80.0))
    monkeypatch.setattr(pipeline, "scan", lambda path, w, h, region=None, progress=None: ([song, card], {}, []))
    monkeypatch.setattr(glide, "glides", lambda *a: False)
    monkeypatch.setattr(pipeline, "views_in", lambda path, crop, pol, gap, paper, seed, t0, *a, **kw:
                        (views[t0], 128.0, 230.0))
    monkeypatch.setattr(pipeline, "plate_obs", plate_obs)
    monkeypatch.setattr(pipeline, "view_move", lambda *a: None)
    monkeypatch.setattr(pipeline, "finish", lambda path, info, r, *a: r)
    r = pipeline.read(Path("unused.mp4"))
    assert r.segments == [song] and r.tab_spans == [(0.0, 60.0)]
    assert [p.seg for p in r.parts] == [0] and [o.t0 for o in r.views] == [0.0, 30.0]


def page(notes: bool) -> np.ndarray:
    """A tab staff with barlines at 100, 400 and 700, a fret digit in the first measure, and one in the second only when notes is set."""
    img = np.full((200, 800, 3), 255, np.uint8)
    for i in range(6):
        cv2.line(img, (20, 60 + i * 14), (780, 60 + i * 14), (60, 60, 60), 1)
    for x in (100, 400, 700):
        cv2.line(img, (x, 60), (x, 130), (0, 0, 0), 2)
    cv2.putText(img, "5", (240, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if notes:
        cv2.putText(img, "7", (540, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # A measure number above the second measure, which doesn't make it full
    cv2.putText(img, "9", (404, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return img


def view(notes: bool) -> Obs:
    systems = parse_view(0, page(notes), (0, 0, 800, 200), "dark", 140.0, 229.0, 14.0)
    cells = [c for ps in systems for c in ps.cells if c.kind == "full"]
    o = Obs(0, 0.0, 5.0, cells, [], 0, (0, 0))
    o.systems = systems
    return o


def test_a_measure_with_only_staff_lines_and_a_number_is_empty():
    o = view(notes=False)
    assert [empty(o, c) for c in o.cells] == [False, True]


def screens(slots: list[list[int]], numbers: dict[int, int] | None = None) -> tuple[Assembly, list[Obs]]:
    """Views 0, 1, ... of one segment, 5 s each, showing measures with notes in the given slots; each slot is first
    seen where it first appears and carries its printed number from numbers, if any."""
    full = view(notes=True)
    obs = []
    for i, s in enumerate(slots):
        o = Obs(i, 5.0 * i, 5.0 * i + 5, [full.cells[0]] * len(s), [], 0, (0, 0))
        o.systems = full.systems
        obs.append(o)
    first: dict[int, int] = {}
    for i, s in enumerate(slots):
        for sid in s:
            first.setdefault(sid, i)
    A = Assembly(slots=[Slot(sid, None, first[sid], (numbers or {}).get(sid)) for sid in sorted(first)],
                 view_slots={i: list(s) for i, s in enumerate(slots)})
    return A, obs


def test_a_screen_sharing_nothing_with_overlapping_neighbors_is_pointed_out():
    A, obs = screens([[0, 1, 2], [1, 2, 3], [2, 3, 4], [5, 6, 7], [6, 7, 8]])
    (w,) = screen_breaks(A, obs)
    assert (w["t0"], w["t1"], w["kind"]) == (15.0, 20.0, "part_switch")
    assert "shares no measure with the one before it" in w["text"]


def test_a_screen_reusing_the_numbers_before_it_with_other_music_says_so():
    # Slots 5-7 are another part's measures, numbered 4-6 while the screen before shows 3-5
    numbers = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 4, 6: 5, 7: 6, 8: 7}
    A, obs = screens([[0, 1, 2], [1, 2, 3], [2, 3, 4], [5, 6, 7], [6, 7, 8]], numbers)
    (w,) = screen_breaks(A, obs)
    assert (w["t0"], w["t1"], w["kind"]) == (15.0, 20.0, "part_switch")
    assert "repeats measure numbers from the one before it, but with different music" in w["text"]


def test_page_turns_with_no_overlap_are_not_pointed_out():
    A, obs = screens([[0, 1], [2, 3], [4, 5], [6, 7]])
    assert screen_breaks(A, obs) == []


def test_a_replay_of_earlier_screens_is_not_pointed_out():
    A, obs = screens([[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 1, 2], [1, 2, 3]])
    assert screen_breaks(A, obs) == []


def one_frame(img):
    def frames(path, fps, start=0.0, length=None):
        yield start, img
    return frames


def test_tab_inside_a_program_window_is_an_editor_screen(monkeypatch):
    from youtabber import pipeline
    img = np.full((720, 1280, 3), 40, np.uint8)
    # Panels and toolbars: bands of lighter gray with straight edges across the window
    for y in (20, 60, 100, 600, 640, 680):
        cv2.rectangle(img, (0, y), (1279, y + 18), (90, 90, 90), -1)
    seg = Segment(10.0, 14.0, 12.0, "dark", y0=300, y1=420, x0=200, x1=1080)
    monkeypatch.setattr(pipeline, "frames", one_frame(img))
    assert pipeline.editor_screen(None, seg)


def test_a_band_over_footage_is_no_editor_screen(monkeypatch):
    from youtabber import pipeline
    rng = np.random.default_rng(0)
    # Footage: soft shapes with no straight edges, and the band's own lines inside the crop
    img = cv2.GaussianBlur(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8), (61, 61), 0)
    for i in range(6):
        cv2.line(img, (0, 560 + i * 16), (1279, 560 + i * 16), (230, 230, 230), 1)
    seg = Segment(10.0, 14.0, 16.0, "light", y0=520, y1=680, x0=0, x1=1280)
    monkeypatch.setattr(pipeline, "frames", one_frame(img))
    assert not pipeline.editor_screen(None, seg)


def test_a_new_line_spacing_straight_after_the_last_segment_is_pointed_out():
    from youtabber.check import layout_changes
    a = Segment(4.0, 306.0, 8.5, "dark", y0=0, y1=360)
    b = Segment(306.0, 312.0, 18.18, "dark", y0=4, y1=360)
    later = Segment(330.0, 340.0, 8.5, "dark", y0=0, y1=360)
    o = Obs(0, 306.0, 311.5, [], [], 1, (0, 0))
    (w,) = layout_changes([a, b, later], [o])
    assert (w["t0"], w["t1"], w["kind"]) == (306.0, 311.5, "part_switch")


def test_a_new_place_on_screen_or_a_short_segment_is_not_pointed_out():
    from youtabber.check import layout_changes
    a = Segment(0.0, 50.0, 14.0, "dark", y0=600, y1=1000)
    moved = Segment(52.0, 90.0, 14.2, "dark", y0=300, y1=700)
    short = Segment(91.0, 94.0, 20.0, "dark", y0=300, y1=700)
    assert layout_changes([a, moved, short], []) == []


def opening(**change) -> list[Segment]:
    """A 10 s opening drawn at line gap 10.4 in the top of the frame, then 60 s of tab at line gap 10 in the bottom
    after a 0.5 s break: an intro by every threshold. `change` replaces fields of the opening (a) or the tab (b),
    as a_<field> or b_<field>."""
    a = Segment(0.0, 10.0, 10.4, "dark", x0=0, x1=100, y0=0, y1=100)
    b = Segment(10.5, 70.5, 10.0, "dark", x0=0, x1=100, y0=200, y1=300)
    a = replace(a, **{k[2:]: v for k, v in change.items() if k.startswith("a_")})
    b = replace(b, **{k[2:]: v for k, v in change.items() if k.startswith("b_")})
    return [a, b]


def test_a_short_opening_in_another_layout_is_an_intro():
    assert intro(opening())
    assert intro(opening(a_t1=15.0, b_t0=15.5, b_t1=80.0))
    assert not intro(opening(a_t1=15.1, b_t0=15.6, b_t1=80.0))   # over 15 s long
    assert not intro(opening(b_t0=10.4))                          # no break before the tab
    assert not intro(opening(b_t1=70.4))                          # under 60 s of tab after it
    # Panels 100 px square, the tab's moved right by 18 px overlaps under 0.7 of their union, by 17 px it doesn't
    assert intro(opening(b_x0=18, b_x1=118, b_y0=0, b_y1=100))
    assert not intro(opening(b_x0=17, b_x1=117, b_y0=0, b_y1=100))
    assert not intro(opening(a_gap=10.3))                         # line gaps under 4% apart
    assert not intro(opening()[:1])


def numbered_view(monkeypatch, boxes: list) -> dict[int, int]:
    """detected_numbers on a one-system view whose tab staff's top line is at row 100 (line gap 10), with cells
    starting at x 0, 200, 400 and 600, when the text detector finds `boxes` (left, right, middle row in the strip,
    number). The strip starts 3 line gaps above the top line, at row 70."""
    monkeypatch.setattr(pipeline.ocr, "numbers_in_line", lambda strip: boxes)
    cells = [SimpleNamespace(system=0, x0=x) for x in (0, 200, 400, 600)]
    o = Obs(0, 0.0, 5.0, cells, [], 0, (0, 0))
    o.image = np.full((200, 800, 3), 255, np.uint8)
    o.systems = [SimpleNamespace(system=SimpleNamespace(notation=None, tab=SimpleNamespace(top=100.0, gap=10.0)))]
    return detected_numbers(o, "dark")


def test_detected_numbers_go_to_the_nearest_barline_and_skip_digits_on_the_top_line(monkeypatch):
    # Rows 26 and 28 are 0.4 and 0.2 line gaps above the top line; x 305 is 9.5 gaps from the nearest barline
    boxes = [(0, 10, 26, 7), (195, 205, 26, 8), (400, 410, 28, 3), (300, 310, 26, 9)]
    assert numbered_view(monkeypatch, boxes) == {0: 7, 1: 8}


def unsure_printed(view: int, links: tuple = ()) -> list:
    """Notices for slot 5, first placed as new under an unresolved fold at 10-12 s, printed from view 0 (10-12 s) or
    view 1 (30-32 s), view 1 carrying on the cells at `links` from the view before."""
    ink = np.zeros((60, 200), bool)
    obs = [Obs(i, t, t + 2, [Cell(i, 0, 0, 0, 200, "full", ink, ink, 0, gap=10.0, mid_y=30.0)], [], 0, (0, 0))
           for i, t in ((0, 10.0), (1, 30.0))]
    A = Assembly(unsure_slots={5: (10.0, 12.0)}, view_links={1: set(links)})
    o = obs[view]
    m = Measure(5, view, o.cells[0], o.t0 + 1, o.t0 + 1, 100.0, 30.0, None, None, False)
    return [(n.t0, n.t1, n.kind) for n in pipeline.unsure_notices(A, obs, [m])]


def test_a_measure_first_placed_under_an_unresolved_fold_and_printed_from_a_later_fold_is_noted():
    assert unsure_printed(1) == [(30.0, 32.0, "unsure_repeat")]


def test_a_measure_printed_where_its_unresolved_fold_is_noted_needs_no_other_notice():
    assert unsure_printed(0) == []


def folded_times(folds: list[int], measures: tuple[int, ...] = (1, 1, 1, 1)) -> list[float]:
    """Play times lone_fold_times gives for views 0-3 (10 s apart, view i holding measures[i] measures, each first
    placed on that view and played 1 s apart from 1 s into it) when the views at `folds` hold unresolved folds."""
    ink = np.zeros((60, 200), bool)
    obs = [Obs(i, 10.0 * i, 10.0 * i + 8,
               [Cell(i, 0, k, 200 * k, 200 * k + 200, "full", ink, ink, 0, gap=10.0, mid_y=30.0) for k in range(n)],
               [], 0, (0, 0)) for i, n in enumerate(measures)]
    cells = [(o, c) for o in obs for c in o.cells]
    A = Assembly(slots=[Slot(s, c, o.index, None) for s, (o, c) in enumerate(cells)],
                 unresolved_folds=[(obs[i].t0, obs[i].t1, 0, 0) for i in folds])
    printed = [Measure(s, o.index, c, o.t0 + 1 + c.index, o.t0 + 1 + c.index, 100.0, 30.0, None, None, False)
               for s, (o, c) in enumerate(cells)]
    return pipeline.lone_fold_times(A, {0: obs}, printed)


def test_only_a_lone_unresolved_fold_gives_measures_that_are_not_new():
    assert folded_times([2]) == [21.0]
    assert folded_times([0, 2]) == [1.0, 21.0]
    # Folds on views in a row follow earlier music page by page
    assert folded_times([1, 2]) == []


def test_every_measure_first_placed_on_a_lone_fold_view_is_not_new():
    assert folded_times([2], (1, 1, 2, 1)) == [21.0, 22.0]


def test_a_view_without_measures_between_two_fold_views_keeps_them_in_a_row():
    assert folded_times([1, 3], (1, 1, 0, 1)) == []
    # With measures on the view between, each fold view stands alone
    assert folded_times([1, 3]) == [11.0, 31.0]


def test_a_measure_carried_on_from_the_view_before_needs_no_other_notice():
    assert unsure_printed(1, links=(0,)) == []


def test_an_opening_cut_in_two_by_a_brief_screen_is_one_intro():
    a, b = opening()
    first, second = replace(a, t1=5.0), replace(a, t0=5.5)
    assert intro([first, second, b]) == 2
    # A second piece in the song's layout is the song, so the opening is the first piece alone
    assert intro([first, replace(second, gap=10.0, y0=200, y1=300), b]) == 1
    # A piece in a third layout right after the opening card doesn't carry the opening on, so there is no intro
    card = Segment(0.0, 5.0, 14.0, "dark", x0=0, x1=100, y0=0, y1=100)
    third = Segment(5.2, 8.0, 20.0, "dark", x0=0, x1=100, y0=100, y1=200)
    song = Segment(9.0, 100.0, 10.0, "dark", x0=0, x1=100, y0=200, y1=300)
    assert intro([card, third, song]) == 0


def test_panels_overlapping_most_of_their_union_or_spaced_alike_are_one_layout():
    a = Segment(0.0, 5.0, 10.0, "dark", x0=0, x1=100, y0=0, y1=100)
    # Moved right by 100/9 px, the panels overlap by 80% of their union
    near = replace(a, x0=100 / 9, x1=100 + 100 / 9)
    assert abs(overlap_share(a, near) - 0.8) < 1e-9 and one_layout(a, near)
    # Moved right by 100/3 px they overlap by half; line gaps 2% apart are still one layout, 17% apart are not
    half = replace(a, x0=100 / 3, x1=100 + 100 / 3)
    assert abs(overlap_share(a, half) - 0.5) < 1e-9
    assert one_layout(a, replace(half, gap=10.2))
    assert not one_layout(a, replace(half, gap=12.0))


def lead_in(lead: int, inked: bool, first: Cell) -> list[Cell]:
    """without_lead_in over `first` then two 100 px measures with fret digits, at line gap 10."""
    rest = [replace(cell(100, marks=True), x0=x, x1=x + 100) for x in (300, 400)]
    return without_lead_in([first] + rest, lead, 10.0, inked)


def test_an_empty_header_ending_at_the_playhead_is_no_measure():
    header = cell(60, marks=False, kind="lead")
    # The header ends at x 160, within half a line gap of the playhead at x 157
    assert header not in lead_in(157, False, header)
    assert header in lead_in(150, False, header)
    digits = cell(60, marks=True, kind="lead")
    assert digits in lead_in(157, False, digits)


def test_a_narrow_empty_cell_from_an_uncolored_playhead_to_the_first_barline_is_no_measure():
    # Starting at the playhead column x 100: 40 px is 0.4 of the typical 100 px measure, 60 px is 0.6
    narrow, wide = cell(40, marks=False), cell(60, marks=False)
    assert narrow not in lead_in(100, True, narrow)
    assert wide in lead_in(100, True, wide)
    # A colored playhead never passes for a barline, so the narrow cell is a measure
    assert narrow in lead_in(100, False, narrow)


def notation(system: int, number: int, read: int | None) -> Cell:
    c = cell(60, marks=True)
    c.system, c.number, c.read, c.notation_only = system, number, read, True
    return c


def test_a_notation_staff_whose_read_number_disagrees_with_the_count_holds_no_measures():
    cells = [notation(0, 5, 9), notation(0, 6, None), notation(1, 7, 7), notation(1, 8, None)]
    assert placed_notation(cells) == cells[2:]
