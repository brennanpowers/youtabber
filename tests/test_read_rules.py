import cv2
import numpy as np

from youtabber.assemble import Assembly, Obs, Slot
from youtabber.check import empty, screen_breaks
from youtabber.parse import parse_view
from youtabber.pipeline import scattered
from youtabber.scan import Segment


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
