import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from test_check import FakeMosaic, cell
from test_numbering import band_view, music

from youtabber import counter, ocr, pipeline, track
from youtabber.assemble import Obs, assemble
from youtabber.parse import Cell
from youtabber.pipeline import part_pairs

USE = {"numbers", "playhead"}


def decisions(A):
    return [(d.kind, d.chosen, d.basis, d.resolved) for d in A.decisions]


def test_a_glide_piece_is_new_music_even_when_it_looks_like_the_piece_before():
    # Two pieces of one glide never share a measure, so a piece that looks like the one before isn't joined to it
    p, o = band_view(0, 1), band_view(1, 1)
    o.move = track.Move("glide")
    A = assemble([p, o], 10.0, False, USE, lambda o: [])
    assert A.view_slots[1] == [4, 5, 6, 7]
    assert decisions(A) == [("join", "glide continues", "track", True)]


def test_a_glide_piece_skips_the_scroll_join():
    p, o = band_view(0, 1), band_view(1, 3)
    o.move = track.Move("glide")
    assert assemble([p, o], 10.0, False, USE, lambda o: []).view_slots[1] == [4, 5, 6, 7]
    p, o = band_view(0, 1), band_view(1, 3)
    assert assemble([p, o], 10.0, False, USE, lambda o: []).view_slots[1] == [2, 3, 4, 5]


def test_glide_pieces_are_never_paired_for_a_band_move():
    # A still view, then two pieces of one glide in the same segment and part
    still, first, second = band_view(0, 1), band_view(1, 5), band_view(2, 9)
    first.mosaic, second.mosaic = object(), object()
    second.move = track.Move("glide")
    # No pair holds a piece, so view_move never replaces a piece's Move
    assert part_pairs([still, first, second]) == []


def piece(t0, middles, moved_at=None, seeded=0, tail_from=10**6):
    cells = [SimpleNamespace(x0=x - 50, x1=x + 50) for x in middles]
    mosaic = SimpleNamespace(moved_at=moved_at, seeded=seeded, tail_from=tail_from)
    return SimpleNamespace(t0=t0, cells=cells, all_cells=cells, mosaic=mosaic, move=None)


def test_each_measure_at_a_redraw_is_kept_in_the_piece_it_glided_through(monkeypatch):
    made = {0.0: piece(0.0, [100, 500, 1000], moved_at=5.0, tail_from=900),
            5.0: piece(5.0, [200, 400, 800], seeded=300)}
    starts = []

    def glide_obs(path, crop, seg, si, found, thr, fthr, index, t0):
        starts.append(t0)
        return made[t0]
    monkeypatch.setattr(pipeline, "glide_obs", glide_obs)
    monkeypatch.setattr(counter, "series", lambda path, t0, t1, stop: [])
    seg = SimpleNamespace(t0=0.0, t1=60.0)
    (first, second), stopped = pipeline.glide_pieces(None, None, seg, 0, {}, 140.0, 229.0, 0)
    assert starts == [0.0, 5.0] and stopped is None
    assert [(c.x0 + c.x1) / 2 for c in first.cells] == [100, 500]
    assert [(c.x0 + c.x1) / 2 for c in second.cells] == [400, 800]
    assert first.move is None and second.move.kind == "glide"
    assert (first.first_piece, first.last_piece) == (True, False)
    assert (second.first_piece, second.last_piece) == (False, True)


def test_reading_a_glide_that_stops_after_a_redraw_gives_the_time_it_stopped(monkeypatch):
    seg = SimpleNamespace(t0=0.0, t1=60.0)
    monkeypatch.setattr(counter, "series", lambda path, t0, t1, stop: [])

    def pieces_of(made):
        monkeypatch.setattr(pipeline, "glide_obs", lambda *args: made.get(args[-1]))
        return pipeline.glide_pieces(None, None, seg, 0, {}, 140.0, 229.0, 0)
    # The piece after the redraw at 5.0 couldn't be read
    found, stopped = pieces_of({0.0: piece(0.0, [100, 500], moved_at=5.0)})
    assert len(found) == 1 and stopped == 5.0
    # The most pieces were read before the segment ended
    monkeypatch.setattr(pipeline, "MAX_GLIDE_PIECES", 1)
    found, stopped = pieces_of({0.0: piece(0.0, [100, 500], moved_at=5.0)})
    assert len(found) == 1 and stopped == 5.0
    # A piece that reaches the end of the segment
    found, stopped = pieces_of({0.0: piece(0.0, [100, 500])})
    assert len(found) == 1 and stopped is None


def test_the_bar_counter_is_read_once_in_a_worker_thread_for_every_piece(monkeypatch):
    made = {0.0: piece(0.0, [100, 500], moved_at=5.0), 5.0: piece(5.0, [200, 400])}
    monkeypatch.setattr(pipeline, "glide_obs", lambda *args: made[args[-1]])
    monkeypatch.setattr(ocr, "engine", lambda: None)
    threads = []

    def series(path, t0, t1, stop):
        threads.append(threading.current_thread())
        return [(0.0, 1.0, 1)]
    monkeypatch.setattr(counter, "series", series)
    seg = SimpleNamespace(t0=0.0, t1=60.0)
    pieces, _ = pipeline.glide_pieces(None, None, seg, 0, {}, 140.0, 229.0, 0)
    assert len(threads) == 1 and threads[0] is not threading.main_thread()
    assert [o.counter for o in pieces] == [[(0.0, 1.0, 1)]] * 2

    # A failure while reading the counter reaches the caller
    def broken(path, t0, t1, stop):
        raise ValueError("unreadable frame")
    monkeypatch.setattr(counter, "series", broken)
    with pytest.raises(ValueError, match="unreadable frame"):
        pipeline.glide_pieces(None, None, seg, 0, {}, 140.0, 229.0, 0)


def test_a_failure_while_a_mosaic_builds_stops_the_counter_read(monkeypatch):
    monkeypatch.setattr(ocr, "engine", lambda: None)
    stopped = []

    def series(path, t0, t1, stop):
        # A counter read that runs until it is told to stop
        stopped.append(stop.wait(5.0))
        return []

    def glide_obs(*args):
        raise ValueError("unreadable mosaic")
    monkeypatch.setattr(counter, "series", series)
    monkeypatch.setattr(pipeline, "glide_obs", glide_obs)
    seg = SimpleNamespace(t0=0.0, t1=60.0)
    began = time.monotonic()
    with pytest.raises(ValueError, match="unreadable mosaic"):
        pipeline.glide_pieces(None, None, seg, 0, {}, 140.0, 229.0, 0)
    assert time.monotonic() - began < 1.0 and stopped == [True]


def test_glided_measures_take_their_numbers_from_the_bar_counter():
    # Canvas x / 100 is play time; the counter went unread while the third measure played
    o = SimpleNamespace(cells=[cell(100 * i, 100 * i + 100) for i in range(4)], mosaic=FakeMosaic(),
                        counter=[(0.0, 0.9, 1), (1.0, 1.9, 2), (3.0, 3.9, 4)])
    pipeline.counted_numbers(o)
    assert [c.number for c in o.cells] == [1, 2, 3, 4]
    # Two measures between counts 1 and 7 can't be numbered, so no measure is
    o = SimpleNamespace(cells=[cell(100 * i, 100 * i + 100) for i in range(4)], mosaic=FakeMosaic(),
                        counter=[(0.0, 0.9, 1), (3.0, 3.9, 7)])
    pipeline.counted_numbers(o)
    assert [c.number for c in o.cells] == [None] * 4


def counted_glide(patterns, numbers):
    """One glide view whose cells hold the music of `patterns` under the counter's `numbers`."""
    cells = []
    for i, (p, n) in enumerate(zip(patterns, numbers)):
        ink = music("a", p)
        cells.append(Cell(0, 0, i, 200 * i, 200 * i + 200, "full", ink, np.zeros_like(ink), 0, gap=10.0, mid_y=30.0,
                          tint=np.zeros_like(ink), number=n, glided=True))
    o = Obs(0, 0.0, 6.0, cells, [], 0, (0, 0))
    o.mosaic = FakeMosaic()
    return assemble([o], 10.0, False, {"numbers"}, lambda o: [])


def test_a_counter_fold_needs_half_its_run_to_look_like_the_measures_it_folds_onto():
    # The counter goes back to 1 over the same music: a replay, folded
    A = counted_glide([1, 2, 3, 4, 1, 2], [1, 2, 3, 4, 1, 2])
    assert len(A.slots) == 4 and A.view_slots[0] == [0, 1, 2, 3, 0, 1]
    assert [d.kind for d in A.decisions] == ["fold"]
    # Over other music the counter misread, so the run is new music
    A = counted_glide([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 1, 2])
    assert len(A.slots) == 6 and A.view_slots[0] == [0, 1, 2, 3, 4, 5]
    assert [d.kind for d in A.decisions] == []
    # One measure of the two differs: the run still folds as a whole
    A = counted_glide([1, 2, 3, 4, 1, 6], [1, 2, 3, 4, 1, 2])
    assert len(A.slots) == 4 and A.view_slots[0] == [0, 1, 2, 3, 0, 1]


def counted_pieces(patterns, numbers):
    """Piece 0 holds patterns 1-4 numbered 1-4; piece 1 glides on with `patterns` under the counter's `numbers`."""
    def view(index, t0, patterns, numbers):
        cells = []
        for i, (p, n) in enumerate(zip(patterns, numbers)):
            ink = music("a", p)
            cells.append(Cell(index, 0, i, 200 * i, 200 * i + 200, "full", ink, np.zeros_like(ink), 0, gap=10.0,
                              mid_y=30.0, tint=np.zeros_like(ink), number=n, glided=True))
        o = Obs(index, t0, t0 + 4.0, cells, [], 0, (0, 0))
        o.mosaic = FakeMosaic()
        return o
    first, second = view(0, 0.0, [1, 2, 3, 4], [1, 2, 3, 4]), view(1, 4.0, patterns, numbers)
    second.move = track.Move("glide")
    return assemble([first, second], 10.0, False, {"numbers"}, lambda o: [])


def test_a_counter_number_naming_the_piece_before_takes_the_same_fold_check():
    # A misread number onto the piece before, over other music, is new music
    A = counted_pieces([6, 7], [4, 5])
    assert len(A.slots) == 6 and A.view_slots[1] == [4, 5]
    assert [d.kind for d in A.decisions] == []
    # The counter goes back over the piece before's music: a replay, folded
    A = counted_pieces([3, 4], [3, 4])
    assert len(A.slots) == 4 and A.view_slots[1] == [2, 3]
    assert [d.kind for d in A.decisions] == ["fold"]
