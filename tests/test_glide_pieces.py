from types import SimpleNamespace

from test_numbering import band_view

from youtabber import pipeline, track
from youtabber.assemble import assemble
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
