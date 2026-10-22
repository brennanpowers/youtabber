"""A glide's last measure with no closing barline, and measures counted on past the bar counter's last reading."""
from types import SimpleNamespace

import numpy as np
from test_check import FakeMosaic
from test_numbering import music

from youtabber import pipeline
from youtabber.assemble import Obs
from youtabber.parse import Cell


def glided(patterns, kinds=None, width=200):
    """Glided cells side by side, `width` wide, holding the music of `patterns`; `kinds` overrides their kinds."""
    kinds = kinds or ["full"] * len(patterns)
    cells = []
    for i, (p, kind) in enumerate(zip(patterns, kinds)):
        ink = music("a", p)
        cells.append(Cell(0, 0, i, width * i, width * i + width, kind, ink, np.zeros_like(ink), 0, gap=10.0,
                          mid_y=30.0, tint=np.zeros_like(ink), glided=True))
    return cells


def test_a_glides_last_measure_without_a_closing_barline_is_timed_half_a_measure_past_its_start():
    # Canvas x / 100 is play time; three measures 100 px wide, then the last one runs on into 700 px of blank staff
    cells = glided([1, 2, 3], width=100) + [Cell(0, 0, 3, 300, 1000, "tail", np.zeros((60, 700), bool),
                                                 np.zeros((60, 700), bool), 0, gap=10.0, glided=True)]
    o = SimpleNamespace(cells=cells, mosaic=FakeMosaic())
    assert pipeline.play_times(o) == [0.5, 1.5, 2.5, 3.5]
    # A last measure narrower than a typical one keeps its middle
    o.cells[3].x1 = 340
    assert pipeline.play_times(o)[3] == 3.2
    # With no whole measure to size it by, its middle too
    o = SimpleNamespace(cells=[cells[3]], mosaic=FakeMosaic())
    o.cells[0].x1 = 1000
    assert pipeline.play_times(o) == [6.5]


class StillEnd(FakeMosaic):
    """A mosaic whose band stood still on its last screen from canvas x 600 on, the playhead at x 0."""
    ref_x, y0 = 0.0, 0

    def seen_at(self, x):
        return (9.0, x - 600.0) if x >= 600 else None


def test_a_glides_last_measure_is_placed_when_its_start_crossed_the_playhead():
    # Its middle (650) sat on the still last screen, but the column half a typical measure past its start crossed
    # the playhead at 3.5 s
    cells = glided([1, 2, 3], width=100) + [Cell(0, 0, 3, 300, 1000, "tail", np.zeros((60, 700), bool),
                                                 np.zeros((60, 700), bool), 0, gap=10.0, glided=True)]
    o = Obs(0, 0.0, 10.0, cells, [], 0, (0, 0))
    o.mosaic = StillEnd()
    m = pipeline.measure_at(3, o, cells[3], [])
    assert (m.t, m.play_t, m.t_ref, m.x) == (3.5, 3.5, 3.5, 0.0)


def test_a_glides_last_measure_takes_the_counter_number_shown_as_it_starts():
    # The counter read 4 while the last measure began and 9 by the middle of its blank staff
    cells = glided([1, 2, 3], width=100) + [Cell(0, 0, 3, 300, 1000, "tail", np.zeros((60, 700), bool),
                                                 np.zeros((60, 700), bool), 0, gap=10.0, glided=True)]
    o = SimpleNamespace(cells=cells, mosaic=FakeMosaic(),
                        counter=[(0.0, 0.9, 1), (1.0, 1.9, 2), (2.0, 2.9, 3), (3.0, 3.9, 4), (6.0, 6.9, 9)])
    pipeline.counted_numbers(o)
    assert [c.number for c in o.cells] == [1, 2, 3, 4]


def test_a_measure_past_the_counters_last_reading_is_counted_on():
    # A number no earlier measure holds is counted on from the measure before it
    o = SimpleNamespace(cells=glided([24, 25, 26, 27]), mosaic=FakeMosaic(),
                        counter=[(0.0, 1.9, 24), (2.0, 3.9, 25), (4.0, 5.9, 26)])
    pipeline.counted_numbers(o)
    assert [c.number for c in o.cells] == [24, 25, 26, 27]
