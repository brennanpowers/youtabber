from types import SimpleNamespace

import numpy as np

from youtabber.panel import panel_bounds
from youtabber.scan import Segment

W, H, CELL = 1920, 1080, 8


def panel_system(x0, x1):
    """A system drawn at columns x0-x1 on a still frame."""
    lines = np.zeros(W, bool)
    lines[x0:x1] = True
    tab = SimpleNamespace(top=400, bottom=470, x0=x0, x1=x1, gap=14)
    return SimpleNamespace(top=400, tab=tab, lines_at=lines, motion=np.zeros((H // CELL, W // CELL)))


def test_a_right_panel_s_bound_is_in_its_own_columns():
    seg = Segment(t0=0, t1=10, gap=14, polarity="dark", y0=0, y1=H, x0=0, x1=W)
    found = {k * 0.5: [panel_system(1000, 1800)] for k in range(20)}
    x0, y0, x1, y1 = panel_bounds(found, seg, (960, W))(5.0)
    assert x0 == 1000 - int(2.5 * 14) - 960
    assert x1 == 1800 + int(1.5 * 14) - 960
    assert (y0, y1) == (0, H)
