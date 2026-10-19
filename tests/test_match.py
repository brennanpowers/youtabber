import numpy as np

from youtabber.match import far_ink, resemble
from youtabber.parse import Cell


def cell(ink: np.ndarray) -> Cell:
    blank = np.zeros_like(ink)
    return Cell(0, 0, 0, 0, ink.shape[1], "full", ink, blank, 0, gap=10.0, ref_y=5, tint=blank)


def test_far_ink_counts_marks_with_nothing_of_the_other_map_nearby():
    a = np.zeros((40, 80), bool)
    a[10:13, 10:13] = True
    b = np.zeros((40, 80), bool)
    b[10:13, 40:43] = True
    assert far_ink(a, b) == 9
    # The same mark 2 px away is near enough to match
    assert far_ink(a, np.roll(a, 2, axis=1)) == 0


def test_the_quick_count_answers_only_past_the_limit_and_never_overstates():
    rng = np.random.default_rng(7)

    def marks() -> np.ndarray:
        m = np.zeros((60, 120), bool)
        for y, x in zip(rng.integers(5, 52, 12), rng.integers(5, 112, 12)):
            m[y:y + 3, x:x + 3] = True
        return m
    quick_answers = 0
    for _ in range(40):
        a = marks()
        b = a.copy() if rng.random() < 0.3 else marks()
        exact = resemble(cell(a), cell(b), 10.0)
        quick = resemble(cell(a), cell(b), 10.0, limit=0.15)
        if quick != exact:
            # Answered from the quick count: still past the limit, and never past the exact share
            quick_answers += 1
            assert 0.15 <= quick <= exact
    assert quick_answers > 0


def test_the_quick_count_keeps_partners_in_the_last_columns_and_rows():
    # 43 px is not a whole number of quick-count squares; each mark's only partner sits in the last 3 px
    for axis in (0, 1):
        for gap in (1, 2):
            a = np.zeros((43, 43), bool)
            b = np.zeros((43, 43), bool)
            a[10:13, 38:40] = True
            b[10:13, 38 + gap:40 + gap] = True
            if axis == 0:
                a, b = a.T.copy(), b.T.copy()
            exact = resemble(cell(a), cell(b), 10.0)
            assert resemble(cell(a), cell(b), 10.0, limit=0.15) == exact
            assert far_ink(a, b) == 0
