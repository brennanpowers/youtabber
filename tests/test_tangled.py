"""A segment where several parts print and most of its screens can't be joined or folded is refused."""
from types import SimpleNamespace as NS

from youtabber.assemble import Decision
from youtabber.pipeline import tangled


def reading(joins, folds, parts_printing=2):
    """One segment, 30-90 s, of two parts with a view each and a segment before it of one part. `joins` and `folds`
    are (resolved, unresolved) counts of decisions about the segment's views; the first `parts_printing` parts of
    the segment print measures."""
    segments = [NS(t0=0.0, t1=20.0), NS(t0=30.0, t1=90.0)]
    parts = [NS(id=0, seg=0), NS(id=1, seg=1), NS(id=2, seg=1)]
    obs = [NS(seg=0, part=0), NS(seg=1, part=1), NS(seg=1, part=2)]
    primary = [NS(view=0)] + [NS(view=1 + k) for k in range(parts_printing)]
    decisions = []
    for kind, (ok, bad) in (("join", joins), ("fold", folds)):
        decisions += [Decision(40.0, kind, "", [], "", True, view=1) for _ in range(ok)]
        decisions += [Decision(40.0, kind, "", [], "", False, view=2) for _ in range(bad)]
    return segments, parts, obs, NS(decisions=decisions), primary


def test_a_segment_of_several_parts_with_most_joins_and_folds_unresolved_is_tangled():
    segments, *rest = reading(joins=(4, 6), folds=(1, 2))
    assert tangled(segments, *rest) is segments[1]


def test_a_segment_just_under_each_threshold_is_not_tangled():
    # Only one part prints
    assert tangled(*reading(joins=(4, 6), folds=(1, 2), parts_printing=1)) is None
    # Nine joins
    assert tangled(*reading(joins=(3, 6), folds=(1, 2))) is None
    # Half the joins unresolved, not more
    assert tangled(*reading(joins=(5, 5), folds=(1, 2))) is None
    # Half the folds unresolved, not more
    assert tangled(*reading(joins=(4, 6), folds=(2, 2))) is None
    # No folds at all: split-screen panels that turn their pages apart leave joins unresolved
    assert tangled(*reading(joins=(4, 7), folds=(0, 0))) is None


def test_decisions_about_another_segment_do_not_count():
    segments, parts, obs, A, primary = reading(joins=(4, 6), folds=(1, 2))
    for d in A.decisions:
        d.view = 0
    assert tangled(segments, parts, obs, A, primary) is None
