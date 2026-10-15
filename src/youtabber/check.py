"""Checks on the assembled score that turn likely mistakes into warnings next to the measures concerned."""
import numpy as np

from youtabber.match import resemble

# Two cells this alike (share of unmatched ink) that were still placed in different slots may be one measure
DUP_RESEMBLE = 0.03
# Earlier views a new slot is compared against
DUP_LOOKBACK_VIEWS = 3
# A screen held this many times longer per measure than the video's typical screen may have lost measures
DWELL_RATIO = 3.0
# ...if it is held at least this long
DWELL_MIN_SECONDS = 4.0
# No measure is played for longer than this: a 4/4 bar at 40 bpm takes 6 s. A screen or a video holding
# its measures longer than this on average has lost measures, whatever the rest of the video does.
MAX_SECONDS_PER_MEASURE = 15.0


def too_few(obs, printed: int) -> str | None:
    """A message when the whole video yields far fewer measures than its time with tab on screen allows."""
    shown = sum(o.t1 - o.t0 for o in obs)
    if shown >= 2 * MAX_SECONDS_PER_MEASURE and printed < shown / MAX_SECONDS_PER_MEASURE:
        return (f"Read only {printed} measures from {shown:.0f} s of tab on screen; at least "
                f"{shown / MAX_SECONDS_PER_MEASURE:.0f} would fit. Measures were missed, so no PDF was written.")
    return None


def possible_duplicates(A, obs, gap: float) -> list[dict]:
    """New slots whose music looks like a slot from the last few views: an overlap that wasn't joined."""
    by_index = {o.index: o for o in obs}
    out = []
    for s in A.slots:
        o = by_index[s.obs]
        for t in A.slots[:s.id]:
            if t.obs >= s.obs or s.obs - t.obs > DUP_LOOKBACK_VIEWS:
                continue
            if s.number is not None and t.number is not None:
                continue  # different printed numbers: different measures
            if resemble(s.first, t.first, gap) < DUP_RESEMBLE:
                first = by_index[t.obs]
                out.append({"t0": round(o.t0, 2), "t1": round(o.t1, 2), "kind": "possible_duplicate",
                            "text": f"A measure at {fmt(o.t0)}-{fmt(o.t1)} looks like one shown at "
                                    f"{fmt(first.t0)}-{fmt(first.t1)}; it may be printed twice."})
                break
    return out


def dwell(obs) -> list[dict]:
    """Screens held much longer per measure than usual, or showing tab but no measure at all."""
    rates = [(o.t1 - o.t0) / len(o.cells) for o in obs if o.cells]
    if not rates:
        return []
    typical = float(np.median(rates))
    out = []
    for o in obs:
        held = o.t1 - o.t0
        if held < DWELL_MIN_SECONDS:
            continue
        if not o.cells or held / len(o.cells) > min(DWELL_RATIO * typical, MAX_SECONDS_PER_MEASURE):
            out.append({"t0": round(o.t0, 2), "t1": round(o.t1, 2), "kind": "few_measures",
                        "text": f"The tab at {fmt(o.t0)}-{fmt(o.t1)} stays on screen a long time for the "
                                f"{len(o.cells)} measures read from it; measures may be missing there."})
    return out


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


# A measure this many times wider than the video's typical measure probably hides a missed barline
WIDE_RATIO = 1.7


def odd_widths(obs) -> list[dict]:
    """Measures much wider than usual, where a barline was likely missed and a measure may be lost."""
    widths = [c.x1 - c.x0 for o in obs for c in o.cells if c.kind == "full"]
    if not widths:
        return []
    typical = float(np.median(widths))
    out = []
    for o in obs:
        for c in o.cells:
            if c.kind == "full" and c.x1 - c.x0 > WIDE_RATIO * typical:
                t = o.mosaic.time_at((c.x0 + c.x1) / 2) if o.mosaic is not None else (o.t0 + o.t1) / 2
                out.append({"t0": round(max(t - 3, o.t0), 2), "t1": round(min(t + 3, o.t1), 2), "kind": "wide_measure",
                            "text": f"A measure around {fmt(t)} is much wider than the others; a barline may have "
                                    f"been missed there."})
    return out
