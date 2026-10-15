"""Checks on the assembled score that turn likely mistakes into warnings next to the measures concerned."""
import numpy as np

from youtabber.match import resemble, resemble_scaled

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

# Glided measures repeat an earlier stretch of the same glide when this many in a row look alike
REPEAT_RUN = 4
# Two glided measures this alike after stretching to one width (share of unmatched ink) look like the same music
REPEAT_RESEMBLE = 0.15
# A gap between consecutive glided measures' play times this many times the median means measures were lost
GAP_RATIO = 1.6
# Tab on screen this long with no printed measure means music was lost
EMPTY_SECONDS = 25.0
# Long stretches are reported in pieces this long, so each points at a findable part of the video
PIECE_SECONDS = 30.0
# More printed measures than the highest measure number plus this many means some printed twice
COUNT_SLACK = 2


def pieces(t0: float, t1: float) -> list[tuple[float, float]]:
    """t0..t1 cut into pieces of at most PIECE_SECONDS."""
    out, a = [], t0
    while a < t1:
        out.append((a, min(t1, a + PIECE_SECONDS)))
        a += PIECE_SECONDS
    return out


def glide_repeats(o, gap: float) -> list[dict]:
    """Stretches of a glide whose measures look like an earlier stretch of the same glide, REPEAT_RUN or more in
    a row. Riffs trip this as well as replays, so it only warns; each stretch is named with its times."""
    cells = o.cells
    n = len(cells)
    like: dict[tuple[int, int], bool] = {}

    def alike(i: int, j: int) -> bool:
        if (i, j) not in like:
            like[(i, j)] = resemble_scaled(cells[j], cells[i], gap) < REPEAT_RESEMBLE
        return like[(i, j)]

    repeating = set()
    for i in range(REPEAT_RUN, n - REPEAT_RUN + 1):
        for j in range(0, i - REPEAT_RUN + 1):
            if all(alike(i + k, j + k) for k in range(REPEAT_RUN)):
                repeating.update(range(i, i + REPEAT_RUN))
                break
    out = []
    run: list[int] = []
    for i in sorted(repeating) + [None]:
        if run and (i is None or i != run[-1] + 1):
            a = o.mosaic.time_at((cells[run[0]].x0 + cells[run[0]].x1) / 2)
            b = o.mosaic.time_at((cells[run[-1]].x0 + cells[run[-1]].x1) / 2)
            for p0, p1 in pieces(a, max(b, a + 1)):
                out.append({"t0": round(p0, 2), "t1": round(p1, 2), "kind": "glide_repeat",
                            "text": f"The measures at {fmt(p0)}-{fmt(p1)} look like ones played earlier in this "
                                    f"smooth scroll. If the video plays them again, they are printed twice."})
            run = []
        if i is not None:
            run.append(i)
    return out


def glide_gaps(o) -> list[dict]:
    """Places in a glide where playback takes much longer than usual from one measure to the next: measures
    were likely lost there."""
    times = [o.mosaic.time_at((c.x0 + c.x1) / 2) for c in o.cells]
    if len(times) < 4:
        return []
    d = np.diff(times)
    med = float(np.median(d))
    out = []
    for k, v in enumerate(d):
        if med > 0 and v > GAP_RATIO * med:
            out.append({"t0": round(times[k], 2), "t1": round(times[k + 1], 2), "kind": "glide_gap",
                        "text": f"Playback takes {v:.1f} s from the measure at {fmt(times[k])} to the next, about "
                                f"{v / med:.1f} measures' time; measures may be missing there."})
    return out


def empty_stretches(spans: list[tuple[float, float]], play_times: list[float]) -> list[dict]:
    """Stretches of tab on screen at least EMPTY_SECONDS long with no printed measure played in them."""
    times = sorted(play_times)
    out = []
    for s0, s1 in spans:
        marks = [s0] + [t for t in times if s0 <= t <= s1] + [s1]
        for a, b in zip(marks, marks[1:]):
            if b - a >= EMPTY_SECONDS:
                for p0, p1 in pieces(a, b):
                    out.append({"t0": round(p0, 2), "t1": round(p1, 2), "kind": "no_measures",
                                "text": f"Tab is on screen at {fmt(p0)}-{fmt(p1)}, but no measure was printed "
                                        f"from it."})
    return out


def number_checks(labels: list[tuple[float, int | None]], once: bool, duration: float) -> list[dict]:
    """Measure numbers of the printed measures, in print order with their play times: more measures than the
    highest number, numbers that skip ahead, and (when replays print once) numbers that come back."""
    read = [(t, n) for t, n in labels if n is not None]
    if not read:
        return []
    out = []
    top = max(n for _, n in read)
    if once and len(labels) > top + COUNT_SLACK:
        out.append({"t0": 0.0, "t1": round(duration, 2), "kind": "count_over",
                    "text": f"{len(labels)} measures are printed, but the highest measure number read is {top}; "
                            f"some measures may be printed twice."})
    for (ta, a), (tb, b) in zip(read, read[1:]):
        if b > a + 1:
            out.append({"t0": round(min(ta, tb), 2), "t1": round(max(ta, tb), 2), "kind": "numbers_skipped",
                        "text": f"Measures {a + 1}-{b - 1} weren't read (between {fmt(ta)} and {fmt(tb)})."})
        elif once and b <= a:
            out.append({"t0": round(tb - 2, 2), "t1": round(tb + 2, 2), "kind": "number_again",
                        "text": f"Measure {b} is printed again at {fmt(tb)}."})
    return out
