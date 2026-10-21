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
        return (f"Read only {printed} measure{'' if printed == 1 else 's'} from {shown:.0f} s of tab on screen; "
                f"at least {shown / MAX_SECONDS_PER_MEASURE:.0f} would fit. Measures were missed, so no PDF was "
                f"written.")
    return None


def possible_duplicates(A, obs, gap: float) -> list[dict]:
    """New slots whose music looks like a slot from the last few views: an overlap that wasn't joined. Views are
    counted back along `obs`, so views of another part shown in between don't count."""
    by_index = {o.index: o for o in obs}
    place = {o.index: i for i, o in enumerate(obs)}
    out = []
    for s in A.slots:
        if s.obs not in place:
            continue  # first read in a view not in `obs`
        o = by_index[s.obs]
        for t in A.slots[:s.id]:
            if t.obs not in place or not 0 < place[s.obs] - place[t.obs] <= DUP_LOOKBACK_VIEWS:
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


def dwell(obs, A) -> list[dict]:
    """Screens held much longer per measure than usual, or showing tab but no measure at all. `A` is the main
    assembly, whose slots show which views scrolled on to new measures."""
    rates = [(o.t1 - o.t0) / len(o.cells) for o in obs if o.cells]
    if not rates:
        return []
    typical = float(np.median(rates))
    out = []
    # A view cut where the playhead jumped back shows the same screen again when it shares no measure with the view
    # before or holds only measures the view before held, so they are judged together. One that shares some
    # measures and adds new ones scrolled on, and starts its own screen
    screens: list[list] = []
    for o in obs:
        prev = screens[-1][-1] if screens else None
        slots = set(A.view_slots.get(o.index, []))
        before = set(A.view_slots.get(prev.index, [])) if prev is not None else set()
        if prev is not None and o.wrapped and o.seg == prev.seg and (not slots & before or slots <= before):
            screens[-1].append(o)
        else:
            screens.append([o])
    for screen in screens:
        t0, t1, n = screen[0].t0, screen[-1].t1, len(screen[0].cells)
        held = t1 - t0
        if held < DWELL_MIN_SECONDS:
            continue
        if not n or held / n > min(DWELL_RATIO * typical, MAX_SECONDS_PER_MEASURE):
            out.append({"t0": round(t0, 2), "t1": round(t1, 2), "kind": "few_measures",
                        "text": f"The tab at {fmt(t0)}-{fmt(t1)} stays on screen a long time for the "
                                f"{n} measure{'' if n == 1 else 's'} read from it; measures may be missing there."})
    return out


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


# A measure this many times wider than the video's typical measure probably hides a missed barline
WIDE_RATIO = 1.7


def odd_widths(obs) -> list[dict]:
    """Measures much wider than usual, where a barline was likely missed and a measure may be lost. Widths are
    counted in line gaps, so tab drawn at different sizes compares fairly. A still view gets one notice however
    many of its measures are wide; a glide gets one per wide measure, at its play time."""
    widths = [(c.x1 - c.x0) / c.gap for o in obs for c in o.cells if c.kind == "full"]
    if not widths:
        return []
    typical = float(np.median(widths))
    out = []
    for o in obs:
        for c in o.cells:
            if c.kind == "full" and (c.x1 - c.x0) / c.gap > WIDE_RATIO * typical:
                t = o.mosaic.time_at((c.x0 + c.x1) / 2) if o.mosaic is not None else (o.t0 + o.t1) / 2
                w = {"t0": round(max(t - 3, o.t0), 2), "t1": round(min(t + 3, o.t1), 2), "kind": "wide_measure",
                     "text": f"A measure around {fmt(t)} is much wider than the others; a barline may have been "
                             f"missed there."}
                if w not in out:
                    out.append(w)
    return out

# Glided measures repeat an earlier stretch of the same glide when this many in a row look alike
REPEAT_RUN = 4
# Two glided measures this alike after stretching to one width (share of unmatched ink) look like the same music
REPEAT_RESEMBLE = 0.15
# A gap between consecutive glided measures' play times this many times the median means measures were lost
GAP_RATIO = 1.6
# Tab on screen this long with no new printed measure may mean music was lost
EMPTY_SECONDS = 25.0
# Long stretches are reported in pieces this long, so each points at a findable part of the video
PIECE_SECONDS = 30.0
# The highest-numbered measure printed more than this many places after its number means some printed twice
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
            like[(i, j)] = resemble_scaled(cells[j], cells[i], gap, REPEAT_RESEMBLE) < REPEAT_RESEMBLE
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


# Neighbouring numbers read further apart than this are a misreading, not lost measures
NUMBER_JUMP_MAX = 8


def glide_number_gaps(o) -> list[dict]:
    """Places in a glide where neighbouring cells' measure numbers, as read, skip a number: a barline was missed
    and two measures print as one. Only neighbours count, since some videos number only the first measure of
    each line. One view can't vouch for its numbers, so they only warn."""
    out = []
    for c, d in zip(o.cells, o.cells[1:]):
        if c.read is None or d.read is None or not 2 <= d.read - c.read <= NUMBER_JUMP_MAX:
            continue
        ta = o.mosaic.time_at((c.x0 + c.x1) / 2)
        tb = o.mosaic.time_at((d.x0 + d.x1) / 2)
        out.append({"t0": round(ta, 2), "t1": round(tb, 2), "kind": "glide_number_gap",
                    "text": f"The measure numbers read between {fmt(ta)} and {fmt(tb)} skip from {c.read} to {d.read}; "
                            f"a barline was likely missed there, so a measure may be printed joined to another."})
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
                                "text": f"Tab is on screen at {fmt(p0)}-{fmt(p1)}, but no new measure was printed "
                                        f"from it. If the video isn't replaying earlier music here, measures are "
                                        f"missing."})
    return out


def number_checks(labels: list[tuple[float, int | None]], once: bool, duration: float) -> list[dict]:
    """Measure numbers of the printed measures, in print order with their play times: a measure printed later
    in the list than its number allows, numbers that skip ahead by more than the unread measures between them,
    and (when replays print once) numbers that come back."""
    read = [(i, t, n) for i, (t, n) in enumerate(labels) if n is not None]
    if not read:
        return []
    out = []
    # The first number read should be about one more than the measures printed before it; a much higher number means
    # the opening measures weren't printed, such as a first screen shown too briefly to read
    i0, t0, n0 = read[0]
    if n0 - i0 > 1 + COUNT_SLACK:
        out.append({"t0": round(max(0.0, t0 - 5), 2), "t1": round(t0, 2), "kind": "numbers_start",
                    "text": f"The first measure number read is {n0}, at {fmt(t0)}. Measures 1-{n0 - i0 - 1} weren't "
                            "printed; check whether the video shows them before that."})
    top = max(n for _, _, n in read)
    # The last measure carrying the top number, counted from 1 in print order
    position = max(i for i, _, n in read if n == top) + 1
    if once and position > top + COUNT_SLACK:
        out.append({"t0": 0.0, "t1": round(duration, 2), "kind": "count_over",
                    "text": f"{position - 1} measures are printed before measure {top}, the highest measure number "
                            f"read; some measures may be printed twice."})
    for (ia, ta, a), (ib, tb, b) in zip(read, read[1:]):
        if b - a - 1 > ib - ia - 1:
            which = f"Measure {a + 1} wasn't" if a + 1 == b - 1 else f"Measures {a + 1}-{b - 1} weren't"
            out.append({"t0": round(min(ta, tb), 2), "t1": round(max(ta, tb), 2), "kind": "numbers_skipped",
                        "text": f"{which} read (between {fmt(ta)} and {fmt(tb)})."})
        elif once and b <= a:
            out.append({"t0": round(tb - 2, 2), "t1": round(tb + 2, 2), "kind": "number_again",
                        "text": f"Measure {b} is printed again at {fmt(tb)}."})
    return out


def numbers_against_cells(obs) -> list[dict]:
    """Views whose trusted measure numbers disagree with the measures found between their barlines: a number read
    over a cell that the count from the view's other numbers gives another number. A barline was likely missed or
    found where there is none, so measures there may be cut in two or merged."""
    out = []
    for o in obs:
        if not o.number_conflicts:
            continue
        if o.mosaic is not None:
            times = [o.mosaic.time_at((c.x0 + c.x1) / 2) for c, _ in o.number_conflicts]
            t0, t1 = max(o.t0, min(times) - 3), min(o.t1, max(times) + 3)
        else:
            t0, t1 = o.t0, o.t1
        read = ", ".join(str(n) for _, n in o.number_conflicts[:4])
        out.append({"t0": round(t0, 2), "t1": round(t1, 2), "kind": "numbers_disagree",
                    "text": f"Measure numbers read at {fmt(t0)}-{fmt(t1)} ({read}) don't match the measures found "
                            f"between the barlines there; a measure may be cut in two or two merged into one."})
    return out


# A measure with less ink than this many square line gaps on its staffs is empty
EMPTY_INK_GAPS = 0.1


def empty(o, c) -> bool:
    """True when a measure has nothing on its staffs between its barlines: only the staff lines (a number above
    it doesn't count)."""
    s = o.systems[c.system].system
    a = max(0, int((s.notation or s.tab).top - 0.5 * c.gap) - c.y0)
    b = max(0, int(s.tab.bottom + 0.5 * c.gap) - c.y0)
    # The cell's ink box ends at its right barline and may start left of its left one
    left = c.x1 + 1 - c.ink.shape[1]
    x0, x1 = int(c.x0 - left + 0.5 * c.gap), int(c.x1 - left - 0.5 * c.gap)
    return int(c.ink[a:b, max(0, x0):max(0, x1)].sum()) < EMPTY_INK_GAPS * c.gap * c.gap


# In a segment where at least this share of screens shows measures from the screen before, a screen that shows
# none of them breaks the pattern
OVERLAP_SHARE = 0.6


def screen_breaks(A, obs) -> list[dict]:
    """Screens that share no measure with the screen before, inside a segment whose screens normally overlap: the
    video switched to another part (a different instrument's track) or jumped, or an overlap was missed."""
    pairs = [(p, o) for p, o in zip(obs, obs[1:]) if p.seg == o.seg and A.view_slots.get(p.index)
             and A.view_slots.get(o.index)]
    shared = {}
    for p, o in pairs:
        # An empty staff end joined to the next screen's first measure is no overlap of music
        before = {sid for sid, c in zip(A.view_slots[p.index], p.cells) if sid is not None and not empty(p, c)}
        shared[o.index] = bool(before & set(A.view_slots[o.index]))
    out = []
    for seg in {o.seg for _, o in pairs}:
        group = [(p, o) for p, o in pairs if o.seg == seg]
        if sum(shared[o.index] for _, o in group) < OVERLAP_SHARE * len(group):
            continue
        for p, o in group:
            # A screen that shows earlier measures again is a replay, which has notices of its own
            if shared[o.index] or any(sid is not None and A.slots[sid].obs != o.index for sid in A.view_slots[o.index]):
                continue
            if view_numbers(A, o) & view_numbers(A, p):
                why = "the screen there repeats measure numbers from the one before it, but with different music"
            else:
                why = "the screen there shares no measure with the one before it, though the screens around it overlap"
            out.append(part_switch(p.t1, o.t1, o.t0, why))
    return out


def view_numbers(A, o) -> set[int]:
    """The printed measure numbers read on a view's slots."""
    return {A.slots[sid].number for sid in A.view_slots[o.index]
            if sid is not None and A.slots[sid].number is not None}


def part_switch(t0: float, t1: float, at: float, why: str) -> dict:
    """The notice for a place where the video may switch to another part."""
    return {"t0": round(t0, 2), "t1": round(t1, 2), "kind": "part_switch",
            "text": f"The video may switch to another part or jump at {fmt(at)}: {why}. Check that the measures "
                    f"around {fmt(at)} follow on."}


# A new segment starting this soon after one at least this long carries on the same video, so a new line spacing
# or theme there may be a switch to another part; shorter segments are cuts between tab and footage
LAYOUT_SECONDS = 5.0


def layout_changes(segments, obs) -> list[dict]:
    """Places where the tab's line spacing or theme changes straight after the previous segment, both lasting at
    least LAYOUT_SECONDS: the video may switch to another part or instrument there."""
    out = []
    for si, (a, b) in enumerate(zip(segments, segments[1:]), start=1):
        same = a.polarity == b.polarity and abs(a.gap - b.gap) <= 0.05 * a.gap
        if same or b.t0 - a.t1 > LAYOUT_SECONDS or min(a.t1 - a.t0, b.t1 - b.t0) < LAYOUT_SECONDS:
            continue
        first = [o for o in obs if o.seg == si]
        t1 = first[0].t1 if first else b.t0 + LAYOUT_SECONDS
        out.append(part_switch(a.t1, t1, b.t0, "the tab's line spacing or theme changes there"))
    return out
