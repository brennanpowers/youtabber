"""Checks on the assembled score that turn likely mistakes into warnings next to the measures concerned."""
import numpy as np

from youtabber.assemble import cell_at, glided_on, measure_time, playhead_track
from youtabber.match import resemble, resemble_scaled
from youtabber.parse import JOINED_SLACK_PX
from youtabber.track import pairs

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


def too_few(obs, printed: int, A) -> str | None:
    """A message when the whole video yields far fewer measures than its time with tab on screen allows and the
    reading itself shows signs of lost measures. A lesson can hold a few measures on screen for minutes, so time
    alone doesn't refuse."""
    shown = sum(o.t1 - o.t0 for o in obs)
    if shown >= 2 * MAX_SECONDS_PER_MEASURE and printed < shown / MAX_SECONDS_PER_MEASURE and not settled(obs, A):
        return (f"Read only {printed} measure{'' if printed == 1 else 's'} from {shown:.0f} s of tab on screen; "
                f"at least {shown / MAX_SECONDS_PER_MEASURE:.0f} would fit. Measures were missed, so no PDF was "
                f"written.")
    return None


def settled(obs, A) -> bool:
    """True when nothing in the reading points to lost measures: every view reads measures, every join between
    screens and every fold of music shown again is settled, and no measure is much wider than the others."""
    return (all(o.cells for o in obs) and all(d.resolved for d in A.decisions if d.kind in ("join", "fold"))
            and not odd_widths(obs))


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
    """Screens held much longer per measure than usual, or showing tab but no measure at all. `A` is the stream's
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


def judged_width(o, c) -> float | None:
    """A cell's width in line gaps, for a cell that can't be much wider than a measure: one between two barlines,
    or on a still screen a system's first cell cut by the panel edge before a barline (so no clef and key widen it),
    or its last cell up to its last mark (blank staff may run on after the last measure). None for other cells."""
    if c.kind == "full":
        return (c.x1 - c.x0) / c.gap
    if o.mosaic is not None:
        return None
    if c.kind == "tail":
        inked = np.flatnonzero(c.ink.any(axis=0))
        return (int(inked[-1]) + 1) / c.gap if inked.size else None
    # A staff with no barline at all is one lead cell as wide as the staff, however many measures it holds
    if c.kind == "lead" and not c.complete and o.systems[c.system].bars:
        return (c.x1 - c.x0) / c.gap
    return None


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
            width = judged_width(o, c)
            if width is not None and width > WIDE_RATIO * typical:
                t = o.mosaic.time_at((c.x0 + c.x1) / 2) if o.mosaic is not None else (o.t0 + o.t1) / 2
                w = {"t0": round(max(t - 3, o.t0), 2), "t1": round(min(t + 3, o.t1), 2), "kind": "wide_measure",
                     "text": f"A measure around {fmt(t)} is much wider than the others; a barline may have been "
                             f"missed there."}
                if w not in out:
                    out.append(w)
    return out


# The playhead crossing a measure this many times slower than the typical measure of its segment crossed two
# measures: a barline between them was missed
SLOW_RATIO = 1.75
# A crossing is timed only when the playhead was first seen in each of the two measures within this share of the
# measure's width from its left barline; a view that opens with the playhead mid-measure gives no timing
ENTRY_SHARE = 0.15


def slow_measures(obs) -> list[dict]:
    """Still screens where the playhead took about two measures' time to cross one measure, so a barline was likely
    missed there. This catches a merged measure no wider than the others, as on a screen of fewer, wider bars.
    A view gets one notice at most, and only a measure at least as wide as the view's typical measure is named:
    a merged measure is wide as well as slow, while a bar played slowly is not."""
    out = []
    for seg in sorted({o.seg for o in obs}):
        still = [o for o in obs if o.seg == seg and o.mosaic is None]
        beat = measure_time(still)
        if beat is None:
            continue
        # A segment's first timed measure is often a slow count-in, so it is never named
        first = True
        for o in still:
            fulls = [c.x1 - c.x0 for c in o.cells if c.kind == "full" and c.complete]
            typical = float(np.median(fulls)) if fulls else None
            entered: dict[int, float] = {}
            for t, x, y in playhead_track(o):
                i = cell_at(o, (x, y))
                if i is not None and i not in entered:
                    c = o.cells[i]
                    # A first sample far inside the measure says nothing about when the playhead got there
                    entered[i] = t if x - c.x0 <= ENTRY_SHARE * (c.x1 - c.x0) else -1.0
            for i, t in entered.items():
                after = entered.get(i + 1, -1.0)
                c = o.cells[i]
                if t < 0 or after < 0 or not c.complete:
                    continue
                if first:
                    first = False
                    continue
                if after - t <= SLOW_RATIO * beat or typical is None or c.x1 - c.x0 < typical:
                    continue
                out.append({"t0": round(t, 2), "t1": round(after, 2), "kind": "long_measure",
                            "text": f"The measure played at {fmt(t)}-{fmt(after)} takes about "
                                    f"{(after - t) / beat:.0f} measures' time; a barline may have been missed there."})
                break
    return out


# Glided measures repeat an earlier stretch of the same glide when this many in a row look alike
REPEAT_RUN = 4
# Two glided measures this alike after stretching to one width (share of unmatched ink) look like the same music
REPEAT_RESEMBLE = 0.15
# A gap between consecutive glided measures' play times this many times the median means measures were lost
GAP_RATIO = 1.6
# Tab on screen this long with no new printed measure may mean music was lost
EMPTY_SECONDS = 25.0
# Tab on screen this long with no still view read means its measures may be missing
UNREAD_SECONDS = 4.0
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


def glide_fade(o, step: float) -> list[dict]:
    """The picture faded out while the band still moved: the measures after the last clear frame get their play
    times from the band's last speed. `step` is the usual play time from one of the view's measures to the next."""
    m = o.mosaic
    faded = float(m.times[-1])
    if m.until is None or not m.until - faded > step > 0 or m.last_move() != len(m.times) - 1:
        return []
    return [{"t0": round(faded, 2), "t1": round(m.until, 2), "kind": "glide_end",
             "text": "The picture faded out while this smooth scroll was still moving, so the last measures are "
                     "placed by its last speed, which is wrong if it stopped during the fade; check them."}]


# Speed changes of a glide that no vertical line confirms, this close in play time, are one stretch
STEP_JOIN_SECONDS = 2.0
# A glide piece whose cells cover less than this share of its mosaic has lost its tab staff somewhere
FOUND_SHARE = 0.8
# A cell at a piece boundary narrower than this share of the typical measure was likely cut by the boundary
PIECE_CUT_SHARE = 0.5


def glide_steps(o) -> list[dict]:
    """Stretches of a glide where its speed changed sharply with no barline or stem to confirm the new step:
    columns may have been dropped there (a measure cut short, two merged) or doubled."""
    times = sorted(min(max(o.mosaic.time_at(x), o.t0), o.t1) for x in o.mosaic.doubtful)
    out, run = [], []
    for t in times + [None]:
        if run and (t is None or t - run[-1] > STEP_JOIN_SECONDS):
            a, b = max(run[0] - 1, o.t0), min(run[-1] + 1, o.t1)
            out.append({"t0": round(a, 2), "t1": round(b, 2), "kind": "glide_step",
                        "text": f"The smooth scroll changed speed sharply at {fmt(a)}-{fmt(b)} where no barline "
                                f"confirms it; a measure there may be cut short, joined to the next or printed "
                                f"twice."})
            run = []
        if t is not None:
            run.append(t)
    return out


def glide_rows(o) -> list[dict]:
    """Stretches of a glide piece with no measure parsed, when the piece's measures cover less than FOUND_SHARE of
    its mosaic: the tab staff likely moved there without the piece ending."""
    spans = sorted((c.x0, c.x1) for ps in o.systems for c in ps.cells)
    width = o.mosaic.image.shape[1]
    covered, reach, holes = 0, 0, []
    for x0, x1 in spans:
        if x0 > reach:
            holes.append((reach, x0))
        covered += max(0, x1 - max(x0, reach))
        reach = max(reach, x1)
    holes.append((reach, width))
    if not spans or covered >= FOUND_SHARE * width:
        return []
    typical = float(np.median([x1 - x0 for x0, x1 in spans]))
    out = []
    for a, b in holes:
        if b - a < typical:
            continue
        ta, tb = o.mosaic.time_at(a), o.mosaic.time_at(b)
        for p0, p1 in pieces(max(ta, o.t0), max(min(tb, o.t1), ta + 1)):
            out.append({"t0": round(p0, 2), "t1": round(p1, 2), "kind": "glide_rows",
                        "text": f"No tab staff was found in the smooth scroll at {fmt(p0)}-{fmt(p1)}; measures "
                                f"played there may be missing."})
    return out


def glide_piece_pairs(obs) -> list[tuple]:
    """Consecutive pieces of one glide: views of one part, the second carrying on the first."""
    return [(a, b) for a, b in pairs(obs, key=lambda v: v.part) if glided_on(b)]


def glide_piece_cuts(obs) -> list[dict]:
    """Places where a glide is read in two pieces (the panel was redrawn) and a measure at the boundary is much
    narrower than usual: the boundary may have cut one measure in two."""
    out = []
    for a, b in glide_piece_pairs(obs):
        if not a.all_cells or not b.all_cells:
            continue
        widths = [c.x1 - c.x0 for c in a.all_cells + b.all_cells if c.kind == "full"]
        if not widths:
            continue
        typical = float(np.median(widths))
        last, first = a.all_cells[-1], b.all_cells[0]
        if min(last.x1 - last.x0, first.x1 - first.x0) < PIECE_CUT_SHARE * typical:
            ta = a.mosaic.time_at((last.x0 + last.x1) / 2)
            tb = b.mosaic.time_at((first.x0 + first.x1) / 2)
            out.append({"t0": round(min(ta, b.t0 - 1), 2), "t1": round(max(tb, b.t0 + 1), 2), "kind": "glide_piece",
                        "text": f"The tab panel was redrawn at {fmt(b.t0)} during the smooth scroll, and a measure "
                                f"there is narrower than usual; it may be printed in two parts or missing."})
    return out


def redraw_gaps(obs) -> list[dict]:
    """Places where a glide is read in two pieces and playback takes much longer than usual from the last measure
    before the redraw to the first after it: measures were likely lost at the redraw."""
    out = []
    for a, b in glide_piece_pairs(obs):
        if not a.cells or not b.cells:
            continue
        times_a = [a.mosaic.time_at((c.x0 + c.x1) / 2) for c in a.cells]
        times_b = [b.mosaic.time_at((c.x0 + c.x1) / 2) for c in b.cells]
        # The usual step comes from the piece before, or the piece after when the one before has one measure
        inside = times_a if len(times_a) > 1 else times_b
        if len(inside) < 2:
            continue
        typical = float(np.median(np.diff(inside)))
        ta, tb = times_a[-1], times_b[0]
        if typical > 0 and tb - ta > GAP_RATIO * typical:
            out.append({"t0": round(min(ta, b.t0), 2), "t1": round(max(tb, b.t0), 2), "kind": "glide_gap",
                        "text": f"The tab panel was redrawn at {fmt(b.t0)} during the smooth scroll, and playback "
                                f"takes {tb - ta:.1f} s from the last measure drawn before it (played at {fmt(ta)}) to "
                                f"the first drawn after it (at {fmt(tb)}); measures may be missing there."})
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


def unread_stretches(spans: list[tuple[float, float]], read: list[tuple[float, float]]) -> list[dict]:
    """Stretches of at least UNREAD_SECONDS inside a span of tab on screen where no still view with a staff was
    read: the screen kept changing, or the staff couldn't be found on the views held there."""
    out = []
    for s0, s1 in spans:
        edge = s0
        for a, b in sorted(read) + [(s1, s1)]:
            if a > s1:
                break
            if a - edge >= UNREAD_SECONDS:
                for p0, p1 in pieces(edge, a):
                    out.append({"t0": round(p0, 2), "t1": round(p1, 2), "kind": "unread_tab",
                                "text": f"Tab is on screen at {fmt(p0)}-{fmt(p1)}, but no still screen of it could "
                                        f"be read there; measures shown there may be missing."})
            edge = max(edge, b)
    return out


def number_checks(labels: list[tuple[float, int | None]], once: bool) -> list[dict]:
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
    if once:
        out += count_over(labels, read)
    for (ia, ta, a), (ib, tb, b) in zip(read, read[1:]):
        if b - a - 1 > ib - ia - 1:
            which = f"Measure {a + 1} wasn't" if a + 1 == b - 1 else f"Measures {a + 1}-{b - 1} weren't"
            out.append({"t0": round(min(ta, tb), 2), "t1": round(max(ta, tb), 2), "kind": "numbers_skipped",
                        "text": f"{which} read (between {fmt(ta)} and {fmt(tb)})."})
        elif once and b <= a:
            out.append({"t0": round(tb - 2, 2), "t1": round(tb + 2, 2), "kind": "number_again",
                        "text": f"Measure {b} is printed again at {fmt(tb)}."})
    return out


def count_over(labels: list[tuple[float, int | None]], read: list[tuple[int, float, int]]) -> list[dict]:
    """Where more measures are printed before a numbered measure than its number allows, by more than COUNT_SLACK:
    the first such place, and each later place where the count runs further ahead. Each notice spans the measures
    printed since the number read before, so it points at the measures that may be printed twice."""
    out = []
    ahead = COUNT_SLACK
    since = labels[0][0]
    for i, t, n in read:
        over = i + 1 - n
        if over > ahead:
            for p0, p1 in pieces(min(since, t), max(since, t)) or [(t, t)]:
                out.append({"t0": round(p0, 2), "t1": round(p1, 2), "kind": "count_over",
                            "text": f"{i} measures are printed before measure {n}, {over} more than its number "
                                    f"allows; some measures printed at {fmt(p0)}-{fmt(p1)} may be printed twice."})
            ahead = over
        since = t
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


# Two staves on one screen whose barlines stand in the same columns, in at least this share of a part's views...
SHARED_BARS_SHARE = 0.5
# ...where the upper staff's numbers are read in at least this many views and the lower staff's in at most this share
# of them, are likely two parts played together rather than one part's systems in turn
SHARED_BARS_MIN_NUMBERED, SHARED_BARS_LOWER_READS = 3, 0.1


def same_columns(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    """Whether two staves' barlines stand in the same columns, within JOINED_SLACK_PX."""
    def near(x, bars):
        return any(abs(x[0] - y[0]) <= JOINED_SLACK_PX for y in bars)
    return bool(a) and bool(b) and all(near(x, b) for x in a) and all(near(x, a) for x in b)


def shared_barlines(obs) -> list[dict]:
    """Stretches where each screen shows two tab staves with barlines in the same columns and measure numbers only
    over the upper one. Those staves are likely two parts played together, but they are read as one part's systems
    one after the other, so their measures may be printed out of order or in place of each other."""
    out = []
    groups: dict = {}
    for o in obs:
        if o.mosaic is None:
            groups.setdefault((o.seg, o.part), []).append(o)
    for views in groups.values():
        paired = [o for o in views if len(o.systems) >= 2 and same_columns(o.systems[0].bars, o.systems[1].bars)]
        if len(paired) < SHARED_BARS_SHARE * len(views):
            continue
        upper = sum(any(c.read is not None for c in o.cells if c.system == 0) for o in paired)
        lower = sum(any(c.read is not None for c in o.cells if c.system == 1) for o in paired)
        if upper < SHARED_BARS_MIN_NUMBERED or lower > SHARED_BARS_LOWER_READS * upper:
            continue
        for p0, p1 in pieces(paired[0].t0, paired[-1].t1):
            out.append({"t0": round(p0, 2), "t1": round(p1, 2), "kind": "shared_barlines",
                        "text": f"At {fmt(p0)}-{fmt(p1)} the screen shows two tab staves with their barlines lined "
                                f"up, and only the upper one has measure numbers. They may be two parts played "
                                f"together, printed here one after the other as if they were one part, so measures "
                                f"may be out of order or missing. Check these measures."})
    return out
