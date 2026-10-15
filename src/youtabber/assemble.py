"""Place each view's measure cells into one score, and decide which re-shown measures are replays.

Each view is joined to the one before it first (CONTINUE: the same measures seen again because the band
scrolled, the lower line rolled up, or a measure was previewed at the edge). Cells the join doesn't
explain are new music, or a replay of earlier slots (REVISIT), which folds only with evidence.
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from youtabber.match import WIDTH_PX, WIDTH_SHARE, resemble, resemble_scaled, same, same_part
from youtabber.observe import unmatched

# A gap with no tab at least this long before a re-shown first page means the song started over
RESTART_GAP_SECONDS = 1.5
# ...and shows music from at most this many measures after a segment's first measure
RESTART_SLACK = 3
# A playhead sample this close to a view's start or end says where playback was at that moment
PLAYHEAD_EDGE_SECONDS = 1.0
# Horizontal registration: share of overlapping ink allowed to disagree
SCROLL_MISMATCH = 0.35
# Shifts tried from the column-profile comparison, best first
PROFILE_CANDIDATES = 8
# A scroll overlap must be at least this many line gaps wide to be told from chance
MIN_OVERLAP_GAPS = 4
# A segment scrolls when this share of its view pairs line up under some shift
SCROLL_SEGMENT_SHARE = 0.8
# Cells overlap after registration when they share this share of the narrower one's width
CELL_OVERLAP = 0.6
# A view's playhead counts as tracked when it was found in this share of the view's frames
TRACKED_SHARE = 0.5
# Line gaps left out at the start of a system's first measure when comparing it (clef, bracket, key)
SYSTEM_START_GAPS = 3.0
# Two sightings this alike (share of unmatched ink) can lend a measure number from one to the other
BORROW_RESEMBLE = 0.05
# Glided measures this alike after scaling (share of unmatched ink) count as the same measure in a run
GLIDE_RESEMBLE = 0.15
GLIDE_CONTINUE = 0.4
# A glide counts as played again when this share of its measures align with an earlier glide
GLIDE_RESTART_SHARE = 0.6
NAV_WORDS = ("D.S", "DS.", "D.C", "DC.", "CODA", "FINE", "SEGNO")


@dataclass
class Obs:
    """One still view, ready for assembly."""
    index: int
    t0: float
    t1: float
    cells: list            # cells in reading order
    playhead: list         # (t, x, y) crop coordinates
    seg: int
    offset: tuple[int, int]
    plate: np.ndarray | None = None
    words: list[str] | None = None
    wrapped: bool = False
    band: np.ndarray | None = None       # ink without staff lines, panel only (crop coordinates)
    tall: np.ndarray | None = None       # the band's marks at least a few pixels tall (no staff-line remnants)
    fixed: np.ndarray | None = None      # ink that sits still on screen in every view (a fixed clef panel)
    rect: tuple = (0, 0, 0, 0)
    single: bool = False                 # one system on screen
    bars: list = field(default_factory=list)
    frames: int = 0                      # frames that went into the view
    repeat_end: bool = False             # shows an end-repeat sign
    mosaic: object = None                # the glide mosaic this view was read from, if any
    all_cells: list = field(default_factory=list)   # every parsed cell, before narrow ones are dropped
    contrast: float = 0.0                # paper against the darkest ink, to spot views caught in a fade
    start: int | None = None             # number of the view's first cell, when its numbers count up
    rows: tuple[int, int] = (0, 0)       # crop rows a glide mosaic was built from
    image: np.ndarray | None = None      # the whole plate or mosaic that cell coordinates refer to
    systems: list = field(default_factory=list)   # the view's parsed systems, top to bottom
    print_rect: tuple = (0, 0, 0, 0)     # paper area to print from (x0, y0, x1, y1), crop coordinates


@dataclass
class Slot:
    id: int
    first: object
    obs: int
    number: int | None


@dataclass
class Decision:
    t: float
    kind: str              # "join", "fold"
    chosen: str
    alternatives: list[str]
    basis: str
    resolved: bool


@dataclass
class Assembly:
    slots: list[Slot] = field(default_factory=list)
    order: list[int] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    unresolved_folds: list[tuple[float, float, int, int]] = field(default_factory=list)
    view_slots: dict = field(default_factory=dict)
    sightings: dict = field(default_factory=dict)
    replay_next: int | None = None       # the slot a replay under way reaches next
    segment_starts: set = field(default_factory=set)
    seen_segments: set = field(default_factory=set)


def number_views(obs: list[Obs], read) -> str:
    """Read measure numbers and keep only those that count up consistently within a view.

    Every cell's number is read; the view's numbers are trusted only when at least two readings agree
    on one starting number for the run of cells. Returns "numbered" or "unnumbered".
    """
    consistent = checked = 0
    for o in obs:
        reads = []
        for i, c in enumerate(o.cells):
            c.number = None
            if c.number_crop is not None:
                n = read(c.number_crop)
                if n is not None:
                    reads.append((i, n))
        o.start = None
        if len(reads) < 2:
            continue
        checked += 1
        start, votes = Counter(n - i for i, n in reads).most_common(1)[0]
        if votes >= 2 and votes >= 0.6 * len(reads):
            consistent += 1
            o.start = start
    numbered = checked >= 3 and consistent >= 0.6 * checked
    for o in obs:
        if numbered and o.start is not None:
            for i, c in enumerate(o.cells):
                c.number = o.start + i
    if numbered:
        borrow_numbers(obs)
    return "numbered" if numbered else "unnumbered"


def borrow_numbers(obs: list[Obs]) -> None:
    """Number the cells of a view whose numbers couldn't be read (a highlight over them, a fade) from the
    neighboring views: cells that look like numbered cells there must agree on one starting number."""
    for k, o in enumerate(obs):
        if o.start is not None or not o.cells:
            continue
        votes = Counter()
        for m in (k - 1, k + 1):
            if not 0 <= m < len(obs) or obs[m].start is None or obs[m].seg != o.seg:
                continue
            for i, c in enumerate(o.cells):
                for d in obs[m].cells:
                    if d.number is not None and resemble(c, d, o.cells[0].gap, any_kind=True) < BORROW_RESEMBLE:
                        votes[d.number - i] += 1
        if votes:
            start, n = votes.most_common(1)[0]
            if n >= 2 and n > sum(votes.values()) / 2:
                o.start = start
                for i, c in enumerate(o.cells):
                    c.number = start + i


def tracked(o: Obs) -> bool:
    """True when the playhead was found in most of a view's frames, so its ends can be trusted."""
    return len(o.playhead) >= TRACKED_SHARE * max(o.frames, 1)


def playhead_at(o: Obs, at_start: bool):
    """(x, y) of the playhead at the start or end of a view, if it was seen there."""
    if not o.playhead:
        return None
    t, x, y = o.playhead[0] if at_start else o.playhead[-1]
    if (t - o.t0 if at_start else o.t1 - t) > PLAYHEAD_EDGE_SECONDS:
        return None
    return x, y


def cell_at(o: Obs, xy) -> int | None:
    if xy is None:
        return None
    x, y = xy
    for i, c in enumerate(o.cells):
        if c.x0 - 2 <= x <= c.x1 + 2 and abs(y - c.mid_y) < 4 * c.gap:
            return i
    return None


def scroll_shifts(p: Obs, o: Obs, gap: float) -> list[tuple[int, float]]:
    """Horizontal shifts dx (content of o sits dx pixels left of where it was in p) that line the two
    views' music up, best first, with their mismatch scores. Candidates come from barline pairs."""
    if not (p.single and o.single) or p.band is None or o.band is None or p.band.shape != o.band.shape:
        return []
    cands = {int(a - b) for a in p.bars for b in o.bars if a - b >= 2 * gap}
    # When no barline shows in both views (the overlap is one measure cut at both ends), shifts come from
    # comparing the two views' column profiles of marks: the closest profiles over the overlap
    pa = (p.tall & ~p.fixed).astype(np.float32).sum(axis=0)
    pb = (o.tall & ~o.fixed).astype(np.float32).sum(axis=0)
    scores = []
    for dx in range(int(2 * gap), len(pa) - int(2 * gap), 2):
        lo, hi = max(p.rect[0], o.rect[0] + dx), min(p.rect[2], o.rect[2] + dx)
        if hi - lo < 2 * gap:
            continue
        a, b = pa[lo:hi], pb[lo - dx:hi - dx]
        mass = a.sum() + b.sum()
        if mass > 0:
            scores.append((float(np.abs(a - b).sum() / mass), dx))
    for _, dx in sorted(scores)[:PROFILE_CANDIDATES]:
        cands.add(dx)
    cands = sorted(cands)
    scored = []
    for dx0 in cands:
        best = None
        for dx in range(dx0 - 3, dx0 + 4):
            # Columns both panels show once o is moved right by dx
            lo, hi = max(p.rect[0], o.rect[0] + dx), min(p.rect[2], o.rect[2] + dx)
            if hi - lo < MIN_OVERLAP_GAPS * gap:
                continue
            a = (p.tall & ~p.fixed)[:, lo:hi]
            b = (o.tall & ~o.fixed)[:, lo - dx:hi - dx]
            total = int(a.sum() + b.sum())
            if total < 20 * gap:
                continue
            score = float(unmatched(a, b, 2).sum()) / total
            if best is None or score < best[1]:
                best = (dx, score)
        if best is not None and best[1] <= SCROLL_MISMATCH:
            scored.append(best)
    scored.sort(key=lambda s: s[1])
    out = []
    for dx, s in scored:
        if all(abs(dx - d) > gap for d, _ in out) and keeps_widths(p, o, dx):
            out.append((dx, s))
    return out


def keeps_widths(p: Obs, o: Obs, dx: int) -> bool:
    """True when every whole measure the shift lines up has the same width in both views.

    Scrolling moves the page without redrawing it; a page that only looks alike (the same music printed
    again elsewhere in the score) is laid out differently. Only measures inside the columns both views
    show are compared, since a measure running past them can't be checked for a barline missed there.
    """
    lo, hi = max(p.rect[0], o.rect[0] + dx), min(p.rect[2], o.rect[2] + dx)
    for i, j in scroll_links(p, o, dx).items():
        a, b = o.cells[i], p.cells[j]
        if a.kind != "full" or b.kind != "full" or min(a.x0 + dx, b.x0) < lo or max(a.x1 + dx, b.x1) > hi:
            continue
        wa, wb = a.x1 - a.x0, b.x1 - b.x0
        if abs(wa - wb) > max(WIDTH_PX, WIDTH_SHARE * max(wa, wb)):
            return False
    return True


def choose_shift(p: Obs, o: Obs, shifts, share: float, use: set[str], gap: float):
    """(dx or None, basis, resolved): whether o is p scrolled by one of the shifts, or a new page.

    The playhead decides when it was seen at both ends: after a scroll it carries on from where it was,
    which in o's coordinates is (where it was in p) - dx. Without it, a segment where nearly every pair
    lines up is taken to scroll.
    """
    if "playhead" in use and tracked(p) and tracked(o):
        a, b = playhead_at(p, at_start=False), playhead_at(o, at_start=True)
        if a is not None and b is not None:
            widths = [c.x1 - c.x0 for c in p.cells if c.kind == "full"] or [4 * gap]
            reach = 1.5 * float(np.median(widths))
            fits = [dx for dx, _ in shifts if -gap <= (b[0] + dx) - a[0] <= reach and abs(b[1] - a[1]) < 3 * gap]
            if len(fits) == 1:
                return fits[0], "playhead", True
            if not fits:
                return None, "playhead", True
    if share >= SCROLL_SEGMENT_SHARE:
        return shifts[0][0], "segment_scrolls", len(shifts) == 1
    return None, "prior", False


def scroll_links(p: Obs, o: Obs, dx: int) -> dict[int, int]:
    """For a horizontal shift, which cell of p each cell of o shows again."""
    links = {}
    for i, c in enumerate(o.cells):
        a0, a1 = c.x0 + dx, c.x1 + dx
        for j, d in enumerate(p.cells):
            ov = min(a1, d.x1) - max(a0, d.x0)
            if ov >= CELL_OVERLAP * min(c.x1 - c.x0, d.x1 - d.x0):
                links[i] = j
                break
    return links


def same_cell(a, b, gap: float) -> bool:
    """Two sightings of one measure: compared whole when cut the same way, else on the part both show."""
    if a.index == 0 and b.index == 0 and a.kind != "tail" and b.kind != "tail":
        # Both start a system: the clef, bracket and first barline there are drawn differently from one
        # screen to the next, so compare the music after them, from the right barline back
        part, whole = (a, b) if a.ink.shape[1] <= b.ink.shape[1] else (b, a)
        if whole.ink.shape[1] - part.ink.shape[1] > 3 * gap:
            return False
        return same_part(part, whole, gap, "right", trim=SYSTEM_START_GAPS)
    if a.kind == b.kind:
        return same(a, b, gap)
    if "tail" in (a.kind, b.kind):
        part, whole = (a, b) if a.kind == "tail" else (b, a)
        return whole.kind == "full" and same_part(part, whole, gap, "left")
    part, whole = (a, b) if a.kind == "lead" else (b, a)
    return same_part(part, whole, gap, "right")


def sequence_links(p: Obs, o: Obs, gap: float) -> list[dict[int, int]]:
    """Possible joins where the last k cells of p are the first k cells of o (largest k first).

    The first and last cell may be partial sightings (a preview cut by the panel edge, or a measure whose
    start is hidden); those are compared on the part both show.
    """
    out = []
    n = min(len(p.cells), len(o.cells))
    for k in range(n, 0, -1):
        ok = True
        for i in range(k):
            a, b = p.cells[len(p.cells) - k + i], o.cells[i]
            ok = same_cell(a, b, gap)
            if not ok:
                break
        if ok:
            out.append({i: len(p.cells) - k + i for i in range(k)})
    return out


def choose_join(p: Obs, o: Obs, options: list, use: set[str]) -> tuple[int, str, bool]:
    """Pick one join among options [(label, links)] (an empty links dict means no overlap)."""
    if len(options) == 1:
        return 0, "only", True
    if "numbers" in use:
        fits = []
        for n, (_, links) in enumerate(options):
            nums = [(o.cells[i].number, p.cells[j].number) for i, j in links.items()]
            if nums and all(a is not None and a == b for a, b in nums):
                fits.append(n)
        if len(fits) == 1:
            return fits[0], "numbers", True
    if "playhead" in use and tracked(p) and tracked(o):
        end = cell_at(p, playhead_at(p, at_start=False))
        begin = cell_at(o, playhead_at(o, at_start=True))
        if end is not None and begin is not None:
            fits = []
            for n, (_, links) in enumerate(options):
                if begin in links:
                    ok = links[begin] in (end, end + 1)
                else:
                    # Playback moved on to music the previous view didn't show
                    ok = end >= len(p.cells) - 1 and all(i < begin for i in links)
                if ok:
                    fits.append(n)
            if len(fits) == 1:
                return fits[0], "playhead", True
    # Without evidence: an overlap of several measures, or of a measure cut by the panel edge, is a roll-up
    # or a preview; one whole measure alike on both sides is as likely a repeated measure on a new page
    first = options[0][1]
    strong = len(first) >= 2 or any(p.cells[j].kind != "full" or o.cells[i].kind != "full" for i, j in first.items())
    return (0 if strong else len(options) - 1), "prior", False


def find_run(slots: list[Slot], cells: list, gap: float, before_view: int, expect: int | None = None) -> int | None:
    """Slot j, first seen on an earlier view, where cells match slots j, j+1, ... in order: `expect` if it
    fits (a replay going on), else the first that does."""
    order = ([expect] if expect is not None else []) + list(range(len(slots)))
    for j in order:
        if j + len(cells) > len(slots) or slots[j + len(cells) - 1].obs >= before_view:
            if j == expect:
                continue
            break
        if all(same_cell(slots[j + i].first, c, gap) for i, c in enumerate(cells)):
            return j
        # Two glides are drawn by different mosaics; a run of several measures alike is the same music
        # Inside a replay under way a looser likeness carries it on, since some glided measures are drawn
        # slightly differently in each pass
        limit = GLIDE_CONTINUE if j == expect else GLIDE_RESEMBLE
        if ((len(cells) >= 3 or j == expect)
                and all(getattr(c, "glided", False) and getattr(slots[j + i].first, "glided", False)
                        and resemble_scaled(slots[j + i].first, c, gap) < limit for i, c in enumerate(cells))):
            return j
    return None


def navigation_before(o: Obs | None, words_of) -> str | None:
    """A D.S. / D.C. / Coda / Fine marking on the page played just before a jump back."""
    if o is None:
        return None
    if o.words is None:
        o.words = words_of(o)
    for w in o.words:
        u = w.upper().replace(" ", "")
        for key in NAV_WORDS:
            if key in u:
                return w
    return None


def replay_evidence(o, run, j, A, gap_before, prev, use, words_of, scrolling) -> tuple[bool, str, bool]:
    """(fold, basis, resolved) for cells that repeat slots j, j+1, ...

    Folding needs evidence that the video went back to the same place in the score: equal printed
    measure numbers, a jump back inside a scrolling band, a D.S./D.C. marking on the page played just
    before, or a restart from the first measure after a break.
    """
    first = run[0]
    if "numbers" in use and first.number is not None and A.slots[j].number is not None:
        return first.number == A.slots[j].number, "numbers", True
    if scrolling:
        return True, "jump_back", True
    if "navigation" in use:
        if prev is not None and prev.repeat_end:
            return True, "repeat_sign", True
        word = navigation_before(prev, words_of)
        if word:
            return True, f"navigation:{word}", True
    # A view right after a break that shows the opening measures of an earlier segment again (an intro or
    # count-in may be cut differently, so within a few measures of its start) is the song starting over
    near_start = j <= RESTART_SLACK or any(0 <= j - s <= RESTART_SLACK for s in A.segment_starts)
    if near_start and gap_before >= RESTART_GAP_SECONDS:
        return True, "restart", True
    return False, "none", False


def assemble(obs: list[Obs], gap: float, fold_unresolved: bool, use: set[str], words_of) -> Assembly:
    """Place views into the score. `use` names the evidence allowed ("numbers", "playhead", "navigation");
    `fold_unresolved` says how to read a re-shown run with no evidence (the main reading prints it again)."""
    A = Assembly()
    by_index = {o.index: o for o in obs}
    prev: Obs | None = None
    prev_slots: list = []
    last_end = None
    scrolled = 0
    # Registration of every consecutive pair, and the share of pairs per segment that line up
    shift_cache: dict = {}
    pairs: dict = {}
    for p, o in zip(obs, obs[1:]):
        if p.seg == o.seg and not o.wrapped and p.cells and o.cells:
            shift_cache[o.index] = scroll_shifts(p, o, gap)
            pairs.setdefault(o.seg, []).append(bool(shift_cache[o.index]))
    scroll_share = {s: sum(v) / len(v) for s, v in pairs.items() if v}
    for o in obs:
        cells = o.cells
        # Time without tab on screen before this view; a view with tab but no usable cells still counts
        gap_before = (o.t0 - last_end) if last_end is not None else 0.0
        last_end = o.t1
        if not cells:
            continue
        slot_of: list = [None] * len(cells)
        if "numbers" in use and all(c.number is not None for c in cells):
            place_numbered(o, slot_of, A, prev, prev_slots, gap)
        else:
            links: dict = {}
            if prev is not None and prev.seg == o.seg and not o.wrapped:
                options = []
                shifts = shift_cache.get(o.index, [])
                if shifts:
                    dx, basis, resolved = choose_shift(prev, o, shifts, scroll_share.get(o.seg, 0.0), use, gap)
                    if dx is not None:
                        links = scroll_links(prev, o, dx)
                        scrolled += 1
                    A.decisions.append(Decision(o.t0, "join", f"shift {dx}" if dx is not None else "page turn",
                                                [f"shift {d}" for d, _ in shifts if d != dx]
                                                + (["page turn"] if dx is not None else []), basis, resolved))
                if not links:
                    options = [(f"overlap {len(lk)}", lk) for lk in sequence_links(prev, o, gap)]
                if options:
                    options.append(("none", {}))
                    n, basis, resolved = choose_join(prev, o, options, use)
                    links = options[n][1]
                    if len(options) >= 2:
                        A.decisions.append(Decision(o.t0, "join", options[n][0],
                                                    [lab for m, (lab, _) in enumerate(options) if m != n],
                                                    basis, resolved))
            for i, j in links.items():
                slot_of[i] = prev_slots[j]
            # A scrolling band that showed earlier music without scrolling onto it jumped back
            place_rest(o, slot_of, A, gap, gap_before, prev, use, words_of, fold_unresolved, scrolled > 0 and not links)
            if "numbers" in use:
                carry_numbers(slot_of, A)
        A.view_slots[o.index] = list(slot_of)
        prev, prev_slots = o, slot_of
    for vi, sl in A.view_slots.items():
        for c, sid in zip(by_index[vi].cells, sl):
            if sid is not None:
                A.sightings.setdefault(sid, []).append((vi, c))
    A.order = [s.id for s in A.slots]
    return A


def carry_numbers(slot_of, A) -> None:
    """In a view without readable numbers, measures next to a numbered one in reading order take the
    following (or preceding) numbers, so a later view with numbers finds them."""
    known = [(i, A.slots[s].number) for i, s in enumerate(slot_of) if s is not None and A.slots[s].number is not None]
    if not known:
        return
    i0, n0 = known[0]
    taken = {s.number for s in A.slots if s.number is not None}
    for i, s in enumerate(slot_of):
        if s is not None and A.slots[s].number is None and n0 + (i - i0) not in taken:
            A.slots[s].number = n0 + (i - i0)


def place_numbered(o, slot_of, A, prev, prev_slots, gap) -> None:
    """Map each cell to the slot with its measure number, making slots for numbers not seen before.

    A number seen on an earlier view (not the one just before, whose overlap is expected) means the video
    went back: a replay with printed-number evidence, folded.
    """
    by_number = {s.number: s.id for s in A.slots if s.number is not None}
    replayed = False
    for i, c in enumerate(o.cells):
        sid = by_number.get(c.number)
        if sid is None and prev is not None:
            # The view before may have had no readable numbers; a measure it showed takes this number
            for pc, ps in zip(prev.cells, prev_slots):
                if ps is not None and A.slots[ps].number is None and same_cell(pc, c, gap):
                    sid = ps
                    A.slots[ps].number = c.number
                    by_number[c.number] = ps
                    break
        if sid is None:
            sid = new_slot(A, c, o)
            by_number[c.number] = sid
        elif sid not in prev_slots:
            replayed = True
        slot_of[i] = sid
    if replayed:
        A.decisions.append(Decision(o.t0, "fold", "fold", ["new"], "numbers", True))


def place_rest(o, slot_of, A, gap, gap_before, prev, use, words_of, fold_unresolved, scrolling) -> None:
    """Place the cells the join with the previous view didn't explain."""
    if all(s is None for s in slot_of) and glide_restart(o, slot_of, A, gap, gap_before):
        return
    i = 0
    cells = o.cells
    while i < len(cells):
        if slot_of[i] is not None:
            i += 1
            continue
        # A run of unexplained cells may repeat earlier slots
        run = []
        k = i
        while k < len(cells) and slot_of[k] is None and len(run) < 4:
            run.append(cells[k])
            k += 1
        # A single cell is enough to carry on a replay already under way
        single_ok = len(cells) == 1 or (A.replay_next is not None and A.replay_next < len(A.slots))
        j = find_run(A.slots, run, gap, o.index, A.replay_next) if len(run) >= 2 or (run and single_ok) else None
        if j is not None and len(run) == 1 and len(cells) > 1 and j != A.replay_next:
            j = None
        if j is not None:
            if A.replay_next == j:
                # The replay already under way goes on
                fold, basis, resolved = True, "replay_continues", True
            else:
                fold, basis, resolved = replay_evidence(o, run, j, A, gap_before, prev, use, words_of, scrolling)
            if not resolved:
                fold = fold_unresolved
                A.unresolved_folds.append((o.t0, o.t1, j, j + len(run) - 1))
            A.decisions.append(Decision(o.t0, "fold", "fold" if fold else "new", ["new" if fold else "fold"],
                                        basis, resolved))
            if fold and o.mosaic is not None and getattr(A.slots[j].first, "glided", False):
                # A glide replaying an earlier glide: line the rest of it up with that glide as a whole
                align_glide(o, slot_of, i, j, A, gap)
                return
            for m, c in enumerate(run):
                slot_of[i + m] = j + m if fold else new_slot(A, c, o)
            A.replay_next = j + len(run) if fold else None
            i += len(run)
            continue
        slot_of[i] = new_slot(A, cells[i], o)
        A.replay_next = None
        i += 1


def glide_alignment(cells, targets, A, gap) -> list[tuple[int, int]]:
    """Best in-order pairing of glided cells with earlier glided slots (index into cells, slot id)."""
    n, m = len(cells), len(targets)
    like = np.array([[resemble_scaled(A.slots[s].first, c, gap) < GLIDE_CONTINUE for s in targets] for c in cells])
    score = np.zeros((n + 1, m + 1), np.int32)
    for a in range(n - 1, -1, -1):
        for b in range(m - 1, -1, -1):
            score[a, b] = max(score[a + 1, b], score[a, b + 1], score[a + 1, b + 1] + (1 if like[a, b] else 0))
    pairs = []
    a = b = 0
    while a < n and b < m:
        if like[a, b] and score[a, b] == score[a + 1, b + 1] + 1:
            pairs.append((a, targets[b]))
            a += 1
            b += 1
        elif score[a + 1, b] >= score[a, b + 1]:
            a += 1
        else:
            b += 1
    return pairs


def align_glide(o, slot_of, i0, j0, A, gap, basis: str = "glide_alignment") -> None:
    """Fold a replayed glide onto the earlier one by the best in-order alignment of their measures.

    Each pass is read from its own mosaic, so a measure may be cut a little differently or be missing in
    one of them; aligning the whole sequences keeps one bad measure from breaking the replay. Measures of
    the replay that align with nothing are not printed again; measures after the earlier glide ends are new.
    """
    first_obs = A.slots[j0].obs
    targets = [s.id for s in A.slots[j0:] if s.obs == first_obs]
    cells = o.cells[i0:]
    pairs = glide_alignment(cells, targets, A, gap)
    last = -1
    for a, s in pairs:
        slot_of[i0 + a] = s
        last = a
    for k in range(last + 1, len(cells)):
        slot_of[i0 + k] = new_slot(A, cells[k], o)
    A.decisions.append(Decision(o.t0, "fold", "fold", ["new"], f"{basis}:{len(pairs)}/{len(cells)}", True))
    A.replay_next = None


def glide_restart(o, slot_of, A, gap, gap_before) -> bool:
    """A glide after a break that plays an earlier glide again from near its start folds onto it."""
    if o.mosaic is None or gap_before < RESTART_GAP_SECONDS:
        return False
    for start in sorted(A.segment_starts):
        if not getattr(A.slots[start].first, "glided", False):
            continue
        targets = [s.id for s in A.slots[start:] if s.obs == A.slots[start].obs]
        pairs = glide_alignment(o.cells, targets, A, gap)
        if (len(pairs) >= GLIDE_RESTART_SHARE * min(len(o.cells), len(targets))
                and pairs[0][0] <= RESTART_SLACK and targets.index(pairs[0][1]) <= RESTART_SLACK):
            align_glide(o, slot_of, 0, start, A, gap, "restart")
            return True
    return False


def new_slot(A, c, o) -> int:
    sid = len(A.slots)
    A.slots.append(Slot(sid, c, o.index, c.number))
    if o.seg not in A.seen_segments:
        # The first measure of a segment: a later return to it after a break is a restart
        A.seen_segments.add(o.seg)
        A.segment_starts.add(sid)
    return sid
