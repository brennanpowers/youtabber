"""Place each view's measure cells into one score, and decide which re-shown measures are replays.

Each view is joined to the one before it first (CONTINUE: the same measures seen again because the band
scrolled, the lower line rolled up, or a measure was previewed at the edge). Cells the join doesn't
explain are new music, or a replay of earlier slots (REVISIT), which folds only with evidence.
"""
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from youtabber.match import WIDTH_PX, WIDTH_SHARE, resemble, resemble_scaled, same, same_part
from youtabber.observe import WRAP_GAPS, unmatched

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
# A view refuses the numbers a look-alike neighbor lends it when one of its own reads is this many measures or fewer
# off them: the neighbor is another page of a riff
BORROW_CONFLICT = 6
# A numbered measure met again on a later screen is the same music when this alike (share of unmatched ink)
REPLAY_RESEMBLE = 0.3
# Glided measures this alike after scaling (share of unmatched ink) count as the same measure in a run
GLIDE_RESEMBLE = 0.15
GLIDE_CONTINUE = 0.4
# A glide counts as played again when this share of its measures align with an earlier glide
GLIDE_RESTART_SHARE = 0.6
# A number read this many measures or fewer off its view's count points at a missed or extra barline
CONFLICT_SPAN = 2
NAV_WORDS = ("D.S", "DS.", "D.C", "DC.", "CODA", "FINE", "SEGNO")
# Two views of a band that didn't move show the same music when at most this share of their marks finds no partner;
# what is left is cursor residue
STILL_MISMATCH = 0.2
# A held page the playhead started over, with at most this share of marks unmatched, is the same page played again;
# a looser match can be a new page drawn on the same grid
SAME_PAGE_MISMATCH = 0.1
# A measured slide is checked on the two plates this many pixels either side of it
SLIDE_REACH = 6
# Over a slide the playhead carries on from where it was by at most this many typical measure widths
CARRY_MEASURES = 1.5
# ...and its two ends lie at most this many line gaps apart in height (a highlight rides up and down with the notes)
CURSOR_DRIFT_GAPS = 5
# A segment's typical time on one measure needs at least this many measures seen played from start to end
MIN_DWELLS = 5
# A page's last measure, cut by the page edge, was played for less than this share of a measure's time
CUT_DWELL = 0.85
# ...or a whole measure before it was played for at least this share, so playback had moved on to the cut one
MOVED_ON = 0.7
# A measure cut by the page edge is narrower than this share of the page's typical whole measure
CUT_WIDTH = 0.95
# A cut measure this wide, or played this long, is close to a whole measure that only lost its barline to the edge
EDGE_WIDTH = 0.75
EDGE_DWELL = 0.6


@dataclass
class Obs:
    """One still view or one glide mosaic, ready for assembly."""
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
    line_contrast: float = 0.0           # paper against the tab staff lines, to spot views caught in a fade
    start: int | None = None             # number of the view's first cell, when its numbers count up
    rows: tuple[int, int] = (0, 0)       # crop rows a glide mosaic was built from
    image: np.ndarray | None = None      # the whole plate or mosaic that cell coordinates refer to
    systems: list = field(default_factory=list)   # the view's parsed systems, top to bottom
    print_rect: tuple = (0, 0, 0, 0)     # paper area to print from (x0, y0, x1, y1), crop coordinates
    number_conflicts: list = field(default_factory=list)   # (cell, number read) where the count gives another
    move: object = None                  # how the band moved from the view before, measured on the frames between
    track: list | None = None            # the playhead samples on the view's music, found when first asked for
    restarted: bool = False              # the band held still while the playhead went back over the page
    part: int = 0                        # id of the part it shows, its place in the reading's parts
    stream: int = 0                      # views of one stream are numbered and joined together
    first_piece: bool = True             # a glide's first and last pieces hold its start and its end
    last_piece: bool = True
    counter: list = field(default_factory=list)   # (start, end, measure) runs of a bar counter shown with a glide


@dataclass
class Slot:
    id: int
    first: object
    obs: int
    number: int | None
    stream: int = 0        # the stream of the views it is read in


@dataclass
class Decision:
    t: float
    kind: str              # "join", "fold"
    chosen: str
    alternatives: list[str]
    basis: str
    resolved: bool
    view: int | None = field(default=None, compare=False)   # index of the view it decides about


@dataclass
class Assembly:
    slots: list[Slot] = field(default_factory=list)
    order: list[int] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    unresolved_folds: list[tuple[float, float, int, int]] = field(default_factory=list)
    view_slots: dict = field(default_factory=dict)
    view_links: dict = field(default_factory=dict)   # view index -> cells the join tied to the view before
    sightings: dict = field(default_factory=dict)
    replay_next: int | None = None       # the slot a replay under way reaches next
    segment_starts: set = field(default_factory=set)
    seen_segments: set = field(default_factory=set)
    edge_joins: list = field(default_factory=list)   # (p, o) indexes of playhead joins that may leave out notes
    sliding: dict = field(default_factory=dict)   # segment -> its band moves on by measured slides
    stream_orders: dict = field(default_factory=dict)   # stream -> its own slots in print order


def number_views(obs: list[Obs], read) -> str:
    """Read measure numbers and keep only those that count up consistently within a view.

    Every cell's number is read; the view's numbers are trusted only when at least two readings agree
    on one starting number for the run of cells. Returns "numbered" or "unnumbered".
    """
    consistent = checked = 0
    all_reads = []
    for o in obs:
        reads = []
        all_reads.append(reads)
        for i, c in enumerate(o.cells):
            c.number = c.read = None
            if c.number_crop is not None:
                n = c.read = read(c.number_crop)
                if n is not None:
                    reads.append((i, n))
        o.start = None
        o.number_conflicts = []
        if len(reads) < 2:
            continue
        checked += 1
        start, votes = Counter(n - i for i, n in reads).most_common(1)[0]
        if votes >= 2 and votes >= 0.6 * len(reads):
            consistent += 1
            o.start = start
    numbered = checked >= 3 and consistent >= 0.6 * checked
    for k, o in enumerate(obs):
        if o.start is not None and not fits_neighbours(obs, k):
            o.start = None
    for o, reads in zip(obs, all_reads):
        if numbered and o.start is not None:
            for i, c in enumerate(o.cells):
                c.number = o.start + i
            # A number a measure or two off the count means a barline was missed or added; one further off is a
            # misread, such as one digit of a two-digit number or a fret number taken for one
            o.number_conflicts = [(o.cells[i], n) for i, n in reads if 0 < abs(n - o.start - i) <= CONFLICT_SPAN]
    if numbered:
        borrow_numbers(obs)
    return "numbered" if numbered else "unnumbered"


def fits_neighbours(obs: list[Obs], k: int) -> bool:
    """False when view k's start carries on from neither the nearest numbered view before nor after it in its layout,
    while a number read on one of its own cells does: the start was voted by misreads."""
    o = obs[k]
    prev = next((obs[m] for m in range(k - 1, -1, -1) if obs[m].seg == o.seg and obs[m].start is not None), None)
    nxt = next((obs[m] for m in range(k + 1, len(obs)) if obs[m].seg == o.seg and obs[m].start is not None), None)

    def agrees(start: int) -> bool:
        after_prev = prev is not None and prev.start <= start <= prev.start + len(prev.cells)
        before_next = nxt is not None and nxt.start - len(o.cells) <= start <= nxt.start
        return after_prev or before_next

    if prev is None and nxt is None or agrees(o.start):
        return True
    # A read a measure or two off the start is a missed or added barline, which number_conflicts reports, not a
    # sign that the start is wrong
    return not any(agrees(c.read - i) for i, c in enumerate(o.cells)
                   if c.read is not None and abs(c.read - i - o.start) > CONFLICT_SPAN)


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
            if n >= 2 and n > sum(votes.values()) / 2 and not read_against(o, start):
                o.start = start
                for i, c in enumerate(o.cells):
                    c.number = start + i


def read_against(o: Obs, start: int) -> bool:
    """True when the numbers read on a view's own cells say it starts a few measures from `start`, and none agrees
    with it: a page of a riff looks like the page before it. A number far off is a misread, such as one digit."""
    off = [c.read - i - start for i, c in enumerate(o.cells) if c.read is not None]
    return bool(off) and 0 not in off and any(abs(d) <= BORROW_CONFLICT for d in off)


def playhead_track(o: Obs) -> list:
    """The view's playhead samples that lie on its music: inside the panel and near its staffs, without a sample
    that jumps away from both its neighbors (footage moving beside the panel)."""
    if o.track is not None:
        return o.track
    if not o.cells:
        o.track = []
        return o.track
    g = o.cells[0].gap
    # Nothing is played right of the last measure; left of the first, a narrow measure may have been left out
    x0, x1 = o.rect[0] - g, max(c.x1 for c in o.cells) + 2 * g
    # From a line gap above a system's top staff to a few below its tab staff's middle
    y0 = min(c.y0 + c.top_y for c in o.cells) - g
    y1 = max(c.mid_y for c in o.cells) + 3 * g
    pts = [(t, x, y) for t, x, y in o.playhead if x0 <= x <= x1 and y0 <= y <= y1]

    def far(a, b):
        return abs(a[1] - b[1]) > 4 * g or abs(a[2] - b[2]) > 2 * g
    o.track = [q for k, q in enumerate(pts)
               if not (0 < k < len(pts) - 1 and far(q, pts[k - 1]) and far(q, pts[k + 1])
                       and not far(pts[k - 1], pts[k + 1]))
               # A first or last sample away from the two next to it
               and not (k == 0 and len(pts) >= 3 and far(q, pts[1]) and not far(pts[1], pts[2]))
               and not (k == len(pts) - 1 and len(pts) >= 3 and far(q, pts[-2]) and not far(pts[-2], pts[-3]))]
    return o.track


def tracked(o: Obs) -> bool:
    """True when the playhead was found in most of a view's frames, so its ends can be trusted."""
    return len(playhead_track(o)) >= TRACKED_SHARE * max(o.frames, 1)


def playhead_at(o: Obs, at_start: bool):
    """(x, y) of the playhead at the start or end of a view, if it was seen there."""
    track = playhead_track(o)
    if not track:
        return None
    t, x, y = track[0] if at_start else track[-1]
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


def choose_shift(p: Obs, o: Obs, shifts, share: float, use: set[str], gap: float, usual: float | None = None):
    """(dx or None, basis, resolved): whether o is p scrolled by one of the shifts, or a new page.

    The playhead decides when it was seen at both ends: after a scroll it carries on from where it was,
    which in o's coordinates is (where it was in p) - dx. Next, in a segment whose band was seen sliding by
    about `usual` px, the shift nearest that is taken. Without either, a segment where nearly every pair
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
    if usual is not None:
        near = min(shifts, key=lambda s: abs(s[0] - usual))
        if abs(near[0] - usual) <= gap:
            # Settled only when the band was seen sliding by about that much here too, in several steps
            m = o.move
            drawn = [d for d, dr in zip(m.shifts, m.drawn) if dr] if m is not None and m.kind == "slide" else []
            agrees = bool(drawn) and abs(drawn[0] - near[0]) <= gap
            # A slide drawn to another of the shifts points away from the usual one, so the pick is left to the
            # rules below
            if agrees or not drawn or not any(abs(drawn[0] - dx) <= gap for dx, _ in shifts):
                return near[0], "usual_slide", agrees
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


def replay_evidence(o, run, j, A, gap_before, prev, use, words_of) -> tuple[bool, str, bool]:
    """(fold, basis, resolved) for cells that repeat slots j, j+1, ...

    Folding needs evidence that the video went back to the same place in the score: equal printed
    measure numbers, an end-repeat sign or a D.S./D.C. marking on the page played just before, a restart
    from the first measure after a break, or a sliding band cut back to its opening. A run that only looks like
    earlier music is an unresolved fold: the main reading prints it again and the folded reading folds it.
    """
    first = run[0]
    if "numbers" in use and first.number is not None and A.slots[j].number is not None:
        return first.number == A.slots[j].number, "numbers", True
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
    # A band that moves on by sliding and instead cuts back to the opening measures was sent back to the start.
    # Settled only when the playhead went from the end of the page before straight to the first measure here
    if near_start and o.move is not None and o.move.kind == "cut" and A.sliding.get(o.seg, False):
        from_end = prev is not None and cell_at(prev, playhead_at(prev, at_start=False)) == len(prev.cells) - 1
        return True, "track_restart", from_end and cell_at(o, playhead_at(o, at_start=True)) == 0
    return False, "none", False


def register(p: Obs, o: Obs, dx: int, gap: float, reach: int) -> tuple[int, float] | None:
    """(shift, mismatch) of the best registration of o on p within `reach` pixels of shift dx, or None when the
    two panels share too little music there."""
    best = None
    for d in range(dx - reach, dx + reach + 1):
        lo, hi = max(p.rect[0], o.rect[0] + d), min(p.rect[2], o.rect[2] + d)
        if hi - lo < MIN_OVERLAP_GAPS * gap:
            continue
        a = (p.tall & ~p.fixed)[:, lo:hi]
        b = (o.tall & ~o.fixed)[:, lo - d:hi - d]
        total = int(a.sum() + b.sum())
        if total < 20 * gap:
            continue
        score = float(unmatched(a, b, 2).sum()) / total
        if best is None or score < best[1]:
            best = (d, score)
    return best


def still_links(p: Obs, o: Obs) -> dict[int, int]:
    """For a band that didn't move, which cell of p each cell of o is: the one on the same staff over the same
    columns."""
    links = {}
    for i, c in enumerate(o.cells):
        for j, d in enumerate(p.cells):
            ov = min(c.x1, d.x1) - max(c.x0, d.x0)
            if abs(c.mid_y - d.mid_y) < c.gap and ov >= CELL_OVERLAP * min(c.x1 - c.x0, d.x1 - d.x0):
                links[i] = j
                break
    return links


def playhead_restarts(p: Obs, o: Obs, gap: float) -> bool | None:
    """Whether the playhead went back over the page between the end of p and the start of o (up a system, or
    back along one), or None when it wasn't seen at both."""
    a, b = playhead_at(p, at_start=False), playhead_at(o, at_start=True)
    if a is None or b is None:
        return None
    # A box or highlight cursor rides up and down a system with its notes; another system is further away
    if b[1] < a[1] - 5 * gap:
        return True
    return abs(b[1] - a[1]) < 5 * gap and b[0] < a[0] - WRAP_GAPS * gap


def track_join(p: Obs, o: Obs, gap: float, cands=()) -> tuple[dict[int, int], str] | None:
    """(links, basis) when the frames between two views show how the band moved: a slide whose measured shift lines
    the two plates up, or a band that held still while only a cursor moved. None when they don't. `cands` are the
    shifts the plates register at (`scroll_shifts`). A basis ending in "?" was not confirmed by the playhead."""
    m = o.move
    if m is None or m.kind not in ("slide", "still") or p.band is None or o.band is None or p.band.shape != o.band.shape:
        return None
    if m.kind == "slide" and abs(m.shifts[0]) >= gap:
        return slide_join(p, o, gap, cands) if p.single and o.single else None
    r = register(p, o, 0, gap, 2)
    if r is None or r[1] > STILL_MISMATCH:
        return None
    restart = playhead_restarts(p, o, gap)
    # A view is cut where the playhead seemed to jump back; when its samples on the music carry on, it didn't
    if o.wrapped and restart is not False:
        return None
    if restart:
        # The same page, with the playhead starting over: played again, or a new page of the same riff
        o.restarted = True
        return None
    return still_links(p, o), "still" if restart is not None else "still?"


def slide_join(p: Obs, o: Obs, gap: float, cands) -> tuple[dict[int, int], str] | None:
    """(links, "slide <dx>") when a measured slide lines the plates up and the playhead carries on across it at that
    shift alone; "slide? <dx>" when the playhead wasn't seen at both ends, or is a box that stayed in one column.
    None when the playhead contradicts every measured shift, or when another candidate fits it as well."""
    m = o.move
    a, b = playhead_at(p, at_start=False), playhead_at(o, at_start=True)
    widths = [c.x1 - c.x0 for c in p.cells if c.kind == "full"] or [4 * gap]
    reach = CARRY_MEASURES * float(np.median(widths))

    def fits(dx):
        # After a slide by dx the playhead carries on from where it was, in o's coordinates (where it was in p) - dx
        return -gap <= (b[0] + dx) - a[0] <= reach and abs(b[1] - a[1]) < CURSOR_DRIFT_GAPS * gap
    for d, drawn in list(zip(m.shifts, m.drawn))[:4]:
        # One jump between two frames can read as a step back by the rest of the band's width
        if d < 0 and not drawn:
            continue
        r = register(p, o, d, gap, SLIDE_REACH)
        if r is None and drawn:
            # Too little music left in common to check: a slide drawn in several steps is taken as measured
            r = (d, 0.0)
        if r is None or r[1] > SCROLL_MISMATCH or not keeps_widths(p, o, r[0]):
            continue
        dx = r[0]
        if a is not None and b is not None:
            if fits(dx):
                # A riff that repeats lines up a period short or long too; when the playhead fits one of those as
                # well, it doesn't tell them apart
                if any(abs(c - dx) > gap and fits(c) for c, _ in cands):
                    return None
                return scroll_links(p, o, dx), f"slide {dx}"
            if abs(b[0] - a[0]) <= gap:
                # A box cursor can stay in one column while the band slides under it: the measure under it after the
                # slide, found on p, is the one it was on or the next
                e, q = cell_at(p, a), cell_at(p, (b[0] + dx, b[1]))
                if e is not None and (q in (e, e + 1) or (q is None and e == len(p.cells) - 1
                                                          and b[0] + dx > p.cells[-1].x0)):
                    return scroll_links(p, o, dx), f"slide? {dx}"
            continue
        # Steps over a riff that repeats can each be measured a period short; without the playhead, only a slide
        # drawn in several steps is taken, and only where the plates leave the join open
        if drawn:
            return scroll_links(p, o, dx), f"slide? {dx}"
    return None


def existing_join(prev, o, gap, shift_cache, scroll_share, usual, use, A) -> dict:
    """The join made without the band's measured movement: a registered shift, else an overlap of look-alike
    cells. Its decisions go to A."""
    links: dict = {}
    options = []
    shifts = shift_cache.get(o.index, [])
    if shifts:
        dx, basis, resolved = choose_shift(prev, o, shifts, scroll_share.get(o.seg, 0.0), use, gap,
                                            usual.get(o.seg))
        if dx is not None:
            links = scroll_links(prev, o, dx)
            # A page shown again whole can line up with itself one measure on; a riff scrolled on by its
            # own length shares at least two
            if o.wrapped and len(links) < 2:
                dx, links = None, {}
        A.decisions.append(Decision(o.t0, "join", f"shift {dx}" if dx is not None else "page turn",
                                    [f"shift {d}" for d, _ in shifts if d != dx]
                                    + (["page turn"] if dx is not None else []), basis, resolved))
    # A view that began where the playhead jumped back may be the page before shown again or a new page
    # that looks like it, so an overlap with that page is no join; so is a still page the playhead started over,
    # unless its plates differ enough that it may be a new page
    same_page = o.restarted and register(prev, o, 0, gap, 2)[1] <= SAME_PAGE_MISMATCH
    if not links and not o.wrapped and not same_page:
        options = [(f"overlap {len(lk)}", lk) for lk in sequence_links(prev, o, gap)]
    if options:
        options.append(("none", {}))
        n, basis, resolved = choose_join(prev, o, options, use)
        # The page may be the one before played again, so whichever way it joins stays unsettled
        resolved = resolved and not o.restarted
        links = options[n][1]
        if len(options) >= 2:
            A.decisions.append(Decision(o.t0, "join", options[n][0],
                                        [lab for m, (lab, _) in enumerate(options) if m != n],
                                        basis, resolved))
    return links


def measure_time(obs: list[Obs]) -> float | None:
    """The typical time the playhead spends on one whole measure, from views where it was seen entering one
    cell and then the next; None when that was seen too rarely to say."""
    dwell = []
    for o in obs:
        entered: dict[int, float] = {}
        for t, x, y in playhead_track(o):
            i = cell_at(o, (x, y))
            if i is not None:
                entered.setdefault(i, t)
        for i, t in entered.items():
            if i + 1 in entered and o.cells[i].complete and entered[i + 1] > t:
                dwell.append(entered[i + 1] - t)
    return float(np.median(dwell)) if len(dwell) >= MIN_DWELLS else None


def playhead_join(p: Obs, o: Obs, beat: float | None, gap: float) -> tuple[dict[int, int], bool, bool] | None:
    """After a page flip or jump that nothing else explained: (links, continues, edge) tying o's first cell to p's
    last measure, cut by the page edge, found from where the playhead was. The cut measure is the one being played
    when the page changed, or the one after a whole measure played to its end. `continues` says o's first cell
    looks like the rest of that measure; `edge` says the join may leave out notes: the cut measure may have been a
    whole one, or o's first cell holds only part of it. None when the playhead doesn't say."""
    if beat is None or not p.cells or not o.cells:
        return None
    e, s = cell_at(p, playhead_at(p, at_start=False)), cell_at(o, playhead_at(o, at_start=True))
    if e is None or s is None:
        return None
    # When the playhead came onto the cell it was on at p's end
    entered = p.t1
    for t, x, y in reversed(playhead_track(p)):
        if cell_at(p, (x, y)) != e:
            break
        entered = t
    played = p.t1 - entered
    if p.cells[e].complete:
        k = e + 1 if played >= MOVED_ON * beat else None
    else:
        k = e if played < CUT_DWELL * beat else None
    # The cut measure is p's last and opens the next page
    if s != 0 or k is None or k != len(p.cells) - 1 or p.cells[k].complete or p.cells[k].kind != "tail":
        return None
    # A cell cut by the page edge is narrower than a measure; a whole last measure that only lost its barline isn't
    widths = [c.x1 - c.x0 for c in p.cells if c.kind == "full"]
    share = (p.cells[k].x1 - p.cells[k].x0) / float(np.median(widths)) if widths else 0.0
    if share >= CUT_WIDTH:
        return None
    # Only one sighting of a measure prints, so when o's first cell starts mid-measure at the page edge the notes
    # of whichever piece doesn't print are lost
    edge = (share >= EDGE_WIDTH or (not p.cells[e].complete and played >= EDGE_DWELL * beat)
            or not o.cells[0].complete)
    # o's first cell shows the rest of the cut measure, or the whole measure drawn again a little differently
    continues = (same_part(p.cells[k], o.cells[0], gap, "left")
                 or resemble(p.cells[k], o.cells[0], gap, any_kind=True) <= REPLAY_RESEMBLE)
    return {0: k}, continues, edge


def move_decision(p: Obs, o: Obs, gap: float) -> Decision:
    """A record of how the band moved between two views and how their plates line up at the measured shifts."""
    m = o.move
    notes = []
    for d, drawn in list(zip(m.shifts, m.drawn))[:3]:
        r = register(p, o, d, gap, SLIDE_REACH)
        notes.append(f"{d}{'d' if drawn else ''}:{r[1]:.3f}" if r else f"{d}:-")
    r0 = register(p, o, 0, gap, 2)
    basis = (f"still {r0[1]:.3f}" if r0 else "still -") + f" restart={playhead_restarts(p, o, gap)}" \
        + f" steps={m.steps} moves={[round(x) for x in m.moves]}"
    return Decision(o.t0, "move", m.kind, notes, basis, True, o.index)


def assemble(obs: list[Obs], gap: float, fold_unresolved: bool, use: set[str], words_of) -> Assembly:
    """Place views into the score. `use` names the evidence allowed ("numbers", "playhead", "navigation");
    `fold_unresolved` says how to read a re-shown run with no evidence (the main reading prints it again)."""
    A = Assembly()
    seg_of = {o.index: o.seg for o in obs}
    prev: Obs | None = None
    prev_slots: list = []
    last_end = None
    # Registration of every consecutive pair, and the share of pairs per segment that line up
    shift_cache: dict = {}
    pairs: dict = {}
    for p, o in zip(obs, obs[1:]):
        if p.seg == o.seg and p.cells and o.cells and not glided_on(o):
            shift_cache[o.index] = scroll_shifts(p, o, gap)
            # A view that began where the playhead jumped back says nothing about whether the band scrolls
            if not o.wrapped:
                pairs.setdefault(o.seg, []).append(bool(shift_cache[o.index]))
    scroll_share = {s: sum(v) / len(v) for s, v in pairs.items() if v}
    # Joins decided by how the band moved between the two views
    track_joins: dict = {}
    slides: dict = {}
    for p, o in zip(obs, obs[1:]):
        if p.seg == o.seg and p.cells and o.cells and not glided_on(o):
            if o.move is not None:
                A.decisions.append(move_decision(p, o, gap))
            tj = track_join(p, o, gap, shift_cache[o.index])
            if tj is not None:
                track_joins[o.index] = tj
                if tj[1].startswith("slide "):
                    slides.setdefault(o.seg, []).append(int(tj[1].split()[1]))
    # The shift each segment's band was usually seen sliding by, from the slides the playhead confirmed
    usual = {s: float(np.median(v)) for s, v in slides.items()}
    # A segment slides when the playhead confirmed most of the slides and cuts its band was seen making
    for sg in {o.seg for o in obs}:
        moved = [o for o in obs if o.seg == sg and o.move is not None and o.move.kind in ("slide", "cut")]
        A.sliding[sg] = len(slides.get(sg, [])) >= 5 and len(slides.get(sg, [])) >= 0.7 * len(moved)
    beats = {s: measure_time([o for o in obs if o.seg == s and o.mosaic is None]) for s in {o.seg for o in obs}}
    for o in obs:
        cells = o.cells
        # Time without tab on screen before this view; a view with tab but no usable cells still counts
        gap_before = (o.t0 - last_end) if last_end is not None else 0.0
        last_end = o.t1
        if not cells:
            continue
        decided = len(A.decisions)
        slot_of: list = [None] * len(cells)
        # Consecutive glide pieces split their measures between them, so no measure is in both: a piece is not
        # joined to the view before, and place_rest places it as new music
        joined = prev is not None and prev.seg == o.seg and not glided_on(o)
        tj = track_joins.get(o.index) if joined else None
        # Only the main reading clears numbers: a view the folded reading folded without evidence may sit on the
        # wrong numbered measures
        if not fold_unresolved and tj is not None and "?" not in tj[1] and any(
                cells[i].number is not None and prev_slots[j] is not None
                and A.slots[prev_slots[j]].number not in (None, cells[i].number) for i, j in tj[0].items()):
            # The playhead confirmed these measures are the ones shown before, so numbers that differ from theirs
            # were misread and the view is placed by the join. Its reads stay as they were
            for c in cells:
                c.number = None
        if "numbers" in use and all(c.number is not None for c in cells):
            place_numbered(o, slot_of, A, prev, prev_slots, gap, seg_of)
        else:
            links: dict = {}
            joins_before = len(A.decisions)
            if joined:
                tj, probed = track_joins.get(o.index), False
                if tj is not None and tj[1].startswith("slide?"):
                    # A slide the playhead didn't confirm only settles a pair the plates and overlaps left open
                    probe = Assembly()
                    plain = existing_join(prev, o, gap, shift_cache, scroll_share, usual, use, probe)
                    if probe.decisions:
                        links, tj, probed = plain, None, True
                        A.decisions.extend(probe.decisions)
                if tj is not None:
                    links = tj[0]
                    if tj[1].startswith("slide?"):
                        # A slide drawn in several steps stands with no playhead seen; a box cursor that stayed in
                        # one column only failed to contradict it
                        resolved = playhead_at(prev, at_start=False) is None or playhead_at(o, at_start=True) is None
                    else:
                        # A band that held still with no playhead seen could also be a page flipped to the same riff
                        resolved = tj[1] != "still?"
                    A.decisions.append(Decision(o.t0, "join", tj[1].replace("?", ""), [], "track", resolved))
                elif not probed:
                    links = existing_join(prev, o, gap, shift_cache, scroll_share, usual, use, A)
                pj = playhead_join(prev, o, beats.get(o.seg), gap) if not links and tj is None else None
                if pj is not None:
                    links, continues, edge = pj
                    # A first cell unlike the cut measure is still the next page unless it opens a run of measures
                    # shown before: then the video may have cut back to the opening
                    resolved = (continues or len(o.cells) < 2
                                or find_run(A.slots, o.cells[:4], gap, o.index) is None)
                    A.decisions.append(Decision(o.t0, "join", f"playhead {sorted(links.items())}",
                                                ["edge"] if edge else [], "track", resolved))
                    if edge:
                        A.edge_joins.append((prev.index, o.index))
            for i, j in links.items():
                slot_of[i] = prev_slots[j]
            A.view_links[o.index] = set(links)
            slots_before, decisions_before = len(A.slots), len(A.decisions)
            seen_before, starts_before = set(A.seen_segments), set(A.segment_starts)
            place_rest(o, slot_of, A, gap, gap_before, prev, use, words_of, fold_unresolved)
            if (o.wrapped and not links and len(A.decisions) == decisions_before
                    and same_count(o, prev, prev_slots) and all(s >= slots_before for s in slot_of)):
                # Nothing tied the view to the music: read it as the view before shown again
                del A.slots[slots_before:]
                A.seen_segments, A.segment_starts = seen_before, starts_before
                fold_view(o, slot_of, prev_slots, A, fold_unresolved)
            if (o.restarted and not links and len(A.decisions) == decisions_before
                    and not any(d.kind == "join" and not d.resolved for d in A.decisions[joins_before:])):
                # A page the playhead started over that nothing tied to the music or folded is printed as new music,
                # though it may be the page before played again
                A.decisions.append(Decision(o.t0, "join", "new", ["played again"], "restart", False))
            if "numbers" in use:
                carry_numbers(slot_of, A)
        A.view_slots[o.index] = list(slot_of)
        for c, sid in zip(cells, slot_of):
            if sid is not None:
                A.sightings.setdefault(sid, []).append((o.index, c))
        prev, prev_slots = o, slot_of
        for d in A.decisions[decided:]:
            d.view = o.index
    A.order = [s.id for s in A.slots]
    return A


def glided_on(o: Obs) -> bool:
    """Whether a view is a glide piece that carries on the piece before."""
    return o.move is not None and o.move.kind == "glide"


def same_count(o: Obs, prev: Obs | None, prev_slots: list) -> bool:
    """Whether a view could be the view before shown again: in its segment, with as many measures, all placed."""
    return prev is not None and prev.seg == o.seg and len(o.cells) == len(prev_slots) and None not in prev_slots


def fold_view(o: Obs, slot_of: list, prev_slots: list, A: Assembly, fold_unresolved: bool) -> None:
    """Read a view that began where the playhead jumped back, which nothing tied to earlier music, as the view before
    shown again, without evidence: the main reading prints it again and the folded one folds it, with a notice
    either way. Cursor tint can keep its measures from matching those of the view before."""
    A.unresolved_folds.append((o.t0, o.t1, prev_slots[0], prev_slots[-1]))
    A.decisions.append(Decision(o.t0, "fold", "fold" if fold_unresolved else "new",
                                ["new" if fold_unresolved else "fold"], "wrap", False))
    for i, c in enumerate(o.cells):
        slot_of[i] = prev_slots[i] if fold_unresolved else new_slot(A, c, o)
    A.replay_next = None


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


def place_numbered(o, slot_of, A, prev, prev_slots, gap, seg_of) -> None:
    """Map each cell to the slot with its measure number, making slots for numbers not seen before.

    A number seen on an earlier view (not the one just before, whose overlap is expected) means the video
    went back: a replay with printed-number evidence, folded. When the music there differs, another part (a
    second guitar track) reuses the numbers, and its measures get slots of their own.

    Glided measures fold a run at a time: a run whose numbers were seen before folds only when at least half its
    measures look like the ones they fold onto. Otherwise its numbers are taken as misread (a counter can drop a
    digit), and the run is printed as new music under numbers already used.
    """
    by_number = {s.number: s.id for s in A.slots if s.number is not None}
    other_part = switched_part(o, A, by_number, prev, prev_slots, gap, seg_of)
    # A glide piece shares no measure with the piece before, so none of its numbers is an expected overlap
    near = [] if glided_on(o) else prev_slots
    replayed = False
    judged, refused = set(), set()
    for i, c in enumerate(o.cells):
        sid = by_number.get(c.number)
        if sid is not None and c.glided and sid not in near and i not in judged:
            run = earlier_run(o.cells, i, by_number, near)
            alike = sum(resemble_scaled(A.slots[by_number[o.cells[k].number]].first, o.cells[k], gap) < GLIDE_CONTINUE
                        for k in run)
            judged.update(run)
            if 2 * alike < len(run):
                refused.update(run)
        if i in refused:
            sid = None
        if other_part and sid is not None and not (sid in prev_slots
                                                   and seen_before(c, same_layout(A, sid, o, seg_of), gap)):
            sid = None
        # A refused run is new music, and a glide piece shares no measure with the piece before
        if sid is None and prev is not None and i not in refused and not glided_on(o):
            # The view before may have had no readable numbers; a measure it showed takes this number
            for j, (pc, ps) in enumerate(zip(prev.cells, prev_slots)):
                if read_elsewhere(prev, j, c.number):
                    continue
                if ps is not None and A.slots[ps].number is None and same_cell(pc, c, gap):
                    sid = ps
                    A.slots[ps].number = c.number
                    by_number[c.number] = ps
                    break
        if sid is None:
            sid = new_slot(A, c, o)
            by_number[c.number] = sid
        elif sid not in near:
            replayed = True
        slot_of[i] = sid
    if replayed:
        A.decisions.append(Decision(o.t0, "fold", "fold", ["new"], "numbers", True))


def earlier_run(cells, i: int, by_number: dict, prev_slots: list) -> list[int]:
    """Cell i and the glided cells right after it whose numbers name earlier slots, other than the view before's."""
    run = [i]
    while run[-1] + 1 < len(cells):
        c = cells[run[-1] + 1]
        sid = by_number.get(c.number)
        if sid is None or sid in prev_slots or not c.glided:
            break
        run.append(run[-1] + 1)
    return run


def read_elsewhere(o, j, number: int) -> bool:
    """True when a number read on view o's cells says its cell j holds another measure than `number`. Such a view had
    too few reads to be numbered, but a look-alike page can't lend its numbers against one it did read."""
    # A read far off is a misread, such as one digit or a fret number
    return any(c.read is not None and 0 < abs(c.read - i + j - number) <= BORROW_CONFLICT
               for i, c in enumerate(o.cells))


def switched_part(o, A, by_number, prev, prev_slots, gap, seg_of) -> bool:
    """Whether most measures a screen shows again, by number, hold other music than before on the same layout,
    judged on at least two of them."""
    shown = [A.slots[s].number for s in prev_slots if s is not None and A.slots[s].number is not None]
    alike = []
    for c in o.cells:
        sid = by_number.get(c.number)
        if sid is None:
            continue
        if sid in prev_slots:
            # The screen before showed this number too: compare the two sightings where both are whole measures
            # drawn at about one width, as `resemble` needs; a cut or squeezed one can't be told from other music
            if prev.seg == o.seg and c.kind == "full":
                w = c.ink.shape[1]
                earlier = [x for x, s in zip(prev.cells, prev_slots) if s == sid and x.kind == "full"
                           and abs(x.ink.shape[1] - w) <= max(3 * WIDTH_PX, 3 * WIDTH_SHARE * w)]
                if earlier:
                    alike.append(seen_before(c, earlier, gap))
            continue
        # A number past the screen before carries the music on; an earlier glimpse of it may look different
        if shown and c.number > max(shown):
            continue
        earlier = same_layout(A, sid, o, seg_of)
        if earlier:
            alike.append(seen_before(c, earlier, gap))
    # A single measure that differs may just be drawn again differently, which is no evidence of another part
    return len(alike) >= 2 and 2 * sum(alike) < len(alike)


def same_layout(A, sid, o, seg_of) -> list:
    """The earlier sightings of a slot drawn in the same layout as view o."""
    return [x for vi, x in A.sightings.get(sid, []) if seg_of[vi] == o.seg]


def seen_before(c, earlier, gap) -> bool:
    """Whether a cell shows the music of some earlier sighting."""
    return any(alike_music(x, c, gap) for x in earlier)


def alike_music(a, b, gap: float) -> bool:
    """Whether two sightings show the same music, allowing for small differences in how they are drawn."""
    if (a.index == 0) != (b.index == 0) and a.kind == b.kind == "full":
        # Only one starts a system, where a clef and key are drawn: compare the music after them
        part, whole = (a, b) if a.ink.shape[1] <= b.ink.shape[1] else (b, a)
        if whole.ink.shape[1] - part.ink.shape[1] > 3 * gap:
            return False
        return same_part(part, whole, gap, "right", trim=SYSTEM_START_GAPS)
    if same_cell(a, b, gap):
        return True
    # A staff whose top line is cut off by the screen edge is placed from its next line down, so the two
    # sightings are also lined up with one moved by up to two line gaps
    reach = int(2 * gap)
    return any(resemble(a, staff_moved(b, dy), gap, any_kind=True) <= REPLAY_RESEMBLE
               for dy in range(-reach, reach + 1, 2))


def staff_moved(c, dy: int):
    """A copy of cell c whose tab staff is taken to start dy rows lower."""
    return replace(c, ref_y=c.ref_y + dy)


def place_rest(o, slot_of, A, gap, gap_before, prev, use, words_of, fold_unresolved) -> None:
    """Place the cells the join with the previous view didn't explain."""
    if glided_on(o) and A.replay_next is None:
        # The band glided on from the piece before, so its measures are new music
        for i, c in enumerate(o.cells):
            if slot_of[i] is None:
                slot_of[i] = new_slot(A, c, o)
        A.decisions.append(Decision(o.t0, "join", "glide continues", [], "track", True))
        return
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
                fold, basis, resolved = replay_evidence(o, run, j, A, gap_before, prev, use, words_of)
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
