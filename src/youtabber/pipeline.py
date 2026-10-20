"""Read a video into a score: find the tab, build clean views, cut measures, place them, and check the result."""
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from youtabber import glide, ocr
from youtabber.assemble import Assembly, Obs, assemble, cell_at, number_views
from youtabber.check import (dwell, empty_stretches, fmt, glide_gaps, glide_number_gaps, glide_repeats, layout_changes,
                             number_checks, numbers_against_cells, odd_widths, pieces, possible_duplicates,
                             screen_breaks, too_few)
from youtabber.observe import (ABOVE_GAPS, BELOW_GAPS, OBS_FPS, PRINT_ABOVE_GAPS, ink_map, ink_threshold, set_paper,
                               to_gray, views_in)
from youtabber.parse import Cell, line_darkness, parse_view
from youtabber.scan import SCAN_FPS, Segment, scan
from youtabber.video import frames, probe

# Every kind of evidence the assembler can weigh
EVIDENCE = {"numbers", "playhead", "navigation"}
# A cell narrower than this share of a typical measure is a header or a sliver, not a measure
EDGE_SHARE = 0.25
# ...unless it is an empty bar drawn short: between two barlines, with less ink on its tab staff than this many
# square line gaps once its left barline is left out...
EMPTY_BAR_INK = 0.1
# ...and, in a system's first cell, less than this many on the notation staff above it, room for a rest but not a clef
# and key signature
EMPTY_NOTATION_INK = 1.0
# Line gaps at a cell's left taken up by its barline, a thick or double one included
BAR_SKIP_GAPS = 0.7
# An empty bar is drawn at least this many line gaps wide; narrower blank cells sit between a barline and a
# column of stems or stacked digits that was read as one
EMPTY_MIN_GAPS = 3.5
# A view is mid-fade when its ink contrast is below FADE_SHARE of its segment's typical view and its staff lines
# stand out from the paper by less than FADE_LINE_SHARE of the typical view's
FADE_SHARE, FADE_LINE_SHARE = 0.85, 0.8
# Only a segment at least this long is tested for gliding; shorter moves are page-turn animations
GLIDE_MIN_SECONDS = 20.0
# Such a segment that doesn't glide but whose still views cover less than this share of it can't be read
HELD_SHARE = 0.4
# Staff positions tried for a glide, most often seen first
GLIDE_GROUPS = 3
# A glide mosaic wider than this many columns is parsed in pieces this wide...
MOSAIC_PIECE = 40000
# ...overlapping by this many columns, several times the widest measure
MOSAIC_OVERLAP = 8000
# A glide's last cell with less ink than this share of its width times the line gap is blank staff, not a measure
BLANK_TAIL_INK = 0.02
# Frame-fixed ink: inked in this share of a segment's views, which must number at least FIXED_MIN_VIEWS
FIXED_SHARE, FIXED_MIN_VIEWS = 0.85, 4
# A page turn between two views takes less than this
TURN_SECONDS = 1.0
# A view is covered by a title card or photo when at least this share of its marks are missing from the next view
# of the same page...
COVER_EXTRA = 0.4
# ...and at most this share of the next view's marks are missing from it...
COVER_MISSING = 0.1
# ...and at least this share of its inked tiles (COVER_TILE line gaps tall and wide) match the next view, under
# COVER_TILE_EXTRA of a tile's marks missing there. A card lies over part of the page; a page of music followed by
# a page of empty bars drawn on the same barlines differs in nearly every tile
COVER_MATCHED, COVER_TILE, COVER_TILE_EXTRA = 0.28, (4, 8), 0.2
# A straight edge is a row where the gray level steps by more than EDGE_STEP across at least EDGE_SHARE_WIDTH of the
# frame; a screen with EDITOR_EDGES or more of them around its tab is a program's window
EDGE_STEP, EDGE_SHARE_WIDTH, EDITOR_EDGES = 25, 1 / 6, 4
# Tab whose every segment is shorter than SCATTERED_SECONDS, which together cover less than SCATTERED_SHARE of the
# video, and whose line spacing varies by more than SCATTERED_GAP_RATIO is no play-along
SCATTERED_SECONDS, SCATTERED_SHARE, SCATTERED_GAP_RATIO = 10.0, 0.25, 1.5

Region = tuple[int, int, int, int]
# Called with a stage name ("find", "read", "assemble", in that order) and the share of that stage done, 0.0 to 1.0
Progress = Callable[[str, float], None]


@dataclass
class Measure:
    """One printed measure: which slot it is, the sighting its image comes from, and when and where it was seen."""
    slot: int
    view: int
    cell: Cell
    t: float                 # view midpoint for still views, playhead crossing for glides
    play_t: float            # when playback reached it
    x: float                 # source pixels: cell center, or the playhead's x for glides
    y: float                 # source pixels: middle of the tab staff
    t_ref: float | None      # glides only: when its center crossed the playhead
    label: int | None        # measure number, when the view's numbers were trusted
    partial: bool


@dataclass
class Candidate:
    name: str                # "primary" or "folded"
    mode: str                # "once" or "all"
    measures: list[Measure]


@dataclass
class Notice:
    t0: float
    t1: float
    kind: str
    text: str
    doubt: bool              # music may be lost or doubled; False when a reading was chosen and the other PDF exists


@dataclass
class Reading:
    duration: float
    segments: list[Segment] = field(default_factory=list)
    outcome: str = "tab"
    fail_reason: str | None = None
    message: str = ""
    tab_spans: list[tuple[float, float]] = field(default_factory=list)
    panels: list[dict] = field(default_factory=list)
    numbering: str = "unnumbered"
    views: list[Obs] = field(default_factory=list)
    assembly: Assembly | None = None
    candidates: list[Candidate] = field(default_factory=list)
    every_pass: list[Candidate] = field(default_factory=list)
    path: list[tuple[int, int, int, float]] = field(default_factory=list)   # (slot, view, cell, play time)
    notices: list[Notice] = field(default_factory=list)

    def fail(self, reason: str, message: str) -> "Reading":
        self.outcome, self.fail_reason, self.message = "fail", reason, message
        self.candidates, self.every_pass, self.path = [], [], []
        return self


def read(path: Path, region: Region | None = None, progress: Progress | None = None) -> Reading:
    """Everything youtabber reads from one video, ready to print, or the reason it can't be printed.

    `progress` hears how far each stage has got; each stage sends 0.0 first and 1.0 when it ends. A reading that
    fails stops sending at the stage it failed in."""
    def tell(stage: str, share: float) -> None:
        if progress is not None:
            progress(stage, share)

    info = probe(path)
    tell("find", 0.0)
    segments, found, scrolls = scan(path, info.width, info.height, region,
                                    progress=share_of(progress, "find", 0.0, info.duration))
    r = Reading(info.duration, segments)
    if scrolls and not segments:
        # No tab ever holds still: whether those lines are a scrolling page or footage, there is no tab to read
        return r.fail("no_tab", f"No tab holds still on screen: the page scrolls up continuously "
                                f"({scroll_times(scrolls)}), and youtabber can't read a page while it moves.")
    if not segments and region is not None:
        x, y, w, h = region
        return r.fail("no_tab", f"No tab staff inside --region {x},{y},{w},{h}. Leave --region out to let youtabber "
                                f"find the tab.")
    if not segments:
        return r.fail("no_tab", f"No tab on screen: no steady set of evenly spaced staff lines on paper in "
                                f"{info.duration:.0f} s of video. youtabber can't read text tab (e|--3--) or tab "
                                f"drawn see-through over the video either.")
    # Tab in short scattered pieces may be a screen recording of an editor, with short play-along clips in it
    clips = None
    if scattered(segments, info.duration):
        held = sum(s.t1 - s.t0 for s in segments)
        segments = [s for s in segments if not editor_screen(path, s)]
        r.segments = segments
        if not segments:
            return r.fail("no_tab", f"Tab shows only in short pieces ({held:.0f} s of {info.duration:.0f} s), on "
                                    f"screens that look like a program's window, as when a tab editor is recorded "
                                    f"while someone types. youtabber reads play-along videos, where the tab stays "
                                    f"on screen while the song plays.")
        clips = (segments[0].t0, segments[-1].t1)
    tell("find", 1.0)
    tell("read", 0.0)
    # The read stage's share is the video time its frames have reached, as a share of the whole video
    reached = share_of(progress, "read", 0.0, info.duration)
    obs: list[Obs] = []
    gaps = []
    for si, seg in enumerate(segments):
        crop = (seg.x0, seg.y0, seg.x1 - seg.x0, seg.y1 - seg.y0)
        set_paper(seg.paper)
        r.tab_spans.append((round(seg.t0, 2), round(seg.t1, 2)))
        seed_at = seed_function(found, seg)
        if reached is not None:
            reached(seg.t0)
        if seg.t1 - seg.t0 >= GLIDE_MIN_SECONDS and glide.glides(path, crop, seg.polarity, seg.paper, seg.t0, seg.t1,
                                                                bands_function(found, seg), seg.gap):
            # The tab glides past a playhead instead of holding still: read it from a mosaic
            thr, fthr = first_thresholds(path, crop, seg, seed_at)
            o = glide_obs(path, crop, seg, si, found, thr, fthr, len(obs))
            if o is None:
                return r.fail("check_failed", f"The tab scrolls smoothly from {fmt(seg.t0)} to {fmt(seg.t1)}, but the "
                                              f"moving band could not be put together, so no PDF was written.")
            obs.append(o)
            r.panels.append({"t0": round(seg.t0, 2), "t1": round(seg.t1, 2),
                             "box": [seg.x0, seg.y0 + o.rows[0], seg.x1 - seg.x0, o.rows[1] - o.rows[0]]})
            gaps.append(seg.gap)
            if reached is not None:
                reached(seg.t1)
            continue
        views, thr, fthr = views_in(path, crop, seg.polarity, seg.gap, seg.paper, seed_at, seg.t0, seg.t1,
                                    "playhead" in EVIDENCE, progress=reached)
        held = sum(v.t1 - v.t0 for v in views)
        if seg.t1 - seg.t0 >= GLIDE_MIN_SECONDS and held < HELD_SHARE * (seg.t1 - seg.t0):
            # Tab that keeps changing without holding still or scrolling smoothly: the still views miss most of it
            return r.fail("check_failed", f"The tab at {fmt(seg.t0)}-{fmt(seg.t1)} holds still for only {held:.0f} of "
                                          f"{seg.t1 - seg.t0:.0f} s and doesn't scroll smoothly either, so most of it "
                                          f"can't be read; no PDF was written.")
        still = []
        for v in views:
            o = plate_obs(v, seg, si, thr, fthr, len(obs))
            obs.append(o)
            # The reported panel is the print area: the print rect's columns, its systems' bands' rows
            x0, y0, x1, y1 = v.print_rect
            if o.systems:
                y0, y1 = min(ps.band[0] for ps in o.systems), max(ps.band[1] for ps in o.systems)
            still.append({"t0": round(v.t0, 2), "t1": round(v.t1, 2),
                          "box": [int(x0 + seg.x0), int(y0 + seg.y0), int(x1 - x0), int(y1 - y0)]})
        # A view's print area stays in place through a page turn to the next view. The scan finds a segment's
        # ends only to within half a scan step, so the first and last areas reach that far past them
        for a, b in zip(still, still[1:]):
            if b["t0"] - a["t1"] <= TURN_SECONDS:
                a["t1"] = b["t0"]
        if still:
            before = segments[si - 1].t1 if si > 0 else 0.0
            after = segments[si + 1].t0 if si + 1 < len(segments) else r.duration
            still[0]["t0"] = round(max(before, min(still[0]["t0"], seg.t0 - 0.5 / SCAN_FPS)), 2)
            still[-1]["t1"] = round(min(after, max(still[-1]["t1"], seg.t1 + 0.5 / SCAN_FPS)), 2)
        r.panels += still
        gaps.append(seg.gap)
    r.views = obs
    tell("read", 1.0)
    tell("assemble", 0.0)
    gap = float(np.median(gaps))
    # A title card or photo laid over the first screen of a page adds marks the next screen doesn't have; the
    # same music shows clean on that next screen, so the covered view gives no measures
    for p, o in zip(obs, obs[1:]):
        if covered(p, o):
            p.all_cells = []
    # A view caught in a fade has weak contrast and loses its barlines; its music is shown again at full
    # contrast, so it is left out. Both its darkest ink and its staff lines are weak. A segment of one or two views
    # is judged against the clearest views of its theme.
    clearest: dict[str, tuple[float, float]] = {}
    for o in obs:
        if o.mosaic is None:
            pol = segments[o.seg].polarity
            ink, line = clearest.get(pol, (0.0, 0.0))
            clearest[pol] = (max(ink, o.contrast), max(line, o.line_contrast))
    for si, seg in enumerate(segments):
        group = [o for o in obs if o.seg == si and o.mosaic is None]
        if len(group) >= 3:
            ink = float(np.median([o.contrast for o in group]))
            line = float(np.median([o.line_contrast for o in group]))
        else:
            ink, line = clearest.get(seg.polarity, (0.0, 0.0))
        for o in group:
            if o.contrast < FADE_SHARE * ink and o.line_contrast < FADE_LINE_SHARE * line:
                o.all_cells = []
    # Ink at the same screen place in nearly every view of a segment is fixed to the frame (a clef and key
    # panel that stays while the music scrolls behind it)
    for si in range(len(segments)):
        group = [o for o in obs if o.seg == si]
        if len(group) >= FIXED_MIN_VIEWS:
            share = np.mean([o.band for o in group], axis=0)
            fixed = cv2.dilate((share >= FIXED_SHARE).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        elif group and group[0].mosaic is None:
            fixed = np.zeros_like(group[0].band)
        else:
            # A glide's mosaic is never lined up with another view, so it needs no map of frame-fixed ink
            fixed = None
        for o in group:
            o.fixed = fixed
    # A cell much narrower than a typical measure is not one: a clef and key signature before a
    # start-repeat barline, or a sliver cut by the panel edge
    full_widths = [c.x1 - c.x0 for o in obs for c in o.all_cells if c.kind == "full"]
    typical = float(np.median(full_widths)) if full_widths else 0.0
    for o in obs:
        o.cells = kept_cells(o.all_cells, typical)
        o.bars = sorted({c.x0 for c in o.all_cells if c.kind != "lead"} | {c.x1 for c in o.all_cells if c.kind != "tail"})
    r.numbering = number_views(obs, ocr.read_number)
    # A measure seen only on its notation staff, its tab out of the panel, can be placed by its number alone
    for o in obs:
        o.cells = [c for c in o.cells if not c.notation_only or c.number is not None]

    def words_of(o: Obs) -> list[str]:
        return ocr.find_words(o.plate) if o.plate is not None else []

    main = assemble(obs, gap, False, EVIDENCE, words_of)
    r.assembly = main
    r.candidates = [Candidate("primary", "once", measures_of(main, obs, main.order))]
    if main.unresolved_folds:
        folded = assemble(obs, gap, True, EVIDENCE, words_of)
        r.candidates.append(Candidate("folded", "once", measures_of(folded, obs, folded.order)))

    notices = []
    for d in main.decisions:
        if not d.resolved and d.kind == "join":
            o = next(x for x in obs if x.t0 == d.t)
            notices.append(Notice(round(d.t - 1, 2), round(o.t1, 2), "unresolved_join",
                                  "Couldn't tell how this screen continues from the one before; check that no "
                                  "measure is missing or printed twice here.", True))
    ranges = []
    for t0, t1, j0, j1 in main.unresolved_folds:
        first_seen = obs[main.slots[j0].obs]
        ranges.append((first_seen.t0, first_seen.t1, t0, t1))
        notices.append(Notice(round(t0, 2), round(t1, 2), "unresolved_fold",
                              f"The measures at {fmt(t0)}-{fmt(t1)} repeat ones first shown at "
                              f"{fmt(first_seen.t0)}-{fmt(first_seen.t1)}, with no sign whether the song plays "
                              f"them again or the video replays them.", False))
    for w in (possible_duplicates(main, obs, gap) + dwell(obs, main) + odd_widths(obs) + numbers_against_cells(obs)
              + screen_breaks(main, obs) + layout_changes(segments, obs)):
        notices.append(Notice(w["t0"], w["t1"], w["kind"], w["text"], True))
    for o in obs:
        if o.mosaic is not None:
            m = o.mosaic
            # Measures read from the first frame alone: on screen between the playhead and the reading column
            # before the scroll moved
            seeded = [c for c in o.cells if (c.x0 + c.x1) / 2 < m.seeded]
            if seeded:
                notices.append(Notice(round(o.t0, 2), round(m.time_at(seeded[-1].x1), 2), "glide_start",
                                      f"The first measure of this smooth scroll was on screen before it moved and "
                                      f"is read from its first frame only; check it." if len(seeded) == 1 else
                                      f"The first {len(seeded)} measures of this smooth scroll were on screen before "
                                      f"it moved and are read from its first frame only; check them.", True))
            # Measures played within one measure's time of the first frame were already at the playhead when the
            # scroll started, so they may be cut off; this finds them even when no playhead was found
            times = [m.time_at((c.x0 + c.x1) / 2) for c in o.cells]
            step = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
            early = [c for c, t in zip(o.cells, times) if t < o.t0 + step and (c.x0 + c.x1) / 2 >= m.seeded]
            if early:
                notices.append(Notice(round(o.t0, 2), round(m.time_at(early[-1].x1), 2), "glide_start",
                                      f"The first measure of this smooth scroll was already at the playhead when "
                                      f"it started and may be cut off or not belong to the song; check it."
                                      if len(early) == 1 else
                                      f"The first {len(early)} measures of this smooth scroll were already at the "
                                      f"playhead when it started and may be cut off or not belong to the song; "
                                      f"check them.", True))
            # Measures read from the last frame alone: still ahead of the reading column when the scroll ended.
            # They are played after it stops, so the notice ends where the scroll ends.
            ending = [c for c in o.cells if (c.x0 + c.x1) / 2 >= m.tail_from]
            if m.tail_from < m.image.shape[1]:
                t0 = min(m.time_at(ending[0].x0 if ending else m.tail_from), o.t1 - 1)
                text = ("The last measures of this smooth scroll were still ahead of the playhead when the scroll "
                        "ended and are read from its last frame only; check them." if ending else
                        "This smooth scroll ended with staff still ahead of the playhead, and no measure was read "
                        "from it; check that the last measures are all printed.")
                notices.append(Notice(round(t0, 2), round(o.t1, 2), "glide_end", text, True))
            for w in glide_repeats(o, gap) + glide_gaps(o) + glide_number_gaps(o):
                notices.append(Notice(w["t0"], w["t1"], w["kind"], w["text"], True))
    primary = r.candidates[0].measures
    for w in (empty_stretches(r.tab_spans, [m.play_t for m in primary])
              + number_checks([(m.play_t, m.label) for m in primary], True, info.duration)):
        notices.append(Notice(w["t0"], w["t1"], w["kind"], w["text"], True))
    # A page scrolling up is tab on screen that nothing reads. It is reported in pieces, so each points at a
    # findable part of the video
    for t0, t1 in scrolls:
        r.tab_spans.append((round(t0, 2), round(t1, 2)))
        for p0, p1 in pieces(t0, t1):
            notices.append(Notice(round(p0, 2), round(p1, 2), "vertical_scroll",
                                  f"The page scrolls up continuously at {fmt(p0)}-{fmt(p1)}. youtabber can't read a "
                                  f"page while it moves, so the measures shown there are missing from this PDF.", True))
    r.tab_spans.sort()
    if clips is not None:
        notices.append(Notice(round(clips[0], 2), round(clips[1], 2), "short_tab",
                              f"Only one stretch of this video was read as play-along tab "
                              f"({fmt(clips[0])}-{fmt(clips[1])}); nothing else is printed. Check that this is the "
                              f"music you wanted.", True))
    r.notices = sorted(notices, key=lambda w: (w.t0, w.t1))

    n = len(r.candidates[0].measures)
    if len(r.candidates) > 1:
        spans = "; ".join(f"{fmt(a)}-{fmt(b)} repeats {fmt(c)}-{fmt(d)}" for c, d, a, b in ranges[:6])
        r.message = (f"Wrote two PDFs that differ where screens repeat without measure numbers or a repeat sign "
                     f"({spans}). The main PDF prints those measures every time; the folded PDF prints them once.")
    else:
        r.message = f"Read {n} measure{'' if n == 1 else 's'}."
    short = too_few(obs, n) if n else None
    if short:
        return r.fail("check_failed", short)
    if not n:
        return r.fail("no_tab", "Found lines that look like a tab staff, but no measures between barlines on them. "
                                "youtabber can't read text tab (e|--3--), tab drawn see-through over the video, or tab "
                                "without barlines.")
    r.path = play_path(main, obs)
    r.every_pass = [Candidate("primary", "all", path_measures(main, obs, r.path))]
    tell("assemble", 1.0)
    return r


def share_of(progress: Progress | None, stage: str, t0: float, t1: float) -> Callable[[float], None] | None:
    """A callback for a frame loop that tells `progress` how far video time t has got through t0..t1, or None
    when nobody is listening, so the loop does nothing extra."""
    if progress is None:
        return None
    span = max(t1 - t0, 1e-6)
    return lambda t: progress(stage, min(1.0, max(0.0, (t - t0) / span)))


def covered(p: Obs, o: Obs) -> bool:
    """True when still view p shows staffs of the next view o plus many marks o doesn't have, gathered over part of
    the page, while nearly all of o's marks are in p. The card can hide a staff of p, so p may show fewer."""
    if p.seg != o.seg or p.mosaic is not None or o.mosaic is not None or not p.systems:
        return False
    # A next view fainter than p is fading out: it lost p's marks to the fade, not to a card
    if o.contrast < FADE_SHARE * p.contrast:
        return False
    if not all(any(abs(a.system.tab.top - b.system.tab.top) <= 0.25 * a.system.tab.gap for b in o.systems)
               for a in p.systems):
        return False
    a, b = p.band, o.band
    near = np.ones((5, 5), np.uint8)
    extra = a & ~cv2.dilate(b.astype(np.uint8), near).astype(bool)
    only_p = extra.sum() / max(int(a.sum()), 1)
    only_o = (b & ~cv2.dilate(a.astype(np.uint8), near).astype(bool)).sum() / max(int(b.sum()), 1)
    if only_p < COVER_EXTRA or only_o > COVER_MISSING:
        return False
    return matched_share(a, extra, p.systems[0].system.tab.gap) >= COVER_MATCHED


def matched_share(ink: np.ndarray, extra: np.ndarray, gap: float) -> float:
    """The share of the band's inked tiles in which few of its marks are extra marks."""
    th, tw = max(1, int(COVER_TILE[0] * gap)), max(1, int(COVER_TILE[1] * gap))
    matched = inked = 0
    for y in range(0, ink.shape[0], th):
        for x in range(0, ink.shape[1], tw):
            n = int(ink[y:y + th, x:x + tw].sum())
            # A tile with less ink than a square line gap holds no mark worth judging
            if n >= gap * gap:
                inked += 1
                matched += int(extra[y:y + th, x:x + tw].sum()) < COVER_TILE_EXTRA * n
    return matched / max(inked, 1)


def scroll_times(scrolls: list[tuple[float, float]]) -> str:
    return ", ".join(f"{fmt(a)}-{fmt(b)}" for a, b in scrolls)


def scattered(segments: list[Segment], duration: float) -> bool:
    """True when tab shows only in short pieces covering a small share of the video, drawn at different sizes:
    a screen recording of a tab editor or a lesson's slides, not a play-along."""
    longest = max(s.t1 - s.t0 for s in segments)
    held = sum(s.t1 - s.t0 for s in segments)
    gaps = [s.gap for s in segments]
    return (longest < SCATTERED_SECONDS and held < SCATTERED_SHARE * duration
            and max(gaps) > SCATTERED_GAP_RATIO * min(gaps))


def editor_screen(path: Path, seg: Segment) -> bool:
    """True when the screen around a segment's tab is a program's window: many straight edges (panel borders,
    toolbars, rows of controls) run across it. Footage of a room can show as many, so this is asked only of tab
    that shows in short scattered pieces, where a play-along band is the exception."""
    mid = (seg.t0 + seg.t1) / 2
    g = next((cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in frames(path, 1, start=mid, length=1.0)), None)
    if g is None:
        return False
    steps = np.abs(np.diff(g.astype(np.int16), axis=0)) > EDGE_STEP
    # The tab's own staff lines don't count
    steps[max(0, seg.y0 - 2):seg.y1 + 2, seg.x0:seg.x1] = False
    long_edges = 0
    for row in steps:
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if len(starts) and (ends - starts).max() >= EDGE_SHARE_WIDTH * g.shape[1]:
            long_edges += 1
    return long_edges >= EDITOR_EDGES


def empty_bar(c: Cell) -> bool:
    """True for a bar with nothing on its tab staff between its two barlines: a rest bar, count-in, closing empty
    bar or bar of notes tied over, which prints however short it is drawn. A measure number above it doesn't count.
    A system's first cell may be a clef and key signature standing alone before a start-repeat barline, so there only
    a rest on the notation staff, smaller than a square line gap, is allowed."""
    # A notation staff whose tab is out of view says nothing about the tab
    if c.kind != "full" or not c.complete or c.notation_only or c.x1 - c.x0 < EMPTY_MIN_GAPS * c.gap:
        return False
    # The ink box runs from a margin left of the left barline to the right barline's first column; only the
    # columns between the two barlines count
    start = c.x0 - (c.x1 + 1 - c.ink.shape[1]) + int(BAR_SKIP_GAPS * c.gap)
    end = c.ink.shape[1] - 2
    bottom = int(round(2 * c.mid_y)) - (c.y0 + c.ref_y) - c.y0
    tab = c.ink[c.ref_y:bottom + 1, start:end].sum()
    # A system cut by the panel top starts above the cell box; its notation rows then start at the box top
    notation = c.ink[max(0, c.top_y):c.ref_y, start:end].sum()
    return tab < EMPTY_BAR_INK * c.gap * c.gap and (c.index > 0 or notation < EMPTY_NOTATION_INK * c.gap * c.gap)


def kept_cells(cells: list[Cell], typical: float) -> list[Cell]:
    """The cells that are measures: those at least EDGE_SHARE of a typical measure wide, and empty bars however
    short."""
    return [c for c in cells if c.x1 - c.x0 >= EDGE_SHARE * typical or empty_bar(c)]


def seed_function(found, seg):
    """The staffs' box at time t in crop coordinates, from the scan, or None when no staff was seen near t."""
    def seed_at(t):
        best = min(found, key=lambda k: abs(k - t))
        if abs(best - t) > 0.6 or not found[best]:
            return None
        sy = found[best]
        return (int(min(x.tab.x0 for x in sy)) - seg.x0, int(min(x.top for x in sy)) - seg.y0,
                int(max(x.tab.x1 for x in sy)) - seg.x0, int(max(x.tab.bottom for x in sy)) - seg.y0)
    return seed_at


def bands_function(found, seg):
    """Crop rows (top, bottom) of each staff system the scan saw near time t."""
    def bands_at(t):
        best = min(found, key=lambda k: abs(k - t))
        if abs(best - t) > 0.6:
            return []
        h = seg.y1 - seg.y0
        out = [(max(0, int(s.top) - seg.y0), min(h, int(s.tab.bottom) + 1 - seg.y0)) for s in found[best]]
        return [(a, b) for a, b in out if b - a >= 4]
    return bands_at


def first_thresholds(path: Path, crop, seg: Segment, seed_at) -> tuple[float, float]:
    """Ink thresholds from the segment's first frame with a staff, as views_in takes them."""
    for t, f in frames(path, OBS_FPS, crop=crop, start=seg.t0, length=seg.t1 - seg.t0):
        seed = seed_at(t)
        if seed is not None:
            g = to_gray(f, seg.polarity)
            return ink_threshold(g[max(0, seed[1]):seed[3], max(0, seed[0]):seed[2]], seg.paper, seg.polarity)
    return 128.0, 230.0


def plate_obs(v, seg: Segment, si: int, thr: float, fthr: float, index: int) -> Obs:
    """One still view, parsed into measure cells."""
    systems = parse_view(index, v.plate, v.rect, seg.polarity, thr, fthr, seg.gap, v.print_rect)
    cells = [c for ps in systems for c in ps.cells]
    x0, y0, x1, y1 = v.rect
    o = Obs(index, v.t0, v.t1, cells, v.playhead, si, (seg.x0, seg.y0), v.plate[y0:y1, x0:x1])
    o.all_cells = cells
    o.wrapped = v.wrapped
    o.frames = len(v.times)
    o.rect = v.rect
    o.single = len(systems) == 1
    # Paper against the darkest ink (the 50th darkest pixel, so a few noisy pixels don't count), and against the
    # tab staff lines: a sparse page has little solid ink, but a page fading in or out also shows its lines faint
    gray = to_gray(v.plate, seg.polarity)
    panel = gray[y0:y1, x0:x1]
    o.contrast = float(np.median(panel) - np.partition(panel.ravel(), 50)[50]) if panel.size > 100 else 0.0
    levels = [line_darkness(gray, ps.system.tab) for ps in systems]
    o.line_contrast = float(np.median([paper - line for line, paper in levels])) if levels else 0.0
    o.repeat_end = any(ps.repeat_end for ps in systems)
    band = ink_map(to_gray(v.plate, seg.polarity), thr, seg.gap)
    keep = np.zeros_like(band)
    keep[y0:y1, x0:x1] = True
    o.band = band & keep
    o.tall = cv2.morphologyEx(o.band.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 1), np.uint8)).astype(bool)
    o.image = v.plate
    o.systems = systems
    o.print_rect = v.print_rect
    return o


def glide_obs(path: Path, crop, seg: Segment, si: int, found, thr: float, fthr: float, index: int) -> Obs | None:
    """One view holding the whole glide of a segment, read from its mosaic."""
    systems_seen = [s for t, ss in found.items() if seg.t0 <= t <= seg.t1 for s in ss]
    if not systems_seen:
        return None
    # Group the systems seen by height on screen; the one that glides is read, others (a fretboard drawn
    # as six lines, a still title staff) are left out
    groups: list[list] = []
    for s in sorted(systems_seen, key=lambda s: s.top):
        if groups and s.top - groups[-1][-1].top < 2 * s.tab.gap:
            groups[-1].append(s)
        else:
            groups.append([s])
    m = rows = None
    for group in sorted(groups, key=len, reverse=True)[:GLIDE_GROUPS]:
        top = float(np.median([s.top for s in group])) - seg.y0
        bottom = float(np.median([s.tab.bottom for s in group])) - seg.y0
        g = float(np.median([s.tab.gap for s in group]))
        rows = (max(0, int(top - ABOVE_GAPS * g)), min(seg.y1 - seg.y0, int(bottom + BELOW_GAPS * g)))
        staff_rows = (max(rows[0], int(top)), min(rows[1], int(bottom) + 1))
        # The mosaic reaches up to the chord names and labels; the playhead and the motion use the analysis rows
        print_rows = (max(0, int(top - PRINT_ABOVE_GAPS * g)), rows[1])
        staff_x = (int(np.median([s.tab.x0 for s in group])) - seg.x0,
                   int(np.median([s.tab.x1 for s in group])) - seg.x0)
        m = glide.build(path, crop, print_rows, staff_rows, seg.polarity, thr, seg.gap, seg.t0, seg.t1,
                        analysis_rows=rows, staff_x=staff_x)
        if m is not None:
            break
    if m is None:
        return None
    h, w = m.image.shape[:2]
    print_top = print_rows[0]
    systems = parse_mosaic(index, m.image, (0, rows[0] - print_top, w, rows[1] - print_top), seg.polarity, thr, fthr,
                           seg.gap)
    # Blank staff that scrolls on after the last barline is read as a tail cell holding only line residue
    cells = [c for ps in systems for c in ps.cells
             if not (c.kind == "tail" and c.ink.sum() < BLANK_TAIL_INK * (c.x1 - c.x0) * c.gap)]
    for c in cells:
        c.glided = True
    o = Obs(index, seg.t0, seg.t1, cells, [], si, (seg.x0, seg.y0), m.image)
    o.all_cells = cells
    o.mosaic = m
    o.rows = print_rows
    o.frames = len(m.times)
    o.rect = (0, 0, w, h)
    o.single = len(systems) == 1
    o.contrast = 255.0
    o.repeat_end = any(ps.repeat_end for ps in systems)
    o.band = np.zeros((h, w), bool)
    o.tall = o.band
    o.image = m.image
    o.systems = systems
    o.print_rect = (0, 0, w, h)
    return o


def parse_mosaic(index: int, image: np.ndarray, rect: tuple[int, int, int, int], polarity: str, thr: float,
                 fthr: float, gap: float) -> list:
    """parse_view of a glide mosaic. A long one is parsed in overlapping pieces, so memory doesn't grow with the
    glide's length: each piece keeps the cells and barlines of its middle, and the pieces' systems are joined."""
    h, w = image.shape[:2]
    if w <= MOSAIC_PIECE:
        # Glides draw notes as tiles over the barlines more often than pages do. A mosaic joins stretches whose
        # staff can sit at different heights, so its lines break up where there are no chord diagrams
        return parse_view(index, image, rect, polarity, thr, fthr, gap, (0, 0, w, h), hidden_bars=True,
                          whole_lines=False)
    x0, y0, x1, y1 = rect
    joined: list = []
    for a in range(0, w - MOSAIC_OVERLAP, MOSAIC_PIECE - MOSAIC_OVERLAP):
        b = min(w, a + MOSAIC_PIECE)
        # The piece's middle: half an overlap in from each cut, so no measure kept was cut by the piece's edge
        lo = a + MOSAIC_OVERLAP // 2 if a > 0 else 0
        hi = b - MOSAIC_OVERLAP // 2 if b < w else w
        piece = parse_view(index, image[:, a:b], (max(0, x0 - a), y0, min(b, x1) - a, y1), polarity, thr, fthr, gap,
                           (0, 0, b - a, h), hidden_bars=True, whole_lines=False)
        for ps in piece:
            # The same system in an earlier piece: the one whose tab staff sits at the same height
            into = next((q for q in joined if abs(q.system.tab.top - ps.system.tab.top) < ps.system.tab.gap), None)
            for c in ps.cells:
                c.x0, c.x1, c.print_x0 = c.x0 + a, c.x1 + a, c.print_x0 + a
            cells = [c for c in ps.cells if lo <= (c.x0 + c.x1) / 2 < hi]
            bars = [(p + a, q + a) for p, q in ps.bars if lo <= p + a < hi]
            if into is None:
                ps.cells, ps.bars = cells, bars
                ps.system.tab.x0 += a
                ps.system.tab.x1 += a
                joined.append(ps)
            else:
                into.cells += cells
                into.bars += bars
                into.system.tab.x1 = ps.system.tab.x1 + a
                into.band = (min(into.band[0], ps.band[0]), max(into.band[1], ps.band[1]))
                into.repeat_end = into.repeat_end or ps.repeat_end
    joined.sort(key=lambda ps: ps.system.tab.top)
    for si, ps in enumerate(joined):
        for k, c in enumerate(ps.cells):
            c.system, c.index, c.band = si, k, ps.band
    return joined


def index_of(o: Obs, c: Cell) -> int:
    """Position of cell c in the view's cells. Cells hold arrays, so they are found by identity, not ==."""
    return next(i for i, x in enumerate(o.cells) if x is c)


def play_times(o: Obs) -> list[float]:
    """When playback reached each cell of a view: the first playhead sample over it, or an even share of the
    view's time when the playhead was never seen there."""
    if o.mosaic is not None:
        return [o.mosaic.time_at((c.x0 + c.x1) / 2) for c in o.cells]
    first: dict[int, float] = {}
    for t, x, y in o.playhead:
        i = cell_at(o, (x, y))
        if i is not None:
            first.setdefault(i, t)
    n = max(len(o.cells), 1)
    return [first.get(i, o.t0 + (i + 0.5) * (o.t1 - o.t0) / n) for i in range(len(o.cells))]


def measure_at(sid: int, o: Obs, c: Cell, play: list[float]) -> Measure:
    """A printed measure for slot sid, seen as cell c of view o."""
    if o.mosaic is not None:
        m = o.mosaic
        # A glided measure is placed when its center crossed the playhead (the panel centre when none was found)
        when = round(m.time_at((c.x0 + c.x1) / 2), 2)
        y = round(c.mid_y + o.offset[1] + m.y0, 1)
        seen = m.seen_at((c.x0 + c.x1) / 2)
        if seen is not None:
            # Never crossed the play column while the band moved: placed where it sat on that still screen
            return Measure(sid, o.index, c, round(seen[0], 2), when, round(seen[1] + o.offset[0], 1), y, None,
                           c.number, not c.complete)
        return Measure(sid, o.index, c, when, when, round(m.ref_x + o.offset[0], 1), y, when, c.number,
                       not c.complete)
    return Measure(sid, o.index, c, round((o.t0 + o.t1) / 2, 2), round(play[index_of(o, c)], 2),
                   round((c.x0 + c.x1) / 2 + o.offset[0], 1), round(c.mid_y + o.offset[1], 1), None, c.number,
                   not c.complete)


def measures_of(A: Assembly, obs: list[Obs], order: list[int]) -> list[Measure]:
    """The measures to print for the slots in `order`, each from its best sighting: a complete one when there
    is one, then one showing its tab staff, then one between two barlines, then the widest, then one from the view
    the measure before printed from, then the earliest."""
    times = {o.index: play_times(o) for o in obs}
    out = []
    prev = None
    for sid in order:
        vi, c = max(A.sightings[sid], key=lambda vc: (vc[1].complete, not vc[1].notation_only, vc[1].kind == "full",
                                                      vc[1].x1 - vc[1].x0, vc[0] == prev, -vc[0]))
        out.append(measure_at(sid, obs[vi], c, times[vi]))
        prev = vi
    return out


def play_path(A: Assembly, obs: list[Obs]) -> list[tuple[int, int, int, float]]:
    """(slot, view, cell, play time) in the order the video plays them. Measures a view shows again from the
    view before (a scroll overlap, a line carried up) are listed once; a replay is listed again."""
    path: list[tuple[int, int, int, float]] = []
    for o in obs:
        sl = A.view_slots.get(o.index)
        if not sl:
            continue
        sl = list(sl)
        known = [i for i, s in enumerate(sl) if s is not None]
        for a, b in zip(known, known[1:]):
            # A cell left unplaced between two placed cells whose slots are as far apart as the cells takes the
            # slot between theirs
            if b - a > 1 and sl[b] - sl[a] == b - a:
                for k in range(a + 1, b):
                    sl[k] = sl[a] + (k - a)
        play = play_times(o)
        step = [(s, o.index, i, play[i]) for i, s in enumerate(sl) if s is not None]
        if o.mosaic is None:
            seen = [i for i in (cell_at(o, (x, y)) for _, x, y in o.playhead) if i is not None]
            if seen:
                played = {x[0] for x in path}
                head = [x for x in step if x[2] < min(seen)]
                if all(x[0] in played for x in head):
                    # Measures left of the leftmost cell the playhead touched were played before this view
                    step = [x for x in step if x[2] >= min(seen)]
        tied = A.view_links.get(o.index)
        if tied is not None:
            # Cells the join tied to the view before were played there
            path.extend(x for x in step if x[2] not in tied)
            continue
        ids = [s for s, _, _, _ in step]
        done = [s for s, _, _, _ in path]
        k = min(len(ids), len(done))
        while k > 0 and ids[:k] != done[-k:]:
            k -= 1
        path.extend(step[k:])
    return path


def path_measures(A: Assembly, obs: list[Obs], path) -> list[Measure]:
    """Measures in play order for the every-pass PDF: each image from its slot's best sighting, the time and
    place from where it was played."""
    best = {m.slot: m for m in measures_of(A, obs, A.order)}
    times = {o.index: play_times(o) for o in obs}
    out = []
    for sid, vi, ci, _ in path:
        here = measure_at(sid, obs[vi], obs[vi].cells[ci], times[vi])
        here.view, here.cell = best[sid].view, best[sid].cell
        out.append(here)
    return out
