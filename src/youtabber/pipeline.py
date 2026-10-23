"""Read a video into a score: find the tab, build clean views, cut measures, place them, and check the result."""
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import heapq
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from youtabber import counter, glide, ocr, track
from youtabber.assemble import GLIDE_CONTINUE, Assembly, Obs, assemble, cell_at, number_views, same_cell
from youtabber.check import (dwell, empty_stretches, fmt, glide_fade, glide_gaps, glide_number_gaps, glide_piece_cuts,
                             glide_repeats, glide_rows, glide_steps, layout_changes, number_checks,
                             numbers_against_cells, odd_widths, pieces, possible_duplicates, redraw_gaps, screen_breaks,
                             too_few)
from youtabber.match import resemble_scaled
from youtabber.observe import (ABOVE_GAPS, BELOW_GAPS, OBS_FPS, PRINT_ABOVE_GAPS, ink_map, ink_threshold, set_paper,
                               to_gray, views_in)
from youtabber.parse import Cell, line_darkness, parse_view
from youtabber.parts import Part, find_parts, merge_order, stream_of
from youtabber.scan import SCAN_FPS, Segment, scan
from youtabber.staff import find_staffs
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
# A glide is read in at most this many pieces, each at least this long, with rows from the staffs seen in its
# first GLIDE_ROWS_SECONDS
MAX_GLIDE_PIECES, GLIDE_PIECE_SECONDS, GLIDE_ROWS_SECONDS = 8, 2.0, 10.0
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
# A first segment is an intro in another layout when it lasts at most INTRO_SECONDS, ends INTRO_BREAK_SECONDS or more
# before the next, which overlaps its panel by less than INTRO_OVERLAP of their union and is spaced INTRO_GAP_SHARE
# or more apart, and the segments after it hold INTRO_REST_SECONDS of tab
INTRO_SECONDS, INTRO_BREAK_SECONDS, INTRO_OVERLAP, INTRO_GAP_SHARE, INTRO_REST_SECONDS = 15.0, 0.5, 0.7, 0.04, 60.0
# The text detector looks for measure numbers from this many line gaps above a system's top staff line to this many
# below it, and gives each to a barline at most NUMBER_REACH_GAPS line gaps from its middle
NUMBER_ABOVE_GAPS, NUMBER_BELOW_GAPS, NUMBER_REACH_GAPS = 3.0, 0.6, 2.5
# A detected number's middle lies at least this many line gaps above the top staff line
NUMBER_CLEAR_GAPS = 0.3
# A measure whose start sits more than this far inside its part's right edge showed on screen
ROOM_PX = 3
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
# Tab on screen for less than BRIEF_SECONDS in all, and for less than BRIEF_SHARE of the video, is no play-along: a
# picture of a page on a promo card, or lines that only look like a staff
BRIEF_SECONDS, BRIEF_SHARE = 10.0, 0.035
# Tab whose views find measures for less than READ_SHARE of their time is lines that only look like a staff
READ_SHARE = 0.12
# A segment shorter than SMALL_SECONDS whose lines are spaced closer than SMALL_GAP_SHARE of the longest segment's is a
# small picture of a page (a thumbnail on a promo card or an end screen), not the tab being played
SMALL_SECONDS, SMALL_GAP_SHARE = 10.0, 0.6
# A segment where several parts print can't be put in order when more than TANGLED_SHARE of its joins between screens
# are left unresolved, over at least TANGLED_JOINS joins, and so are more than TANGLED_SHARE of its folds
TANGLED_SHARE, TANGLED_JOINS = 0.5, 10
# A video without tab staffs shows text tab when this many of its frames, taken this many seconds apart, show a staff
# of typed dashes
TEXT_TAB_FRAMES, TEXT_TAB_STEP = 3, 5.0
TEXT_TAB_MESSAGE = ("The video shows text tab: rows of typed dashes with fret numbers on them (e|--3--). youtabber "
                    "reads tab drawn with staff lines, not typed as text.")

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
    t_ref: float | None      # glides only: when it crossed the playhead, timed by glide_middle
    label: int | None        # measure number, when the view's numbers were trusted
    partial: bool
    stream: int = 0          # the stream its slot belongs to
    pieces: list | None = None   # (view, cell) of each piece printed side by side, for a measure split by a page turn


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
    parts: list[Part] = field(default_factory=list)   # every segment's parts; a part's id is its place here
    intro: tuple[float, float] | None = None   # an opening stretch of other tab left out, such as a how-to-read demo

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
    if not segments and text_tab_shown(path):
        return r.fail("text_tab", TEXT_TAB_MESSAGE)
    if not segments:
        return r.fail("no_tab", f"No tab on screen: no steady set of evenly spaced staff lines on paper in "
                                f"{info.duration:.0f} s of video. youtabber can't read text tab (e|--3--) or tab "
                                f"drawn see-through over the video either.")
    held = sum(s.t1 - s.t0 for s in segments)
    if held < BRIEF_SECONDS and held < BRIEF_SHARE * info.duration:
        return r.fail("no_tab", f"No tab to read: lines like a tab staff show for only {held:.0f} s of "
                                f"{info.duration:.0f} s of video, too briefly for the song to be played from them. "
                                f"They are likely a small picture of a page, such as on a promo card, or lines "
                                f"that only look like tab, such as the frets of a guitar neck.")
    # Tab in short scattered pieces may be a screen recording of an editor, with short play-along clips in it
    clips = None
    if scattered(segments, info.duration):
        segments = [s for s in segments if not editor_screen(path, s)]
        r.segments = segments
        if not segments:
            return r.fail("no_tab", f"Tab shows only in short pieces ({held:.0f} s of {info.duration:.0f} s), on "
                                    f"screens that look like a program's window, as when a tab editor is recorded "
                                    f"while someone types. youtabber reads play-along videos, where the tab stays "
                                    f"on screen while the song plays.")
        clips = (segments[0].t0, segments[-1].t1)
    else:
        segments = r.segments = drawn_full_size(segments)
        if intro(segments):
            r.intro = (segments[0].t0, segments[0].t1)
            segments = r.segments = segments[1:]
    tell("find", 1.0)
    tell("read", 0.0)
    # The read stage's share is the video time its frames have reached, as a share of the whole video
    reached = share_of(progress, "read", 0.0, info.duration)
    obs: list[Obs] = []
    gaps = []
    # (segment start, segment end, why it isn't printed) of each part other than the main one that can't be read
    dropped: list[tuple[float, float, str]] = []
    # (from, to) of each stretch of a glide left unread after the tab panel was redrawn
    unread: list[tuple[float, float]] = []
    for si, seg in enumerate(segments):
        crop = (seg.x0, seg.y0, seg.x1 - seg.x0, seg.y1 - seg.y0)
        set_paper(seg.paper)
        r.tab_spans.append((round(seg.t0, 2), round(seg.t1, 2)))
        seed_at = seed_function(found, seg)
        if reached is not None:
            reached(seg.t0)
        seg_parts = find_parts(path, seg, si, found, 1 / SCAN_FPS)
        for part in seg_parts:
            part.id = len(r.parts)
            part.stream = stream_of(part, seg_parts)
            r.parts.append(part)
        read_views = []
        thr_of = {}
        for part in seg_parts:
            part_found = found if part.kind == "page" else part.found
            part_seed = seed_at if part.kind == "page" else seed_function(part.found, seg)
            if part.seen >= GLIDE_MIN_SECONDS and glide.glides(path, crop, seg.polarity, seg.paper, seg.t0, seg.t1,
                                                               bands_function(part_found, seg), seg.gap):
                # The part's tab glides past a playhead instead of holding still: read it from mosaics
                thr, fthr = first_thresholds(path, crop, seg, part_seed)
                glided, stopped = glide_pieces(path, crop, seg, si, part_found, thr, fthr, len(obs))
                if not glided and part.stream != 0:
                    # Any other part is left out, so the main part still prints
                    dropped.append((seg.t0, seg.t1, "it scrolls smoothly and its moving band could not be put "
                                                    "together"))
                    continue
                if not glided:
                    return r.fail("check_failed", f"The tab scrolls smoothly from {fmt(seg.t0)} to {fmt(seg.t1)}, "
                                                  f"but the moving band could not be put together, so no PDF was "
                                                  f"written.")
                if stopped is not None:
                    unread.append((stopped, seg.t1))
                for o in glided:
                    o.part, o.stream = part.id, part.stream
                    obs.append(o)
                    r.panels.append({"t0": round(o.t0, 2), "t1": round(o.t1, 2),
                                     "box": [seg.x0, seg.y0 + o.rows[0], seg.x1 - seg.x0, o.rows[1] - o.rows[0]]})
                if reached is not None:
                    reached(seg.t1)
                continue
            views, thr, fthr = views_in(path, crop, seg.polarity, seg.gap, seg.paper, part_seed, seg.t0, seg.t1,
                                        "playhead" in EVIDENCE, progress=reached, lead=lead_before(segments, si))
            held = sum(v.t1 - v.t0 for v in views)
            shown = part.seen if part.kind == "band" else seg.t1 - seg.t0
            if shown >= GLIDE_MIN_SECONDS and held < HELD_SHARE * shown:
                if part.stream == 0:
                    # Tab that keeps changing without holding still or scrolling smoothly: the still views miss
                    # most of it
                    return r.fail("check_failed", f"The tab at {fmt(seg.t0)}-{fmt(seg.t1)} holds still for only "
                                                  f"{held:.0f} of {shown:.0f} s and doesn't scroll smoothly either, so "
                                                  f"most of it can't be read; no PDF was written.")
                # Any other part is left out, so the main part still prints
                dropped.append((seg.t0, seg.t1, f"it holds still for only {held:.0f} of {shown:.0f} s"))
                continue
            thr_of[part.id] = thr
            read_views += [(v, part, thr, fthr) for v in views]
        # The parts' views in one list by time, the upper part first when two start together
        still = []
        mine = []
        for v, part, thr, fthr in sorted(read_views, key=lambda x: (x[0].t0, x[1].rows[0])):
            o = plate_obs(v, seg, si, thr, fthr, len(obs))
            if v.brief and not o.all_cells:
                # A first or last screen held briefly with no measures is a page coming or going, not time with tab
                continue
            o.part, o.stream = part.id, part.stream
            obs.append(o)
            mine.append(o)
            # The reported panel is the print area: the print rect's columns, its systems' bands' rows
            x0, y0, x1, y1 = v.print_rect
            if o.systems:
                y0, y1 = min(ps.band[0] for ps in o.systems), max(ps.band[1] for ps in o.systems)
            still.append({"t0": round(v.t0, 2), "t1": round(v.t1, 2),
                          "box": [int(x0 + seg.x0), int(y0 + seg.y0), int(x1 - x0), int(y1 - y0)]})
        # A view's print area stays in place through a page turn to the part's next view. The scan finds a
        # segment's ends only to within half a scan step, so each part's first and last areas reach that far past them
        for part in seg_parts:
            areas = [a for a, o in zip(still, mine) if o.part == part.id]
            for a, b in zip(areas, areas[1:]):
                if b["t0"] - a["t1"] <= TURN_SECONDS:
                    a["t1"] = b["t0"]
            if areas:
                before = segments[si - 1].t1 if si > 0 else 0.0
                after = segments[si + 1].t0 if si + 1 < len(segments) else r.duration
                areas[0]["t0"] = round(max(before, min(areas[0]["t0"], seg.t0 - 0.5 / SCAN_FPS)), 2)
                areas[-1]["t1"] = round(min(after, max(areas[-1]["t1"], seg.t1 + 0.5 / SCAN_FPS)), 2)
        r.panels += still
        gaps.append(seg.gap)
        # How the part's band moved between each pair of its views, read from the frames between them
        for p, o in part_pairs(mine):
            part = r.parts[o.part]
            o.move = view_move(path, crop, seg, p, o, thr_of[part.id], (part.rows[0] - seg.y0, part.rows[1] - seg.y0))
    r.views = obs
    tell("read", 1.0)
    return finish(path, info, r, segments, scrolls, clips, obs, gaps, dropped, unread, tell)


def finish(path, info, r, segments, scrolls, clips, obs, gaps, dropped, unread, tell) -> Reading:
    """Assemble the read views into measures, notices and the play path. `dropped` holds the parts left out because
    they can't be read and `unread` the stretches of a glide left unread, as `read` lists them."""
    tell("assemble", 0.0)
    gap = float(np.median(gaps))
    # A title card or photo laid over the first screen of a page adds marks the next screen doesn't have; the
    # same music shows clean on that next screen, so the covered view gives no measures
    for p, o in part_pairs(obs):
        if covered(p, o):
            p.all_cells = []
    # A view caught in a fade has weak contrast and loses its barlines; its music is shown again at full
    # contrast, so it is left out. Both its darkest ink and its staff lines are weak. A part of one or two views
    # is judged against the clearest views of its theme.
    clearest: dict[str, tuple[float, float]] = {}
    for o in obs:
        if o.mosaic is None:
            pol = segments[o.seg].polarity
            ink, line = clearest.get(pol, (0.0, 0.0))
            clearest[pol] = (max(ink, o.contrast), max(line, o.line_contrast))
    for part in r.parts:
        seg = segments[part.seg]
        group = [o for o in obs if o.part == part.id and o.mosaic is None]
        if len(group) >= 3:
            ink = float(np.median([o.contrast for o in group]))
            line = float(np.median([o.line_contrast for o in group]))
        else:
            ink, line = clearest.get(seg.polarity, (0.0, 0.0))
        for o in group:
            if o.contrast < FADE_SHARE * ink and o.line_contrast < FADE_LINE_SHARE * line:
                o.all_cells = []
    # Ink at the same screen place in nearly every view of a part is fixed to the frame (a clef and key
    # panel that stays while the music scrolls behind it)
    for part in r.parts:
        group = [o for o in obs if o.part == part.id]
        if not group or any(o.mosaic is not None for o in group):
            # A glide's mosaic is never lined up with another view, so it needs no map of frame-fixed ink
            fixed = None
        elif len(group) >= FIXED_MIN_VIEWS:
            share = np.mean([o.band for o in group], axis=0)
            fixed = cv2.dilate((share >= FIXED_SHARE).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        else:
            fixed = np.zeros_like(group[0].band)
        for o in group:
            o.fixed = fixed
    drop_fades(obs)
    # A cell much narrower than a typical measure is not one: a clef and key signature before a
    # start-repeat barline, or a sliver cut by the panel edge
    full_widths = [c.x1 - c.x0 for o in obs for c in o.all_cells if c.kind == "full"]
    typical = float(np.median(full_widths)) if full_widths else 0.0
    for o in obs:
        o.cells = kept_cells(o.all_cells, typical)
        o.bars = sorted({c.x0 for c in o.all_cells if c.kind != "lead"} | {c.x1 for c in o.all_cells if c.kind != "tail"})
    # Each stream is numbered and assembled on its own
    streams = sorted({o.stream for o in obs}) or [0]
    stream_obs = {k: [o for o in obs if o.stream == k] for k in streams}
    r.numbering = number_streams(stream_obs, ocr.read_number,
                                 lambda o: detected_numbers(o, segments[o.seg].polarity))
    for o in obs:
        if o.mosaic is not None and o.counter:
            counted_numbers(o)
    # A measure seen only on its notation staff, its tab out of the panel, can be placed by its number alone
    for o in obs:
        o.cells = [c for c in o.cells if not c.notation_only or c.number is not None]

    def words_of(o: Obs) -> list[str]:
        return ocr.find_words(o.plate) if o.plate is not None else []

    place = {p.id: p.rows[0] for p in r.parts}
    right = {p.id: segments[p.seg].x1 - segments[p.seg].x0 for p in r.parts}
    by_stream, main, folded = readings(stream_obs, streams, obs, gap, place, right, words_of)
    r.assembly = main
    r.candidates = [Candidate("primary", "once", measures_of(main, obs, main.order))]
    if folded is not None:
        r.candidates.append(Candidate("folded", "once", measures_of(folded, obs, folded.order)))

    notices = join_notices(main, obs) + edge_notices(main, obs, r.candidates[0].measures)
    if r.intro is not None:
        notices.append(Notice(round(r.intro[0], 2), round(r.intro[1], 2), "intro",
                              f"The tab at {fmt(r.intro[0])}-{fmt(r.intro[1])} is drawn in another layout than the "
                              f"rest and is left out as an intro; check that no measure of the song is missing there.",
                              True))
    for t0, t1 in main.other_music:
        notices.append(Notice(round(t0, 2), round(t1, 2), "version_change",
                              "The measure numbers here repeat ones already printed, but the music looks different: "
                              "this may be another version or section of the song, printed over the earlier "
                              "measures. Check these measures.", True))
    ranges = []
    for t0, t1, j0, j1 in main.unresolved_folds:
        first_seen = obs[main.slots[j0].obs]
        ranges.append((first_seen.t0, first_seen.t1, t0, t1))
        notices.append(Notice(round(t0, 2), round(t1, 2), "unresolved_fold",
                              f"The measures at {fmt(t0)}-{fmt(t1)} repeat ones first shown at "
                              f"{fmt(first_seen.t0)}-{fmt(first_seen.t1)}, with no sign whether the song plays "
                              f"them again or the video replays them.", False))
    notices += repeat_notices(main, obs)
    primary = r.candidates[0].measures
    printing = {obs[m.view].part for m in primary}
    # A part that prints nothing while another part of its segment prints is lines that only look like tab, such as
    # guitar strings in footage, so its views are left out of the checks
    idle = {p.id for p in r.parts if p.id not in printing and any(q.seg == p.seg and q.id in printing for q in r.parts)}
    # Each stream is checked on its own views, so another part's views shown in between aren't its neighbours
    for k, A in zip(streams, by_stream):
        mine = [o for o in stream_obs[k] if o.part not in idle]
        for w in (possible_duplicates(A, mine, gap) + dwell(mine, A) + odd_widths(mine) + numbers_against_cells(mine)
                  + screen_breaks(A, mine)):
            notices.append(Notice(w["t0"], w["t1"], w["kind"], w["text"], True))
    for w in layout_changes(segments, obs) + glide_piece_cuts(obs) + redraw_gaps(obs):
        notices.append(Notice(w["t0"], w["t1"], w["kind"], w["text"], True))
    for a, b in unread:
        for p0, p1 in pieces(a, b):
            notices.append(Notice(round(p0, 2), round(p1, 2), "glide_rows",
                                  f"The smooth scroll from {fmt(p0)} to {fmt(p1)} wasn't read after the tab panel was "
                                  f"redrawn; measures played there are missing.", True))
    notices += part_notices(r.parts, obs, primary)
    for t0, t1, why in dropped:
        notices.append(Notice(round(t0, 2), round(t1, 2), "parts",
                              f"Another part of the tab shows at {fmt(t0)}-{fmt(t1)}, but {why}, so it isn't "
                              f"printed.", True))
    for o in obs:
        if o.mosaic is not None:
            m = o.mosaic
            # Measures read from the first frame alone: on screen between the playhead and the reading column
            # before the scroll moved
            seeded = [c for c in o.cells if (c.x0 + c.x1) / 2 < m.seeded] if o.first_piece else []
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
            early = [c for c, t in zip(o.cells, times)
                     if t < o.t0 + step and (c.x0 + c.x1) / 2 >= m.seeded] if o.first_piece else []
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
            if m.tail_from < m.image.shape[1] and o.last_piece:
                t0 = min(m.time_at(ending[0].x0 if ending else m.tail_from), o.t1 - 1)
                text = ("The last measures of this smooth scroll were still ahead of the playhead when the scroll "
                        "ended and are read from its last frame only; check them." if ending else
                        "This smooth scroll ended with staff still ahead of the playhead, and no measure was read "
                        "from it; check that the last measures are all printed.")
                notices.append(Notice(round(t0, 2), round(o.t1, 2), "glide_end", text, True))
            for w in (glide_repeats(o, gap) + glide_gaps(o) + (glide_fade(o, step) if o.last_piece else [])
                      + glide_number_gaps(o) + glide_steps(o) + glide_rows(o)):
                notices.append(Notice(w["t0"], w["t1"], w["kind"], w["text"], True))
    for w in empty_stretches(r.tab_spans, [m.play_t for m in primary]):
        notices.append(Notice(w["t0"], w["t1"], w["kind"], w["text"], True))
    # Parts shown at once count their measures apart, so each stream's numbers are checked on their own
    for k in streams:
        for w in stream_number_checks(k, primary, r.parts, stream_obs[k], info.duration):
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
    # Lines that only look like tab, such as chord diagrams or a picking pattern, give a stray cell now and then
    shown = sum(o.t1 - o.t0 for o in obs)
    read_time = sum(o.t1 - o.t0 for o in obs if o.cells)
    if n and read_time < READ_SHARE * shown:
        return r.fail("no_tab", f"No tab to read: lines like a tab staff show for {shown:.0f} s, but measures "
                                f"were found on them for only {read_time:.0f} s. They are likely chord diagrams or "
                                f"other drawings that only look like tab.")
    short = too_few(obs, n, main) if n else None
    if short:
        return r.fail("check_failed", short)
    tangle = tangled(segments, r.parts, obs, main, primary)
    if tangle is not None:
        return r.fail("check_failed", f"Several parts of the tab show at once at {fmt(tangle.t0)}-{fmt(tangle.t1)}, "
                                      f"and most screens there couldn't be joined to the one before, so their "
                                      f"measures can't be put in order; no PDF was written.")
    # Short stretches of other lines can hold segments while the tab itself is typed as text
    if not n and text_tab_shown(path):
        return r.fail("text_tab", TEXT_TAB_MESSAGE)
    if not n:
        return r.fail("no_tab", "Found lines that look like a tab staff, but no measures between barlines on them. "
                                "youtabber can't read text tab (e|--3--), tab drawn see-through over the video, or tab "
                                "without barlines.")
    if len(streams) == 1:
        r.path = play_path(main, obs)
    else:
        # Each stream's path with the slot ids the joined assembly gave it, merged by play time
        paths = [[(s + off, v, c, t) for s, v, c, t in play_path(A, stream_obs[k])]
                 for A, k, off in zip(by_stream, streams, slot_offsets(by_stream))]
        r.path = list(heapq.merge(*paths, key=lambda x: x[3]))
    r.every_pass = [Candidate("primary", "all", path_measures(main, obs, r.path))]
    tell("assemble", 1.0)
    return r


def stream_number_checks(k: int, primary: list[Measure], parts: list[Part], views: list[Obs],
                         duration: float) -> list[dict]:
    """The measure-number notices of stream k's printed measures, from its parts' scan sightings and its views."""
    ws = number_checks([(m.play_t, m.label) for m in primary if m.stream == k], True, duration)
    if k == 0:
        return ws
    # A part beside the main one can enter late or leave the screen; numbers it never showed aren't missed
    shown = sorted(t for p in parts if p.stream == k for t, ss in p.found.items() if ss)
    first = min((o.t0 for o in views if o.cells), default=0.0)
    return [w for w in ws if on_screen(w, shown, first)]


def on_screen(w: dict, shown: list[float], first: float) -> bool:
    """False for a numbers notice about measures the part never showed: numbers before the part first came on
    screen, or numbers skipped while it was off screen for longer than a page turn."""
    if w["kind"] == "numbers_start":
        return bool(shown) and shown[0] < first - TURN_SECONDS
    if w["kind"] == "numbers_skipped":
        ts = [w["t0"]] + [t for t in shown if w["t0"] < t < w["t1"]] + [w["t1"]]
        return all(b - a <= TURN_SECONDS for a, b in zip(ts, ts[1:]))
    return True


def readings(stream_obs: dict[int, list[Obs]], streams: list[int], obs: list[Obs], gap: float,
             place: dict[int, float], right: dict[int, float],
             words_of) -> tuple[list[Assembly], Assembly, Assembly | None]:
    """The main reading (each stream's assembly and the joined one), and the folded reading when the main one leaves
    folds unresolved."""
    assigned = numbers_of(obs)
    by_stream = [assemble(stream_obs[k], gap, False, EVIDENCE, words_of) for k in streams]
    main = joined(by_stream, obs, place, right, streams)
    if not main.unresolved_folds:
        return by_stream, main, None
    # The main reading clears numbers that a confirmed track join contradicts. The folded reading starts from the
    # numbers as read, before the main reading cleared any, and the main reading's numbers are put back afterwards
    # for its measures and play path
    cleared = numbers_of(obs)
    set_numbers(obs, assigned)
    folded = joined([assemble(stream_obs[k], gap, True, EVIDENCE, words_of) for k in streams], obs, place, right,
                    streams)
    set_numbers(obs, cleared)
    return by_stream, main, folded


def numbers_of(obs: list[Obs]) -> list[list[int | None]]:
    """Each view's cell numbers."""
    return [[c.number for c in o.cells] for o in obs]


def set_numbers(obs: list[Obs], numbers: list[list[int | None]]) -> None:
    """Give each view's cells the numbers `numbers_of` saved."""
    for o, ns in zip(obs, numbers):
        for c, n in zip(o.cells, ns):
            c.number = n


def number_streams(stream_obs: dict[int, list[Obs]], read, more=None) -> str:
    """Number each stream's views apart from the others', since parts shown at once count their measures apart.
    "numbered" when any stream is."""
    found = [number_views(views, read, more) for views in stream_obs.values()]
    return "numbered" if "numbered" in found else "unnumbered"


def detected_numbers(o: Obs, polarity: str) -> dict[int, int]:
    """Measure numbers the OCR's text detector finds above each system of a still view, by cell: each number goes to
    the cell whose left barline is nearest its middle, within NUMBER_REACH_GAPS line gaps."""
    out: dict[int, int] = {}
    if o.image is None:
        return out
    gray = to_gray(o.image, polarity)
    for si, ps in enumerate(o.systems):
        st = ps.system.notation or ps.system.tab
        y0, y1 = int(max(0, st.top - NUMBER_ABOVE_GAPS * st.gap)), int(st.top + NUMBER_BELOW_GAPS * st.gap)
        cells = [(i, c) for i, c in enumerate(o.cells) if c.system == si]
        if y1 - y0 < 8 or not cells:
            continue
        for left, right, row, n in ocr.numbers_in_line(gray[y0:y1]):
            # A measure number sits above the staff; digits on its top line are frets
            if y0 + row > st.top - NUMBER_CLEAR_GAPS * st.gap:
                continue
            mid = (left + right) / 2
            i, c = min(cells, key=lambda ic: abs(ic[1].x0 - mid))
            if abs(c.x0 - mid) <= NUMBER_REACH_GAPS * st.gap:
                out.setdefault(i, n)
    return out


def slot_offsets(assemblies: list[Assembly]) -> list[int]:
    """Where each assembly's slots start in the joined assembly."""
    return [sum(len(A.slots) for A in assemblies[:k]) for k in range(len(assemblies))]


def joined(assemblies: list[Assembly], obs: list[Obs], place: dict[int, float], right: dict[int, float],
           streams: list[int]) -> Assembly:
    """One assembly from the assemblies of `streams`: slot ids made unique, measures printed in the order they first
    show, those first shown on one screen in the order of their parts' rows (`place`, part id -> source row).
    `right` is each part's right edge in crop columns."""
    if len(assemblies) == 1:
        A = assemblies[0]
        for s in A.slots:
            s.stream = streams[0]
        A.stream_orders = {streams[0]: list(A.order)}
        return A
    out = Assembly()
    keys = []
    for A, k, off in zip(assemblies, streams, slot_offsets(assemblies)):
        out.slots += [replace(s, id=s.id + off, stream=k) for s in A.slots]
        out.stream_orders[k] = [sid + off for sid in A.order]
        when = first_shown(A, obs, right)
        keys += [(when[sid], place[obs[A.slots[sid].obs].part], sid + off) for sid in A.order]
        out.decisions += A.decisions
        out.unresolved_folds += [(t0, t1, j0 + off, j1 + off) for t0, t1, j0, j1 in A.unresolved_folds]
        out.maybe_repeats += [(t0, t1, j + off) for t0, t1, j in A.maybe_repeats]
        out.view_slots.update({v: [None if s is None else s + off for s in sl] for v, sl in A.view_slots.items()})
        out.view_links.update(A.view_links)
        out.sightings.update({s + off: v for s, v in A.sightings.items()})
        out.edge_joins += A.edge_joins
        out.other_music += A.other_music
        out.stitches.update(A.stitches)
    out.order = [keys[i][2] for i in merge_order(keys)]
    out.decisions.sort(key=lambda d: d.t)
    return out


def first_shown(A: Assembly, obs: list[Obs], right: dict[int, float]) -> dict[int, float]:
    """When each slot first showed: the start of the view it was first read in, or of the part's view before when that
    view showed its start at the right edge (`right`, part id -> crop column), cut too short to read. A measure first
    read in a glide mosaic showed by the time its start reached the play column, kept within its view."""
    when = {}
    for s in A.slots:
        o = obs[s.obs]
        when[s.id] = o.t0 if o.mosaic is None else min(o.t1, max(o.t0, o.mosaic.time_at(s.first.x0)))
    last: dict[int, Obs] = {}
    for o in sorted((obs[i] for i in A.view_slots), key=lambda o: o.t0):
        p, last[o.part] = last.get(o.part), o
        # A glide view's slots are timed from its mosaic above
        if p is None or o.mosaic is not None or o.t0 - p.t1 > TURN_SECONDS or not p.cells:
            continue
        sl = A.view_slots[o.index]
        shown = set(A.view_slots.get(p.index) or [])
        tied = {i for i, sid in enumerate(sl) if sid is not None and sid in shown}
        if tied:
            # The music slid by the median shift of the barlines of the cells the two views share. The next measure
            # showed in p when its start, put in p's columns by that shift, lay inside the part's right edge
            k = max(tied) + 1
            dx = shared_shift(p, o, A.view_slots[p.index], sl, tied)
            room = dx is not None and k < len(o.cells) and o.cells[k].x0 + dx < right[o.part] - ROOM_PX
        else:
            # A page turn: the view before showed the start of o's first new measure when it dropped a tail cell
            # too narrow to keep
            k = next((i for i, sid in enumerate(sl) if sid is not None and A.slots[sid].obs == o.index), len(sl))
            room = any(c.kind == "tail" and not any(c is d for d in p.cells) for c in p.all_cells)
        if room and k < len(sl) and sl[k] is not None and A.slots[sl[k]].obs == o.index:
            when[sl[k]] = p.t0
    return when


def shared_shift(p: Obs, o: Obs, p_slots: list, o_slots: list, tied: set[int]) -> float | None:
    """The median shift from o's columns to p's, over the barlines of o's cells `tied` to p's cells. A lead cell
    starts at the panel edge and a tail cell ends there, so those ends aren't barlines and are skipped."""
    moved = []
    for i in tied:
        a, b = p.cells[p_slots.index(o_slots[i])], o.cells[i]
        if "lead" not in (a.kind, b.kind):
            moved.append(a.x0 - b.x0)
        if "tail" not in (a.kind, b.kind):
            moved.append(a.x1 - b.x1)
    return float(np.median(moved)) if moved else None


def tangled(segments: list[Segment], parts: list[Part], obs: list[Obs], A: Assembly,
            primary: list[Measure]) -> Segment | None:
    """The first segment where two or more parts print measures and most joins between its screens and most runs of
    music shown again are left unresolved, so its measures can't be put in order. Unresolved joins alone are common
    where split-screen panels turn their pages apart."""
    printing = {obs[m.view].part for m in primary}
    for si in sorted({p.seg for p in parts}):
        if sum(p.id in printing for p in parts if p.seg == si) < 2:
            continue
        mine = [d for d in A.decisions if d.view is not None and obs[d.view].seg == si]
        joins = [d for d in mine if d.kind == "join"]
        folds = [d for d in mine if d.kind == "fold"]
        if (len(joins) >= TANGLED_JOINS and sum(not d.resolved for d in joins) > TANGLED_SHARE * len(joins)
                and sum(not d.resolved for d in folds) > TANGLED_SHARE * len(folds)):
            return segments[si]
    return None


def part_notices(parts: list[Part], obs: list[Obs], primary: list[Measure]) -> list[Notice]:
    """A `parts` notice for each segment where two or more parts print measures, over the first view of the second
    part to show. A part none of whose views gives a measure of `primary` counts for nothing."""
    printing = {obs[m.view].part for m in primary}
    first: dict[int, Obs] = {}
    for o in obs:
        if o.part not in first or o.t0 < first[o.part].t0:
            first[o.part] = o
    notices = []
    for seg in sorted({p.seg for p in parts}):
        shown = sorted((first[p.id] for p in parts if p.seg == seg and p.id in printing), key=lambda o: o.t0)
        if len(shown) >= 2:
            o = shown[1]
            notices.append(Notice(round(o.t0, 2), round(o.t1, 2), "parts",
                                  f"{len(shown)} parts show at once here, each in its own band or panel; each is read "
                                  f"on its own and their measures are printed in the order they first show, top "
                                  f"before bottom and left before right.", True))
    return notices


def join_notices(A: Assembly, obs: list[Obs]) -> list[Notice]:
    """A notice for each join left unresolved, from a second before the screen to its end. A view the playhead
    started over reaches back to the start of its stream's view before."""
    notices = []
    for d in A.decisions:
        if not d.resolved and d.kind == "join":
            o = obs[d.view]
            # A page the playhead started over may repeat its stream's page before, so the notice covers that page too
            start = (next(x for x in reversed(obs[:o.index]) if x.cells and x.stream == o.stream).t0 if o.restarted
                     else d.t - 1)
            notices.append(Notice(round(start, 2), round(o.t1, 2), "unresolved_join",
                                  "Couldn't tell how this screen continues from the one before; check that no "
                                  "measure is missing or printed twice here.", True))
    return notices


def repeat_notices(A: Assembly, obs: list[Obs]) -> list[Notice]:
    """A notice for each view printed as new that may repeat earlier measures it matches only where those were cut
    off or highlighted."""
    # A view with an unresolved fold already has its own notice and a folded PDF
    folded_views = {(t0, t1) for t0, t1, _, _ in A.unresolved_folds}
    notices = []
    for t0, t1, j in A.maybe_repeats:
        if (t0, t1) in folded_views:
            continue
        first_seen = obs[A.slots[j].obs]
        notices.append(Notice(round(t0, 2), round(t1, 2), "possible_duplicate",
                              f"The measures at {fmt(t0)}-{fmt(t1)} may repeat ones first shown at "
                              f"{fmt(first_seen.t0)}-{fmt(first_seen.t1)}: they match only where those were cut off or "
                              f"highlighted, so they are printed again. Check whether they repeat.", True))
    return notices


def edge_notices(A: Assembly, obs: list[Obs], printed: list[Measure] = ()) -> list[Notice]:
    """A notice for each join by the playhead that may leave out notes. It covers the page before, where the cut
    measure was played, the next page, the view that first showed the measure and the one its image prints from."""
    printed_view = {m.slot: m.view for m in printed}
    whole = {m.slot for m in printed if m.pieces or m.cell.complete}
    notices = []
    for pi, vi in A.edge_joins:
        sid = A.view_slots[vi][0]
        st = A.stitches.get(vi)
        resolved = next((d.resolved for d in A.decisions if d.view == vi and d.chosen.startswith("playhead")), False)
        # A measure printed whole, or from both pieces, after a join the playhead settled has no note left out
        if resolved and st is not None and (sid in whole or (st[2] == "whole" and printed_view.get(sid) == vi)):
            continue
        views = [obs[i] for i in (pi, vi, A.slots[sid].obs, printed_view.get(sid, vi))]
        notices.append(Notice(round(min(v.t0 for v in views), 2), round(max(v.t1 for v in views), 2), "edge_join",
                              "A measure cut by the page edge was joined to the next page by where the playhead was; "
                              "check that no measure, or part of one, is missing here.", True))
    return notices


def share_of(progress: Progress | None, stage: str, t0: float, t1: float) -> Callable[[float], None] | None:
    """A callback for a frame loop that tells `progress` how far video time t has got through t0..t1, or None
    when nobody is listening, so the loop does nothing extra."""
    if progress is None:
        return None
    span = max(t1 - t0, 1e-6)
    return lambda t: progress(stage, min(1.0, max(0.0, (t - t0) / span)))


def text_tab_shown(path: Path) -> bool:
    """True when enough frames show a staff whose lines are typed dashes, as wide as a tab staff."""
    seen = 0
    for _, f in frames(path, 1 / TEXT_TAB_STEP):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        seen += any(s.dashed and s.x1 - s.x0 >= g.shape[1] / 5 for s in find_staffs(g, keep_dashed=True))
        if seen >= TEXT_TAB_FRAMES:
            return True
    return False


def lead_before(segments: list[Segment], si: int) -> float:
    """How far before segment si's start its still views are looked for. The scan sees a segment start only to
    within one of its steps, so a first screen shown for less than a step may lie just before it; the look back
    stops where the segment before ends."""
    before = segments[si - 1].t1 if si > 0 else 0.0
    return max(0.0, min(1 / SCAN_FPS, segments[si].t0 - before))


def drop_fades(obs: list[Obs]) -> None:
    """Clear the cells of each brief first or last screen with fewer than two whole measures, or whose measures the
    screen next to it shows again in order: it is a page fading or drawing in or out."""
    for p, o in part_pairs(obs):
        if p.brief and (sum(c.kind == "full" for c in p.all_cells) < 2 or shown_again(p, o)):
            p.all_cells = []
        if o.brief and (sum(c.kind == "full" for c in o.all_cells) < 2 or shown_again(o, p)):
            o.all_cells = []


def shown_again(a: Obs, b: Obs) -> bool:
    """Whether view b shows all of view a's whole measures again, in order and next to each other. Cells cut by the
    panel edge are left out: a page drawing in shows them blank."""
    cells = [c for c in a.all_cells if c.kind == "full"]
    other = [c for c in b.all_cells if c.kind == "full"]
    if not cells:
        return False
    gap = cells[0].gap
    return any(all(same_cell(c, other[k + i], gap) for i, c in enumerate(cells))
               for k in range(len(other) - len(cells) + 1))


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


def drawn_full_size(segments: list[Segment]) -> list[Segment]:
    """The segments left once short ones drawn much smaller than the longest are dropped: a thumbnail of the page on
    a promo card or an end screen, not the tab being played."""
    longest = max(segments, key=lambda s: s.t1 - s.t0)
    return [s for s in segments if s.t1 - s.t0 >= SMALL_SECONDS or s.gap >= SMALL_GAP_SHARE * longest.gap]


def intro(segments: list[Segment]) -> bool:
    """True when the first segment is a short opening in another layout, such as a channel's how-to-read-tab demo:
    at most INTRO_SECONDS long, followed after a break by at least INTRO_REST_SECONDS of tab drawn in another panel
    at another line spacing."""
    if len(segments) < 2:
        return False
    a, b = segments[0], segments[1]
    ix = max(0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - ix
    return (a.t1 - a.t0 <= INTRO_SECONDS and b.t0 - a.t1 >= INTRO_BREAK_SECONDS
            and sum(s.t1 - s.t0 for s in segments[1:]) >= INTRO_REST_SECONDS
            and ix < INTRO_OVERLAP * union and abs(a.gap - b.gap) >= INTRO_GAP_SHARE * b.gap)


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


def view_move(path: Path, crop, seg: Segment, p: Obs, o: Obs, thr: float, within: tuple[int, int]):
    """How the band moved from still view p to still view o: measured on the systems' rows and the panels'
    columns, from p's last frame to o's first. The rows are kept inside `within`, the part's crop rows, so another
    part shown at the same time doesn't count."""
    bands = [ps.band for ps in p.systems + o.systems]
    rows = (min(b[0] for b in bands), max(b[1] for b in bands)) if bands else (min(p.rect[1], o.rect[1]),
                                                                              max(p.rect[3], o.rect[3]))
    rows = (max(rows[0], within[0]), min(rows[1], within[1]))
    cols = (min(p.rect[0], o.rect[0]), max(p.rect[2], o.rect[2]))
    return track.transition(path, crop, rows, cols, seg.polarity, thr, seg.gap, p.t1 - 1 / OBS_FPS, o.t0)


def part_pairs(obs: list[Obs]) -> list[tuple[Obs, Obs]]:
    """Each still view paired with the one before it of the same part in the same segment. Parts shown at once
    have their views interleaved in time, so a view's neighbor in `obs` may belong to another part."""
    return track.pairs([o for o in obs if o.mosaic is None], key=lambda o: (o.seg, o.part))


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
    o.repeat_bars = [(k, x) for k, ps in enumerate(systems) for x in ps.repeat_bars]
    band = ink_map(to_gray(v.plate, seg.polarity), thr, seg.gap)
    keep = np.zeros_like(band)
    keep[y0:y1, x0:x1] = True
    o.band = band & keep
    o.tall = cv2.morphologyEx(o.band.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 1), np.uint8)).astype(bool)
    o.image = v.plate
    o.systems = systems
    o.print_rect = v.print_rect
    o.brief = v.brief
    return o


def glide_pieces(path: Path, crop, seg: Segment, si: int, found, thr: float, fthr: float,
                 index: int) -> tuple[list[Obs], float | None]:
    """The glide of a segment as views read from mosaics, one per stretch where its staffs keep their rows on
    screen. Where the panel is redrawn with the staffs at other heights, the next piece starts on the new rows.
    Consecutive pieces both show the screen at the redraw; each measure is kept in the piece it glided through.
    Every piece gets the bar counter's runs over the whole glide. Also gives the time reading stopped short of the
    segment's end after a redraw, or None when it read to the end or to within GLIDE_PIECE_SECONDS of it."""
    # The bar counter is read in a worker thread while the mosaics are built: ffmpeg and the OCR model run outside
    # the GIL, so the two overlap. Leaving the with block waits for the worker, so it is always told to stop first,
    # which ends it early on an error or Ctrl-C, or when no piece was read and its result isn't needed.
    # The recognizer is loaded before the worker starts, as loading it from two threads at once isn't safe.
    ocr.engine()
    stop = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        reading = pool.submit(counter.series, path, seg.t0, seg.t1, stop)
        try:
            pieces: list[Obs] = []
            t = seg.t0
            stopped = None
            while t < seg.t1 - GLIDE_PIECE_SECONDS:
                if len(pieces) == MAX_GLIDE_PIECES:
                    stopped = t
                    break
                o = glide_obs(path, crop, seg, si, found, thr, fthr, index + len(pieces), t)
                if o is None:
                    stopped = t
                    break
                pieces.append(o)
                if o.mosaic.moved_at is None:
                    break
                t = o.mosaic.moved_at
            for a, b in zip(pieces, pieces[1:]):
                # Music right of the reading column when a piece ends glides through the next piece
                a.cells = a.all_cells = [c for c in a.cells if (c.x0 + c.x1) / 2 < a.mosaic.tail_from]
                b.cells = b.all_cells = [c for c in b.cells if (c.x0 + c.x1) / 2 >= b.mosaic.seeded]
                b.move = track.Move("glide")
            shown = reading.result() if pieces else []
        finally:
            stop.set()
    for k, o in enumerate(pieces):
        o.first_piece, o.last_piece = k == 0, k == len(pieces) - 1
        o.counter = shown
    return pieces, stopped


def glide_obs(path: Path, crop, seg: Segment, si: int, found, thr: float, fthr: float, index: int,
              t0: float) -> Obs | None:
    """One view holding a segment's glide from t0 on, read from its mosaic; it ends early where the staffs move."""
    # The rows come from the staffs seen in the piece's first seconds, after any redraw at t0
    near = [s for t, ss in found.items() if t0 + 0.5 <= t <= min(seg.t1, t0 + GLIDE_ROWS_SECONDS) for s in ss]
    systems_seen = near or [s for t, ss in found.items() if t0 <= t <= seg.t1 for s in ss]
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
        m = glide.build(path, crop, print_rows, staff_rows, seg.polarity, thr, seg.gap, t0, seg.t1,
                        analysis_rows=rows, staff_x=staff_x, redrawn=t0 > seg.t0,
                        tab_rows=(float(np.median([s.tab.top for s in group])) - seg.y0, bottom))
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
    o = Obs(index, t0, seg.t1 if m.moved_at is None else m.moved_at, cells, [], si, (seg.x0, seg.y0), m.image)
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


def counted_numbers(o: Obs) -> None:
    """Number a glide's measures by the bar counter shown when each was played. A measure the counter skipped is
    numbered from its neighbors when they are as many measures apart as cells, and measures played before the
    counter's first reading or after its last count on from their neighbor. If any stays unnumbered, the glide's
    numbers are left as they were."""
    times = [o.mosaic.time_at(glide_middle(o, c)) for c in o.cells]
    nums = [counter.at(o.counter, t) for t in times]
    known = [i for i, n in enumerate(nums) if n is not None]
    if not known:
        return
    for i, j in zip(known, known[1:]):
        if nums[j] - nums[i] == j - i:
            for k in range(i + 1, j):
                nums[k] = nums[i] + k - i
    first, last = o.counter[0][0], o.counter[-1][1]
    for k in range(known[0] - 1, -1, -1):
        if times[k] < first and nums[k + 1] is not None:
            nums[k] = nums[k + 1] - 1
    for k in range(known[-1] + 1, len(nums)):
        if times[k] > last and nums[k - 1] is not None:
            nums[k] = nums[k - 1] + 1
    # A measure the counter skipped where it jumps back (a section played again) is either the next measure before
    # the jump or the one before the measure jumped to; it is the latter when it looks like that earlier measure
    for k in range(1, len(nums) - 1):
        a, b = nums[k - 1], nums[k + 1]
        if nums[k] is None and a is not None and b is not None and b <= a:
            earlier = next((c for c, n in zip(o.cells[:k], nums[:k]) if n == b - 1), None)
            again = earlier is not None and resemble_scaled(earlier, o.cells[k], o.cells[k].gap) < GLIDE_CONTINUE
            nums[k] = b - 1 if again else a + 1
    if None in nums:
        return
    for c, n in zip(o.cells, nums):
        c.number = n


def glide_middle(o: Obs, c: Cell) -> float:
    """The mosaic column a glided measure is timed by: its middle, or for a last measure with no closing barline,
    which runs on into blank staff, half a typical measure past its start when that comes first."""
    if c.kind != "tail":
        return (c.x0 + c.x1) / 2
    widths = [d.x1 - d.x0 for d in o.cells if d.kind == "full"]
    return min((c.x0 + c.x1) / 2, c.x0 + float(np.median(widths)) / 2) if widths else (c.x0 + c.x1) / 2


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
        return [o.mosaic.time_at(glide_middle(o, c)) for c in o.cells]
    first: dict[int, float] = {}
    for t, x, y in o.playhead:
        i = cell_at(o, (x, y))
        if i is not None:
            first.setdefault(i, t)
    n = max(len(o.cells), 1)
    return [first.get(i, o.t0 + (i + 0.5) * (o.t1 - o.t0) / n) for i in range(len(o.cells))]


def measure_at(sid: int, o: Obs, c: Cell, play: list[float]) -> Measure:
    """A printed measure for slot sid, seen as cell c of view o. A slot is only seen in views of its own stream, so
    the measure takes the view's stream."""
    if o.mosaic is not None:
        m = o.mosaic
        x = glide_middle(o, c)
        # A glided measure is placed when column x crossed the playhead (the panel centre when none was found)
        when = round(m.time_at(x), 2)
        y = round(c.mid_y + o.offset[1] + m.y0, 1)
        seen = m.seen_at(x)
        if seen is not None:
            # Never crossed the play column while the band moved: placed where it sat on that still screen
            return Measure(sid, o.index, c, round(seen[0], 2), when, round(seen[1] + o.offset[0], 1), y, None,
                           c.number, not c.complete, o.stream)
        return Measure(sid, o.index, c, when, when, round(m.ref_x + o.offset[0], 1), y, when, c.number,
                       not c.complete, o.stream)
    return Measure(sid, o.index, c, round((o.t0 + o.t1) / 2, 2), round(play[index_of(o, c)], 2),
                   round((c.x0 + c.x1) / 2 + o.offset[0], 1), round(c.mid_y + o.offset[1], 1), None, c.number,
                   not c.complete, o.stream)


def measures_of(A: Assembly, obs: list[Obs], order: list[int]) -> list[Measure]:
    """The measures to print for the slots in `order`, each from its best sighting: a complete one when there
    is one, then one showing its tab staff, then one between two barlines, then the widest, then one from the view
    the measure before printed from, then the earliest."""
    times = {o.index: play_times(o) for o in obs}
    split = {A.view_slots[oi][0]: (pi, k, oi) for oi, (pi, k, kind) in A.stitches.items() if kind == "stitch"}
    out = []
    prev = None
    for sid in order:
        vi, c = max(A.sightings[sid], key=lambda vc: (vc[1].complete, not vc[1].notation_only, vc[1].kind == "full",
                                                      vc[1].x1 - vc[1].x0, vc[0] == prev, -vc[0]))
        m = measure_at(sid, obs[vi], c, times[vi])
        if not c.complete and sid in split:
            # Each page shows only a piece of this measure, so both pieces print, the earlier page's first
            pi, k, oi = split[sid]
            m.pieces = [(pi, obs[pi].cells[k]), (oi, obs[oi].cells[0])]
        out.append(m)
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
        here.view, here.cell, here.pieces = best[sid].view, best[sid].cell, best[sid].pieces
        out.append(here)
    return out
