"""Decide whether two measure cells show the same written measure."""
import cv2
import numpy as np

from youtabber.observe import faint_change, significant_change, unmatched

# Two cells of the same measure have widths within this many pixels, or this share of the width
WIDTH_PX, WIDTH_SHARE = 4, 0.03
# Pixel shifts tried when lining two cells up (rendering and compression move edges by a pixel or two)
SHIFTS = (-2, -1, 0, 1, 2)


def _crop_pair(a, b):
    """The two cells' maps cut to the rows both cover, lined up on the tab staff's top line."""
    above = min(a.ref_y, b.ref_y)
    below = min(a.ink.shape[0] - a.ref_y, b.ink.shape[0] - b.ref_y)
    w = min(a.ink.shape[1], b.ink.shape[1])
    ra = slice(a.ref_y - above, a.ref_y + below)
    rb = slice(b.ref_y - above, b.ref_y + below)
    # Pixels tinted by a cursor in either sighting are left out of both
    care = ~(a.tint[ra, :w] | b.tint[rb, :w])
    return (a.ink[ra, :w] & care, a.faint[ra, :w] & care), (b.ink[rb, :w] & care, b.faint[rb, :w] & care)


def same(a, b, gap: float) -> bool:
    """True when cells a and b hold the same marks, faint ones (light-gray measure numbers) included."""
    if abs(a.gap - b.gap) > 0.1 * gap:
        return False
    wa, wb = a.ink.shape[1], b.ink.shape[1]
    if abs(wa - wb) > max(WIDTH_PX, WIDTH_SHARE * max(wa, wb)):
        return False
    (ia, fa), (ib, fb) = _crop_pair(a, b)
    # Ink counts are compared on marks a few pixels tall, so staff-line remnants under a tint don't count
    k = np.ones((3, 1), np.uint8)
    na = int(cv2.morphologyEx(ia.astype(np.uint8), cv2.MORPH_OPEN, k).sum())
    nb = int(cv2.morphologyEx(ib.astype(np.uint8), cv2.MORPH_OPEN, k).sum())
    if abs(na - nb) > 0.25 * max(na, nb, 1) + 30:
        return False
    h, w = ia.shape
    m = 3
    if h <= 2 * m or w <= 2 * m:
        return False
    core_a = ia[m:h - m, m:w - m]
    core_fa = fa[m:h - m, m:w - m]
    # Line the cells up where their solid ink agrees best, then require every mark to agree there
    best = None
    for dy in SHIFTS:
        for dx in SHIFTS:
            core_b = ib[m + dy:h - m + dy, m + dx:w - m + dx]
            if core_b.shape != core_a.shape:
                continue
            miss = int(unmatched(core_a, core_b).sum())
            if best is None or miss < best[0]:
                best = (miss, dx, dy)
    if best is None:
        return False
    _, dx, dy = best
    core_b = ib[m + dy:h - m + dy, m + dx:w - m + dx]
    core_fb = fb[m + dy:h - m + dy, m + dx:w - m + dx]
    return not significant_change(core_a, core_b, gap) and not faint_change(core_fa, core_fb, gap)


def same_part(part, whole, gap: float, align: str, trim: float = 1.0) -> bool:
    """True when a partial sighting shows the same marks as the matching end of a complete one.

    `align` is "left" when the partial cell starts at the measure's left barline (a preview cut by the
    panel's right edge) and "right" when it ends at the right barline (a measure whose start is hidden).
    """
    if abs(part.gap - whole.gap) > 0.1 * gap:
        return False
    w = part.ink.shape[1]
    if w > whole.ink.shape[1] + WIDTH_PX or w < 2 * gap:
        return False
    sl = slice(0, w) if align == "left" else slice(whole.ink.shape[1] - w, whole.ink.shape[1])

    class View:
        pass
    v = View()
    v.ink, v.faint, v.ref_y, v.gap, v.tint = whole.ink[:, sl], whole.faint[:, sl], whole.ref_y, whole.gap, whole.tint[:, sl]
    # Keep clear of the cut edge, where a mark may be half shown
    m = int(trim * gap)
    (ia, fa), (ib, fb) = _crop_pair(part, v)
    if align == "left":
        ia, fa, ib, fb = ia[:, :-m], fa[:, :-m], ib[:, :-m], fb[:, :-m]
    else:
        ia, fa, ib, fb = ia[:, m:], fa[:, m:], ib[:, m:], fb[:, m:]
    if ia.shape[1] < gap:
        return False
    best = min(((int(unmatched(ia[2:-2, 2:-2], ib[2 + dy:ib.shape[0] - 2 + dy, 2 + dx:ib.shape[1] - 2 + dx]).sum()), dx, dy)
                for dy in SHIFTS for dx in SHIFTS), key=lambda s: s[0])
    _, dx, dy = best
    cb = ib[2 + dy:ib.shape[0] - 2 + dy, 2 + dx:ib.shape[1] - 2 + dx]
    return not significant_change(ia[2:-2, 2:-2], cb, gap)


def resemble(a, b, gap: float, any_kind: bool = False) -> float:
    """Share of the two cells' solid ink that finds no partner in the other, at the best small shift.

    A loose measure of likeness for the checker: low values mean the cells may be the same measure even
    when `same` says no.
    """
    if (a.kind != b.kind and not any_kind) or abs(a.gap - b.gap) > 0.1 * gap:
        return 1.0
    wa, wb = a.ink.shape[1], b.ink.shape[1]
    if abs(wa - wb) > max(3 * WIDTH_PX, 3 * WIDTH_SHARE * max(wa, wb)):
        return 1.0
    (ia, _), (ib, _) = _crop_pair(a, b)
    h, w = ia.shape
    m = 3
    if h <= 2 * m or w <= 2 * m:
        return 1.0
    core_a = ia[m:h - m, m:w - m]
    total = max(int(core_a.sum()) + int(ib[m:h - m, m:w - m].sum()), 1)
    best = min(int(unmatched(core_a, ib[m + dy:h - m + dy, m + dx:w - m + dx], 1).sum())
               for dy in SHIFTS for dx in SHIFTS)
    return best / total


def resemble_scaled(a, b, gap: float) -> float:
    """`resemble` after stretching b to a's width, for mosaics whose measured scroll speed differs a little
    between two glides of the same music."""
    if abs(a.gap - b.gap) > 0.1 * gap:
        return 1.0
    wa, wb = a.ink.shape[1], b.ink.shape[1]
    if not 0.85 < wa / max(wb, 1) < 1.15:
        return 1.0

    class Scaled:
        pass
    s = Scaled()
    s.ink = cv2.resize(b.ink.astype(np.uint8), (wa, b.ink.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    s.faint = np.zeros_like(s.ink)
    s.tint = cv2.resize(b.tint.astype(np.uint8), (wa, b.ink.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    s.ref_y, s.gap, s.kind = b.ref_y, b.gap, a.kind
    return resemble(a, s, gap)
