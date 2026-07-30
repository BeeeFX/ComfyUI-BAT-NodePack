VAE_TEMPORAL_STRIDE = 4


def _pixels_to_latent(num_frames, with_ref_or_end_frame):
    base = (num_frames - 1) // VAE_TEMPORAL_STRIDE + 1
    return base + (1 if with_ref_or_end_frame else 0)


def _latent_to_pixels(lvl, with_ref_or_end_frame):
    base = lvl - (1 if with_ref_or_end_frame else 0)
    return (base - 1) * VAE_TEMPORAL_STRIDE + 1


def _context_to_latent(context_frames, context_stride, context_overlap):
    cf_lat = (context_frames - 1) // VAE_TEMPORAL_STRIDE + 1
    cs_lat = max(1, context_stride // VAE_TEMPORAL_STRIDE)
    co_lat = context_overlap // VAE_TEMPORAL_STRIDE
    return cf_lat, cs_lat, co_lat


def _is_clean(lvl, cf_lat, co_lat):
    if lvl <= cf_lat:
        return True
    delta_lat = cf_lat - co_lat
    if delta_lat <= 0:
        return False
    return (lvl - cf_lat) % delta_lat == 0


def _num_windows(lvl, cf_lat, co_lat):
    if lvl <= cf_lat:
        return 1
    delta_lat = cf_lat - co_lat
    if delta_lat <= 0:
        return -1
    base = (lvl - cf_lat) // delta_lat + 1
    if (lvl - cf_lat) % delta_lat != 0:
        base += 1
    return base


def _nearest_clean_num_frames(target_num_frames, cf_lat, co_lat, with_ref_or_end_frame, count=3):
    delta_lat = cf_lat - co_lat
    if delta_lat <= 0:
        return [], []
    target_lvl = _pixels_to_latent(target_num_frames, with_ref_or_end_frame)
    base_lvl = target_lvl - (1 if with_ref_or_end_frame else 0)

    valid_base_lvls = []
    n_windows = 1
    while True:
        candidate = cf_lat + (n_windows - 1) * delta_lat
        candidate_base = candidate - (1 if with_ref_or_end_frame else 0)
        if candidate_base > base_lvl + 200 and len(valid_base_lvls) > 6:
            break
        valid_base_lvls.append((candidate_base, n_windows))
        n_windows += 1
        if n_windows > 1000:
            break

    valid_base_lvls = [(b, w) for b, w in valid_base_lvls if b >= 1]

    below = [(b, w) for b, w in valid_base_lvls if b <= base_lvl]
    above = [(b, w) for b, w in valid_base_lvls if b > base_lvl]

    below_pixels = [((b - 1) * VAE_TEMPORAL_STRIDE + 1, w) for b, w in below[-count:]]
    above_pixels = [((b - 1) * VAE_TEMPORAL_STRIDE + 1, w) for b, w in above[:count]]
    return below_pixels, above_pixels


def _pick_best_num_frames(target_num_frames, cf_lat, co_lat, with_ref_or_end_frame, prefer_round_up):
    """Return a single best clean num_frames pixel value (or None)."""
    below, above = _nearest_clean_num_frames(target_num_frames, cf_lat, co_lat, with_ref_or_end_frame, count=1)
    above_pix = above[0][0] if above else None
    below_pix = below[-1][0] if below else None
    if prefer_round_up and above_pix is not None:
        return above_pix
    # nearest absolute
    candidates = [v for v in (above_pix, below_pix) if v is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda v: abs(v - target_num_frames))


def _nearest_overlap_for_target(num_frames, context_frames, with_ref_or_end_frame, max_pixels=80):
    cf_lat, _, _ = _context_to_latent(context_frames, 4, 0)
    lvl = _pixels_to_latent(num_frames, with_ref_or_end_frame)
    if lvl <= cf_lat:
        return []
    diff = lvl - cf_lat
    candidates = []
    for co_lat_try in range(0, cf_lat):
        delta_try = cf_lat - co_lat_try
        if delta_try <= 0:
            continue
        if diff % delta_try == 0:
            for ov_pix in range(co_lat_try * VAE_TEMPORAL_STRIDE,
                                (co_lat_try + 1) * VAE_TEMPORAL_STRIDE):
                if 0 <= ov_pix <= max_pixels:
                    candidates.append((ov_pix, co_lat_try, diff // delta_try + 1))
                    break
    return candidates


def _nearest_context_frames_for_target(num_frames, context_overlap, with_ref_or_end_frame,
                                       min_pixels=21, max_pixels=161):
    co_lat = context_overlap // VAE_TEMPORAL_STRIDE
    lvl = _pixels_to_latent(num_frames, with_ref_or_end_frame)
    candidates = []
    for cf_pix in range(min_pixels, max_pixels + 1, VAE_TEMPORAL_STRIDE):
        cf_lat = (cf_pix - 1) // VAE_TEMPORAL_STRIDE + 1
        if cf_lat <= co_lat:
            continue
        if cf_lat > lvl:
            continue
        delta_lat = cf_lat - co_lat
        diff = lvl - cf_lat
        if diff < 0:
            continue
        if diff % delta_lat == 0:
            candidates.append((cf_pix, cf_lat, diff // delta_lat + 1))
    return candidates


def _pick_recommendation(num_frames, context_frames, context_stride, context_overlap,
                         use_ref_or_end_frame, priority, prefer_round_up):
    """Return (rec_num_frames, rec_cf, rec_stride, rec_overlap) per priority mode.

    - sliding_context_priority: keep cf/stride/overlap, find nearest valid num_frames
    - batch_priority: keep num_frames, find smallest adjustment to (overlap or cf)
    - balanced: pick whichever single-axis adjustment is smallest in pixel-frame terms

    When the user is actually configuring sliding context (input lvl > cf_lat), we filter
    out cf/overlap candidates that collapse to 1 window — those defeat the point.
    """
    cf_lat, _, co_lat = _context_to_latent(context_frames, context_stride, context_overlap)
    lvl = _pixels_to_latent(num_frames, use_ref_or_end_frame)
    user_wants_windowing = lvl > cf_lat

    if _is_clean(lvl, cf_lat, co_lat) and user_wants_windowing:
        return num_frames, context_frames, context_stride, context_overlap

    cand_n = _pick_best_num_frames(num_frames, cf_lat, co_lat, use_ref_or_end_frame, prefer_round_up)

    ov_cands = _nearest_overlap_for_target(num_frames, context_frames, use_ref_or_end_frame)
    if user_wants_windowing:
        ov_cands = [t for t in ov_cands if t[2] >= 2]
    cand_ov = None
    if ov_cands:
        cand_ov = min(ov_cands, key=lambda t: abs(t[0] - context_overlap))[0]

    cf_cands = _nearest_context_frames_for_target(num_frames, context_overlap, use_ref_or_end_frame)
    if user_wants_windowing:
        cf_cands = [t for t in cf_cands if t[2] >= 2]
    cand_cf = None
    if cf_cands:
        cand_cf = min(cf_cands, key=lambda t: abs(t[0] - context_frames))[0]

    n_diff = abs(cand_n - num_frames) if cand_n is not None else float("inf")
    ov_diff = abs(cand_ov - context_overlap) if cand_ov is not None else float("inf")
    cf_diff = abs(cand_cf - context_frames) if cand_cf is not None else float("inf")

    if priority == "sliding_context_priority":
        if cand_n is not None:
            return cand_n, context_frames, context_stride, context_overlap
        return num_frames, context_frames, context_stride, context_overlap

    if priority == "batch_priority":
        if ov_diff <= cf_diff and cand_ov is not None:
            return num_frames, context_frames, context_stride, cand_ov
        if cand_cf is not None:
            return num_frames, cand_cf, context_stride, context_overlap
        return num_frames, context_frames, context_stride, context_overlap

    # balanced
    best = min(n_diff, ov_diff, cf_diff)
    if best == n_diff and cand_n is not None:
        return cand_n, context_frames, context_stride, context_overlap
    if best == ov_diff and cand_ov is not None:
        return num_frames, context_frames, context_stride, cand_ov
    if best == cf_diff and cand_cf is not None:
        return num_frames, cand_cf, context_stride, context_overlap
    return num_frames, context_frames, context_stride, context_overlap


class VoltWanContextCalculator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_frames": ("INT", {"default": 101, "min": 1, "max": 100000, "step": 1,
                                       "tooltip": "Total pixel frames you want to generate."}),
                "context_frames": ("INT", {"default": 81, "min": 5, "max": 1001, "step": 4,
                                           "tooltip": "WanVideoContextOptions context_frames (pixel frames)."}),
                "context_stride": ("INT", {"default": 4, "min": 4, "max": 100, "step": 4,
                                           "tooltip": "WanVideoContextOptions context_stride (pixel frames)."}),
                "context_overlap": ("INT", {"default": 16, "min": 0, "max": 100, "step": 4,
                                            "tooltip": "WanVideoContextOptions context_overlap (pixel frames)."}),
                "use_ref_or_end_frame": ("BOOLEAN", {"default": True,
                                                     "tooltip": "True if your VACE has ref_images, or you're using an end_frame / I2V — adds +1 latent frame."}),
                "priority": (["balanced", "batch_priority", "sliding_context_priority"], {
                    "default": "balanced",
                    "tooltip": "balanced = smallest adjustment overall. batch_priority = keep num_frames, adjust sliding context. sliding_context_priority = keep sliding context, adjust num_frames.",
                }),
                "prefer_round_up": ("BOOLEAN", {"default": True,
                                                "tooltip": "When picking nearest valid num_frames, prefer the next valid value >= current (so the input batch isn't cropped)."}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("report", "rec_num_frames", "rec_context_frames", "rec_context_stride", "rec_context_overlap", "num_windows")
    FUNCTION = "analyze"
    CATEGORY = "BAT/wan"
    OUTPUT_NODE = True
    DESCRIPTION = "Recommend optimal num_frames / context_frames / stride / overlap values for the WAN sliding-context video model based on the input clip and target priorities."

    def analyze(self, num_frames, context_frames, context_stride, context_overlap,
                use_ref_or_end_frame, priority, prefer_round_up):
        cf_lat, cs_lat, co_lat = _context_to_latent(context_frames, context_stride, context_overlap)
        lvl = _pixels_to_latent(num_frames, use_ref_or_end_frame)
        delta_lat = cf_lat - co_lat

        lines = []
        lines.append("=" * 64)
        lines.append("WAN sliding-context analysis")
        lines.append("=" * 64)
        lines.append(f"num_frames        = {num_frames} pixel frames")
        lines.append(f"context_frames    = {context_frames} pixel  ({cf_lat} latent)")
        lines.append(f"context_stride    = {context_stride} pixel  ({cs_lat} latent)")
        lines.append(f"context_overlap   = {context_overlap} pixel  ({co_lat} latent)")
        lines.append(f"ref / end_frame   = {use_ref_or_end_frame}  (latent +1 if True)")
        lines.append(f"latent_video_len  = {lvl} latent frames")
        lines.append(f"window step delta = {delta_lat} latent ({delta_lat * VAE_TEMPORAL_STRIDE} pixel)")
        lines.append(f"priority          = {priority}  (round_up={prefer_round_up})")
        lines.append("")

        rec_n, rec_cf, rec_st, rec_ov = _pick_recommendation(
            num_frames, context_frames, context_stride, context_overlap,
            use_ref_or_end_frame, priority, prefer_round_up
        )
        rec_cf_lat, _, rec_co_lat = _context_to_latent(rec_cf, rec_st, rec_ov)
        rec_lvl = _pixels_to_latent(rec_n, use_ref_or_end_frame)
        rec_n_win = _num_windows(rec_lvl, rec_cf_lat, rec_co_lat)

        if delta_lat <= 0:
            lines.append("INVALID: context_overlap must be smaller than context_frames")
            lines.append("(otherwise the window can't advance and tiling is undefined).")
            return ("\n".join(lines), num_frames, context_frames, context_stride, context_overlap, -1)

        if lvl <= cf_lat:
            implied_max_pixels = _latent_to_pixels(cf_lat, use_ref_or_end_frame)
            lines.append("STATUS: num_frames fits inside one context window — sliding context is")
            lines.append("        effectively DISABLED for this run (only one window will be used).")
            lines.append(f"        Maximum num_frames before windowing kicks in: {implied_max_pixels} pixels.")
            lines.append("        For longer outputs, increase num_frames past that threshold.")
            lines.append("")
            lines.append(f">>> Recommended: num_frames={num_frames}, cf={context_frames}, stride={context_stride}, overlap={context_overlap} (1 window)")
            return ("\n".join(lines), num_frames, context_frames, context_stride, context_overlap, 1)

        clean = _is_clean(lvl, cf_lat, co_lat)
        n_win = _num_windows(lvl, cf_lat, co_lat)

        if clean:
            lines.append(f"STATUS: CLEAN TILING ✓  ({n_win} windows, no backtracking on the last window)")
        else:
            extra = (lvl - cf_lat) % delta_lat
            lines.append(f"STATUS: NOT CLEAN — last window will back up by {extra} latent")
            lines.append(f"        ({extra * VAE_TEMPORAL_STRIDE} pixel) frames to fit. Output will still")
            lines.append("        cover all frames, but the last window has redundant overlap, which")
            lines.append("        wastes a small amount of compute and can produce slight blending")
            lines.append(f"        artifacts in that region. ({n_win} windows total.)")
        lines.append("")

        if not clean:
            below, above = _nearest_clean_num_frames(num_frames, cf_lat, co_lat, use_ref_or_end_frame)
            lines.append("--- Option A: change num_frames to a value that tiles cleanly ---")
            if below:
                lines.append("  Closest BELOW your target:")
                for n_pix, nw in below:
                    lines.append(f"    num_frames = {n_pix:>5}  →  {nw} windows")
            if above:
                lines.append("  Closest ABOVE your target:")
                for n_pix, nw in above:
                    lines.append(f"    num_frames = {n_pix:>5}  →  {nw} windows")
            lines.append("")

            ov_candidates = _nearest_overlap_for_target(num_frames, context_frames, use_ref_or_end_frame)
            if ov_candidates:
                ov_candidates_sorted = sorted(ov_candidates, key=lambda t: abs(t[0] - context_overlap))[:5]
                lines.append("--- Option B: keep num_frames, change context_overlap ---")
                for ov_pix, ov_lat, nw in ov_candidates_sorted:
                    flag = "  (current)" if ov_pix == context_overlap else ""
                    lines.append(f"    context_overlap = {ov_pix:>3} pixel ({ov_lat} latent)  →  {nw} windows{flag}")
                lines.append("")

            cf_candidates = _nearest_context_frames_for_target(num_frames, context_overlap, use_ref_or_end_frame)
            if cf_candidates:
                cf_candidates_sorted = sorted(cf_candidates, key=lambda t: abs(t[0] - context_frames))[:5]
                lines.append("--- Option C: keep num_frames, change context_frames ---")
                for cf_pix, cf_lat_c, nw in cf_candidates_sorted:
                    flag = "  (current)" if cf_pix == context_frames else ""
                    lines.append(f"    context_frames  = {cf_pix:>3} pixel ({cf_lat_c} latent)  →  {nw} windows{flag}")
                lines.append("")

        lines.append(f"--- Canonical clean num_frames for cf={context_frames}, ov={context_overlap} ---")
        seq_lvl = cf_lat
        natural = []
        max_to_show = 12
        for w in range(1, max_to_show + 1):
            n_pix = _latent_to_pixels(seq_lvl, use_ref_or_end_frame)
            if n_pix >= 1:
                natural.append((n_pix, w))
            seq_lvl += delta_lat
        formatted = ", ".join(f"{n}({w}w)" for n, w in natural)
        lines.append(f"    {formatted}")
        lines.append("    (format: num_frames(window_count))")
        lines.append("")

        lines.append(">>> RECOMMENDED (per priority='{}'):".format(priority))
        lines.append(f"    num_frames     = {rec_n}     {'(unchanged)' if rec_n == num_frames else f'(was {num_frames})'}")
        lines.append(f"    context_frames = {rec_cf}     {'(unchanged)' if rec_cf == context_frames else f'(was {context_frames})'}")
        lines.append(f"    context_stride = {rec_st}     {'(unchanged)' if rec_st == context_stride else f'(was {context_stride})'}")
        lines.append(f"    context_overlap= {rec_ov}     {'(unchanged)' if rec_ov == context_overlap else f'(was {context_overlap})'}")
        lines.append(f"    → {rec_n_win} windows, clean tiling")

        return ("\n".join(lines), rec_n, rec_cf, rec_st, rec_ov, rec_n_win)
