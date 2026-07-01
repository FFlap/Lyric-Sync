#!/usr/bin/env python3
"""Per-word hybrid: acoustic base + line-windowed where it clearly helps.

Post-blend line optimization (no hardcoded words):
- Score candidate boundaries from acoustic + line-windowed proposals.
- Preserve natural LW micro-gaps when acoustic left dead air.
- Split low-confidence vowel shorts absorbed into an oversized previous span (onset-guided).
"""
import argparse
import json
import pathlib
import re

import librosa
import numpy as np

from clean_lyrics import parse_lyrics

VOWELS = set("aeiouy")


def norm_alpha(word):
    return re.sub(r"[^a-zA-Z]", "", word).lower()


def word_weight(word):
    n = norm_alpha(word)
    if not n:
        return 0.7
    if len(n) <= 1:
        return 0.5
    if len(n) <= 2:
        return 0.65
    syll = max(1, sum(1 for c in n if c in VOWELS))
    return min(2.4, 0.75 + 0.22 * len(n) + 0.18 * syll)


def min_duration(word):
    return max(0.028, 0.032 * word_weight(word))


def max_short_hold_duration(word):
    n = norm_alpha(word)
    if len(n) <= 2:
        return 0.36
    if len(n) <= 4:
        return 0.48
    return 0.62


def expected_duration(word, ac, lw):
    ac_d = float(ac["end"]) - float(ac["start"])
    lw_d = float(lw["end"]) - float(lw["start"])
    ac_s = float(ac.get("score", 0))
    lw_s = float(lw.get("score", 0))
    denom = ac_s + lw_s + 1e-9
    blended = (ac_d * ac_s + lw_d * lw_s) / denom
    return max(min_duration(word), blended)


def has_plausible_early_lw_short(ac, lw):
    if not is_vowel_short(ac["word"]):
        return False

    ac_start = float(ac["start"])
    lw_start = float(lw["start"])
    if ac_start - lw_start < 0.18:
        return False

    ac_dur = float(ac["end"]) - ac_start
    lw_dur = float(lw["end"]) - lw_start
    if lw_dur < max(0.04, min_duration(ac["word"]) * 0.9):
        return False

    ac_score = float(ac.get("score", 0))
    lw_score = float(lw.get("score", 0))
    if ac_score > 0.45 or lw_score < ac_score + 0.08:
        return False

    return ac_dur > 0.42 or ac_dur > expected_duration(ac["word"], ac, lw) * 1.12


def is_cv_short(word):
    n = norm_alpha(word)
    return len(n) == 2 and n and n[0] not in VOWELS


def is_vowel_short(word):
    n = norm_alpha(word)
    return len(n) <= 2 and n and n[0] in VOWELS


def token_count(line):
    return len(re.findall(r"\S+", line))


def prefer_line_windowed(ac, lw, prev_ac, prev_lw):
    ac_dur = float(ac["end"]) - float(ac["start"])
    lw_dur = float(lw["end"]) - float(lw["start"])
    ac_score = float(ac.get("score", 0))
    lw_score = float(lw.get("score", 0))
    lw_later = float(lw["start"]) - float(ac["start"])

    if lw_later > 0.35:
        return False

    if has_plausible_early_lw_short(ac, lw):
        return True

    lw_earlier = float(ac["start"]) - float(lw["start"])

    if ac_dur > 0.26 and lw_dur < ac_dur * 0.55:
        # Acoustic smeared long, but LW is earlier: trust LW onset, not short duration.
        if lw_earlier <= 0.12:
            return False
    if lw_dur < 0.14 and ac_dur > 0.22:
        return False

    gap_ac = float(ac["start"]) - float(prev_ac["end"]) if prev_ac else 0.0
    start_delta = float(ac["start"]) - float(lw["start"])

    if is_cv_short(ac["word"]) and gap_ac < 0.06 and start_delta > 0.04:
        if lw_score >= ac_score - 0.2:
            return True

    if lw_score > ac_score + 0.22 and abs(start_delta) < 0.35:
        return True

    if ac_score < 0.42 and lw_score > ac_score + 0.1 and abs(start_delta) <= 0.30:
        return True

    # CTC smeared dead air before the sung syllable; LW places the word later.
    if lw_later > 0.28 and lw_dur >= 0.06 and ac_dur > max(lw_dur * 2.2, expected_duration(ac["word"], ac, lw) * 1.5):
        if lw_score >= ac_score - 0.25:
            return True

    # Acoustic often smears a following word onto the previous syllable. If LW
    # finds a clearly later, higher-confidence onset after a tight acoustic join,
    # preserve that later attack.
    if prev_ac is not None:
        acoustic_gap = float(ac["start"]) - float(prev_ac["end"])
        if (
            lw_later > 0.45
            and acoustic_gap < 0.04
            and lw_dur >= min_duration(ac["word"])
            and lw_score >= ac_score + 0.08
        ):
            return True

    # LW line-window often catches phrase onsets earlier than full-song CTC (e.g. chorus after verse).
    if lw_earlier > 0.18 and lw_dur >= 0.06 and lw_score >= ac_score - 0.15:
        return True

    return False


def is_ctc_overspan(ac, lw):
    ac_dur = float(ac["end"]) - float(ac["start"])
    lw_dur = float(lw["end"]) - float(lw["start"])
    lw_later = float(lw["start"]) - float(ac["start"])
    exp = expected_duration(ac["word"], ac, lw)
    return ac_dur > exp * 1.45 and (
        lw_later > 0.28
        or (lw_dur >= 0.06 and ac_dur > lw_dur * 2.2)
    )


def should_tighten_voicing(w, ac, lw):
    if is_ctc_overspan(ac, lw):
        return True
    span = float(w["end"]) - float(w["start"])
    ac_dur = float(ac["end"]) - float(ac["start"])
    if is_vowel_short(w["word"]) and float(w.get("score", 1)) < 0.35 and span > 0.42:
        return True
    if span > 0.55 and ac_dur > expected_duration(w["word"], ac, lw) * 1.35:
        return True
    return False


def voiced_clusters(times, rms, thr):
    clusters = []
    in_cluster = False
    start = None
    for i, r in enumerate(rms):
        if r > thr and not in_cluster:
            in_cluster = True
            start = float(times[i])
        elif r <= thr and in_cluster:
            in_cluster = False
            clusters.append((start, float(times[i - 1])))
    if in_cluster:
        clusters.append((start, float(times[-1])))
    return clusters


def pick_voicing_cluster(clusters, times, rms, in_span, word):
    if len(clusters) == 1:
        return clusters[0]
    if is_vowel_short(word):
        best, best_e = clusters[0], -1.0
        for v0, v1 in clusters:
            mask = (times >= v0) & (times <= v1) & in_span
            energy = float(np.sum(rms[mask]))
            if energy > best_e:
                best_e = energy
                best = (v0, v1)
        return best
    gap = clusters[1][0] - clusters[0][1]
    if gap > 0.18:
        return clusters[0]
    return (clusters[0][0], clusters[-1][1])


def tighten_to_voicing(w, ac, lw, y, sr):
    """Shrink CTC span to actual vocal energy inside the window."""
    if y is None:
        return w

    t0, t1 = float(w["start"]), float(w["end"])
    pad = 0.02
    margin = 0.06
    search_lo = max(0.0, t0 - margin)
    search_hi = min(len(y) / sr, t1 + margin)
    i0, i1 = int(search_lo * sr), int(search_hi * sr)
    seg = y[i0:i1]
    if len(seg) < 512:
        return w

    hop = 128
    frame = int(0.04 * sr)
    rms = librosa.feature.rms(y=seg, frame_length=frame, hop_length=hop, center=True)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop) + search_lo
    if len(rms) == 0:
        return w

    in_span = (times >= t0) & (times <= t1)
    if not np.any(in_span):
        return w
    peak = float(np.max(rms[in_span]))
    if peak < 1e-7:
        return w

    thr = peak * 0.36
    clusters = voiced_clusters(times[in_span], rms[in_span], thr)
    if not clusters:
        return w

    v0, v1 = pick_voicing_cluster(clusters, times[in_span], rms[in_span], np.ones(len(times[in_span]), dtype=bool), w["word"])

    new_start = max(t0, round(v0 - pad, 3))
    new_end = min(t1, round(v1 + pad, 3))

    if is_vowel_short(w["word"]) and float(ac.get("score", 1)) < 0.25:
        attack_start = vowel_short_attack_start(y, sr, float(ac["start"]), t1 + margin)
        new_start = max(new_start, attack_start)

    md = min_duration(w["word"])
    if new_end - new_start < md:
        return w

    w["start"] = new_start
    w["end"] = new_end
    w["source"] = w.get("source", "acoustic") + "+voicing"
    return w


def realign_overspanned_words(words, ac_words, lw_words, y, sr):
    for i, w in enumerate(words):
        ac, lw = ac_words[i], lw_words[i]
        if is_ctc_overspan(ac, lw) and float(lw["start"]) - float(ac["start"]) > 0.28:
            lw_dur = float(lw["end"]) - float(lw["start"])
            if lw_dur >= 0.06:
                anchored, start = onset_anchors_ac_start(y, sr, ac, lw)
                if anchored:
                    w["start"] = start
                    lw_dur = float(lw["end"]) - float(lw["start"])
                    exp = expected_duration(w["word"], ac, lw)
                    dur = min(float(ac["end"]) - start, max(min_duration(w["word"]), lw_dur, exp))
                    dur = min(dur, exp * 1.2 + 0.06)
                    w["end"] = round(start + dur, 3)
                    w["source"] = w.get("source", "acoustic") + "+onset_realign"
                else:
                    w["start"] = round(float(lw["start"]), 3)
                    w["end"] = round(float(lw["end"]), 3)
                    w["source"] = w.get("source", "acoustic") + "+lw_realign"
                continue
        if should_tighten_voicing(w, ac, lw):
            tighten_to_voicing(w, ac, lw, y, sr)
    return words


def refine_line_openers(words, ac_words, lw_words, lyric_lines, y, sr):
    """First word of a line after a pause: snap to the phrase onset, not a late LW tail."""
    if y is None:
        return words
    pos = 0
    for line in lyric_lines:
        count = token_count(line)
        if count == 0:
            continue
        idx = pos
        pos += count
        if idx == 0:
            continue

        prev = words[idx - 1]
        cur = words[idx]
        ac, lw = ac_words[idx], lw_words[idx]
        prev_end = float(prev["end"])
        lw_start = float(lw["start"])
        gap = lw_start - prev_end
        if gap < 0.35:
            continue

        anchored, ac_anchor = onset_anchors_ac_start(y, sr, ac, lw)

        lo = prev_end + 0.08
        hi = lw_start + 0.12
        onset = best_phrase_onset(y, sr, lo, hi)
        if onset is None:
            continue
        if anchored and onset > ac_anchor + 0.06:
            onset = ac_anchor
        if float(cur["start"]) < onset + 0.15:
            continue
        if onset >= float(cur["start"]) - 0.04:
            continue
        if onset < prev_end + min_duration(prev["word"]) * 0.5:
            continue

        duration = max(
            min_duration(cur["word"]),
            float(cur["end"]) - float(cur["start"]),
        )
        cur["start"] = onset
        cur["end"] = round(onset + duration, 3)
        cur["source"] = cur.get("source", "acoustic") + "+line_open"
    return words


def refine_swallowed_line_openers(words, ac_words, lw_words, lyric_lines, y, sr):
    """Pull short vowel line openers back when the previous line swallowed the attack."""
    if y is None:
        return words

    pos = 0
    for line in lyric_lines:
        count = token_count(line)
        if count == 0:
            continue
        idx = pos
        pos += count
        if idx == 0:
            continue

        prev = words[idx - 1]
        cur = words[idx]
        if not is_vowel_short(cur["word"]):
            continue

        cl = lw_words[idx]
        pa = ac_words[idx - 1]
        pl = lw_words[idx - 1]
        cur_start = float(cur["start"])
        if float(cl["start"]) - cur_start < 0.25:
            continue

        prev_dur = float(prev["end"]) - float(prev["start"])
        exp_prev = expected_duration(prev["word"], pa, pl)
        if prev_dur < max(0.85, exp_prev * 1.18):
            continue

        lo = max(float(prev["start"]) + min_duration(prev["word"]) * 1.5, cur_start - 0.58)
        hi = cur_start - 0.03
        if hi <= lo + 0.05:
            continue

        onset = best_phrase_onset(y, sr, lo, hi)
        if onset is None:
            continue

        pull = cur_start - onset
        if pull < 0.12 or pull > 0.58:
            continue

        onset_e = local_rms_peak(y, sr, onset, radius=0.14)
        cur_e = local_rms_peak(y, sr, cur_start, radius=0.14)
        if cur_e > 1e-7 and onset_e < cur_e * 0.35:
            continue

        boundary = round(float(onset), 3)
        min_prev_end = round(float(prev["start"]) + min_duration(prev["word"]), 3)
        max_cur_start = round(float(cur["end"]) - min_duration(cur["word"]), 3)
        if boundary < min_prev_end or boundary > max_cur_start:
            continue

        prev["end"] = boundary
        cur["start"] = boundary
        prev["source"] = prev.get("source", "acoustic") + "+line_open_pull"
        cur["source"] = cur.get("source", "acoustic") + "+line_open_pull"
    return words


def pick_word(ac, lw, use_lw):
    if use_lw:
        start = round(float(lw["start"]), 3)
        end = round(float(lw["end"]), 3)
        source = "line_windowed"
        score = round(float(lw.get("score", 0)), 3)
    else:
        start = round(float(ac["start"]), 3)
        end = round(float(ac["end"]), 3)
        source = "acoustic"
        score = round(float(ac.get("score", 0)), 3)

    return {
        "i": ac["i"],
        "word": ac["word"],
        "start": start,
        "end": end,
        "score": score,
        "source": source,
    }


def has_line_windowed_early_drift(acoustic, line_windowed):
    """Detect when LW is globally early enough to make words visibly too fast."""
    if len(acoustic) < 8 or len(line_windowed) < 8:
        return False

    deltas = []
    for ac, lw in zip(acoustic, line_windowed):
        lw_duration = float(lw["end"]) - float(lw["start"])
        if lw_duration < 0.045:
            continue
        deltas.append(float(lw["start"]) - float(ac["start"]))

    if len(deltas) < 8:
        return False

    early = sum(1 for delta in deltas if delta < -0.30)
    late = sum(1 for delta in deltas if delta > 0.30)
    mean_delta = sum(deltas) / len(deltas)
    return (
        early >= max(6, int(len(deltas) * 0.28))
        and early >= late * 3 + 3
        and mean_delta < -0.18
    )


def boundary_candidates(prev_start, cur_end, prev, cur, pa, ca, pl, cl):
    raw = [
        pa["end"],
        ca["start"],
        pl["end"],
        cl["start"],
        prev["end"],
        cur["start"],
        (float(pa["end"]) + float(ca["start"])) / 2,
        (float(pl["end"]) + float(cl["start"])) / 2,
        prev_start + expected_duration(prev["word"], pa, pl),
        prev_start + (float(pl["end"]) - float(pl["start"])),
    ]
    out = set()
    min_p = min_duration(prev["word"])
    min_c = min_duration(cur["word"])
    for v in raw:
        b = round(float(v), 3)
        if b - prev_start >= min_p - 0.001 and cur_end - b >= min_c - 0.001:
            out.add(b)
    return sorted(out)


def score_boundary(b, prev_start, cur_end, prev, cur, pa, ca, pl, cl):
    min_p = min_duration(prev["word"])
    min_c = min_duration(cur["word"])
    if b - prev_start < min_p - 0.001 or cur_end - b < min_c - 0.001:
        return -1e9

    ac_b = (float(pa["end"]) + float(ca["start"])) / 2
    lw_b = (float(pl["end"]) + float(cl["start"])) / 2
    ac_conf = (float(pa.get("score", 0)) + float(ca.get("score", 0))) / 2
    lw_conf = (float(pl.get("score", 0)) + float(cl.get("score", 0))) / 2

    s = 0.0
    s -= abs(b - ac_b) * (0.35 + 0.65 * ac_conf)
    s -= abs(b - lw_b) * (0.35 + 0.65 * lw_conf)

    prev_dur = b - prev_start
    lw_prev_dur = float(pl["end"]) - float(pl["start"])
    exp_prev = expected_duration(prev["word"], pa, pl)
    if prev.get("source") == "line_windowed":
        s -= max(0.0, prev_dur - lw_prev_dur) * 3.0
    s -= abs(prev_dur - exp_prev) * 0.55

    cur_dur = cur_end - b
    exp_cur = expected_duration(cur["word"], ca, cl)
    s -= abs(cur_dur - exp_cur) * 0.45

    ac_hole = float(ca["start"]) - b
    lw_stride = float(cl["start"]) - float(pl["end"])
    if ac_hole > 0.15 and lw_stride < 0.22:
        s += 0.22 - min(0.22, abs(b - float(cl["start"])))
    if abs(b - float(cl["start"])) < 0.04:
        s += 0.14

    return s


def optimize_line_boundaries(chunk, ac_chunk, lw_chunk):
    n = len(chunk)
    if n < 2:
        return chunk

    chunk[-1]["end"] = round(max(float(chunk[-1]["end"]), float(ac_chunk[-1]["end"])), 3)

    for j in range(n - 1):
        prev, cur = chunk[j], chunk[j + 1]
        pa, ca = ac_chunk[j], ac_chunk[j + 1]
        pl, cl = lw_chunk[j], lw_chunk[j + 1]
        prev_start = float(prev["start"])
        cur_end = max(float(cur["end"]), float(ca["end"]))

        if has_plausible_early_lw_short(ca, cl):
            boundary = round(float(cl["start"]), 3)
            if (
                boundary - prev_start >= min_duration(prev["word"]) - 0.001
                and cur_end - boundary >= min_duration(cur["word"]) - 0.001
            ):
                max_prev_duration = max_short_hold_duration(prev["word"])
                if boundary - prev_start > max_prev_duration:
                    prev["start"] = round(boundary - max_prev_duration, 3)
                prev["end"] = boundary
                cur["start"] = boundary
                cur["end"] = round(cur_end, 3)
                prev["source"] = prev.get("source", "acoustic") + "+early_lw_short"
                cur["source"] = cur.get("source", "acoustic") + "+early_lw_short"
                continue

        if (
            "line_windowed" in cur.get("source", "")
            and float(cl["start"]) - float(ca["start"]) > 0.45
            and float(cl.get("score", 0)) >= float(ca.get("score", 0)) + 0.08
        ):
            boundary = round(float(cl["start"]), 3)
            if (
                boundary - prev_start >= min_duration(prev["word"]) - 0.001
                and cur_end - boundary >= min_duration(cur["word"]) - 0.001
            ):
                max_prev_duration = max_short_hold_duration(prev["word"])
                if boundary - prev_start > max_prev_duration:
                    prev["start"] = round(boundary - max_prev_duration, 3)
                prev["end"] = boundary
                cur["start"] = boundary
                cur["end"] = round(cur_end, 3)
                prev["source"] = prev.get("source", "acoustic") + "+later_lw"
                cur["source"] = cur.get("source", "acoustic") + "+later_lw"
                continue

        acoustic_gap = float(ca["start"]) - float(pa["end"])
        acoustic_boundary = (float(pa["end"]) + float(ca["start"])) / 2
        window_boundary = (float(pl["end"]) + float(cl["start"])) / 2
        if (
            "line_windowed" not in prev.get("source", "")
            and "line_windowed" not in cur.get("source", "")
            and abs(acoustic_gap) <= 0.06
            and abs(window_boundary - acoustic_boundary) > 0.30
        ):
            boundary = round(acoustic_boundary, 3)
            if (
                boundary - prev_start >= min_duration(prev["word"]) - 0.001
                and cur_end - boundary >= min_duration(cur["word"]) - 0.001
            ):
                prev["end"] = boundary
                cur["start"] = boundary
                cur["end"] = round(cur_end, 3)
                continue

        best_b, best_s = None, -1e9
        for b in boundary_candidates(prev_start, cur_end, prev, cur, pa, ca, pl, cl):
            sc = score_boundary(b, prev_start, cur_end, prev, cur, pa, ca, pl, cl)
            if sc > best_s:
                best_s, best_b = sc, b

        if best_b is not None:
            prev["end"] = best_b
            cur["start"] = best_b
            cur["end"] = round(cur_end, 3)
            cur["source"] = cur.get("source", "acoustic") + "+opt"

    return chunk


def attack_before_next_word(y, sr, lo, hi, next_lw_start):
    """Onset for the next word: skip prefix bleed, land near LW start."""
    onsets = local_onsets(y, sr, lo, hi)
    candidates = [t for t in onsets if t >= lo + 0.05]
    if not candidates:
        return best_phrase_onset(y, sr, lo, hi)
    target = float(next_lw_start)
    # Ignore onsets in the first half of the gap (usually the prefix syllable).
    floor = lo + (target - lo) * 0.55
    strong = [t for t in candidates if t >= floor]
    pool = strong if strong else candidates
    within = [t for t in pool if target - 0.25 <= t <= target + 0.05]
    if within:
        return round(min(within), 3)
    return round(min(pool, key=lambda t: abs(t - target)), 3)


def split_before_next_attack(chunk, ac_chunk, lw_chunk, y, sr):
    """Short prefix word (e.g. 'I') smeared long: next syllable onset starts the next word."""
    if y is None or len(chunk) < 2:
        return chunk
    for j in range(len(chunk) - 1):
        cur, nxt = chunk[j], chunk[j + 1]
        if len(norm_alpha(cur["word"])) > 1:
            continue
        span = float(cur["end"]) - float(cur["start"])
        exp = expected_duration(cur["word"], ac_chunk[j], lw_chunk[j])
        if span < exp * 1.25:
            continue
        ca = ac_chunk[j + 1]
        cl = lw_chunk[j + 1]
        lo = float(cur["start"]) + min_duration(cur["word"])
        hi = min(float(ca["start"]) + 0.12, float(nxt["end"]) - min_duration(nxt["word"]))
        if hi <= lo + 0.05:
            continue
        attack = attack_before_next_word(y, sr, lo, hi, float(cl["start"]))
        if attack is None or attack <= lo + 0.05:
            continue
        if attack < float(cur["end"]) - 0.04:
            cur["end"] = attack
        if attack < float(nxt["start"]) - 0.04:
            cur["end"] = attack
            nxt["start"] = attack
            cur["source"] = cur.get("source", "acoustic") + "+split_next"
            nxt["source"] = nxt.get("source", "acoustic") + "+split_next"
    return chunk


def shift_line_to_early_anchor(chunk, ac_chunk, lw_chunk):
    """Keep acoustic word spacing when LW finds a credible earlier line attack."""
    if not chunk or not ac_chunk or not lw_chunk:
        return chunk

    first, first_ac, first_lw = chunk[0], ac_chunk[0], lw_chunk[0]
    ac_start = float(first_ac["start"])
    lw_start = float(first_lw["start"])
    shift = lw_start - ac_start
    if not (-1.15 <= shift <= -0.35):
        return chunk

    ac_score = float(first_ac.get("score", 0))
    lw_score = float(first_lw.get("score", 0))
    ac_first_duration = float(first_ac["end"]) - ac_start
    lw_first_duration = float(first_lw["end"]) - lw_start
    if lw_first_duration < 0.06 or lw_score < 0.4:
        return chunk

    acoustic_duration = float(ac_chunk[-1]["end"]) - ac_start
    current_duration = float(chunk[-1]["end"]) - float(first["start"])
    starts_at_lw = abs(float(first["start"]) - lw_start) <= 0.12
    stretched_from_mixed_sources = (
        len(chunk) >= 8
        and starts_at_lw
        and current_duration > acoustic_duration + 0.35
        and ac_first_duration >= 0.3
        and lw_first_duration <= ac_first_duration * 0.8
        and lw_score >= ac_score - 0.16
    )
    compact_confident_opener = (
        len(chunk) >= 8
        and ac_first_duration >= 0.3
        and lw_first_duration <= ac_first_duration * 0.35
        and lw_score >= ac_score - 0.16
    )
    swallowed_short_opener = (
        is_vowel_short(first["word"])
        and len(norm_alpha(first["word"])) == 1
        and ac_first_duration >= max(0.35, lw_first_duration * 2.8)
        and lw_score >= max(0.55, ac_score - 0.2)
    )
    if not (
        stretched_from_mixed_sources
        or compact_confident_opener
        or swallowed_short_opener
    ):
        return chunk
    if not has_consistent_line_shift(ac_chunk, lw_chunk, shift):
        return chunk

    for word, acoustic in zip(chunk, ac_chunk):
        word["start"] = round(float(acoustic["start"]) + shift, 3)
        word["end"] = round(float(acoustic["end"]) + shift, 3)
        word["score"] = round(float(acoustic.get("score", 0)), 3)
        word["source"] = "acoustic+line_shift"
    return chunk


def conservative_line_anchor(ac_chunk, lw_chunk, previous_lw_chunk, onset_times):
    """Return an earlier line anchor only when ordering and audio agree."""
    if len(ac_chunk) < 2 or not lw_chunk or not previous_lw_chunk:
        return None

    acoustic = ac_chunk[0]
    windowed = lw_chunk[0]
    ac_start = float(acoustic["start"])
    lw_start = float(windowed["start"])
    shift = lw_start - ac_start
    if not (-1.15 <= shift <= -0.30):
        return None
    if len(norm_alpha(acoustic["word"])) < 4:
        return None

    tail_deltas = [
        float(windowed_word["start"]) - float(acoustic_word["start"])
        for acoustic_word, windowed_word in zip(ac_chunk[1:], lw_chunk[1:])
        if float(windowed_word["end"]) - float(windowed_word["start"]) >= 0.045
    ]
    if len(tail_deltas) < 2:
        return None
    if abs(float(np.median(tail_deltas)) - shift) < 0.28:
        return None

    previous_lw_end = float(previous_lw_chunk[-1]["end"])
    if lw_start < previous_lw_end - 0.04:
        return None

    ac_score = float(acoustic.get("score", 0))
    lw_score = float(windowed.get("score", 0))
    lw_duration = float(windowed["end"]) - lw_start
    if lw_score < max(0.45, ac_score - 0.10):
        return None
    if lw_duration < min_duration(acoustic["word"]):
        return None
    if not any(abs(float(onset) - lw_start) <= 0.18 for onset in onset_times):
        return None
    return round(lw_start, 3)


def apply_conservative_line_anchor(
    previous_chunk,
    previous_acoustic,
    previous_windowed,
    current_chunk,
    anchor,
):
    """Translate a line and fit the preceding tail before its new anchor."""
    if not current_chunk:
        return

    shift = float(anchor) - float(current_chunk[0]["start"])
    for word in current_chunk:
        word["start"] = round(float(word["start"]) + shift, 3)
        word["end"] = round(float(word["end"]) + shift, 3)
        word["source"] = word.get("source", "acoustic") + "+line_anchor"

    boundary = round(float(anchor), 3)
    for index in range(len(previous_chunk) - 1, -1, -1):
        word = previous_chunk[index]
        if (
            float(word["end"]) <= boundary
            and float(word["start"]) <= boundary - min_duration(word["word"])
        ):
            break

        duration = expected_duration(
            word["word"],
            previous_acoustic[index],
            previous_windowed[index],
        )
        word["end"] = boundary
        if float(word["start"]) > boundary - min_duration(word["word"]):
            word["start"] = round(boundary - duration, 3)
        word["source"] = word.get("source", "acoustic") + "+line_anchor_fit"
        boundary = round(float(word["start"]), 3)

        if index:
            previous = previous_chunk[index - 1]
            if float(previous["end"]) > boundary:
                previous["end"] = boundary


def apply_supported_line_anchors(line_chunks, y, sr):
    if y is None or sr is None:
        return

    for index in range(1, len(line_chunks)):
        previous_chunk, previous_acoustic, previous_windowed = line_chunks[index - 1]
        current_chunk, current_acoustic, current_windowed = line_chunks[index]
        if not previous_chunk or not current_chunk or not current_windowed:
            continue

        lw_start = float(current_windowed[0]["start"])
        onset_times = local_onsets(y, sr, lw_start - 0.20, lw_start + 0.20)
        anchor = conservative_line_anchor(
            current_acoustic,
            current_windowed,
            previous_windowed,
            onset_times,
        )
        if anchor is None:
            continue
        if float(previous_chunk[-1]["end"]) - float(anchor) > 1.0:
            continue
        apply_conservative_line_anchor(
            previous_chunk,
            previous_acoustic,
            previous_windowed,
            current_chunk,
            anchor,
        )


def has_matching_line_opener(ac_chunk, lw_chunk):
    """Whether both aligners found the same compact opener on offset timelines."""
    if len(ac_chunk) < 4 or not lw_chunk:
        return False
    ac = ac_chunk[0]
    lw = lw_chunk[0]
    shift = float(lw["start"]) - float(ac["start"])
    if not (-1.25 <= shift <= -0.35):
        return False
    ac_duration = float(ac["end"]) - float(ac["start"])
    lw_duration = float(lw["end"]) - float(lw["start"])
    ratio = lw_duration / max(ac_duration, 0.001)
    return (
        0.55 <= ratio <= 1.45
        and float(lw.get("score", 0))
        >= max(0.35, float(ac.get("score", 0)) - 0.15)
    )


def has_consistent_line_shift(ac_chunk, lw_chunk, shift, tolerance=0.24):
    """Require several words to agree before moving a whole line timeline."""
    if len(ac_chunk) < 4 or len(lw_chunk) < 4:
        return False

    supported = 0
    usable = 0
    for ac, lw in zip(ac_chunk, lw_chunk):
        lw_duration = float(lw["end"]) - float(lw["start"])
        if lw_duration < 0.045:
            continue
        usable += 1
        delta = float(lw["start"]) - float(ac["start"])
        if abs(delta - shift) <= tolerance:
            supported += 1

    needed = max(3, min(5, int(round(usable * 0.45))))
    return usable >= 4 and supported >= needed


def apply_acoustic_line_shift(chunk, ac_chunk, shift, source):
    for word, acoustic in zip(chunk, ac_chunk):
        word["start"] = round(float(acoustic["start"]) + shift, 3)
        word["end"] = round(float(acoustic["end"]) + shift, 3)
        word["score"] = round(float(acoustic.get("score", 0)), 3)
        word["source"] = source


def propagate_consistent_line_shifts(line_chunks, y=None, sr=None):
    """Carry a validated timeline correction through an adjacent drift block."""
    matching = [has_matching_line_opener(ac, lw) for _, ac, lw in line_chunks]
    for i in range(len(line_chunks) - 1):
        if not (matching[i] and matching[i + 1]):
            continue
        for j in (i, i + 1):
            chunk, ac_chunk, lw_chunk = line_chunks[j]
            anchor = float(lw_chunk[0]["start"])
            if j > 0 and y is not None and sr is not None:
                previous_end = float(line_chunks[j - 1][0][-1]["end"])
                if previous_end - anchor > 0.75:
                    onset = strongest_onset(
                        y,
                        sr,
                        anchor + 0.20,
                        min(float(ac_chunk[0]["start"]) + 0.05, anchor + 1.30),
                    )
                    if onset is not None:
                        anchor = onset
            shift = anchor - float(ac_chunk[0]["start"])
            if not has_consistent_line_shift(ac_chunk, lw_chunk, shift):
                continue
            apply_acoustic_line_shift(
                chunk, ac_chunk, shift, "acoustic+line_shift_pair"
            )

    previous_shift = None
    for line_index, (chunk, ac_chunk, lw_chunk) in enumerate(line_chunks):
        if not chunk or not ac_chunk or not lw_chunk:
            previous_shift = None
            continue

        shift = float(lw_chunk[0]["start"]) - float(ac_chunk[0]["start"])
        already_shifted = all("line_shift" in w.get("source", "") for w in chunk)
        if already_shifted:
            previous_shift = shift
            continue

        if (
            previous_shift is None
            or not (-1.25 <= shift <= -0.35)
            or abs(shift - previous_shift) > 0.35
            or not has_consistent_line_shift(ac_chunk, lw_chunk, shift)
        ):
            previous_shift = None
            continue

        proposed_start = float(ac_chunk[0]["start"]) + shift
        if line_index > 0:
            previous_end = float(line_chunks[line_index - 1][0][-1]["end"])
            if previous_end - proposed_start > 0.75:
                previous_shift = None
                continue

        apply_acoustic_line_shift(
            chunk, ac_chunk, shift, "acoustic+line_shift_follow"
        )
        previous_shift = shift

    for line_index in range(1, len(line_chunks)):
        previous = line_chunks[line_index - 1][0]
        current = line_chunks[line_index][0]
        if not previous or not current:
            continue
        propagated = all(
            "line_shift_follow" in word.get("source", "") for word in current
        )
        if propagated and float(previous[-1]["end"]) - float(current[0]["start"]) > 0.75:
            repair([previous[-1], *current])
    return line_chunks


def delay_oversized_vowel_line_opener(chunk, ac_chunk, lw_chunk, y, sr):
    """Skip a hidden vowel prefix absorbed into the visible line opener."""
    if y is None or sr is None or len(chunk) < 2:
        return chunk
    ac = ac_chunk[0]
    lw = lw_chunk[0]
    ac_start = float(ac["start"])
    lw_start = float(lw["start"])
    ac_duration = float(ac["end"]) - ac_start
    lw_duration = float(lw["end"]) - lw_start
    if not (
        is_vowel_short(ac["word"])
        and ac_duration > 0.8
        and lw_duration < 0.2
        and abs(ac_start - lw_start) < 0.15
    ):
        return chunk

    onset = strongest_onset(
        y,
        sr,
        max(ac_start, lw_start) + 0.25,
        min(ac_start + 1.0, float(ac["end"]) - 0.2),
    )
    if onset is None:
        return chunk

    shift = onset - lw_start
    for word, windowed in zip(chunk, lw_chunk):
        word["start"] = round(float(windowed["start"]) + shift, 3)
        word["end"] = round(float(windowed["end"]) + shift, 3)
        word["score"] = round(float(windowed.get("score", 0)), 3)
        word["source"] = "line_windowed+delayed_attack"
    return chunk


def apply_lw_stride_gaps(chunk, ac_chunk, lw_chunk):
    """Keep LW's within-line gap when acoustic smeared two words into dead air."""
    for j in range(len(chunk) - 1):
        prev, cur = chunk[j], chunk[j + 1]
        pa, ca = ac_chunk[j], ac_chunk[j + 1]
        pl, cl = lw_chunk[j], lw_chunk[j + 1]

        if (
            abs(float(pl["start"]) - float(pa["start"])) > 0.60
            or abs(float(cl["start"]) - float(ca["start"])) > 0.60
        ):
            continue

        lw_gap = round(float(cl["start"]) - float(pl["end"]), 3)
        if not (0.04 < lw_gap < 0.20):
            continue

        ac_hole = float(ca["start"]) - float(pl["end"])
        if ac_hole < 0.18:
            continue

        prev_end = round(float(pl["end"]), 3)
        cur_start = round(float(cl["start"]), 3)
        if prev_end < float(prev["start"]) + min_duration(prev["word"]) - 0.001:
            continue
        if float(ca["end"]) - cur_start < min_duration(cur["word"]) - 0.001:
            continue

        prev["end"] = prev_end
        cur["start"] = cur_start
        cur["end"] = round(max(float(cur["end"]), float(ca["end"])), 3)
        prev["source"] = prev.get("source", "acoustic") + "+lw_gap"
        cur["source"] = cur.get("source", "acoustic") + "+lw_gap"
    return chunk


def rms_attack_start(y, sr, lo, hi, frac=0.68):
    """First time local RMS crosses frac of its peak: later than raw onset (avoids bleed)."""
    hop = 128
    frame = int(0.04 * sr)
    i0, i1 = int(max(0, lo) * sr), int(min(len(y) / sr, hi) * sr)
    seg = y[i0:i1]
    if len(seg) < frame:
        return None
    rms = librosa.feature.rms(y=seg, frame_length=frame, hop_length=hop, center=True)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop) + lo
    if len(rms) == 0:
        return None
    peak = float(np.max(rms))
    if peak < 1e-7:
        return None
    target = peak * frac
    for t, r in zip(times, rms):
        if float(r) >= target:
            return round(float(t), 3)
    return None


def vowel_short_attack_start(y, sr, ctc_start, search_end):
    """Start a short vowel interjection at sung attack, not CTC bleed from the previous word."""
    t = float(ctc_start)
    hi = min(search_end, t + 0.30)
    # Skip the first ~40ms after CTC: often tail bleed from the previous word.
    attack = rms_attack_start(y, sr, t + 0.04, hi, frac=0.78)
    if attack is not None:
        return attack
    attack = rms_attack_start(y, sr, t + 0.02, hi, frac=0.72)
    if attack is not None:
        return attack
    lo = t + 0.04
    onsets = local_onsets(y, sr, lo, hi)
    oa = nearest_onset_after(onsets, lo, 0.12)
    return round(float(oa), 3) if oa is not None else round(lo, 3)


def local_onsets(y, sr, t0, t1, delta=0.05):
    i0, i1 = int(max(0, t0) * sr), int(min(len(y) / sr, t1) * sr)
    seg = y[i0:i1]
    if len(seg) < 512:
        return []
    hop = 128
    raw = librosa.onset.onset_detect(y=seg, sr=sr, hop_length=hop, backtrack=True, delta=delta)
    return [float(t) for t in librosa.frames_to_time(raw, sr=sr, hop_length=hop) + t0]


def strongest_onset(y, sr, t0, t1):
    i0, i1 = int(max(0, t0) * sr), int(min(len(y) / sr, t1) * sr)
    seg = y[i0:i1]
    if len(seg) < 512:
        return None
    hop = 128
    strength = librosa.onset.onset_strength(y=seg, sr=sr, hop_length=hop)
    frames = librosa.onset.onset_detect(
        onset_envelope=strength,
        sr=sr,
        hop_length=hop,
        backtrack=True,
        delta=0.03,
        units="frames",
    )
    if len(frames) == 0:
        return None
    frame = max(frames, key=lambda value: float(strength[value]))
    return round(
        float(librosa.frames_to_time(frame, sr=sr, hop_length=hop) + t0), 3
    )


def nearest_onset_before(times, cur_start, search_before=0.25):
    t = float(cur_start)
    pool = [x for x in times if t - search_before <= x <= t + 0.012]
    if not pool:
        return None
    before = [x for x in pool if x <= t + 0.004]
    return max(before) if before else None


def nearest_onset_after(times, cur_start, search_after=0.18):
    t = float(cur_start)
    after = [x for x in times if t - 0.012 <= x <= t + search_after]
    return min(after) if after else None


def onset_anchors_ac_start(y, sr, ac, lw, window=0.22):
    """True when CTC start lines up with a vocal onset: prefer it over a later LW start."""
    ac_start = float(ac["start"])
    lw_later = float(lw["start"]) - ac_start
    lw_dur = float(lw["end"]) - float(lw["start"])
    ac_dur = float(ac["end"]) - ac_start
    lw_position = lw_later / max(ac_dur, 0.1)
    # LW syllable at the final stretch of a bloated span (e.g. "And"): trust LW, not CTC start.
    if lw_later > 0.35 and lw_dur < ac_dur * 0.45 and lw_position > 0.78:
        return False, round(float(lw["start"]), 3)
    if y is None:
        return False, round(ac_start, 3)
    lo = ac_start - window
    hi = ac_start + 0.10
    onsets = local_onsets(y, sr, lo, hi)
    ob = nearest_onset_before(onsets, ac_start, search_before=window)
    if ob is not None and ac_start - ob <= 0.22:
        return True, round(ac_start, 3)
    oa = nearest_onset_after(onsets, ac_start, 0.10)
    if oa is not None and oa - ac_start <= 0.12:
        return True, round(oa, 3)
    return False, round(ac_start, 3)


def local_rms_peak(y, sr, t, radius=0.12):
    hop = 128
    frame = int(0.04 * sr)
    lo = max(0.0, float(t))
    hi = min(len(y) / sr, float(t) + radius)
    i0, i1 = int(lo * sr), int(hi * sr)
    seg = y[i0:i1]
    if len(seg) < frame:
        return 0.0
    rms = librosa.feature.rms(y=seg, frame_length=frame, hop_length=hop, center=True)[0]
    return float(np.max(rms)) if len(rms) else 0.0


def best_phrase_onset(y, sr, lo, hi):
    onsets = local_onsets(y, sr, lo, hi)
    if not onsets:
        return None
    best_t, best_e = None, -1.0
    for t in onsets:
        e = local_rms_peak(y, sr, t, radius=0.14)
        if e > best_e:
            best_e = e
            best_t = t
    return round(float(best_t), 3) if best_t is not None else None


def is_absorbed_successor(pa, ca, pl, cl, cur_word):
    cur_score = float(ca.get("score", 0))
    if cur_score > 0.35:
        return False
    if not is_vowel_short(cur_word) and len(norm_alpha(cur_word)) > 3:
        return False
    if abs(float(ca["start"]) - float(pa["end"])) > 0.06:
        return False
    prev_ac_dur = float(pa["end"]) - float(pa["start"])
    prev_lw_dur = float(pl["end"]) - float(pl["start"])
    threshold = max(prev_lw_dur * 1.35 + 0.15, 0.65)
    return prev_ac_dur > threshold


def refine_absorbed_successors(words, ac_words, lw_words, y, sr):
    for i in range(1, len(words)):
        prev, cur = words[i - 1], words[i]
        pa, ca = ac_words[i - 1], ac_words[i]
        pl, cl = lw_words[i - 1], lw_words[i]
        if not is_absorbed_successor(pa, ca, pl, cl, cur["word"]):
            continue

        t = float(ca["start"])

        if has_plausible_early_lw_short(ca, cl) and float(cur["start"]) <= t - 0.12:
            boundary = round(float(cur["start"]), 3)
            next_start = float(words[i + 1]["start"]) if i + 1 < len(words) else None
            cur_end = max(float(cur["end"]), float(ca["end"]), boundary + min_duration(cur["word"]))
            if next_start is not None:
                cur_end = min(cur_end, next_start)
            prev["end"] = boundary
            cur["end"] = round(cur_end, 3)
            prev["source"] = prev.get("source", "acoustic") + "+early_lw_short"
            cur["source"] = cur.get("source", "acoustic") + "+early_lw_short"
            continue

        if is_vowel_short(cur["word"]):
            if y is not None:
                boundary = vowel_short_attack_start(y, sr, t, t + 0.30)
            else:
                boundary = round(t + 0.02, 3)
            if boundary - t > 0.16:
                continue
            prev["end"] = boundary
            cur["start"] = boundary
            prev["source"] = prev.get("source", "acoustic") + "+absorb_split"
            cur["source"] = cur.get("source", "acoustic") + "+absorb_split"
            tighten_to_voicing(cur, ca, cl, y, sr)
            continue

        lo = max(
            float(prev["start"]) + min_duration(prev["word"]) * 1.5,
            t - 0.38,
        )
        hi = t + 0.01
        onsets = local_onsets(y, sr, lo, hi) if y is not None else []
        onset = nearest_onset_before(onsets, t, search_before=0.32)

        if onset is not None:
            boundary = round(max(lo, onset), 3)
        else:
            overflow = float(pa["end"]) - float(pa["start"]) - max(
                float(pl["end"]) - float(pl["start"]),
                expected_duration(prev["word"], pa, pl),
            )
            pull = min(0.14, max(0.04, overflow * 0.12))
            boundary = round(t - pull, 3)
            boundary = max(lo, boundary)

        if boundary >= t - 0.008:
            continue

        prev["end"] = boundary
        cur["start"] = boundary
        prev["source"] = prev.get("source", "acoustic") + "+absorb_split"
        cur["source"] = cur.get("source", "acoustic") + "+absorb_split"
    return words


def repair(words):
    for i in range(1, len(words)):
        prev, cur = words[i - 1], words[i]
        if cur["start"] < prev["end"]:
            min_prev_end = round(float(prev["start"]) + min_duration(prev["word"]), 3)
            if float(cur["start"]) >= min_prev_end:
                boundary = round(float(cur["start"]), 3)
            else:
                boundary = round((float(prev["end"]) + float(cur["start"])) / 2, 3)
            boundary = max(min_prev_end, boundary)
            prev["end"] = boundary
            cur["start"] = boundary
        if cur["end"] < cur["start"] + min_duration(cur["word"]):
            cur["end"] = round(float(cur["start"]) + min_duration(cur["word"]), 3)
        for w in (prev, cur):
            w["start"] = round(float(w["start"]), 3)
            w["end"] = round(float(w["end"]), 3)
    return words


def blend(acoustic, line_windowed, lyric_lines, y=None, sr=None):
    blended = []
    picks = []
    pos = 0
    line_chunks = []
    # Earlier line-window timelines are often adjacent repeats or backing vocals.
    # Accept them only through the audio-supported anchor path below.
    suppress_line_windowed = True
    previous_lw_chunk = None

    for line in lyric_lines:
        count = token_count(line)
        ac_chunk = acoustic[pos : pos + count]
        lw_chunk = line_windowed[pos : pos + count]
        chunk = []
        for j, (ac, lw) in enumerate(zip(ac_chunk, lw_chunk)):
            prev_ac = ac_chunk[j - 1] if j else (acoustic[pos - 1] if pos else None)
            use_lw = prefer_line_windowed(ac, lw, prev_ac, None)
            if (
                j == 0
                and previous_lw_chunk
                and float(lw["start"]) < float(previous_lw_chunk[-1]["end"]) - 0.04
            ):
                use_lw = False
            if suppress_line_windowed and float(lw["start"]) < float(ac["start"]) - 0.12:
                use_lw = False
            w = pick_word(ac, lw, use_lw)
            chunk.append(w)
            if use_lw:
                picks.append(
                    {
                        "i": w["i"],
                        "word": w["word"],
                        "acoustic": [ac["start"], ac["end"]],
                        "line_windowed": [lw["start"], lw["end"]],
                    }
                )
        line_chunks.append((chunk, ac_chunk, lw_chunk))
        blended.extend(chunk)
        pos += count
        previous_lw_chunk = lw_chunk

    for chunk, ac_chunk, lw_chunk in line_chunks:
        optimize_line_boundaries(chunk, ac_chunk, lw_chunk)
        apply_lw_stride_gaps(chunk, ac_chunk, lw_chunk)
        split_before_next_attack(chunk, ac_chunk, lw_chunk, y, sr)
        if not suppress_line_windowed:
            shift_line_to_early_anchor(chunk, ac_chunk, lw_chunk)
        delay_oversized_vowel_line_opener(chunk, ac_chunk, lw_chunk, y, sr)

    if not suppress_line_windowed:
        propagate_consistent_line_shifts(line_chunks, y, sr)
    blended = repair(blended)
    blended = refine_absorbed_successors(blended, acoustic, line_windowed, y, sr)
    blended = repair(blended)
    blended = realign_overspanned_words(blended, acoustic, line_windowed, y, sr)
    blended = refine_line_openers(blended, acoustic, line_windowed, lyric_lines, y, sr)
    blended = refine_swallowed_line_openers(blended, acoustic, line_windowed, lyric_lines, y, sr)

    # Reapply validated line anchors after cross-line refinements. Overlapping
    # singers cannot be represented by one globally non-overlapping word stream.
    for chunk, ac_chunk, lw_chunk in line_chunks:
        if not suppress_line_windowed:
            shift_line_to_early_anchor(chunk, ac_chunk, lw_chunk)
        delay_oversized_vowel_line_opener(chunk, ac_chunk, lw_chunk, y, sr)
    if not suppress_line_windowed:
        propagate_consistent_line_shifts(line_chunks, y, sr)
    else:
        apply_supported_line_anchors(line_chunks, y, sr)
    for chunk, _, _ in line_chunks:
        repair(chunk)
    return blended, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acoustic", default="work/acoustic_refined.json")
    ap.add_argument("--line-windowed", default="work/align/whisperx_line_windowed.json")
    ap.add_argument("--lyrics", default="lyrics.txt")
    ap.add_argument("--audio", default="work/stems/bs_roformer/vocals.normalized.wav")
    ap.add_argument("--out", default="work/hybrid.json")
    args = ap.parse_args()
    root = pathlib.Path(__file__).resolve().parents[1]

    def rp(p):
        path = pathlib.Path(p)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    acoustic = json.loads(rp(args.acoustic).read_text())
    line_windowed = json.loads(rp(args.line_windowed).read_text())
    lyric_lines, _ = parse_lyrics(rp(args.lyrics).read_text())
    if not lyric_lines:
        raise SystemExit(f"No lyric lines in {args.lyrics}")
    if len(acoustic) != len(line_windowed):
        raise SystemExit(f"Word count mismatch: {len(acoustic)} vs {len(line_windowed)}")

    audio_path = rp(args.audio)
    y, sr = (librosa.load(str(audio_path), sr=16000, mono=True) if audio_path.exists() else (None, None))

    blended, picks = blend(acoustic, line_windowed, lyric_lines, y, sr)
    out = rp(args.out)
    out.write_text(json.dumps(blended, indent=2))
    (out.parent / "hybrid_picks.json").write_text(json.dumps(picks, indent=2))
    print(json.dumps({"words": len(blended), "line_windowed_picks": len(picks), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
