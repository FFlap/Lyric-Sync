#!/usr/bin/env python3
"""Word-level lyric alignment: wav2vec2 CTC + Whisper phrase anchors.

Pipeline:
 1. wav2vec2 CTC emissions over the whole vocal stem (chunked).
 2. Full-song forced alignment of the lyrics -> base word spans + confidences.
 3. Whisper transcription of voiced windows -> fuzzy-matched phrase anchors.
    CTC alignment drifts through melisma (stretched syllables carry almost no
    letter evidence); anchors pin the phrase timeline so drift cannot cascade.
 4. Between anchors: keep the base alignment when it agrees, otherwise re-align
    inside the window, otherwise distribute words across voiced audio by
    syllable weight snapped to onsets.
 5. Boundary polish: sung words hold through voiced gaps (karaoke hold), ends
    trimmed at silence, minimum durations enforced.
 6. Parenthetical echo/backing runs with no audio evidence of their own are
    overlaid on the main phrase they repeat instead of being crammed between
    phrases.

Output schema matches the old hybrid.json: [{i, word, start, end, score, source}].
"""
from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from clean_lyrics import parse_lyrics  # noqa: E402

SR = 16000
FRAME_SEC = 0.02  # wav2vec2 stride (320 samples @ 16k)
HOP_SEC = 0.01  # energy hop

VOWELS = set("aeiouy")

NUM_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


# ---------------------------------------------------------------- text utils

def norm_token(tok: str) -> str:
    t = tok.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9']", "", t)
    return "".join(NUM_WORDS[c] if c.isdigit() else c for c in t)


def syllables(norm: str) -> int:
    if not norm:
        return 1
    groups = re.findall(r"[aeiouy]+", norm)
    return max(1, len(groups))


def min_duration(norm: str) -> float:
    return 0.04 + 0.02 * min(len(norm), 4)


def lyric_tokens(lines: list[str]) -> list[dict]:
    """Flatten lyric lines into tokens with paren (ad-lib/backing) flags."""
    tokens = []
    for li, line in enumerate(lines):
        depth = 0
        for raw in re.findall(r"\S+", line):
            opens = raw.count("(")
            closes = raw.count(")")
            in_paren = depth > 0 or opens > 0
            depth = max(0, depth + opens - closes)
            tokens.append(
                {
                    "i": len(tokens) + 1,
                    "word": raw,
                    "norm": norm_token(raw),
                    "line": li,
                    "paren": in_paren,
                }
            )
    return tokens


# ------------------------------------------------------------- audio features

class AudioFeatures:
    def __init__(self, y: np.ndarray):
        import librosa

        self.y = y
        self.duration = len(y) / SR
        hop = int(HOP_SEC * SR)
        rms = librosa.feature.rms(y=y, frame_length=4 * hop, hop_length=hop, center=True)[0]
        self.rms = rms
        ref = np.percentile(rms, 98) + 1e-9
        self.rms_db = 20.0 * np.log10(np.maximum(rms, 1e-9) / ref)
        self.voiced = self.rms_db > -34.0
        self.onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
        frames = librosa.onset.onset_detect(
            onset_envelope=self.onset_env, sr=SR, hop_length=hop, backtrack=True, delta=0.04
        )
        self.onsets = librosa.frames_to_time(frames, sr=SR, hop_length=hop)
        # spectral novelty (MFCC delta): vowel/articulation changes inside
        # continuous voicing that leave no amplitude onset (melisma word attacks)
        import scipy.signal

        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, hop_length=hop)[1:]
        nov = np.linalg.norm(np.diff(mfcc, axis=1), axis=0)
        nov = np.convolve(nov, np.ones(7) / 7, mode="same")
        nov = (nov - np.min(nov)) / (np.ptp(nov) + 1e-9)
        peaks, _ = scipy.signal.find_peaks(nov, height=0.30, distance=10)
        self.novelty = [(float(p * HOP_SEC), float(nov[p])) for p in peaks]

    def t2f(self, t: float) -> int:
        return int(np.clip(round(t / HOP_SEC), 0, len(self.rms) - 1))

    def is_voiced_between(self, t0: float, t1: float, frac: float = 0.72) -> bool:
        a, b = self.t2f(t0), self.t2f(t1)
        if b <= a:
            return True
        return float(np.mean(self.voiced[a:b])) >= frac

    def last_voiced_before(self, t0: float, t1: float) -> float:
        """End of the voiced stretch beginning at t0 (<= t1)."""
        a, b = self.t2f(t0), self.t2f(t1)
        if b <= a:
            return t1
        seg = self.voiced[a:b]
        run_end = a
        i = a
        # allow 60ms unvoiced blips inside the run
        gap = 0
        for i in range(a, b):
            if self.voiced[i]:
                run_end = i
                gap = 0
            else:
                gap += 1
                if gap > 6:
                    break
        del seg
        return run_end * HOP_SEC

    def voiced_intervals(self, t0: float, t1: float, min_len: float = 0.08) -> list[tuple[float, float]]:
        a, b = self.t2f(t0), self.t2f(t1)
        out = []
        start = None
        for i in range(a, b + 1):
            v = self.voiced[i] if i < len(self.voiced) else False
            if v and start is None:
                start = i
            elif not v and start is not None:
                if (i - start) * HOP_SEC >= min_len:
                    out.append((start * HOP_SEC, i * HOP_SEC))
                start = None
        if start is not None and (b - start) * HOP_SEC >= min_len:
            out.append((start * HOP_SEC, b * HOP_SEC))
        return out

    def snap_to_onset(self, t: float, radius: float = 0.15) -> float:
        if len(self.onsets) == 0:
            return t
        idx = int(np.argmin(np.abs(self.onsets - t)))
        cand = float(self.onsets[idx])
        return cand if abs(cand - t) <= radius else t


# ------------------------------------------------------------- CTC alignment

class CtcAligner:
    def __init__(self, device: str = "cpu"):
        import torch
        import torchaudio

        self.torch = torch
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H
        self.device = torch.device(device)
        self.model = self.bundle.get_model().to(self.device).eval()
        self.labels = self.bundle.get_labels()
        self.char_to_id = {c: i for i, c in enumerate(self.labels)}
        self.sep = self.char_to_id["|"]

    def emissions(self, y: np.ndarray, chunk_s: float = 30.0, ctx_s: float = 4.0):
        """Log-prob emissions for the whole waveform, chunked with context."""
        torch = self.torch
        wav = torch.from_numpy(y).float().unsqueeze(0)
        n = wav.shape[-1]
        chunk, ctx, frame = int(chunk_s * SR), int(ctx_s * SR), int(FRAME_SEC * SR)
        outs = []
        with torch.inference_mode():
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                lo, hi = max(0, start - ctx), min(n, end + ctx)
                em, _ = self.model(wav[:, lo:hi].to(self.device))
                em = torch.log_softmax(em, dim=-1).cpu()
                f0 = (start - lo) // frame
                f1 = f0 + (end - start) // frame
                outs.append(em[0, f0:f1])
        return torch.cat(outs, dim=0)

    def encode(self, norm_words: list[str]) -> list[int]:
        ids = []
        for wi, w in enumerate(norm_words):
            if wi:
                ids.append(self.sep)
            ids.extend(self.char_to_id[c] for c in w.upper() if c.upper() in self.char_to_id)
        return ids

    def align(self, emission, norm_words: list[str], t_offset: float = 0.0) -> list[dict]:
        """Forced-align words against an emission slice. Returns spans in seconds."""
        import torchaudio.functional as AF

        torch = self.torch
        norm_words = [w for w in norm_words]
        ids = self.encode(norm_words)
        if not ids or emission.shape[0] < len(ids):
            return []
        targets = torch.tensor([ids], dtype=torch.int32)
        aligned, scores = AF.forced_align(emission.unsqueeze(0), targets, blank=0)
        spans = AF.merge_tokens(aligned[0], scores[0].exp())
        words, cur, wi = [], [], 0
        counts = [sum(1 for c in w.upper() if c.upper() in self.char_to_id) for w in norm_words]

        def flush():
            nonlocal cur, wi
            if cur:
                words.append(
                    {
                        "start": round(t_offset + cur[0].start * FRAME_SEC, 3),
                        "end": round(t_offset + cur[-1].end * FRAME_SEC, 3),
                        "score": round(float(np.mean([c.score for c in cur])), 3),
                    }
                )
            else:
                words.append(None)
            cur = []
            wi += 1

        for s in spans:
            if s.token == self.sep:
                flush()
            else:
                cur.append(s)
        flush()
        # words with zero alignable chars produce no span; keep list aligned
        out = []
        j = 0
        for c in counts:
            if c == 0:
                out.append(None)
            else:
                out.append(words[j] if j < len(words) else None)
                j += 1
        return out


# ------------------------------------------------------------ whisper anchors

_WHISPER_MODEL = None


def get_whisper(model_name: str):
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel

        _WHISPER_MODEL = WhisperModel(model_name, device="cpu", compute_type="int8")
    return _WHISPER_MODEL


def transcribe_slice(model, y: np.ndarray, a: float, b: float) -> list[dict]:
    seg = y[int(a * SR) : int(b * SR)]
    segments, _ = model.transcribe(
        seg,
        language="en",
        word_timestamps=True,
        condition_on_previous_text=False,
        beam_size=5,
        # no sampling fallback: identical inputs must transcribe identically,
        # otherwise word placement flaps between runs on hard passages
        temperature=0.0,
    )
    words = []
    for s in segments:
        for w in s.words or []:
            norm = norm_token(w.word)
            if not norm:
                continue
            words.append(
                {
                    "norm": norm,
                    "start": round(a + w.start, 3),
                    "end": round(a + w.end, 3),
                    "p": round(float(w.probability), 3),
                }
            )
    return words


def transcribe_slice_stable(model, y: np.ndarray, a: float, b: float, duration: float) -> list[dict]:
    """Slice transcription stabilized over boundary offsets.

    Whisper output on hard slices swings with the exact cut points; words that
    persist across offset runs (same text, start within 0.18s) are trustworthy,
    the rest are boundary artifacts and get dropped.
    """
    def collect(offsets):
        out = []
        for da, db in offsets:
            aa, bb = max(0.0, a + da), min(duration, b + db)
            if bb - aa < 0.4:
                continue
            out.append(transcribe_slice(model, y, aa, bb))
        return out

    runs = collect(((0.0, 0.0), (-0.35, 0.3), (0.3, -0.35), (-0.8, 0.0), (0.0, 0.7)))
    if not runs:
        return []

    def same_word(w, o) -> bool:
        if o["norm"] != w["norm"]:
            return False
        if abs(o["start"] - w["start"]) <= 0.18:
            return True
        # sustained words: starts wobble with slice bounds, spans overlap;
        # two heavily-overlapping same-text words are one sung event
        inter = min(o["end"], w["end"]) - max(o["start"], w["start"])
        union = max(o["end"], w["end"]) - min(o["start"], w["start"])
        return min(o["end"] - o["start"], w["end"] - w["start"]) >= 0.5 and inter / union >= 0.5

    # cluster across the union of all framings: a word heard consistently in
    # any two framings is trustworthy, wherever it first appeared
    pool = [(ri, w) for ri, r in enumerate(runs) for w in r]
    pool.sort(key=lambda t: t[1]["start"])
    used = [False] * len(pool)
    stable = []
    for i, (ri, w) in enumerate(pool):
        if used[i]:
            continue
        used[i] = True
        starts = [w["start"]]
        run_ids = {ri}
        for j in range(i + 1, len(pool)):
            rj, o = pool[j]
            if not used[j] and same_word(w, o):
                used[j] = True
                starts.append(o["start"])
                run_ids.add(rj)
        if len(run_ids) >= 2:
            stable.append({**w, "start": round(float(np.median(starts)), 3), "stable": True})
    stable.sort(key=lambda w: w["start"])
    # same-text stable entries with overlapping spans are one sung event
    # heard at slightly different offsets - keep a single merged word
    merged: list[dict] = []
    for w in stable:
        if (
            merged
            and merged[-1]["norm"] == w["norm"]
            and w["start"] < merged[-1]["end"] - 0.1
        ):
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
            merged[-1]["p"] = max(merged[-1]["p"], w["p"])
        else:
            merged.append(dict(w))
    return merged


def dedupe_hyp(words: list[dict]) -> list[dict]:
    """Overlapping windows re-hear the same word; keep one per (norm, ~time)."""
    words = sorted(words, key=lambda w: w["start"])
    out: list[dict] = []
    for w in words:
        dup = any(
            o["norm"] == w["norm"] and abs(o["start"] - w["start"]) < 0.2 for o in out[-6:]
        )
        if not dup:
            out.append(w)
    return out


def transcribe_windows(y: np.ndarray, feats: AudioFeatures, model_name: str) -> list[dict]:
    """Whisper words with timestamps, transcribing voiced windows independently."""
    intervals = feats.voiced_intervals(0.0, feats.duration, min_len=0.25)
    windows: list[tuple[float, float]] = []
    for v0, v1 in intervals:
        if windows and v0 - windows[-1][1] < 1.2 and v1 - windows[-1][0] < 26.0:
            windows[-1] = (windows[-1][0], v1)
        else:
            windows.append((v0, v1))
    merged = [(max(0.0, a - 0.35), min(feats.duration, b + 0.35)) for a, b in windows if b - a > 0.3]

    model = get_whisper(model_name)
    words = []
    for a, b in merged:
        words.extend(transcribe_slice(model, y, a, b))
    return dedupe_hyp(words)


_SIM_CACHE: dict[tuple[str, str], float] = {}

# common sung/written variants that plain edit distance underrates
_SIM_OVERRIDES = {
    ("ya", "you"): 0.9,
    ("ya", "yah"): 0.9,
    ("cause", "because"): 0.9,
    ("cause", "cos"): 0.85,
    ("gonna", "going"): 0.85,
    ("wanna", "want"): 0.85,
    ("gotta", "got"): 0.85,
    ("til", "until"): 0.85,
    ("em", "them"): 0.85,
}


def word_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    key = (a, b) if a <= b else (b, a)
    hit = _SIM_CACHE.get(key)
    if hit is None:
        override = _SIM_OVERRIDES.get(key) or _SIM_OVERRIDES.get((key[1], key[0]))
        hit = override or difflib.SequenceMatcher(None, key[0], key[1]).ratio()
        _SIM_CACHE[key] = hit
    return hit


def interpolated_base_times(base: list[dict]) -> list[float]:
    """Per-token rough time from the base alignment, gaps interpolated."""
    n = len(base)
    times = [b["start"] if b else None for b in base]
    known = [(k, t) for k, t in enumerate(times) if t is not None]
    if not known:
        return [0.0] * n
    out = []
    ki = 0
    for k in range(n):
        if times[k] is not None:
            out.append(float(times[k]))
            continue
        while ki + 1 < len(known) and known[ki + 1][0] < k:
            ki += 1
        lo = known[ki]
        hi = known[ki + 1] if ki + 1 < len(known) else lo
        if hi[0] == lo[0]:
            out.append(float(lo[1]))
        else:
            f = (k - lo[0]) / (hi[0] - lo[0])
            out.append(float(lo[1] + f * (hi[1] - lo[1])))
    return out


def match_transcript(
    tokens: list[dict],
    hyp: list[dict],
    base_times: list[float] | None = None,
    pen_scale: float = 2.4,
) -> list[tuple[int, int, float]]:
    """Global fuzzy alignment lyric tokens vs transcript words.

    Repeated lyric lines make pure text matching degenerate, so matches are
    softly penalized for disagreeing with the base CTC timeline. The penalty
    saturates well below the cost of skipping, letting anchors still correct
    genuine CTC drift (~1-2s) while rejecting wrong-repetition matches (~3s+).

    Returns matched (token_idx, hyp_idx, similarity) pairs, monotone in both.
    """
    n, m = len(tokens), len(hyp)
    if n == 0 or m == 0:
        return []
    GAP = -0.35
    NEG = -1e9
    score = np.full((n + 1, m + 1), 0.0)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 up(skip tok), 2 left(skip hyp)
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + GAP
        move[i][0] = 1
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + GAP
        move[0][j] = 2
    sims = np.full((n, m), NEG)
    for i in range(1, n + 1):
        tn = tokens[i - 1]["norm"]
        bt = base_times[i - 1] if base_times else None
        for j in range(1, m + 1):
            sim = word_similarity(tn, hyp[j - 1]["norm"]) if tn else 0.0
            sims[i - 1][j - 1] = sim
            pen = 0.0
            if bt is not None:
                dt = abs(hyp[j - 1]["start"] - bt)
                pen = 0.35 * min(1.0, (dt / pen_scale) ** 2)
            # exact matches outrank fuzzy false friends ("two"/"to") even
            # when the fuzzy one sits closer to the (possibly wrong) base time
            exact = 0.15 if sim >= 0.999 else 0.0
            diag = score[i - 1][j - 1] + (sim - 0.55) + exact - pen
            up = score[i - 1][j] + GAP
            left = score[i][j - 1] + GAP
            best = max(diag, up, left)
            score[i][j] = best
            move[i][j] = 0 if best == diag else (1 if best == up else 2)
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        mv = move[i][j]
        if i > 0 and j > 0 and mv == 0:
            pairs.append((i - 1, j - 1, float(sims[i - 1][j - 1])))
            i, j = i - 1, j - 1
        elif i > 0 and (mv == 1 or j == 0):
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def pick_anchors(
    tokens: list[dict],
    hyp: list[dict],
    pairs: list[tuple[int, int, float]],
    base: list[dict],
) -> list[dict]:
    """Strong matched words that can pin the timeline."""
    strong = []
    for k, (ti, hj, sim) in enumerate(pairs):
        if sim < 0.78 or hyp[hj]["p"] < 0.25:
            continue
        # one edit turns any 2-3 letter word into another ("to"/"two");
        # short words may only anchor on an exact match
        if len(tokens[ti]["norm"]) <= 3 and sim < 0.999:
            continue
        # whisper sometimes merges several sung words into one token
        # ("gotta" spanning "I got a"); its span then says nothing about
        # where the matched lyric word starts
        hyp_dur = hyp[hj]["end"] - hyp[hj]["start"]
        expected = 0.12 + 0.14 * syllables(tokens[ti]["norm"])
        if hyp_dur > max(1.0, 3.5 * expected):
            continue
        # neighbor support: at least one adjacent pair also matches decently
        support = 0
        for kk in (k - 1, k + 1):
            if 0 <= kk < len(pairs):
                pt, ph, ps = pairs[kk]
                if ps >= 0.6 and abs(pt - ti) <= 2 and abs(ph - hj) <= 2:
                    support += 1
        if support == 0:
            continue
        strong.append({"tok": ti, "t": hyp[hj]["start"], "t_end": hyp[hj]["end"], "sim": sim})
    # enforce monotonic times via longest increasing subsequence (by time)
    if not strong:
        return []
    best_len = [1] * len(strong)
    prev = [-1] * len(strong)
    for a in range(len(strong)):
        for b in range(a):
            if strong[b]["t"] <= strong[a]["t"] - 0.01 and best_len[b] + 1 > best_len[a]:
                best_len[a] = best_len[b] + 1
                prev[a] = b
    end = int(np.argmax(best_len))
    keep = []
    while end != -1:
        keep.append(strong[end])
        end = prev[end]
    keep.reverse()
    # drop anchors that wildly disagree with several neighbors' base positions?
    # (kept simple: LIS already removes stragglers)
    return keep


# --------------------------------------------------------------- reconciling

def detect_holds(
    emission,
    feats: AudioFeatures,
    hyp: list[dict],
) -> list[tuple[float, float]]:
    """Sustained sung holds: melisma spans where word attacks cannot happen.

    A hold is voiced audio where the CTC posterior is (almost) all blank for
    a sustained stretch. Note changes inside one melisma briefly dip the
    blank, so nearby runs are bridged. A single transcript word starting at a
    hold's head is the hold-owner's own attack; several word starts strictly
    inside mean the region is articulated singing the CTC merely cannot read,
    so the hold is vetoed.
    """
    blank = emission[:, 0].exp().numpy()
    smooth = np.convolve(blank, np.ones(9) / 9, mode="same")
    n = len(smooth)

    runs: list[tuple[float, float]] = []
    start = None
    for f in range(n + 1):
        ok = f < n and smooth[f] >= 0.85 and bool(feats.voiced[feats.t2f(f * FRAME_SEC)])
        if ok and start is None:
            start = f * FRAME_SEC
        elif not ok and start is not None:
            if f * FRAME_SEC - start >= 0.6:
                runs.append((start, f * FRAME_SEC))
            start = None

    def voiced_frac(a: float, b: float) -> float:
        i, j = feats.t2f(a), feats.t2f(b)
        if j <= i:
            return 1.0
        return float(np.mean(feats.voiced[i:j]))

    # bridge only across voiced re-articulations (note changes inside one
    # melisma); an unvoiced dip is a real phrase boundary
    bridged: list[tuple[float, float]] = []
    for h0, h1 in runs:
        if bridged and h0 - bridged[-1][1] < 0.5 and voiced_frac(bridged[-1][1], h0) >= 0.7:
            bridged[-1] = (bridged[-1][0], h1)
        else:
            bridged.append((h0, h1))

    # A blank run can span a melisma AND the following words (CTC blind to
    # both). The transcript tells them apart: a single long word over the
    # head is the melisma itself, while a covered string of word starts
    # marks where articulated singing resumes - truncate the hold there.
    holds = []
    for h0, h1 in bridged:
        inner = sorted(w["start"] for w in hyp if h0 + 0.25 < w["start"] < h1 - 0.05)
        cut = h1
        for s in inner:
            tail_cov = sum(
                max(0.0, min(w["end"], h1) - max(w["start"], s)) for w in hyp
            ) / max(h1 - s, 0.01)
            if tail_cov >= 0.5:
                # whisper word starts lag the true attack over melisma
                cut = s - 0.25
                break
        if cut - h0 >= 0.6:
            holds.append((h0, cut))
    return holds


def subtract_holds(
    ivals: list[tuple[float, float]],
    holds: list[tuple[float, float]],
    keep_head: float = 0.15,
) -> list[tuple[float, float]]:
    """Remove hold interiors from intervals, keeping each hold's attack head."""
    out = ivals
    for h0, h1 in holds:
        cut0, cut1 = h0 + keep_head, h1
        nxt = []
        for a, b in out:
            if cut1 <= a or cut0 >= b:
                nxt.append((a, b))
                continue
            if a < cut0:
                nxt.append((a, cut0))
            if cut1 < b:
                nxt.append((cut1, b))
        out = nxt
    return [(a, b) for a, b in out if b - a > 0.02]


def in_hold(t: float, holds: list[tuple[float, float]], head: float = 0.15) -> bool:
    return any(h0 + head < t < h1 for h0, h1 in holds)


def hold_chains(
    holds: list[tuple[float, float]],
    t0: float,
    t1: float,
    feats: AudioFeatures | None = None,
    max_gap: float = 0.9,
) -> list[tuple[float, float]]:
    """Merge nearby holds: re-articulation islets belong to one melisma.

    An unvoiced dip between holds is a breath - a real boundary - so merging
    only crosses voiced gaps.
    """
    sel = [
        (max(h0, t0), min(h1, t1))
        for h0, h1 in holds
        if min(h1, t1) - max(h0, t0) > 0.2
    ]

    def gap_voiced(a: float, b: float) -> bool:
        if feats is None:
            return True
        i, j = feats.t2f(a), feats.t2f(b)
        if j <= i:
            return True
        return float(np.mean(feats.voiced[i:j])) >= 0.7

    chains: list[tuple[float, float]] = []
    for h0, h1 in sel:
        if chains and h0 - chains[-1][1] < max_gap and gap_voiced(chains[-1][1], h0):
            chains[-1] = (chains[-1][0], h1)
        else:
            chains.append((h0, h1))
    return chains


def chain_owner(
    aligner: "CtcAligner", emission, chain: tuple[float, float], norms: list[str]
) -> str | None:
    """The word whose letters keep firing inside a melisma chain owns it.

    A held "o-o-on" keeps spiking 'o' (and finally 'n') across the chain; the
    words sung before/after leave no letter trace inside. Blank dominates the
    raw posterior, so letters are renormalized against non-blank mass. The
    owner norm must clearly beat every other word. Duplicate norms (repeated
    lines) are fine: ownership is by word text, order resolves which instance.
    """
    f0 = max(0, int(chain[0] / FRAME_SEC))
    f1 = min(emission.shape[0], int(chain[1] / FRAME_SEC))
    if f1 - f0 < 10:
        return None
    probs = emission[f0:f1].exp()
    letters = probs[:, 1:]
    rel = letters / (letters.sum(dim=1, keepdim=True) + 1e-9)
    cov: dict[str, float] = {}
    for norm in set(norms):
        ids = [
            aligner.char_to_id[c.upper()] - 1
            for c in set(norm)
            if c.upper() in aligner.char_to_id and aligner.char_to_id[c.upper()] > 0
        ]
        if not ids:
            continue
        per_frame = rel[:, ids].max(dim=1).values
        top = per_frame.sort(descending=True).values[: max(3, (f1 - f0) // 2)]
        cov[norm] = float(top.mean())
    if not cov:
        return None
    ranked = sorted(cov.items(), key=lambda kv: kv[1], reverse=True)
    if ranked[0][1] < 0.18:
        return None
    # vowel-only words (I, a, oh) light up inside any held vowel; they need a
    # decisive margin before claiming a melisma
    margin = 1.8 if not any(c not in VOWELS for c in ranked[0][0] if c.isalpha()) else 1.35
    if len(ranked) > 1 and ranked[0][1] < margin * ranked[1][1]:
        return None
    return ranked[0][0]


def first_letter_spikes(
    aligner: "CtcAligner", emission, norm: str, t_lo: float, t_hi: float
) -> list[float]:
    """Times where the CTC probability of the word's first letter spikes."""
    if not norm:
        return []
    ch = norm[0].upper()
    idx = aligner.char_to_id.get(ch)
    if idx is None:
        return []
    import scipy.signal

    f0 = max(0, int(t_lo / FRAME_SEC))
    f1 = min(emission.shape[0], int(t_hi / FRAME_SEC) + 1)
    if f1 - f0 < 3:
        return []
    probs = emission[f0:f1, idx].exp().numpy()
    peaks, _ = scipy.signal.find_peaks(probs, height=0.08, distance=5)
    return [round((f0 + int(p)) * FRAME_SEC, 3) for p in peaks]

def distribute_words(
    tokens: list[dict],
    t0: float,
    t1: float,
    feats: AudioFeatures,
    extra_onsets: list[float] | None = None,
    letter_spikes: list[list[float]] | None = None,
    holds: list[tuple[float, float]] | None = None,
    owned_chains: list[tuple[float, float, str]] | None = None,
    next_line: int | None = None,
) -> list[dict]:
    """Place words across voiced audio in [t0, t1].

    Priors are syllable-proportional over voiced time; a small DP then snaps
    each word start to an articulation event (librosa onset or a whisper word
    onset), keeping the order and minimum durations. Sung melisma stretches a
    single syllable for seconds, so proportional placement alone lands words
    mid-vowel; articulation snapping recovers the actual attacks.
    """
    n = len(tokens)
    if n == 0:
        return []
    t1 = max(t1, t0 + 0.05 * n)
    ivals = feats.voiced_intervals(t0, t1, min_len=0.06)
    if not ivals:
        ivals = [(t0, t1)]
    if holds:
        # words are articulated outside sustained holds; a hold extends the
        # word that starts at its head, so priors spread over articulated time
        reduced = subtract_holds(ivals, holds)
        if reduced and sum(b - a for a, b in reduced) >= 0.12 * n:
            ivals = reduced
    total = sum(b - a for a, b in ivals)
    weights = [syllables(t["norm"]) + 0.35 for t in tokens]
    wsum = sum(weights)
    acc = 0.0
    offsets = []
    for w in weights:
        offsets.append(acc / wsum * total)
        acc += w
    offsets.append(total)

    def voiced_time(offset: float) -> float:
        rem = offset
        for a, b in ivals:
            if rem <= (b - a):
                return a + rem
            rem -= b - a
        return ivals[-1][1]

    priors = [voiced_time(offsets[k]) for k in range(n)]
    seg_end = voiced_time(total)

    # words that belong to the same lyric line as the next pinned word
    # cluster near it (a count-in "1" sits with its anchored "2, 3", not in
    # stray audio seconds earlier)
    if next_line is not None and t1 - t0 > 1.2:
        pull = 0.55
        for k in range(n - 1, -1, -1):
            if tokens[k].get("line") != next_line:
                break
            priors[k] = max(priors[k], t1 - pull)
            pull += 0.55

    # candidate articulation events with salience
    env = feats.onset_env
    env_max = float(np.max(env)) + 1e-9

    def env_at(t: float) -> float:
        return float(env[feats.t2f(t)]) / env_max

    # candidates hugging the far pin belong to the next phrase, not this one
    # (kept tight: rapid-fire words can attack ~0.12s before the next pin)
    hi_edge = t1 - 0.10 if t1 - 0.10 > t0 else t1
    holds = holds or []

    def usable(t: float) -> bool:
        return t0 - 0.05 <= t <= hi_edge and not in_hold(t, holds)

    cands: dict[float, float] = {}
    for t in feats.onsets:
        if usable(float(t)):
            cands[round(float(t), 3)] = 0.5 * env_at(float(t))
    # voicing resuming after a breath is a certain word boundary - but only
    # when actual singing resumes, not flickering instrumental bleed, and
    # not the previous word's own re-voicing just past its capped end
    for v0, v1 in feats.voiced_intervals(t0, t1, min_len=0.15):
        if usable(float(v0)) and v0 > t0 + 0.25:
            i0 = feats.t2f(v0)
            i1 = feats.t2f(min(v0 + 0.25, v1))
            if i1 > i0 and float(np.mean(feats.rms_db[i0:i1])) >= -24.0:
                key = round(float(v0), 3)
                cands[key] = max(cands.get(key, 0.0), 0.6)
    for t, strength in feats.novelty:
        if usable(float(t)):
            key = round(float(t), 3)
            cands[key] = max(cands.get(key, 0.0), 0.42 * strength)
    stable_ts = []
    for item in extra_onsets or []:
        if isinstance(item, tuple):
            t, is_stable = item
        else:
            t, is_stable = item, False
        if usable(float(t)):
            key = round(float(t), 3)
            if is_stable:
                # stability-confirmed attacks act as pseudo-anchors: low base
                # salience, strong bonus for the word whose prior is nearest
                # (otherwise n words chase n junk candidates)
                stable_ts.append(key)
                cands[key] = max(cands.get(key, 0.0), 0.2)
            else:
                cands[key] = max(cands.get(key, 0.0), 0.45)
    for words_spikes in letter_spikes or []:
        for t in words_spikes:
            if usable(float(t)):
                cands.setdefault(round(float(t), 3), 0.15)
    for p in priors:
        if not in_hold(p, holds):
            cands.setdefault(round(p, 3), 0.0)
    if not cands:
        for p in priors:
            cands.setdefault(round(p, 3), 0.0)
    cand_times = sorted(cands)
    sal = [cands[t] for t in cand_times]
    m = len(cand_times)

    # per-word bonus: candidate coincides with a CTC spike of the word's first
    # letter. CTC spikes lag the acoustic attack (vowels by up to ~300ms), so
    # a candidate slightly before the spike is also a plausible attack.
    bonus = np.zeros((n, m))
    for t in stable_ts:
        k_star = int(np.argmin([abs(p - t) for p in priors]))
        for j, ct in enumerate(cand_times):
            if abs(ct - t) <= 0.03:
                bonus[k_star][j] = max(bonus[k_star][j], 0.75)
    if letter_spikes:
        for k in range(min(n, len(letter_spikes))):
            # vowel spikes fire on every sung vowel - only consonant attacks
            # (plosives/fricatives) are precise enough to bonus
            if tokens[k]["norm"][:1] in VOWELS:
                continue
            for t in letter_spikes[k]:
                for j, ct in enumerate(cand_times):
                    if -0.05 <= t - ct <= 0.15:
                        bonus[k][j] = max(bonus[k][j], 0.5)

    # melisma ownership: only the owning word (by text) may start at or inside
    # its chain, and taking the chain head effectively overrides other paths.
    # The head pull requires a real acoustic event there - a blank run can
    # begin in a soft tail well before the owner's actual attack.
    legal = np.ones((n, m), dtype=bool)
    for c0, c1, owner_norm in owned_chains or []:
        for j, ct in enumerate(cand_times):
            if c0 - 0.05 < ct < c1:
                for k in range(n):
                    if tokens[k]["norm"] != owner_norm:
                        legal[k][j] = False
                    elif ct <= c0 + 0.15 and sal[j] >= 0.15:
                        bonus[k][j] = max(bonus[k][j], 3.0)

    # DP: monotone assignment of words to candidate starts.
    # In hold-dominated (melisma) audio the proportional prior is structurally
    # wrong - the singer speaks the words quickly and stretches one syllable -
    # so the prior yields to articulation evidence there.
    hold_cover = sum(
        max(0.0, min(h1, t1) - max(h0, t0)) for h0, h1 in holds
    ) / max(t1 - t0, 0.01)
    POS_W = 0.12 if hold_cover >= 0.4 else 0.26  # per-second deviation from prior
    NEG = -1e9
    dp = np.full((n, m), NEG)
    back = np.zeros((n, m), dtype=np.int32)
    for j in range(m):
        if legal[0][j]:
            dp[0][j] = sal[j] + bonus[0][j] - POS_W * abs(cand_times[j] - priors[0])
    for k in range(1, n):
        md = min_duration(tokens[k - 1]["norm"])
        best_j, best_v = -1, NEG
        jj = 0
        for j in range(m):
            # candidates for word k must leave room for word k-1
            while jj < m and cand_times[jj] <= cand_times[j] - md:
                if dp[k - 1][jj] > best_v:
                    best_v, best_j = dp[k - 1][jj], jj
                jj += 1
            if best_j < 0 or not legal[k][j]:
                continue
            local = sal[j] + bonus[k][j] - POS_W * abs(cand_times[j] - priors[k])
            dp[k][j] = best_v + local
            back[k][j] = best_j
    ends = [j for j in range(m) if dp[n - 1][j] > NEG / 2]
    starts = priors
    if ends:
        j = max(ends, key=lambda jj: dp[n - 1][jj])
        picked = [0] * n
        for k in range(n - 1, -1, -1):
            picked[k] = j
            j = back[k][j]
        starts = [cand_times[p] for p in picked]

    out = []
    for k, tok in enumerate(tokens):
        s = max(t0, starts[k])
        e = starts[k + 1] if k + 1 < n else seg_end
        e = max(e, s + min_duration(tok["norm"]))
        out.append(
            {
                "start": round(s, 3),
                "end": round(min(e, t1) if k + 1 == n else e, 3),
                "score": 0.1,
                "source": "distributed",
            }
        )
    for k in range(1, len(out)):
        if out[k]["start"] < out[k - 1]["end"]:
            out[k]["start"] = out[k - 1]["end"]
            out[k]["end"] = max(out[k]["end"], out[k]["start"] + 0.05)
    return out


def reconcile(
    tokens: list[dict],
    base: list[dict],
    anchors: list[dict],
    aligner: CtcAligner,
    emission,
    feats: AudioFeatures,
    hyp: list[dict] | None = None,
    holds: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Combine base FA with anchors; re-align or distribute drifting segments."""
    n = len(tokens)
    out: list[dict | None] = [None] * n

    # segment boundaries: token indices of anchors (plus virtual ends)
    pins = [{"tok": -1, "t": 0.0, "t_end": 0.0}] + anchors + [
        {"tok": n, "t": feats.duration, "t_end": feats.duration}
    ]

    # place anchor words themselves: trust base when it agrees, else whisper time
    for a in anchors:
        ti = a["tok"]
        b = base[ti]
        if b and abs(b["start"] - a["t"]) <= 0.30:
            # base start agrees; its end may still be clipped mid-melisma, so
            # extend toward whisper's word end (bounded - whisper ends smear)
            end = max(b["end"], min(a["t_end"], b["start"] + 1.2))
            out[ti] = {**b, "end": round(end, 3), "source": "ctc"}
        else:
            end = a["t_end"]
            if b and a["t"] < b["end"] and 0 < b["end"] - a["t"] < 2.0:
                end = max(end, b["end"])
            # whisper word ends smear across melisma; keep anchors compact and
            # let the karaoke-hold polish re-extend through voiced audio
            cap = a["t"] + max(0.35, 0.20 * syllables(tokens[ti]["norm"]) + 0.15)
            end = min(end, cap)
            out[ti] = {
                "start": a["t"],
                "end": max(end, a["t"] + min_duration(tokens[ti]["norm"])),
                "score": 0.5,
                "source": "anchor",
            }

    for p in range(len(pins) - 1):
        lo_tok, hi_tok = pins[p]["tok"], pins[p + 1]["tok"]
        seg = list(range(lo_tok + 1, hi_tok))
        if not seg:
            continue
        t_lo = out[lo_tok]["end"] if lo_tok >= 0 and out[lo_tok] else pins[p]["t_end"]
        t_hi = out[hi_tok]["start"] if hi_tok < n and out[hi_tok] else pins[p + 1]["t"]
        t_lo, t_hi = float(t_lo), float(max(t_hi, t_lo + 0.02))

        seg_base = [base[k] for k in seg]
        ok = all(b is not None for b in seg_base)
        if ok:
            inside = all(b["start"] >= t_lo - 0.35 and b["end"] <= t_hi + 0.35 for b in seg_base)
            scores = [b["score"] for b in seg_base]
            if inside and float(np.mean(scores)) >= 0.28:
                for k, b in zip(seg, seg_base):
                    out[k] = {**b, "source": "ctc"}
                continue

        # windowed re-alignment on the emission slice between the pins
        f0 = int(max(0.0, t_lo - 0.05) / FRAME_SEC)
        f1 = int(min(feats.duration, t_hi + 0.05) / FRAME_SEC)
        words = [tokens[k]["norm"] for k in seg]
        re_al = []
        if f1 - f0 > len("".join(words)) + len(words):
            re_al = aligner.align(emission[f0:f1], words, t_offset=f0 * FRAME_SEC)
        re_scores = [w["score"] for w in re_al if w]
        if re_al and all(w is not None for w in re_al) and float(np.mean(re_scores)) >= 0.35:
            for k, w in zip(seg, re_al):
                out[k] = {**w, "source": "ctc_window"}
            continue

        # even when the segment as a whole is weak, individually confident
        # re-aligned words ("got" with its plosive spike) are deterministic
        # evidence - pin them and distribute only the weak words between
        pinned: list[int] = []
        if re_al and len(re_al) == len(seg):
            for j, w in enumerate(re_al):
                # short words match stray letters anywhere ("her" pinning on
                # the next chorus' "she"), so they need far stronger evidence
                floor = 0.30 if len(tokens[seg[j]]["norm"]) >= 4 else 0.50
                if w is None or w["score"] < floor:
                    continue
                # a confident pin must also look like the word: a short word
                # smeared over a second is Viterbi filling space, not evidence
                max_dur = max(0.5, 2.5 * (0.12 + 0.14 * syllables(tokens[seg[j]]["norm"])))
                if w["end"] - w["start"] > max_dur:
                    continue
                out[seg[j]] = {**w, "source": "ctc_window"}
                pinned.append(j)

        def fill(sub: list[int], lo: float, hi: float) -> None:
            # weak evidence: distribute across voiced audio between pins,
            # snapping to articulations (whisper word onsets included even
            # when the transcript text didn't match the lyrics) and to CTC
            # spikes of each word's first letter
            in_window = [w for w in hyp or [] if lo - 0.1 <= w["start"] <= hi + 0.1]
            stable_in = [w for w in in_window if w.get("stable")]
            # once stability-checked words exist here, unchecked first-pass
            # words in the same hard region are more likely junk than signal
            extra = [(w["start"], True) for w in stable_in] or [
                (w["start"], False) for w in in_window
            ]
            spikes = [
                first_letter_spikes(aligner, emission, tokens[k]["norm"], lo, hi) for k in sub
            ]
            sub_norms = [tokens[k]["norm"] for k in sub]
            owned = []
            for chain in hold_chains(holds or [], lo, hi, feats):
                owner_norm = chain_owner(aligner, emission, chain, sub_norms)
                if owner_norm is not None:
                    owned.append((chain[0], chain[1], owner_norm))
            nxt = sub[-1] + 1
            next_line = tokens[nxt]["line"] if nxt < n else None
            dist = distribute_words(
                [tokens[k] for k in sub], lo, hi, feats, extra, spikes, holds, owned, next_line
            )
            for k, w in zip(sub, dist):
                out[k] = w

        if not pinned:
            fill(seg, t_lo, t_hi)
        else:
            # distribute each weak stretch between its surrounding pins
            j = 0
            while j < len(seg):
                if j in pinned:
                    j += 1
                    continue
                j0 = j
                while j < len(seg) and j not in pinned:
                    j += 1
                lo = float(out[seg[j0 - 1]]["end"]) if j0 > 0 else t_lo
                hi = float(out[seg[j]]["start"]) if j < len(seg) else t_hi
                fill(seg[j0:j], lo, max(hi, lo + 0.02))

    # any remaining gaps (shouldn't happen) -> distribute
    for k in range(n):
        if out[k] is None:
            prev_end = out[k - 1]["end"] if k else 0.0
            out[k] = {
                "start": prev_end,
                "end": prev_end + min_duration(tokens[k]["norm"]),
                "score": 0.0,
                "source": "fallback",
            }
    return out  # type: ignore[return-value]


def weak_segments(tokens: list[dict], words: list[dict]) -> list[tuple[list[int], float, float]]:
    """Long distribution-fallback runs: (token indices, t_start, t_end)."""
    runs, run = [], []
    for k, w in enumerate(words):
        if w.get("source") == "distributed":
            run.append(k)
        else:
            if run:
                runs.append(run)
            run = []
    if run:
        runs.append(run)
    out = []
    for run in runs:
        a, b = float(words[run[0]]["start"]), float(words[run[-1]]["end"])
        if len(run) >= 2 and b - a >= 1.2:
            out.append((run, a, b))
    return out


def merge_anchors(primary: list[dict], extra: list[dict]) -> list[dict]:
    """Combine anchor sets, keep one per token, enforce monotone times.

    On similarity ties the extra (tight-slice) anchor wins: focused slices
    hear dense/doubled vocals more accurately than the long first-pass windows.
    """
    by_tok: dict[int, dict] = {}
    for a in primary:
        cur = by_tok.get(a["tok"])
        if cur is None or a["sim"] > cur["sim"]:
            by_tok[a["tok"]] = a
    for a in extra:
        cur = by_tok.get(a["tok"])
        if cur is None or a["sim"] >= cur["sim"]:
            by_tok[a["tok"]] = a
    merged = [by_tok[t] for t in sorted(by_tok)]
    # longest increasing subsequence over time
    if not merged:
        return []
    best_len = [1] * len(merged)
    prev = [-1] * len(merged)
    for i in range(len(merged)):
        for j in range(i):
            if merged[j]["t"] <= merged[i]["t"] - 0.01 and best_len[j] + 1 > best_len[i]:
                best_len[i] = best_len[j] + 1
                prev[i] = j
    end = int(np.argmax(best_len))
    keep = []
    while end != -1:
        keep.append(merged[end])
        end = prev[end]
    keep.reverse()
    return keep


# ------------------------------------------------------------ boundary polish

def polish(tokens: list[dict], words: list[dict], feats: AudioFeatures) -> list[dict]:
    n = len(words)

    # a word placed in silence before its audible attack (whisper/prior fuzz)
    # snaps forward to the attack; small pre-roll covers unvoiced consonants
    for k, w in enumerate(words):
        if w.get("source") not in ("anchor", "distributed"):
            continue
        s = float(w["start"])
        if feats.voiced[feats.t2f(s + 0.02)]:
            continue
        limit = min(s + 0.55, float(words[k + 1]["start"]) if k + 1 < n else feats.duration)
        a, b = feats.t2f(s), feats.t2f(limit)
        attack = None
        for f in range(a, b):
            if feats.voiced[f]:
                attack = f * HOP_SEC
                break
        if attack is None:
            continue
        new_start = round(max(s, attack - 0.07), 3)
        if new_start > s + 0.05:
            w["start"] = new_start
            md = min_duration(tokens[k]["norm"])
            if float(w["end"]) < new_start + md:
                w["end"] = round(new_start + md, 3)

    # monotonic repair
    for k in range(1, n):
        prev, cur = words[k - 1], words[k]
        if cur["start"] < prev["end"]:
            lo = prev["start"] + min_duration(tokens[k - 1]["norm"])
            boundary = max(min(prev["end"], cur["start"]), lo)
            prev["end"] = round(boundary, 3)
            cur["start"] = round(boundary, 3)
        if cur["end"] < cur["start"] + min_duration(tokens[k]["norm"]):
            cur["end"] = round(cur["start"] + min_duration(tokens[k]["norm"]), 3)

    # karaoke hold: extend a word through voiced audio until the next word
    for k in range(n - 1):
        cur, nxt = words[k], words[k + 1]
        gap = nxt["start"] - cur["end"]
        if gap <= 0.02:
            continue
        old_end = float(cur["end"])
        if tokens[k]["line"] != tokens[k + 1]["line"] and gap > 1.5:
            hold_end = feats.last_voiced_before(cur["end"], min(nxt["start"], cur["end"] + 4.0))
            if hold_end > cur["end"]:
                cur.setdefault("core_end", old_end)
                cur["end"] = round(min(hold_end + 0.05, nxt["start"] - 0.02), 3)
            continue
        if feats.is_voiced_between(cur["end"], nxt["start"]):
            cur.setdefault("core_end", old_end)
            cur["end"] = round(nxt["start"] - 0.01, 3)
        else:
            hold_end = feats.last_voiced_before(cur["end"], nxt["start"])
            if hold_end > cur["end"]:
                cur.setdefault("core_end", old_end)
                cur["end"] = round(min(hold_end + 0.06, nxt["start"] - 0.01), 3)

    for w in words:
        w["start"] = round(max(0.0, w["start"]), 3)
        w["end"] = round(min(feats.duration, max(w["end"], w["start"] + 0.03)), 3)
    return words


# ----------------------------------------------------------------- echo runs

def paren_runs(tokens: list[dict]) -> list[list[int]]:
    runs, cur = [], []
    for k, t in enumerate(tokens):
        if t["paren"]:
            cur.append(k)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def find_twin(tokens: list[dict], run: list[int]) -> list[int] | None:
    """Preceding non-paren token span with the same normalized words."""
    target = [tokens[k]["norm"] for k in run]
    lo = max(0, run[0] - 40)
    cand_idx = [k for k in range(lo, run[0]) if not tokens[k]["paren"]]
    words = [tokens[k]["norm"] for k in cand_idx]
    for s in range(len(words) - len(target), -1, -1):
        if words[s : s + len(target)] == target:
            return cand_idx[s : s + len(target)]
    return None


def realign_pickup(
    tokens: list[dict],
    words: list[dict],
    feats: AudioFeatures,
    aligner: "CtcAligner",
    emission,
    run: list[int],
    prev_end: float,
) -> bool:
    """Try to place a paren run as a sung pickup absorbed by the next word.

    Whisper merges pickups into the next word ("twice step" heard as one
    "Step"), so the next word's anchor swallows the paren audio. Re-aligning
    [run + following words] against the emissions lets the letter evidence
    split them; accepted only when the following words land confidently and
    strictly later than before (i.e., they really had stolen audio).
    """
    n = len(words)
    last = run[-1]
    follow = [j for j in range(last + 1, min(last + 3, n)) if not tokens[j]["paren"]]
    if not follow:
        return False
    t_lo = prev_end + 0.02
    t_hi = float(words[follow[-1]]["end"]) + 0.1
    f0 = int(max(0.0, t_lo) / FRAME_SEC)
    f1 = int(min(feats.duration, t_hi) / FRAME_SEC)
    group = run + follow
    norms = [tokens[k]["norm"] for k in group]
    if f1 - f0 <= sum(len(w) for w in norms) + len(norms):
        return False
    re_al = aligner.align(emission[f0:f1], norms, t_offset=f0 * FRAME_SEC)
    if not re_al or any(w is None for w in re_al):
        return False
    if any(re_al[j + 1]["start"] < re_al[j]["end"] - 0.02 for j in range(len(re_al) - 1)):
        return False
    follow_re = re_al[len(run) :]
    if min(w["score"] for w in follow_re) < 0.15:
        return False
    if follow_re[0]["start"] < float(words[follow[0]]["start"]) + 0.25:
        return False
    if re_al[0]["start"] < prev_end - 0.05:
        return False

    # The follow words' re-aligned starts are the trustworthy part; the run's
    # own spans smear (its letters barely register). The pickup is sung in
    # the voiced stretch directly before the follow word's attack.
    v_end = follow_re[0]["start"] - 0.01
    f_end = feats.t2f(v_end)
    f_cur = f_end
    gap = 0
    while f_cur > feats.t2f(max(prev_end, v_end - 2.0)):
        if feats.voiced[f_cur]:
            gap = 0
        else:
            gap += 1
            if gap > 8:
                break
        f_cur -= 1
    v_start = max(prev_end + 0.01, (f_cur + gap) * HOP_SEC)
    expected = sum(0.12 + 0.1 * syllables(tokens[k]["norm"]) for k in run)
    if v_end - v_start < 0.4 * expected:
        return False
    # a pickup is an interjection after a break; continuous singing from the
    # previous word into this stretch means the audio is a response between
    # the words (gap placement), not an absorbed pickup
    if v_start - prev_end < 0.2 or feats.is_voiced_between(prev_end, v_start, frac=0.6):
        return False
    v_start = max(v_start, v_end - 1.6 * expected)
    weights = [syllables(tokens[k]["norm"]) + 0.3 for k in run]
    wsum = sum(weights)
    t = v_start
    for k, wt in zip(run, weights):
        dur = (v_end - v_start) * wt / wsum
        words[k] = {
            "start": round(t, 3),
            "end": round(t + dur, 3),
            "score": 0.15,
            "source": "ctc_pickup",
        }
        t += dur
    for j, (k, w) in enumerate(zip(follow, follow_re)):
        end = w["end"]
        if j + 1 < len(follow_re):
            nxt_start = follow_re[j + 1]["start"]
            if feats.is_voiced_between(end, nxt_start):
                end = nxt_start - 0.01
        else:
            end = max(end, float(words[k]["end"]))
            if k + 1 < n:
                end = min(end, float(words[k + 1]["start"]) - 0.01)
            end = max(end, w["start"] + min_duration(tokens[k]["norm"]))
        words[k] = {**w, "end": round(end, 3), "source": "ctc_pickup"}
    return True


def overlay_echoes(
    tokens: list[dict],
    words: list[dict],
    feats: AudioFeatures,
    base: list[dict] | None = None,
    aligner: "CtcAligner | None" = None,
    emission=None,
    holds: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Echo/backing paren runs without their own audio get overlaid on the main phrase."""
    n = len(words)
    base = base or [None] * n
    attack_times = np.array(
        sorted(set(list(feats.onsets) + [t for t, _ in feats.novelty]))
    )

    def near_attack(t: float, tol: float = 0.22) -> bool:
        if len(attack_times) == 0:
            return True
        return bool(np.min(np.abs(attack_times - t)) <= tol)

    for run in paren_runs(tokens):
        first, last = run[0], run[-1]
        w0, w1 = words[first], words[last]
        span = w1["end"] - w0["start"]
        expected = sum(0.12 + 0.1 * syllables(tokens[k]["norm"]) for k in run)
        good_scores = [words[k]["score"] for k in run]
        has_evidence = (
            float(np.mean(good_scores)) >= 0.30
            and span >= 0.5 * expected
            and all(words[k].get("source", "").startswith(("ctc", "anchor")) for k in run)
        ) or any(words[k].get("source") == "anchor" for k in run)
        if has_evidence:
            continue

        # a karaoke-held previous word may have swallowed the run's audio;
        # its pre-hold core end is the true floor, the tail is reclaimable
        prev_w = words[first - 1] if first > 0 else None
        prev_end = float(prev_w.get("core_end", prev_w["end"])) if prev_w else 0.0
        next_start = words[last + 1]["start"] if last + 1 < n else feats.duration

        def clamp_prev(run_start: float) -> None:
            if prev_w is None:
                return
            if float(prev_w["end"]) > run_start - 0.01:
                lo = float(prev_w["start"]) + min_duration(tokens[first - 1]["norm"])
                prev_w["end"] = round(max(lo, run_start - 0.01), 3)

        # A short run may be a sung pickup that the next word's anchor
        # absorbed ("(TWICE) Step" heard as one "Step"). Letter evidence
        # arbitrates: accepted only when re-alignment moves the next word
        # confidently later. Checked before gap placement because stray gap
        # audio (e.g. the tail of a repeated shout) also attracts the run.
        if (
            len(run) <= 3
            and aligner is not None
            and emission is not None
            and realign_pickup(tokens, words, feats, aligner, emission, run, prev_end)
        ):
            clamp_prev(float(words[first]["start"]))
            continue

        # A real (non-overlapping) echo shows up as voiced audio between the
        # preceding word and the next word. If that audio exists, place the
        # echo there instead of overlaying it on the main phrase.
        gap_voiced = sum(
            b - a for a, b in feats.voiced_intervals(prev_end + 0.04, next_start - 0.02)
        )
        if gap_voiced >= 0.6 * expected:
            # a hold beginning right at the previous word's core end is that
            # word's own melisma tail; the response starts after it
            gap_lo = prev_end
            for h0, h1 in holds or []:
                if prev_end - 0.1 <= h0 <= prev_end + 0.2 and h1 < next_start - 0.1:
                    gap_lo = max(gap_lo, h1 + 0.03)
            run_base = [base[k] for k in run]
            base_fits = (
                all(b is not None for b in run_base)
                and run_base[0]["start"] >= gap_lo - 0.10
                and run_base[-1]["end"] <= next_start + 0.10
                and run_base[-1]["end"] - run_base[0]["start"] >= 0.4 * expected
                and near_attack(float(run_base[0]["start"]))
                and not in_hold(float(run_base[0]["start"]), holds or [])
                and all(
                    run_base[j]["start"] >= run_base[j - 1]["end"] - 0.05
                    for j in range(1, len(run_base))
                )
            )
            if base_fits:
                for k, b in zip(run, run_base):
                    words[k] = {**b, "source": "ctc_echo"}
                clamp_prev(float(run_base[0]["start"]))
            else:
                spikes = None
                if aligner is not None and emission is not None:
                    spikes = [
                        first_letter_spikes(
                            aligner, emission, tokens[k]["norm"], gap_lo, next_start
                        )
                        for k in run
                    ]
                dist = distribute_words(
                    [tokens[k] for k in run],
                    gap_lo + 0.04,
                    next_start - 0.02,
                    feats,
                    None,
                    spikes,
                    holds,
                    None,
                    tokens[last + 1]["line"] if last + 1 < n else None,
                )
                for k, w in zip(run, dist):
                    words[k] = {**w, "source": "distributed_echo"}
                clamp_prev(float(dist[0]["start"]))
            continue

        twin = find_twin(tokens, run)
        # a doubled-vocal overlay only makes sense when the echo directly
        # follows the phrase it repeats; distant twins would teleport it
        if twin and run[0] - twin[-1] <= 8:
            main_start = words[twin[0]]["start"]
            main_end = words[twin[-1]]["end"]
            # echoes trail the main phrase slightly; keep room before the next word
            lag = min(0.25, max(0.08, (next_start - main_end) * 0.4))
            avail_end = max(main_end + lag, min(main_end + lag + 0.2, next_start - 0.02))
            scale_src = main_end - main_start
            scale_dst = max(0.3, avail_end - (main_start + lag))
            ratio = scale_dst / max(scale_src, 0.05)
            overlay_start = main_start + lag
            if overlay_start >= prev_end - max(2.0, scale_src * 1.5):
                for k, m in zip(run, twin):
                    ms, me = words[m]["start"], words[m]["end"]
                    words[k] = {
                        "start": round(overlay_start + (ms - main_start) * ratio, 3),
                        "end": round(overlay_start + (me - main_start) * ratio, 3),
                        "score": words[m]["score"],
                        "source": "echo_overlay",
                    }
                # echoes may overlap preceding words by design
                continue

        # No separate audio and no adjacent twin. Inside continuous singing
        # the run is a buried backing shout ("over (No)") - show it right
        # after the previous word. Before a silent break it is an annotation
        # of the previous word - overlay that word's span.
        if next_start - prev_end > 0.5 and first > 0:
            prev_w = words[first - 1]
            span0, span1 = float(prev_w["start"]), float(prev_w["end"])
            step = (span1 - span0) / len(run)
            for j, k in enumerate(run):
                words[k] = {
                    "start": round(span0 + j * step, 3),
                    "end": round(span0 + (j + 1) * step, 3),
                    "score": 0.05,
                    "source": "echo_annotation",
                }
            continue
        t = prev_end + 0.02
        clamp_prev(t)
        for k in run:
            dur = max(min_duration(tokens[k]["norm"]), 0.30 * (0.12 + 0.1 * syllables(tokens[k]["norm"])))
            words[k] = {
                "start": round(t, 3),
                "end": round(t + dur, 3),
                "score": 0.05,
                "source": "echo_attach",
            }
            t += dur
    return words


def line_evidence(idxs: list[int], words: list[dict], feats: AudioFeatures) -> float:
    """How well-supported a line's word timings are."""
    scores = [float(words[k]["score"]) for k in idxs]
    src = [
        1.0 if words[k].get("source", "").startswith(("anchor", "ctc")) else 0.0
        for k in idxs
    ]
    attacks = np.array(sorted(set(list(feats.onsets) + [t for t, _ in feats.novelty])))
    if len(attacks):
        atk = [
            1.0 if float(np.min(np.abs(attacks - float(words[k]["start"])))) <= 0.12 else 0.0
            for k in idxs
        ]
    else:
        atk = [0.0]
    return float(np.mean(scores) + 0.3 * np.mean(src) + 0.3 * np.mean(atk))


def _line_mfcc(y: np.ndarray, t0: float, t1: float):
    import librosa

    i0, i1 = int(max(0.0, t0) * SR), int(min(len(y) / SR, t1) * SR)
    if i1 - i0 < SR // 4:
        return None
    m = librosa.feature.mfcc(y=y[i0:i1], sr=SR, n_mfcc=13, hop_length=160)
    m = m - m.mean(axis=1, keepdims=True)
    return m


def repeat_similarity(y: np.ndarray, a0: float, a1: float, b0: float, b1: float):
    """(similarity, lag_seconds) between two spans of near-equal duration.

    Correlates MFCC sequences over lags up to +-1.2s; a repeated (copy-paste
    or re-performed) line correlates strongly at the true offset.
    """
    pad = 0.6
    # the shorter span is the reference: sliding a mis-carved longer span's
    # center (often melisma-heavy) gives falsely low similarity
    swapped = (a1 - a0) > (b1 - b0)
    if swapped:
        (a0, a1), (b0, b1) = (b0, b1), (a0, a1)
    ma = _line_mfcc(y, a0 - 0.1, a1 + 0.1)
    mb = _line_mfcc(y, b0 - pad, b1 + pad)
    if ma is None or mb is None:
        return 0.0, 0.0
    ta, tb = ma.shape[1], mb.shape[1]
    win = ta
    if win < 50 or tb < win:
        return 0.0, 0.0
    a_ref = ma / (np.linalg.norm(ma) + 1e-9)
    best, best_lag = 0.0, 0
    for lag in range(0, tb - win + 1, 2):
        seg = mb[:, lag : lag + win]
        c = float(np.sum(a_ref * (seg / (np.linalg.norm(seg) + 1e-9))))
        if c > best:
            best, best_lag = c, lag
    a_off = a0 - 0.1
    b_off = (b0 - pad) + best_lag * 0.01
    delta = b_off - a_off
    return (best, -delta) if swapped else (best, delta)


def ownership_consistency(
    idxs: list[int],
    words: list[dict],
    tokens: list[dict],
    owned_chains: list[tuple[float, float, str]],
) -> float:
    """+/- per melisma chain in the line span: does the word covering the
    chain match the chain's letter-evidence owner?"""
    a = float(words[idxs[0]]["start"])
    b = float(words[idxs[-1]]["end"])
    score = 0.0
    for c0, c1, owner in owned_chains:
        if c1 <= a or c0 >= b:
            continue
        best_k, best_cov = None, 0.0
        for k in idxs:
            cov = min(float(words[k]["end"]), c1) - max(float(words[k]["start"]), c0)
            if cov > best_cov:
                best_cov, best_k = cov, k
        if best_k is None or best_cov < 0.3:
            continue
        score += 0.4 if tokens[best_k]["norm"] == owner else -0.4
    return score


def _transfer_pattern(
    tokens: list[dict],
    words: list[dict],
    feats: AudioFeatures,
    y: np.ndarray,
    hyp: list[dict] | None,
    tmpl: list[int],
    idxs: list[int],
    label: str = "",
    min_sim: float = 0.55,
    max_conflict_frac: float = 0.20,
    debug: bool = False,
) -> bool:
    """Map the template instance's word pattern onto the target instance."""
    n = len(words)
    t0, t1 = float(words[tmpl[0]]["start"]), float(words[tmpl[-1]]["end"])
    b0, b1 = float(words[idxs[0]]["start"]), float(words[idxs[-1]]["end"])
    sim, lag = repeat_similarity(y, t0, t1, b0, b1)
    if debug:
        print(
            f"      repeat {label}: tmpl@{t0:.1f} -> inst@{b0:.1f} sim={sim:.2f} lag={lag:.2f}",
            flush=True,
        )
    if sim < min_sim:
        return False
    dur_t, dur_b = t1 - t0, b1 - b0
    prev_end = float(words[idxs[0] - 1]["end"]) if idxs[0] > 0 else 0.0
    next_start = float(words[idxs[-1] + 1]["start"]) if idxs[-1] + 1 < n else feats.duration
    if abs(dur_b - dur_t) <= 0.15 * max(dur_t, 0.5):
        # audio repeats verbatim: shift the template pattern
        new = [(float(words[m]["start"]) + lag, float(words[m]["end"]) + lag) for m in tmpl]
    else:
        # same phrase, looser phrasing: rescale relative positions
        scale = dur_b / max(dur_t, 0.05)
        new = [
            (
                b0 + (float(words[m]["start"]) - t0) * scale,
                b0 + (float(words[m]["end"]) - t0) * scale,
            )
            for m in tmpl
        ]

    # anchor-backed or transcript-confirmed words are ground truth; if the
    # template pattern contradicts one, the instances are not sung alike
    stable_nearby = [
        w for w in hyp or [] if w.get("stable") and b0 - 0.3 <= w["start"] <= b1 + 0.3
    ]

    def confirmed(k: int) -> bool:
        src = words[k].get("source", "")
        if src == "anchor":
            return True
        if src.startswith("ctc") and float(words[k]["score"]) >= 0.30:
            return True
        s = float(words[k]["start"])
        return any(
            hw["norm"] == tokens[k]["norm"] and abs(s - hw["start"]) <= 0.10
            for hw in stable_nearby
        )

    # confirmed words that disagree with the mapping stay put; when many of
    # them disagree the instances are not sung alike and nothing transfers
    conflict_set = {
        j
        for j, k in enumerate(idxs)
        if confirmed(k) and abs(new[j][0] - float(words[k]["start"])) > 0.20
    }
    if len(conflict_set) > max(1, int(max_conflict_frac * len(idxs))):
        if debug:
            print(f"        rejected: {len(conflict_set)} confirmed words disagree", flush=True)
        return False

    # seams: a karaoke-held previous end is reclaimable; the last word's end
    # clips to the next word
    prev_w = words[idxs[0] - 1] if idxs[0] > 0 else None
    if prev_w is not None and new[0][0] < prev_end:
        prev_core = float(prev_w["start"]) + min_duration(tokens[idxs[0] - 1]["norm"])
        if new[0][0] < prev_core:
            if debug:
                print(f"        rejected: start {new[0][0]:.2f} inside prev core", flush=True)
            return False
        prev_w["end"] = round(new[0][0] - 0.01, 3)
    if new[-1][0] > next_start - 0.04:
        if debug:
            print(f"        rejected: last word start {new[-1][0]:.2f} past next", flush=True)
        return False

    attacks = np.array(sorted(set(list(feats.onsets) + [t for t, _ in feats.novelty])))

    def attack_supported(t: float) -> bool:
        return len(attacks) > 0 and float(np.min(np.abs(attacks - t))) <= 0.12

    for j, (k, (s, e)) in enumerate(zip(idxs, new)):
        # a target word sitting on a real attack does not yield to a template
        # position that has none - but only if keeping it stays adjacent to
        # the mapped neighbors (melisma re-articulations also read as attacks)
        old_s, old_e = float(words[k]["start"]), float(words[k]["end"])
        if j in conflict_set:
            continue
        keep = (
            abs(s - old_s) > 0.25
            and attack_supported(old_s)
            and not attack_supported(s)
        )
        if keep and j > 0 and old_s - new[j - 1][1] > 0.6:
            keep = False
        if keep and j + 1 < len(new) and new[j + 1][0] - old_e > 0.6:
            keep = False
        if keep:
            continue
        words[k] = {
            "start": round(s, 3),
            "end": round(e, 3),
            "score": float(words[tmpl[j]]["score"]),
            "source": "repeat_transfer",
        }
    if float(words[idxs[-1]]["end"]) > next_start:
        words[idxs[-1]]["end"] = round(
            max(next_start - 0.01, float(words[idxs[-1]]["start"]) + 0.05), 3
        )
    for j in range(1, len(idxs)):
        if float(words[idxs[j]]["start"]) < float(words[idxs[j - 1]]["end"]):
            words[idxs[j]]["start"] = words[idxs[j - 1]]["end"]
            if float(words[idxs[j]]["end"]) < float(words[idxs[j]]["start"]) + 0.04:
                words[idxs[j]]["end"] = round(float(words[idxs[j]]["start"]) + 0.04, 3)
    return True


def harmonize_repeated_lines(
    tokens: list[dict],
    words: list[dict],
    feats: AudioFeatures,
    y: np.ndarray,
    hyp: list[dict] | None = None,
    owned_chains: list[tuple[float, float, str]] | None = None,
    debug: bool = False,
) -> list[dict]:
    """Identical lyric lines sung identically should carry identical timing.

    Line level: a clearly weak instance takes the best-evidenced instance's
    pattern. Block level: consecutive repeated line runs (whole verse
    sections) are compared as units, with the earliest occurrence as the
    reference on near-ties - later verses are re-carvings of the same audio
    and inherit the first statement's word pattern where unconfirmed.
    """
    by_line: dict[int, list[int]] = {}
    for k, t in enumerate(tokens):
        by_line.setdefault(t["line"], []).append(k)
    line_ids = sorted(by_line)
    keys = {li: tuple(tokens[k]["norm"] for k in by_line[li]) for li in line_ids}

    groups: dict[tuple, list[list[int]]] = {}
    for li in line_ids:
        if len(keys[li]) >= 3:
            groups.setdefault(keys[li], []).append(by_line[li])

    # -- line level: rescue clearly weak instances
    for key, instances in groups.items():
        if len(instances) < 2:
            continue
        ev = [
            line_evidence(idxs, words, feats)
            + ownership_consistency(idxs, words, tokens, owned_chains or [])
            for idxs in instances
        ]
        t_i = int(np.argmax(ev))
        for i, idxs in enumerate(instances):
            if i == t_i or ev[i] >= 0.60 or ev[t_i] < ev[i] + 0.10:
                continue
            _transfer_pattern(
                tokens, words, feats, y, hyp, instances[t_i], idxs,
                label=" ".join(key[:4]), debug=debug,
            )

    # -- block level: consecutive repeated sections
    pos = {li: i for i, li in enumerate(line_ids)}
    done: set[tuple[int, int]] = set()
    for ai, a_li in enumerate(line_ids):
        for bi in range(ai + 1, len(line_ids)):
            b_li = line_ids[bi]
            if keys[a_li] != keys[b_li] or len(keys[a_li]) < 2:
                continue
            length = 0
            while (
                ai + length < len(line_ids)
                and bi + length < len(line_ids)
                and ai + length < bi
                and keys[line_ids[ai + length]] == keys[line_ids[bi + length]]
            ):
                length += 1
            if length < 2 or (ai, bi) in done:
                continue
            for off in range(length):
                done.add((ai + off, bi + off))
            tmpl_idxs = [k for off in range(length) for k in by_line[line_ids[ai + off]]]
            tgt_idxs = [k for off in range(length) for k in by_line[line_ids[bi + off]]]
            if len(tmpl_idxs) != len(tgt_idxs) or len(tmpl_idxs) < 6:
                continue
            # blocks bulldoze whole sections: demand near-verbatim audio
            # and near-total confirmed agreement
            _transfer_pattern(
                tokens, words, feats, y, hyp, tmpl_idxs, tgt_idxs,
                label=f"block {' '.join(keys[a_li][:3])}.. x{length}",
                min_sim=0.68, max_conflict_frac=0.08, debug=debug,
            )
    return words


# ----------------------------------------------------------------------- main

def align_song(
    y: np.ndarray,
    lines: list[str],
    device: str = "cpu",
    whisper_model: str = "large-v3-turbo",
    debug_dir: pathlib.Path | None = None,
) -> list[dict]:
    tokens_all = lyric_tokens(lines)
    # pure-punctuation tokens carry no phonetic content; they get zero-width
    # spans at the end and are excluded from every alignment stage
    tokens = [t for t in tokens_all if t["norm"]]
    feats = AudioFeatures(y)

    print("==> Word alignment (wav2vec2 CTC)", flush=True)
    aligner = CtcAligner(device=device)
    emission = aligner.emissions(y)
    base = aligner.align(emission, [t["norm"] for t in tokens])

    print("==> Whisper anchors", flush=True)
    anchors: list[dict] = []
    hyp: list[dict] = []
    whisper_ok = False
    try:
        hyp = transcribe_windows(y, feats, whisper_model)
        pairs = match_transcript(tokens, hyp, interpolated_base_times(base))
        anchors = pick_anchors(tokens, hyp, pairs, base)
        whisper_ok = True
    except Exception as exc:  # noqa: BLE001 - anchors are an enhancement, not required
        print(f"    whisper anchoring unavailable: {exc}", flush=True)

    print(f"    {len(anchors)} anchors from {len(hyp)} transcript words", flush=True)

    print("==> Refine (reconcile + polish)", flush=True)
    holds = detect_holds(emission, feats, hyp)
    words = reconcile(tokens, base, anchors, aligner, emission, feats, hyp, holds)

    # Second pass: re-transcribe stubborn regions with tight slices. Long
    # windows garble dense/doubled vocals that a focused slice hears cleanly.
    if whisper_ok:
        weak = weak_segments(tokens, words)
        if weak:
            model = get_whisper(whisper_model)
            local_anchors: list[dict] = []
            stable_words: list[dict] = []
            added = 0
            for run, a, b in weak:
                # a crammed run often sits next to a wrongly-stretched anchor
                # word holding the audio the run really belongs to - include it
                prev_k, next_k = run[0] - 1, run[-1] + 1
                if prev_k >= 0 and words[prev_k]["end"] - words[prev_k]["start"] > 1.5:
                    a = min(a, float(words[prev_k]["start"]))
                if next_k < len(words) and words[next_k]["end"] - words[next_k]["start"] > 1.5:
                    b = max(b, float(words[next_k]["end"]))
                b = min(b, a + 16.0)
                extra = transcribe_slice_stable(
                    model, y, max(0.0, a - 0.8), min(feats.duration, b + 1.2), feats.duration
                )
                if not extra:
                    continue
                added += len(extra)
                hyp.extend(extra)
                stable_words.extend(extra)
                # include neighboring tokens: their first-pass anchors may be
                # the misplaced ones that caused this weak segment
                lo_k = max(0, run[0] - 3)
                hi_k = min(len(tokens), run[-1] + 4)
                ctx = list(range(lo_k, hi_k))
                seg_tokens = [tokens[k] for k in ctx]
                seg_times = [float(words[k]["start"]) for k in ctx]
                seg_pairs = match_transcript(seg_tokens, extra, seg_times, pen_scale=4.5)
                seg_anchors = pick_anchors(seg_tokens, extra, seg_pairs, [base[k] for k in ctx])
                for sa in seg_anchors:
                    sa["tok"] += lo_k
                local_anchors.extend(seg_anchors)
                if debug_dir is not None:
                    print(
                        f"      weak {a:.1f}-{b:.1f}s: heard "
                        + " ".join(f"{w2['norm']}[{w2['start']:.2f}]" for w2 in extra)
                        + " | anchors "
                        + " ".join(
                            f"{tokens[sa['tok']]['word']}@{sa['t']:.2f}" for sa in seg_anchors
                        ),
                        flush=True,
                    )
            if added:
                hyp[:] = dedupe_hyp(hyp)
                anchors = merge_anchors(anchors, local_anchors)
                holds = detect_holds(emission, feats, hyp)
                print(
                    f"    second pass: {len(weak)} weak segments, +{added} words, "
                    f"+{len(local_anchors)} local anchors -> {len(anchors)}",
                    flush=True,
                )
                words = reconcile(tokens, base, anchors, aligner, emission, feats, hyp, holds)

    words = polish(tokens, words, feats)
    words = overlay_echoes(tokens, words, feats, base, aligner, emission, holds)

    # echo/pickup moves can strand voiced audio after a word whose hold was
    # computed against the old layout; re-extend ends only (starts are final)
    for k in range(len(words) - 1):
        cur, nxt = words[k], words[k + 1]
        gap = float(nxt["start"]) - float(cur["end"])
        if gap <= 0.02:
            continue
        hold_end = feats.last_voiced_before(float(cur["end"]), min(float(nxt["start"]), float(cur["end"]) + 2.0))
        if hold_end > float(cur["end"]):
            cur["end"] = round(min(hold_end + 0.05, float(nxt["start"]) - 0.01), 3)

    owned_all: list[tuple[float, float, str]] = []
    for chain in hold_chains(holds or [], 0.0, feats.duration, feats):
        near = sorted(
            {
                tokens[k]["norm"]
                for k in range(len(tokens))
                if float(words[k]["start"]) < chain[1] + 0.8
                and float(words[k]["end"]) > chain[0] - 0.8
            }
        )
        if near:
            owner = chain_owner(aligner, emission, chain, near)
            if owner is not None:
                owned_all.append((chain[0], chain[1], owner))
    words = harmonize_repeated_lines(
        tokens, words, feats, y, hyp, owned_all, debug=debug_dir is not None
    )

    by_tok = {tok["i"]: w for tok, w in zip(tokens, words)}
    out = []
    prev_end = 0.0
    for tok in tokens_all:
        w = by_tok.get(tok["i"])
        if w is None:
            w = {"start": prev_end, "end": prev_end, "score": 0.0, "source": "punct"}
        prev_end = float(w["end"])
        out.append(
            {
                "i": tok["i"],
                "word": tok["word"],
                "start": round(float(w["start"]), 3),
                "end": round(float(w["end"]), 3),
                "score": round(float(w["score"]), 3),
                "source": w.get("source", "ctc"),
            }
        )
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "anchors.json").write_text(json.dumps(anchors, indent=2))
        (debug_dir / "whisper_words.json").write_text(json.dumps(hyp, indent=2))
        base_dump = [
            {"i": t["i"], "word": t["word"], **(b or {})} for t, b in zip(tokens, base)
        ]
        (debug_dir / "ctc_base.json").write_text(json.dumps(base_dump, indent=2))
    return out


def main() -> int:
    import librosa

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--whisper-model", default="large-v3-turbo")
    ap.add_argument("--debug-dir", default=None)
    args = ap.parse_args()

    lines, _ = parse_lyrics(pathlib.Path(args.lyrics).read_text())
    if not lines:
        print(f"No lyric lines in {args.lyrics}", file=sys.stderr)
        return 1
    y, _ = librosa.load(args.audio, sr=SR, mono=True)
    debug_dir = pathlib.Path(args.debug_dir) if args.debug_dir else None
    words = align_song(
        y, lines, device=args.device, whisper_model=args.whisper_model, debug_dir=debug_dir
    )
    pathlib.Path(args.out).write_text(json.dumps(words, indent=2))
    print(json.dumps({"words": len(words), "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
