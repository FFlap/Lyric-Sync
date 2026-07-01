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
    words.sort(key=lambda w: w["start"])
    return words


_SIM_CACHE: dict[tuple[str, str], float] = {}


def word_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    key = (a, b) if a <= b else (b, a)
    hit = _SIM_CACHE.get(key)
    if hit is None:
        hit = difflib.SequenceMatcher(None, key[0], key[1]).ratio()
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
            diag = score[i - 1][j - 1] + (sim - 0.55) - pen
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

def distribute_words(
    tokens: list[dict],
    t0: float,
    t1: float,
    feats: AudioFeatures,
    extra_onsets: list[float] | None = None,
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

    # candidate articulation events with salience
    env = feats.onset_env
    env_max = float(np.max(env)) + 1e-9

    def env_at(t: float) -> float:
        return float(env[feats.t2f(t)]) / env_max

    # candidates hugging the far pin belong to the next phrase, not this one
    hi_edge = t1 - 0.25 if t1 - 0.25 > t0 else t1
    cands: dict[float, float] = {}
    for t in feats.onsets:
        if t0 - 0.05 <= t <= hi_edge:
            cands[round(float(t), 3)] = 0.5 * env_at(float(t))
    for t in extra_onsets or []:
        if t0 - 0.05 <= t <= hi_edge:
            key = round(float(t), 3)
            cands[key] = max(cands.get(key, 0.0), 0.6)
    for p in priors:
        cands.setdefault(round(p, 3), 0.0)
    cand_times = sorted(cands)
    sal = [cands[t] for t in cand_times]
    m = len(cand_times)

    # DP: monotone assignment of words to candidate starts
    POS_W = 0.30  # per-second deviation from prior
    NEG = -1e9
    dp = np.full((n, m), NEG)
    back = np.zeros((n, m), dtype=np.int32)
    for j in range(m):
        dp[0][j] = sal[j] - POS_W * abs(cand_times[j] - priors[0])
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
            if best_j < 0:
                continue
            local = sal[j] - POS_W * abs(cand_times[j] - priors[k])
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
            out[ti] = {**b, "source": "ctc"}
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

        # weak evidence: distribute across voiced audio between pins,
        # snapping to articulations (whisper word onsets included even when
        # the transcript text didn't match the lyrics)
        extra = [w["start"] for w in hyp or [] if t_lo - 0.1 <= w["start"] <= t_hi + 0.1]
        dist = distribute_words([tokens[k] for k in seg], t_lo, t_hi, feats, extra)
        for k, w in zip(seg, dist):
            out[k] = w

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
        if len(run) >= 4 and b - a >= 2.0:
            out.append((run, a, b))
    return out


def merge_anchors(primary: list[dict], extra: list[dict]) -> list[dict]:
    """Combine anchor sets, keep one per token, enforce monotone times."""
    by_tok: dict[int, dict] = {}
    for a in primary + extra:
        cur = by_tok.get(a["tok"])
        if cur is None or a["sim"] > cur["sim"]:
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
        if tokens[k]["line"] != tokens[k + 1]["line"] and gap > 1.5:
            hold_end = feats.last_voiced_before(cur["end"], min(nxt["start"], cur["end"] + 4.0))
            if hold_end > cur["end"]:
                cur["end"] = round(min(hold_end + 0.05, nxt["start"] - 0.02), 3)
            continue
        if feats.is_voiced_between(cur["end"], nxt["start"]):
            cur["end"] = round(nxt["start"] - 0.01, 3)
        else:
            hold_end = feats.last_voiced_before(cur["end"], nxt["start"])
            if hold_end > cur["end"]:
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


def overlay_echoes(
    tokens: list[dict],
    words: list[dict],
    feats: AudioFeatures,
    base: list[dict] | None = None,
) -> list[dict]:
    """Echo/backing paren runs without their own audio get overlaid on the main phrase."""
    n = len(words)
    base = base or [None] * n
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
        )
        if has_evidence:
            continue

        # A real (non-overlapping) echo shows up as voiced audio between the
        # preceding word and the next word. If that audio exists, place the
        # echo there instead of overlaying it on the main phrase.
        prev_end = words[first - 1]["end"] if first > 0 else 0.0
        next_start = words[last + 1]["start"] if last + 1 < n else feats.duration
        gap_voiced = sum(
            b - a for a, b in feats.voiced_intervals(prev_end + 0.04, next_start - 0.02)
        )
        if gap_voiced >= 0.6 * expected:
            run_base = [base[k] for k in run]
            base_fits = (
                all(b is not None for b in run_base)
                and run_base[0]["start"] >= prev_end - 0.10
                and run_base[-1]["end"] <= next_start + 0.10
                and all(
                    run_base[j]["start"] >= run_base[j - 1]["end"] - 0.05
                    for j in range(1, len(run_base))
                )
            )
            if base_fits:
                for k, b in zip(run, run_base):
                    words[k] = {**b, "source": "ctc_echo"}
            else:
                dist = distribute_words(
                    [tokens[k] for k in run], prev_end + 0.04, next_start - 0.02, feats
                )
                for k, w in zip(run, dist):
                    words[k] = {**w, "source": "distributed_echo"}
            continue

        twin = find_twin(tokens, run)
        if not twin:
            continue
        main_start = words[twin[0]]["start"]
        main_end = words[twin[-1]]["end"]
        next_start = words[last + 1]["start"] if last + 1 < n else feats.duration
        # echoes trail the main phrase slightly; keep room before the next word
        lag = min(0.25, max(0.08, (next_start - main_end) * 0.4))
        avail_end = max(main_end + lag, min(main_end + lag + 0.2, next_start - 0.02))
        scale_src = main_end - main_start
        scale_dst = max(0.3, avail_end - (main_start + lag))
        ratio = scale_dst / max(scale_src, 0.05)
        for k, m in zip(run, twin):
            ms, me = words[m]["start"], words[m]["end"]
            words[k] = {
                "start": round(main_start + lag + (ms - main_start) * ratio, 3),
                "end": round(main_start + lag + (me - main_start) * ratio, 3),
                "score": words[m]["score"],
                "source": "echo_overlay",
            }
        # keep global monotonicity with the previous non-run word untouched:
        # echoes may overlap preceding words by design.
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
    words = reconcile(tokens, base, anchors, aligner, emission, feats, hyp)

    # Second pass: re-transcribe stubborn regions with tight slices. Long
    # windows garble dense/doubled vocals that a focused slice hears cleanly.
    if whisper_ok:
        weak = weak_segments(tokens, words)
        if weak:
            model = get_whisper(whisper_model)
            local_anchors: list[dict] = []
            added = 0
            for run, a, b in weak:
                extra = transcribe_slice(model, y, max(0.0, a - 0.5), min(feats.duration, b + 0.5))
                if not extra:
                    continue
                added += len(extra)
                hyp.extend(extra)
                seg_tokens = [tokens[k] for k in run]
                seg_times = [float(words[k]["start"]) for k in run]
                seg_pairs = match_transcript(seg_tokens, extra, seg_times, pen_scale=4.5)
                seg_anchors = pick_anchors(seg_tokens, extra, seg_pairs, [base[k] for k in run])
                for sa in seg_anchors:
                    sa["tok"] += run[0]
                local_anchors.extend(seg_anchors)
            if local_anchors:
                hyp.sort(key=lambda w: w["start"])
                anchors = merge_anchors(anchors, local_anchors)
                print(
                    f"    second pass: {len(weak)} weak segments, +{added} words, "
                    f"+{len(local_anchors)} local anchors -> {len(anchors)}",
                    flush=True,
                )
                words = reconcile(tokens, base, anchors, aligner, emission, feats, hyp)

    words = polish(tokens, words, feats)
    words = overlay_echoes(tokens, words, feats, base)

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
