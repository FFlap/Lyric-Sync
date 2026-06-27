#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import re

import librosa
import numpy as np
import soundfile as sf

VOWELS = set('aeiouy')


def load_words(path):
    return json.loads(pathlib.Path(path).read_text())


def norm(word):
    return re.sub(r"[^a-z0-9']+", '', word.lower())


def word_weight(word):
    n = norm(word)
    if not n:
        return 0.7
    if len(n) <= 1:
        return 0.5
    if len(n) <= 2:
        return 0.65
    syll = max(1, sum(1 for c in n if c in VOWELS))
    return min(2.4, 0.75 + 0.22 * len(n) + 0.18 * syll)


def candidates_from_audio(audio_path, hop=0.01):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    frame = int(0.04 * sr)
    hop_len = int(hop * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop_len, center=True)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    rms_s = np.convolve(rms_db, np.ones(5) / 5, mode='same')
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_len, aggregate=np.median)
    flux = flux / (np.max(flux) + 1e-9)
    # Boundary evidence: energy transitions and spectral/onset flux.
    grad = np.abs(np.gradient(rms_s))
    grad = grad / (np.max(grad) + 1e-9)
    score = 0.55 * flux + 0.45 * grad
    times = librosa.frames_to_time(np.arange(len(score)), sr=sr, hop_length=hop_len)
    return times, score, rms_s


def best_boundary(times, score, lo, hi, fallback, min_score=0.18):
    lo_i = int(np.searchsorted(times, lo, side='left'))
    hi_i = int(np.searchsorted(times, hi, side='right'))
    if hi_i <= lo_i:
        return fallback, 0.0, False
    local = score[lo_i:hi_i]
    j = int(np.argmax(local))
    cand = float(times[lo_i + j])
    strength = float(local[j])
    return (cand, strength, strength >= min_score)


def refine(words, times, score):
    refined = [dict(w) for w in words]
    changes = []
    n = len(refined)
    for i in range(1, n):
        prev = refined[i - 1]
        cur = refined[i]
        gap_start = float(prev['end'])
        gap_end = float(cur['start'])
        midpoint = (gap_start + gap_end) / 2
        # Search around the CTC boundary. Wider search for large gaps, bounded to avoid line jumps.
        radius = min(0.28, max(0.08, abs(gap_end - gap_start) / 2 + 0.06))
        lo = max(float(prev['start']) + 0.025, midpoint - radius)
        hi = min(float(cur['end']) - 0.025, midpoint + radius)
        cand, strength, ok = best_boundary(times, score, lo, hi, lo + 0.30 * (hi - lo))
        if not ok:
            cand = midpoint
        min_prev = max(0.035, 0.035 * word_weight(prev['word']))
        min_cur = max(0.035, 0.035 * word_weight(cur['word']))
        cand = max(float(prev['start']) + min_prev, min(float(cur['end']) - min_cur, cand))
        old_prev_end = float(prev['end'])
        old_cur_start = float(cur['start'])
        # Only split silence/gap or overlap boundaries. Avoid shrinking clean separated words too much.
        if abs(cand - old_prev_end) > 0.012 or abs(cand - old_cur_start) > 0.012:
            prev['end'] = round(cand, 3)
            cur['start'] = round(cand, 3)
            changes.append({
                'between': [prev['i'], cur['i']],
                'prev_word': prev['word'],
                'word': cur['word'],
                'old_prev_end': round(old_prev_end, 3),
                'old_cur_start': round(old_cur_start, 3),
                'new_boundary': round(cand, 3),
                'acoustic_strength': round(strength, 3),
                'used_acoustic_peak': ok,
            })
    # Repair any accidental duration issues and preserve monotonic order.
    for i, w in enumerate(refined):
        if i and w['start'] < refined[i - 1]['end']:
            w['start'] = refined[i - 1]['end']
        if w['end'] < w['start'] + 0.025:
            w['end'] = round(w['start'] + 0.025, 3)
        w['start'] = round(float(w['start']), 3)
        w['end'] = round(float(w['end']), 3)
    return refined, changes


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['i', 'word', 'start', 'end', 'score'])
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--words', default='work/align/whisperx_full.json')
    ap.add_argument('--audio', default='stems/htdemucs/source/vocals.wav')
    ap.add_argument('--out', default='work/acoustic_refined.json')
    args = ap.parse_args()
    words = load_words(args.words)
    times, score, rms = candidates_from_audio(args.audio)
    refined, changes = refine(words, times, score)
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(refined, indent=2))
    write_csv(out.with_suffix('.csv'), refined)
    out.with_name(out.stem + '_changes.json').write_text(json.dumps(changes, indent=2))
    print(json.dumps({
        'words': len(refined),
        'changes': len(changes),
        'first': refined[0],
        'last': refined[-1],
        'hard': [w for w in refined if 'open' in w['word'].lower() or 'own' in w['word'].lower()],
    }, indent=2))

if __name__ == '__main__':
    main()
