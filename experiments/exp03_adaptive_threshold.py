#!/usr/bin/env python3
"""
Experiment #3: Adaptive Threshold Showdown

Compare silence detection methods on real podcast audio:
1. Fixed threshold (pydub default)
2. Adaptive: Noise floor + margin
3. Adaptive: Mean - X std dev

Ground truth: manually labeled silence segments from franck_60s.wav
"""

import json
import time
import numpy as np
import librosa
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_silence

EXPERIMENT_ID = "exp03_adaptive_threshold"
AUDIO_PATH = "/home/agentic/podcast-editor-ai/samples/franck_60s.wav"

# Ground truth: best estimate from manual inspection
# These are the clearest silence gaps in the recording
GROUND_TRUTH = [
    {"start": 5.6, "end": 6.2},
    {"start": 13.2, "end": 13.9},
    {"start": 19.3, "end": 19.8},
    {"start": 24.6, "end": 25.1},
    {"start": 29.6, "end": 30.1},
]

def detect_fixed(audio, threshold_db=-45):
    """Fixed threshold method."""
    segments = detect_silence(audio, min_silence_len=500, silence_thresh=threshold_db)
    return [{"start": s/1000, "end": e/1000} for s, e in segments]

def detect_adaptive_floor(audio_path, margin_db=8):
    """Adaptive: noise floor (5th percentile) + margin."""
    audio = AudioSegment.from_file(audio_path)
    y, sr = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    noise_floor = np.percentile(rms_db, 5)
    threshold = noise_floor + margin_db
    
    segments = detect_silence(audio, min_silence_len=500, silence_thresh=threshold)
    return [{"start": s/1000, "end": e/1000} for s, e in segments], threshold

def detect_adaptive_std(audio_path, n_std=1.0):
    """Adaptive: mean - N standard deviations."""
    audio = AudioSegment.from_file(audio_path)
    y, sr = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    mean_db = rms_db.mean()
    std_db = rms_db.std()
    threshold = mean_db - (n_std * std_db)
    
    segments = detect_silence(audio, min_silence_len=500, silence_thresh=threshold)
    return [{"start": s/1000, "end": e/1000} for s, e in segments], threshold

def calc_f1(detected, ground_truth, tolerance=0.5):
    """Calculate F1 score against ground truth."""
    tp = 0
    for det in detected:
        for gt in ground_truth:
            if abs(det["start"] - gt["start"]) < tolerance:
                tp += 1
                break
    
    precision = tp / len(detected) if detected else 0
    recall = tp / len(ground_truth) if ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "tp": tp,
        "segments": len(detected)
    }

def run_experiment():
    print(f"🧪 Experiment: {EXPERIMENT_ID}")
    print("=" * 60)
    print(f"Audio: franck_60s.wav (real podcast)")
    print(f"Ground truth: {len(GROUND_TRUTH)} silence segments")
    print()
    
    audio = AudioSegment.from_file(AUDIO_PATH)
    results = {"experiment_id": EXPERIMENT_ID, "methods": {}}
    
    # Method 1: Fixed threshold
    print("📏 Method 1: Fixed threshold (-45dB)")
    start = time.time()
    fixed_det = detect_fixed(audio, -45)
    fixed_time = time.time() - start
    fixed_acc = calc_f1(fixed_det, GROUND_TRUTH)
    results["methods"]["fixed"] = {
        "type": "fixed",
        "threshold_db": -45,
        "time_sec": round(fixed_time, 3),
        "accuracy": fixed_acc
    }
    print(f"   Segments: {fixed_acc['segments']}, F1={fixed_acc['f1']}, Time={fixed_time:.3f}s")
    print()
    
    # Method 2: Adaptive noise floor
    print("📈 Method 2: Adaptive (noise floor + 8dB)")
    start = time.time()
    floor_det, floor_thresh = detect_adaptive_floor(AUDIO_PATH, 8)
    floor_time = time.time() - start
    floor_acc = calc_f1(floor_det, GROUND_TRUTH)
    results["methods"]["adaptive_floor"] = {
        "type": "adaptive_floor",
        "threshold_db": round(floor_thresh, 1),
        "time_sec": round(floor_time, 3),
        "accuracy": floor_acc
    }
    print(f"   Threshold: {floor_thresh:.1f}dB")
    print(f"   Segments: {floor_acc['segments']}, F1={floor_acc['f1']}, Time={floor_time:.3f}s")
    print()
    
    # Method 3: Adaptive std dev
    print("📉 Method 3: Adaptive (mean - 1.0 std)")
    start = time.time()
    std_det, std_thresh = detect_adaptive_std(AUDIO_PATH, 1.0)
    std_time = time.time() - start
    std_acc = calc_f1(std_det, GROUND_TRUTH)
    results["methods"]["adaptive_std"] = {
        "type": "adaptive_std",
        "threshold_db": round(std_thresh, 1),
        "time_sec": round(std_time, 3),
        "accuracy": std_acc
    }
    print(f"   Threshold: {std_thresh:.1f}dB")
    print(f"   Segments: {std_acc['segments']}, F1={std_acc['f1']}, Time={std_time:.3f}s")
    print()
    
    # Determine winner
    print("=" * 60)
    methods = [
        ("fixed", fixed_acc["f1"], fixed_time),
        ("adaptive_floor", floor_acc["f1"], floor_time),
        ("adaptive_std", std_acc["f1"], std_time)
    ]
    
    # Sort by F1 (desc), then by time (asc)
    methods.sort(key=lambda x: (-x[1], x[2]))
    winner = methods[0][0]
    
    winner_names = {
        "fixed": "FIXED THRESHOLD",
        "adaptive_floor": "ADAPTIVE (NOISE FLOOR)",
        "adaptive_std": "ADAPTIVE (STD DEV)"
    }
    
    print(f"🏆 Winner: {winner_names[winner]}")
    results["winner"] = winner
    
    # Save results
    Path("experiments/results").mkdir(parents=True, exist_ok=True)
    with open(f"experiments/results/{EXPERIMENT_ID}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved: experiments/results/{EXPERIMENT_ID}.json")
    
    return results

if __name__ == "__main__":
    run_experiment()
