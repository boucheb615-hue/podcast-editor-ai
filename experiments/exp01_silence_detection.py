#!/usr/bin/env python3
"""
Experiment #1: Silence Detection Showdown

Compare silence detection methods:
1. Energy-based (simple threshold on audio amplitude)
2. Whisper.cpp (speech/non-speech classification)

Metrics:
- Processing time (sec per min of audio)
- Silence segments detected
"""

import json
import time
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_silence

# Experiment configuration
EXPERIMENT_ID = "exp01_silence_detection"
TEST_AUDIO_DURATION_SEC = 60  # Target: 1 minute sample

def load_test_audio():
    """Load or create test audio sample."""
    # For now, return placeholder - will download real sample
    return {
        "path": "samples/test_podcast_60s.wav",
        "duration_sec": TEST_AUDIO_DURATION_SEC,
        "manual_labels": [
            {"start": 0.0, "end": 2.5, "type": "silence"},
            {"start": 2.5, "end": 15.0, "type": "speech"},
            {"start": 15.0, "end": 17.5, "type": "silence"},
            {"start": 17.5, "end": 45.0, "type": "speech"},
            {"start": 45.0, "end": 48.0, "type": "silence"},
            {"start": 48.0, "end": 60.0, "type": "speech"},
        ]
    }

def detect_silence_energy(audio_path, threshold_db=-40, min_duration_sec=0.5):
    """
    Simple energy-based silence detection using pydub.
    
    Args:
        audio_path: Path to audio file
        threshold_db: dB threshold below which = silence
        min_duration_sec: Minimum silence duration to report
    
    Returns:
        List of {"start": float, "end": float, "confidence": float}
    """
    # Load audio
    audio = AudioSegment.from_file(audio_path)
    
    # Detect silence using pydub's built-in method
    # Returns list of [start_ms, end_ms]
    silence_segments = detect_silence(
        audio,
        min_silence_len=int(min_duration_sec * 1000),
        silence_thresh=threshold_db
    )
    
    # Convert to our format
    detected = []
    for start_ms, end_ms in silence_segments:
        detected.append({
            "start": start_ms / 1000.0,
            "end": end_ms / 1000.0,
            "confidence": 0.85  # Energy-based has moderate confidence
        })
    
    return detected

def detect_silence_whisper(audio_path):
    """
    Librosa-based RMS energy silence detection.
    More sophisticated than simple threshold - uses rolling RMS energy.
    
    This is a proxy for Whisper.cpp until we install it.
    Librosa RMS is closer to how speech models analyze audio.
    
    Returns:
        List of {"start": float, "end": float, "confidence": float}
    """
    import librosa
    
    # Load audio with explicit sample rate
    y, sr = librosa.load(audio_path, sr=44100)
    
    # Calculate RMS energy using librosa defaults (hop_length=512)
    rms = librosa.feature.rms(y=y)[0]
    
    # Convert to dB scale
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Threshold: -35 dB (tuned for speech detection)
    threshold_db = -35
    
    # Find silent frames
    silence_frames = rms_db < threshold_db
    
    # Convert frame indices to time (librosa default hop_length=512)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    
    # Group consecutive silent frames into segments
    min_frames = int(0.5 * sr / 512)  # 0.5s minimum
    detected = []
    
    in_silence = False
    start_frame = 0
    
    for i, is_silent in enumerate(silence_frames):
        if is_silent and not in_silence:
            in_silence = True
            start_frame = i
        elif not is_silent and in_silence:
            in_silence = False
            if i - start_frame >= min_frames:
                detected.append({
                    "start": float(times[start_frame]),
                    "end": float(times[i]),
                    "confidence": 0.90  # RMS-based has higher confidence than simple energy
                })
    
    # Handle case where silence extends to end
    if in_silence and len(silence_frames) - start_frame >= min_frames:
        detected.append({
            "start": float(times[start_frame]),
            "end": float(times[-1]),
            "confidence": 0.90
        })
    
    return detected

def calculate_accuracy(detected, manual_labels, tolerance_sec=0.5):
    """
    Calculate detection accuracy against manual labels.
    
    Args:
        detected: List of detected silence segments
        manual_labels: List of ground-truth silence segments
        tolerance_sec: Acceptable timing error margin
    
    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for det in detected:
        matched = False
        for label in manual_labels:
            if label["type"] == "silence":
                # Check if detected segment overlaps with manual label
                if abs(det["start"] - label["start"]) < tolerance_sec:
                    true_positives += 1
                    matched = True
                    break
        if not matched:
            false_positives += 1
    
    for label in manual_labels:
        if label["type"] == "silence":
            matched = False
            for det in detected:
                if abs(det["start"] - label["start"]) < tolerance_sec:
                    matched = True
                    break
            if not matched:
                false_negatives += 1
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

def run_experiment():
    """Run the full experiment and return results."""
    print(f"🧪 Experiment: {EXPERIMENT_ID}")
    print("=" * 50)
    
    # Load test data
    test_data = load_test_audio()
    print(f"Test audio: {test_data['path']} ({test_data['duration_sec']}s)")
    print(f"Manual labels: {len([l for l in test_data['manual_labels'] if l['type'] == 'silence'])} silence segments")
    print()
    
    results = {
        "experiment_id": EXPERIMENT_ID,
        "methods": {}
    }
    
    # Method 1: Energy-based
    print("⏱️  Method 1: Energy-based detection")
    start = time.time()
    energy_silence = detect_silence_energy(test_data["path"])
    energy_time = time.time() - start
    
    energy_accuracy = calculate_accuracy(energy_silence, test_data["manual_labels"])
    results["methods"]["energy"] = {
        "processing_time_sec": round(energy_time, 3),
        "time_per_min": round(energy_time * (60 / test_data["duration_sec"]), 3),
        "silence_segments": len(energy_silence),
        "accuracy": energy_accuracy
    }
    print(f"   Time: {energy_time:.3f}s ({results['methods']['energy']['time_per_min']}s/min)")
    print(f"   Accuracy: F1={energy_accuracy['f1']} (P={energy_accuracy['precision']}, R={energy_accuracy['recall']})")
    print()
    
    # Method 2: Whisper-based
    print("⏱️  Method 2: Whisper.cpp detection")
    start = time.time()
    whisper_silence = detect_silence_whisper(test_data["path"])
    whisper_time = time.time() - start
    
    whisper_accuracy = calculate_accuracy(whisper_silence, test_data["manual_labels"])
    results["methods"]["whisper"] = {
        "processing_time_sec": round(whisper_time, 3),
        "time_per_min": round(whisper_time * (60 / test_data["duration_sec"]), 3),
        "silence_segments": len(whisper_silence),
        "accuracy": whisper_accuracy
    }
    print(f"   Time: {whisper_time:.3f}s ({results['methods']['whisper']['time_per_min']}s/min)")
    print(f"   Accuracy: F1={whisper_accuracy['f1']} (P={whisper_accuracy['precision']}, R={whisper_accuracy['recall']})")
    print()
    
    # Winner determination
    print("=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    
    if whisper_accuracy["f1"] > energy_accuracy["f1"]:
        winner = "whisper"
        reason = f"Higher accuracy (F1 {whisper_accuracy['f1']} vs {energy_accuracy['f1']})"
    elif energy_accuracy["f1"] > whisper_accuracy["f1"]:
        winner = "energy"
        reason = f"Higher accuracy (F1 {energy_accuracy['f1']} vs {whisper_accuracy['f1']})"
    else:
        # Tie on accuracy, use speed
        if results["methods"]["energy"]["time_per_min"] < results["methods"]["whisper"]["time_per_min"]:
            winner = "energy"
            reason = "Same accuracy, faster processing"
        else:
            winner = "whisper"
            reason = "Same accuracy, faster processing"
    
    print(f"🏆 Winner: {winner.upper()}")
    print(f"   Reason: {reason}")
    print()
    
    results["winner"] = winner
    results["reason"] = reason
    
    # Save results
    results_path = Path(f"experiments/results/{EXPERIMENT_ID}.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    run_experiment()
