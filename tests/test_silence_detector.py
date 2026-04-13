#!/usr/bin/env python3
"""
Test suite for silence detector.

Validates F1 score against ground truth labels.
"""

import sys
sys.path.insert(0, '.')

from audio.silence_detector import SilenceDetector

# Ground truth for franck_60s.wav
GROUND_TRUTH = [
    {"start": 5.6, "end": 6.2},
    {"start": 13.2, "end": 13.9},
    {"start": 19.3, "end": 19.8},
    {"start": 24.6, "end": 25.1},
    {"start": 29.6, "end": 30.1},
]

def calculate_f1(detected, ground_truth, tolerance=0.5):
    """Calculate F1 score for silence detection."""
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
        "fp": len(detected) - tp,
        "fn": len(ground_truth) - tp
    }

def test_franck_audio():
    """Test on franck_60s.wav sample."""
    detector = SilenceDetector()
    detected = detector.detect("samples/franck_60s.wav")
    
    metrics = calculate_f1(detected, GROUND_TRUTH)
    
    print("Test: franck_60s.wav")
    print("=" * 50)
    print(f"Detected: {len(detected)} segments")
    print(f"Ground truth: {len(GROUND_TRUTH)} segments")
    print(f"True positives: {metrics['tp']}")
    print(f"False positives: {metrics['fp']}")
    print(f"False negatives: {metrics['fn']}")
    print()
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1']}")
    print()
    
    # Expected: F1 >= 0.75 (we achieved 0.77 in experiments)
    if metrics["f1"] >= 0.75:
        print("✓ PASS: F1 >= 0.75")
        return True
    else:
        print("✗ FAIL: F1 < 0.75")
        return False

if __name__ == "__main__":
    success = test_franck_audio()
    sys.exit(0 if success else 1)
