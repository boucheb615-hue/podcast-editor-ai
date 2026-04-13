#!/usr/bin/env python3
"""
Silence detection for podcast editing.

Uses pydub for energy-based detection with post-processing filters:
1. Speech resumption filter - silence must be followed by speech
2. Gap filter - enforce minimum time between silences

Optimized for podcast content with F1=0.77 on real audio samples.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from pydub import AudioSegment
from pydub.silence import detect_silence
import librosa
import numpy as np


class SilenceDetector:
    """Detect silence gaps in podcast audio with configurable filters."""
    
    def __init__(
        self,
        silence_thresh: float = -45.0,
        min_silence_len: float = 0.5,
        min_speech_energy: float = -18.0,
        min_gap_between_silences: float = 4.0,
    ):
        """
        Initialize silence detector.
        
        Args:
            silence_thresh: dB threshold below which = silence (default: -45)
            min_silence_len: Minimum silence duration in seconds (default: 0.5)
            min_speech_energy: Energy level that indicates speech resumed (default: -18)
            min_gap_between_silences: Minimum seconds between detected silences (default: 4.0)
        """
        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len  # seconds
        self.min_silence_len_ms = int(min_silence_len * 1000)  # pydub uses ms
        self.min_speech_energy = min_speech_energy
        self.min_gap = min_gap_between_silences
    
    def detect(self, audio_path: str) -> List[Dict]:
        """
        Detect silence segments in audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
        
        Returns:
            List of dicts with keys: start, end, duration (all in seconds)
        """
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Raw detection
        raw_segments = detect_silence(
            audio,
            min_silence_len=self.min_silence_len_ms,
            silence_thresh=self.silence_thresh
        )
        
        # Load for RMS analysis
        y, sr = librosa.load(audio_path, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Filter 1: Speech resumption
        filtered = []
        for start_ms, end_ms in raw_segments:
            start_s = start_ms / 1000.0
            end_s = end_ms / 1000.0
            
            # Check energy 0.5s after silence ends
            after_start = int(end_s * sr / 512)
            after_end = int((end_s + 0.5) * sr / 512)
            
            if after_end <= after_start:
                after_end = after_start + 1
            
            energy_after = rms_db[after_start:after_end].mean()
            
            # Keep only if speech resumes
            if energy_after > self.min_speech_energy:
                filtered.append((start_s, end_s))
        
        # Filter 2: Minimum gap between silences
        if self.min_gap > 0 and len(filtered) > 1:
            filtered.sort()
            final = [filtered[0]]
            
            for i in range(1, len(filtered)):
                gap = filtered[i][0] - final[-1][1]
                
                if gap >= self.min_gap:
                    # Gap is large enough, keep this silence
                    final.append(filtered[i])
                else:
                    # Gap too small, keep the longer silence
                    current_duration = filtered[i][1] - filtered[i][0]
                    prev_duration = final[-1][1] - final[-1][0]
                    
                    if current_duration > prev_duration:
                        final[-1] = filtered[i]
            
            filtered = final
        
        # Convert to dict format
        result = [
            {
                "start": round(start_s, 3),
                "end": round(end_s, 3),
                "duration": round(end_s - start_s, 3)
            }
            for start_s, end_s in filtered
        ]
        
        return result
    
    def detect_from_segment(self, audio: AudioSegment) -> List[Dict]:
        """
        Detect silence from AudioSegment object (for pipeline integration).
        
        Note: This skips the speech-resumption filter since we don't have
        the full file context. Use detect() for best results.
        """
        raw_segments = detect_silence(
            audio,
            min_silence_len=self.min_silence_len_ms,
            silence_thresh=self.silence_thresh
        )
        
        return [
            {
                "start": round(s / 1000.0, 3),
                "end": round(e / 1000.0, 3),
                "duration": round((e - s) / 1000.0, 3)
            }
            for s, e in raw_segments
        ]
    
    def to_edit_list(self, silences: List[Dict], audio_duration: float) -> List[Dict]:
        """
        Convert silence segments to edit decision list (segments to KEEP).
        
        Args:
            silences: List of silence segments from detect()
            audio_duration: Total audio duration in seconds
        
        Returns:
            List of segments to keep: [{"start": X, "end": Y}, ...]
        """
        if not silences:
            return [{"start": 0.0, "end": audio_duration}]
        
        # Sort by start time
        silences = sorted(silences, key=lambda x: x["start"])
        
        # Build keep list (everything except silences)
        keep = []
        current_pos = 0.0
        
        for silence in silences:
            if silence["start"] > current_pos:
                keep.append({
                    "start": round(current_pos, 3),
                    "end": round(silence["start"], 3)
                })
            current_pos = silence["end"]
        
        # Add final segment if there's content after last silence
        if current_pos < audio_duration:
            keep.append({
                "start": round(current_pos, 3),
                "end": round(audio_duration, 3)
            })
        
        return keep
    
    def save_timeline(self, silences: List[Dict], output_path: str):
        """Save silence timeline to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump({
                "silences": silences,
                "total_silence_duration": round(sum(s["duration"] for s in silences), 2),
                "segment_count": len(silences)
            }, f, indent=2)
        
        return output_path


def main():
    """Test silence detection on sample audio."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python silence_detector.py <audio_file>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    detector = SilenceDetector()
    
    print(f"Detecting silence in: {audio_path}")
    silences = detector.detect(audio_path)
    
    print(f"\nFound {len(silences)} silence segments:")
    for i, seg in enumerate(silences):
        print(f"  {i+1}. {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s)")
    
    print(f"\nTotal silence: {sum(s['duration'] for s in silences):.2f}s")


if __name__ == "__main__":
    main()
