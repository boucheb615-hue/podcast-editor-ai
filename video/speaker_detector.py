#!/usr/bin/env python3
"""
Speaker detection for auto-zoom.

Combines face detection with audio energy to identify active speaker.
Outputs zoom timeline for FFmpeg crop/zoom filters.
"""

import cv2
import numpy as np
import librosa
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FaceDetection:
    """Detected face in a frame."""
    frame_time: float
    center_x: int
    center_y: int
    width: int
    height: int
    confidence: float


@dataclass
class SpeakerSegment:
    """Active speaker segment."""
    start: float
    end: float
    speaker_x: int  # Face center X
    speaker_y: int  # Face center Y
    confidence: float


class SpeakerDetector:
    """Detect active speaker using face detection + audio correlation."""
    
    def __init__(
        self,
        min_face_size: int = 50,
        audio_window_sec: float = 1.0,
        zoom_margin: float = 0.2,  # 20% margin around face
    ):
        """
        Initialize speaker detector.
        
        Args:
            min_face_size: Minimum face size in pixels (default: 50)
            audio_window_sec: Audio analysis window in seconds (default: 1.0)
            zoom_margin: Margin around detected face for zoom (default: 0.2)
        """
        self.min_face_size = min_face_size
        self.audio_window_sec = audio_window_sec
        self.zoom_margin = zoom_margin
        
        # Load OpenCV face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_faces_in_video(self, video_path: str, frame_interval: float = 1.0) -> List[FaceDetection]:
        """
        Detect faces in video at regular intervals.
        
        Args:
            video_path: Path to video file
            frame_interval: Analyze every N seconds (default: 1.0)
        
        Returns:
            List of FaceDetection objects
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"Video: {width}x{height}, {fps:.1f}fps, {duration:.1f}s")
        print(f"Analyzing every {frame_interval}s ({int(fps * frame_interval)} frames)")
        
        faces = []
        frame_interval_int = int(fps * frame_interval)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval_int == 0:
                frame_time = frame_count / fps
                
                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(self.min_face_size, self.min_face_size)
                )
                
                for (x, y, fw, fh) in detected:
                    faces.append(FaceDetection(
                        frame_time=frame_time,
                        center_x=int(x + fw / 2),
                        center_y=int(y + fh / 2),
                        width=int(fw),
                        height=int(fh),
                        confidence=0.8  # OpenCV doesn't provide confidence
                    ))
            
            frame_count += 1
        
        cap.release()
        print(f"Detected {len(faces)} faces across {frame_count} frames")
        
        return faces
    
    def get_audio_energy(self, audio_path: str, window_sec: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Calculate audio energy per time window.
        
        Args:
            audio_path: Path to audio file (or video with audio)
            window_sec: Analysis window in seconds
        
        Returns:
            (energy_array, sample_rate)
        """
        y, sr = librosa.load(audio_path, sr=None)
        
        # Calculate RMS energy
        hop_length = int(sr * window_sec)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        return rms, sr
    
    def correlate_audio_face(self, faces: List[FaceDetection], audio_energy: np.ndarray, 
                            fps: float, frame_interval: float) -> List[SpeakerSegment]:
        """
        Correlate face positions with audio energy to find active speaker.
        
        Simple heuristic: when audio energy is high, the largest/closest face
        is likely the active speaker.
        
        Args:
            faces: List of detected faces
            audio_energy: Audio energy per window
            fps: Video FPS
            frame_interval: Frame sampling interval
        
        Returns:
            List of SpeakerSegment objects
        """
        if not faces:
            return []
        
        # Group faces by time window
        time_windows = {}
        for face in faces:
            window_idx = int(face.frame_time / frame_interval)
            if window_idx not in time_windows:
                time_windows[window_idx] = []
            time_windows[window_idx].append(face)
        
        # For each window with high audio, pick the dominant face
        segments = []
        current_speaker = None
        segment_start = None
        
        for i, energy in enumerate(audio_energy):
            if i in time_windows:
                window_faces = time_windows[i]
                # Pick largest face (closest to camera = likely speaker)
                dominant_face = max(window_faces, key=lambda f: f.width * f.height)
                
                if energy > np.median(audio_energy):  # Above median = speaking
                    if current_speaker is None:
                        current_speaker = dominant_face
                        segment_start = i * frame_interval
                    else:
                        # Check if same speaker (similar position)
                        dist = np.sqrt(
                            (dominant_face.center_x - current_speaker.center_x)**2 +
                            (dominant_face.center_y - current_speaker.center_y)**2
                        )
                        if dist < 100:  # Same speaker within 100px
                            current_speaker = dominant_face
                        else:
                            # Speaker changed, save segment
                            if segment_start is not None:
                                segments.append(SpeakerSegment(
                                    start=segment_start,
                                    end=i * frame_interval,
                                    speaker_x=current_speaker.center_x,
                                    speaker_y=current_speaker.center_y,
                                    confidence=0.8
                                ))
                            current_speaker = dominant_face
                            segment_start = i * frame_interval
                else:
                    # Low energy = silence, end segment
                    if current_speaker is not None and segment_start is not None:
                        segments.append(SpeakerSegment(
                            start=segment_start,
                            end=i * frame_interval,
                            speaker_x=current_speaker.center_x,
                            speaker_y=current_speaker.center_y,
                            confidence=0.8
                        ))
                    current_speaker = None
                    segment_start = None
        
        # Handle final segment
        if current_speaker is not None and segment_start is not None:
            segments.append(SpeakerSegment(
                start=segment_start,
                end=len(audio_energy) * frame_interval,
                speaker_x=current_speaker.center_x,
                speaker_y=current_speaker.center_y,
                confidence=0.8
            ))
        
        return segments
    
    def generate_zoom_timeline(self, segments: List[SpeakerSegment], 
                               video_width: int, video_height: int,
                               output_width: int = 1280, output_height: int = 720,
                               zoom_scale: float = 1.5) -> List[Dict]:
        """
        Generate zoom timeline for FFmpeg.
        
        Args:
            segments: Speaker segments
            video_width: Original video width
            video_height: Original video height
            output_width: Desired output width (zoomed)
            output_height: Desired output height
            zoom_scale: Zoom level (1.0 = no zoom, 2.0 = 2x zoom in)
        
        Returns:
            List of zoom commands for FFmpeg
        """
        timeline = []
        
        # Calculate crop dimensions based on zoom scale
        # Higher zoom = smaller crop region
        crop_width = int(video_width / zoom_scale)
        crop_height = int(video_height / zoom_scale)
        
        for seg in segments:
            # Calculate crop region centered on speaker
            # Ensure crop stays within video bounds
            x_start = seg.speaker_x - crop_width // 2
            y_start = seg.speaker_y - crop_height // 2
            
            # Clamp to valid range
            x_start = max(0, min(x_start, video_width - crop_width))
            y_start = max(0, min(y_start, video_height - crop_height))
            
            timeline.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "crop_x": x_start,
                "crop_y": y_start,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "speaker_x": seg.speaker_x,
                "speaker_y": seg.speaker_y
            })
        
        return timeline
    
    def save_zoom_timeline(self, timeline: List[Dict], output_path: str):
        """Save zoom timeline to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump({
                "zoom_segments": timeline,
                "segment_count": len(timeline),
                "total_zoom_duration": round(sum(s["end"] - s["start"] for s in timeline), 2)
            }, f, indent=2)
        
        return output_path


def main():
    """Test speaker detection on proxy video."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python speaker_detector.py <video_path> [output_json]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    detector = SpeakerDetector()
    
    print(f"Detecting speakers in: {video_path}")
    
    # Detect faces
    faces = detector.detect_faces_in_video(video_path, frame_interval=1.0)
    
    # Get audio energy
    audio_energy, sr = detector.get_audio_energy(video_path)
    
    # Correlate
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    segments = detector.correlate_audio_face(faces, audio_energy, fps, frame_interval=1.0)
    
    print(f"\nFound {len(segments)} speaker segments:")
    for i, seg in enumerate(segments[:10]):
        print(f"  {i+1}. {seg.start:.1f}s - {seg.end:.1f}s @ ({seg.speaker_x},{seg.speaker_y})")
    if len(segments) > 10:
        print(f"  ... and {len(segments)-10} more")
    
    if output_path:
        # Generate zoom timeline
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        timeline = detector.generate_zoom_timeline(segments, width, height)
        detector.save_zoom_timeline(timeline, output_path)
        print(f"\nSaved zoom timeline: {output_path}")


if __name__ == "__main__":
    main()
