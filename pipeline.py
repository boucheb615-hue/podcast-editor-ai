#!/usr/bin/env python3
"""
Podcast Editor AI - Main Pipeline

End-to-end workflow:
1. Generate proxy (720p) for fast processing
2. Detect silence → cut list
3. Detect speakers → zoom timeline
4. Combine → edit decision list (EDL)
5. Apply EDL to original → HD output (1080p)

Usage:
    python pipeline.py <input_video> [output_video]
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Optional, Tuple

# Import our modules
from audio.silence_detector import SilenceDetector
from video.speaker_detector import SpeakerDetector
from video.proxy_video import ProxyGenerator, HDExporter
from video.zoom_filter import SilenceZoomPipeline


class PodcastEditorPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(
        self,
        proxy_resolution: Tuple[int, int] = (1280, 720),
        output_resolution: Tuple[int, int] = (1920, 1080),
        work_dir: Optional[str] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            proxy_resolution: Proxy video resolution (default: 1280x720)
            output_resolution: Final output resolution (default: 1920x1080)
            work_dir: Working directory for temp files (default: ./output/)
        """
        self.proxy_width, self.proxy_height = proxy_resolution
        self.output_width, self.output_height = output_resolution
        self.work_dir = Path(work_dir) if work_dir else Path("output")
        
        # Initialize components
        self.proxy_gen = ProxyGenerator(
            proxy_width=self.proxy_width,
            proxy_height=self.proxy_height,
        )
        
        self.silence_detector = SilenceDetector(
            silence_thresh=-45.0,
            min_silence_len=0.5,
            min_speech_energy=-18.0,
            min_gap_between_silences=4.0,
        )
        
        self.speaker_detector = SpeakerDetector(
            min_face_size=50,
            audio_window_sec=1.0,
            zoom_margin=0.2,
        )
        
        self.zoom_pipeline = SilenceZoomPipeline(
            output_width=self.output_width,
            output_height=self.output_height,
        )
        
        self.hd_exporter = HDExporter(
            output_width=self.output_width,
            output_height=self.output_height,
        )
    
    def run(self, input_path: str, output_path: Optional[str] = None,
            skip_proxy: bool = False, keep_intermediates: bool = False) -> str:
        """
        Run full pipeline on input video.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video (default: <input>_edited.mp4)
            skip_proxy: Skip proxy generation, work on original (slow!)
            keep_intermediates: Keep temp files (for debugging)
        
        Returns:
            Path to final output video
        """
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_edited.mp4"
        else:
            output_path = Path(output_path)
        
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Podcast Editor AI Pipeline")
        print("=" * 60)
        print(f"Input: {input_path.name}")
        print(f"Output: {output_path.name}")
        print(f"Proxy resolution: {self.proxy_width}x{self.proxy_height}")
        print(f"Output resolution: {self.output_width}x{self.output_height}")
        print()
        
        # Step 1: Generate proxy (or use original)
        if skip_proxy:
            print("⏭️  Skipping proxy generation (using original)")
            proxy_path = input_path
        else:
            print("📹 Step 1/5: Generating proxy...")
            proxy_path = self.work_dir / f"{input_path.stem}_proxy.mp4"
            self.proxy_gen.generate_proxy(str(input_path), str(proxy_path))
            print(f"   Proxy: {proxy_path.name}")
        print()
        
        # Step 2: Detect silence
        print("🔇 Step 2/5: Detecting silence...")
        silence_segments = self.silence_detector.detect(str(proxy_path))
        print(f"   Found {len(silence_segments)} silence segments")
        print(f"   Total silence: {sum(s['duration'] for s in silence_segments):.1f}s")
        
        # Save silence timeline
        silence_timeline_path = self.work_dir / "silence_timeline.json"
        self.silence_detector.save_timeline(silence_segments, str(silence_timeline_path))
        
        # Convert to "keep" segments (everything except silence)
        proxy_info = self.proxy_gen.get_video_info(str(proxy_path))
        keep_segments = self.silence_detector.to_edit_list(
            silence_segments, proxy_info["duration"]
        )
        print(f"   Content segments to keep: {len(keep_segments)}")
        print()
        
        # Step 3: Detect speakers
        print("👤 Step 3/5: Detecting speakers...")
        faces = self.speaker_detector.detect_faces_in_video(
            str(proxy_path), frame_interval=1.0
        )
        
        audio_energy, sr = self.speaker_detector.get_audio_energy(str(proxy_path))
        
        # Get FPS
        cap = __import__('cv2').VideoCapture(str(proxy_path))
        fps = cap.get(__import__('cv2').CAP_PROP_FPS)
        cap.release()
        
        speaker_segments = self.speaker_detector.correlate_audio_face(
            faces, audio_energy, fps, frame_interval=1.0
        )
        print(f"   Found {len(speaker_segments)} speaker segments")
        
        # Generate zoom timeline (1.5x zoom for speaker focus)
        zoom_timeline = self.speaker_detector.generate_zoom_timeline(
            speaker_segments,
            proxy_info["width"],
            proxy_info["height"],
            zoom_scale=1.5
        )
        
        zoom_timeline_path = self.work_dir / "zoom_timeline.json"
        self.speaker_detector.save_zoom_timeline(zoom_timeline, str(zoom_timeline_path))
        print()
        
        # Step 4: Combine into EDL
        print("📝 Step 4/5: Building edit decision list...")
        
        # Convert zoom timeline to dict format
        zoom_dicts = [
            {
                "start": z["start"],
                "end": z["end"],
                "crop_x": z["crop_x"],
                "crop_y": z["crop_y"],
                "crop_width": z["crop_width"],
                "crop_height": z["crop_height"],
            }
            for z in zoom_timeline
        ]
        
        edl = self.zoom_pipeline.build_edl(keep_segments, zoom_dicts)
        print(f"   EDL segments: {len(edl)}")
        
        edl_path = self.work_dir / "edit_decision_list.json"
        self.zoom_pipeline.save_edl(edl, str(edl_path))
        print()
        
        # Step 5: Export final video
        print("🎬 Step 5/5: Exporting HD video...")
        
        # Generate FFmpeg command
        cmd = self.zoom_pipeline.generate_ffmpeg_command(
            str(input_path),  # Use original for best quality
            edl,
            str(output_path)
        )
        
        print(f"   Running FFmpeg...")
        print(f"   Command: ffmpeg {' '.join(cmd[2:])}")
        print()
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Export failed:")
            print(result.stderr)
            raise RuntimeError("FFmpeg export failed")
        
        print(f"✅ Export complete!")
        
        # Verify output
        output_info = self.proxy_gen.get_video_info(str(output_path))
        print(f"   Output: {output_info['width']}x{output_info['height']}, {output_info['duration']:.1f}s")
        print(f"   File: {output_path}")
        print()
        
        # Cleanup
        if not keep_intermediates and not skip_proxy:
            print("🧹 Cleaning up intermediates...")
            proxy_path.unlink(missing_ok=True)
            print(f"   Removed: {proxy_path.name}")
        
        print()
        print("=" * 60)
        print("Pipeline complete!")
        print("=" * 60)
        
        return str(output_path)


def main():
    """Run pipeline from command line."""
    if len(sys.argv) < 2:
        print("Podcast Editor AI - Automatic podcast video editing")
        print()
        print("Usage:")
        print("  python pipeline.py <input_video> [output_video]")
        print()
        print("Options:")
        print("  --skip-proxy     Skip proxy generation (use original)")
        print("  --keep           Keep intermediate files")
        print("  --help           Show this help")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Parse options
    skip_proxy = "--skip-proxy" in sys.argv
    keep = "--keep" in sys.argv
    
    # Run pipeline
    pipeline = PodcastEditorPipeline()
    output = pipeline.run(input_path, output_path, skip_proxy=skip_proxy, keep_intermediates=keep)
    
    print(f"\nFinal output: {output}")


if __name__ == "__main__":
    main()
