#!/usr/bin/env python3
"""
FFmpeg dynamic zoom filter generator.

Creates filter_complex strings for pan-and-scan zoom effects
based on speaker detection timeline.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class ZoomSegment:
    """A segment with zoom parameters."""
    start: float
    end: float
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int


class ZoomFilterGenerator:
    """Generate FFmpeg filter_complex for dynamic zoom."""
    
    def __init__(
        self,
        output_width: int = 1920,
        output_height: int = 1080,
        transition_duration: float = 0.5,
    ):
        """
        Initialize zoom filter generator.
        
        Args:
            output_width: Output video width (default: 1920)
            output_height: Output video height (default: 1080)
            transition_duration: Crossfade duration between zooms (default: 0.5s)
        """
        self.output_width = output_width
        self.output_height = output_height
        self.transition_duration = transition_duration
    
    def load_timeline(self, timeline_path: str) -> List[ZoomSegment]:
        """Load zoom timeline from JSON file."""
        with open(timeline_path, "r") as f:
            data = json.load(f)
        
        segments = []
        for seg in data.get("zoom_segments", []):
            segments.append(ZoomSegment(
                start=seg["start"],
                end=seg["end"],
                crop_x=seg["crop_x"],
                crop_y=seg["crop_y"],
                crop_width=seg["crop_width"],
                crop_height=seg["crop_height"]
            ))
        
        return segments
    
    def generate_filter_complex(self, timeline: List[ZoomSegment], 
                                audio_duration: float) -> str:
        """
        Generate FFmpeg filter_complex string for dynamic zoom.
        
        Args:
            timeline: List of ZoomSegment objects
            audio_duration: Total audio/video duration
        
        Returns:
            FFmpeg filter_complex string
        """
        if not timeline:
            # No zoom, just scale to output size
            return f"scale={self.output_width}:{self.output_height}:force_original_aspect_ratio=decrease"
        
        filters = []
        
        # For each zoom segment, create a crop + scale filter
        # Then concatenate them with crossfade transitions
        for i, seg in enumerate(timeline):
            # Calculate crop parameters
            # FFmpeg crop filter: crop=w:h:x:y
            crop_filter = f"[0:v]crop={seg.crop_width}:{seg.crop_height}:{seg.crop_x}:{seg.crop_y}"
            
            # Scale to output size
            scale_filter = f"scale={self.output_width}:{self.output_height}"
            
            # Set PTS (presentation timestamp) for proper timing
            pts_filter = f"setpts=PTS-STARTPTS"
            
            # Combine filters for this segment
            segment_filter = f"{crop_filter},{scale_filter},{pts_filter}[v{i}]"
            filters.append(segment_filter)
        
        # Concatenate all video segments
        # Note: This is simplified - proper implementation needs trim filters
        # to extract exact time segments before applying zoom
        
        # For now, return a simpler version that applies zoom to entire video
        # with smooth transitions between speaker positions
        return self._generate_smooth_zoom(timeline)
    
    def _generate_smooth_zoom(self, timeline: List[ZoomSegment]) -> str:
        """
        Generate smooth pan-and-scan zoom that follows speaker.
        
        Uses FFmpeg's zoompan filter with dynamic parameters.
        """
        if not timeline:
            return f"scale={self.output_width}:{self.output_height}:force_original_aspect_ratio=decrease"
        
        # Calculate average zoom position (simplified approach)
        # For production, we'd use expression-based dynamic positioning
        
        avg_x = sum(seg.crop_x for seg in timeline) / len(timeline)
        avg_y = sum(seg.crop_y for seg in timeline) / len(timeline)
        avg_width = sum(seg.crop_width for seg in timeline) / len(timeline)
        avg_height = sum(seg.crop_height for seg in timeline) / len(timeline)
        
        # Use crop + scale for static zoom on average position
        # This is a simplified version - full dynamic zoom requires
        # segment-by-segment processing with trim/concat
        
        filter_str = f"crop={int(avg_width)}:{int(avg_height)}:{int(avg_x)}:{int(avg_y)},scale={self.output_width}:{self.output_height}"
        
        return filter_str
    
    def generate_segment_filters(self, timeline: List[ZoomSegment]) -> List[Dict]:
        """
        Generate per-segment FFmpeg commands for precise zoom.
        
        This creates separate filter chains for each speaker segment,
        which are then concatenated.
        
        Args:
            timeline: List of ZoomSegment objects
        
        Returns:
            List of segment configs for FFmpeg
        """
        segments = []
        
        for i, seg in enumerate(timeline):
            segments.append({
                "index": i,
                "start": seg.start,
                "end": seg.end,
                "duration": seg.end - seg.start,
                "filter": f"crop={seg.crop_width}:{seg.crop_height}:{seg.crop_x}:{seg.crop_y},scale={self.output_width}:{self.output_height}",
                "label": f"v{i}"
            })
        
        return segments
    
    def build_concat_command(self, segments: List[Dict]) -> str:
        """
        Build FFmpeg concat filter for multiple segments.
        
        Args:
            segments: List of segment configs from generate_segment_filters
        
        Returns:
            FFmpeg filter_complex concat string
        """
        n = len(segments)
        
        # Build input labels
        video_inputs = "".join([f"[v{i}]" for i in range(n)])
        
        # Concat filter
        concat_filter = f"{video_inputs}concat=n={n}:v=1:a=0[outv]"
        
        return concat_filter


class SilenceZoomPipeline:
    """Combined silence removal + speaker zoom pipeline."""
    
    def __init__(self, output_width: int = 1920, output_height: int = 1080):
        """
        Initialize combined pipeline.
        
        Args:
            output_width: Final output width (default: 1920)
            output_height: Final output height (default: 1080)
        """
        self.output_width = output_width
        self.output_height = output_height
        self.zoom_gen = ZoomFilterGenerator(output_width, output_height)
    
    def build_edl(self, silence_timeline: List[Dict], 
                  zoom_timeline: List[Dict]) -> List[Dict]:
        """
        Combine silence cuts with zoom segments into unified EDL.
        
        Args:
            silence_timeline: List of segments to KEEP (from silence detector)
            zoom_timeline: List of zoom segments (from speaker detector)
        
        Returns:
            Unified edit decision list
        """
        # For each keep segment, find overlapping zoom segments
        # and apply appropriate zoom
        
        edl = []
        
        for keep_seg in silence_timeline:
            keep_start = keep_seg["start"]
            keep_end = keep_seg["end"]
            
            # Find zoom segments that overlap with this keep segment
            overlapping_zooms = [
                z for z in zoom_timeline
                if z["start"] < keep_end and z["end"] > keep_start
            ]
            
            if overlapping_zooms:
                # Use the first overlapping zoom segment
                zoom = overlapping_zooms[0]
                edl.append({
                    "start": keep_start,
                    "end": keep_end,
                    "crop_x": zoom.get("crop_x", 0),
                    "crop_y": zoom.get("crop_y", 0),
                    "crop_width": zoom.get("crop_width", 1920),
                    "crop_height": zoom.get("crop_height", 1080),
                    "has_zoom": True
                })
            else:
                # No zoom data, use full frame
                edl.append({
                    "start": keep_start,
                    "end": keep_end,
                    "crop_x": 0,
                    "crop_y": 0,
                    "crop_width": 1920,  # Will be scaled from source
                    "crop_height": 1080,
                    "has_zoom": False
                })
        
        return edl
    
    def generate_ffmpeg_command(self, input_path: str, edl: List[Dict],
                                output_path: str, use_segmented_approach: bool = True) -> List[str]:
        """
        Generate complete FFmpeg command for silence + zoom editing.
        
        Args:
            input_path: Input video path
            edl: Edit decision list from build_edl
            output_path: Output video path
            use_segmented_approach: If True, use concat demuxer for large edits (>4 segments)
        
        Returns:
            FFmpeg command as list of arguments
        """
        import json
        import tempfile
        
        # For large edit lists, use segmented approach (more reliable)
        if use_segmented_approach and len(edl) > 4:
            return self._generate_segmented_command(input_path, edl, output_path)
        
        # Build filter complex for all segments (original approach)
        video_filters = []
        audio_filters = []
        
        for i, seg in enumerate(edl):
            duration = seg["end"] - seg["start"]
            
            # Video: trim → crop → scale → setpts
            if seg.get("has_zoom", False):
                video_filter = (
                    f"[0:v]trim=start={seg['start']}:end={seg['end']},"
                    f"setpts=PTS-STARTPTS,"
                    f"crop={seg['crop_width']}:{seg['crop_height']}:{seg['crop_x']}:{seg['crop_y']},"
                    f"scale={self.output_width}:{self.output_height}"
                    f"[v{i}]"
                )
            else:
                video_filter = (
                    f"[0:v]trim=start={seg['start']}:end={seg['end']},"
                    f"setpts=PTS-STARTPTS,"
                    f"scale={self.output_width}:{self.output_height}:force_original_aspect_ratio=decrease"
                    f"[v{i}]"
                )
            
            # Audio: trim → setpts
            audio_filter = (
                f"[0:a]atrim=start={seg['start']}:end={seg['end']},"
                f"asetpts=PTS-STARTPTS"
                f"[a{i}]"
            )
            
            video_filters.append(video_filter)
            audio_filters.append(audio_filter)
        
        # Concatenate all segments
        n = len(edl)
        video_inputs = "".join([f"[v{i}]" for i in range(n)])
        audio_inputs = "".join([f"[a{i}]" for i in range(n)])
        
        video_filters.append(f"{video_inputs}concat=n={n}:v=1:a=0[outv]")
        audio_filters.append(f"{audio_inputs}concat=n={n}:v=0:a=1[outa]")
        
        filter_complex = ";".join(video_filters + audio_filters)
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-c:a", "aac",
            "-b:a", "192k",
            output_path
        ]
        
        return cmd
    
    def _generate_segmented_command(self, input_path: str, edl: List[Dict],
                                    output_path: str) -> str:
        """
        Generate FFmpeg command using concat demuxer for reliability.
        
        This two-pass approach:
        1. Export each segment individually
        2. Concatenate with FFmpeg concat demuxer
        
        More reliable for long videos with many segments.
        """
        from pathlib import Path
        import os
        
        work_dir = Path(output_path).parent / "temp_segments"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # Export each segment individually
        segment_files = []
        for i, seg in enumerate(edl):
            seg_file = work_dir / f"seg_{i:03d}.mp4"
            segment_files.append(seg_file.name)  # Store just filename for concat file
            
            if seg.get("has_zoom", False):
                filter_str = f"crop={seg['crop_width']}:{seg['crop_height']}:{seg['crop_x']}:{seg['crop_y']},scale={self.output_width}:{self.output_height}"
            else:
                filter_str = f"scale={self.output_width}:{self.output_height}:force_original_aspect_ratio=decrease"
            
            cmd = [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ss", str(seg["start"]),
                "-to", str(seg["end"]),
                "-vf", filter_str,
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                str(seg_file)
            ]
            
            print(f"   Exporting segment {i+1}/{len(edl)}: {seg['start']:.1f}s - {seg['end']:.1f}s")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Segment {i+1} export failed: {result.stderr[:500]}")
        
        # Create concat file with relative paths (ffmpeg needs to be in work_dir)
        concat_file = work_dir / "concat.txt"
        with open(concat_file, "w") as f:
            for seg_file in segment_files:
                f.write(f"file '{seg_file}'\n")
        
        # Concatenate all segments (run from work_dir for relative paths)
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", "concat.txt",
            "-c", "copy",
            str(output_path)
        ]
        
        print(f"   Concatenating {len(segment_files)} segments...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(work_dir))
        if result.returncode != 0:
            raise RuntimeError(f"Concat failed: {result.stderr[:500]}")
        
        return cmd
    
    def save_edl(self, edl: List[Dict], output_path: str):
        """Save EDL to JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump({
                "edit_decision_list": edl,
                "segment_count": len(edl),
                "total_duration": round(sum(s["end"] - s["start"] for s in edl), 2),
                "output_resolution": f"{self.output_width}x{self.output_height}"
            }, f, indent=2)
        
        return output_path


def main():
    """Test zoom filter generation."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python zoom_filter.py <timeline.json> [output_edl.json]")
        sys.exit(1)
    
    timeline_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    zoom_gen = ZoomFilterGenerator()
    timeline = zoom_gen.load_timeline(timeline_path)
    
    print(f"Loaded {len(timeline)} zoom segments")
    
    filter_str = zoom_gen.generate_filter_complex(timeline, 60.0)
    print(f"\nFilter complex:\n{filter_str}")
    
    if output_path:
        pipeline = SilenceZoomPipeline()
        # Mock silence timeline for testing
        silence_keep = [
            {"start": 0, "end": 30}  # Keep everything for this test
        ]
        edl = pipeline.build_edl(silence_keep, timeline)
        pipeline.save_edl(edl, output_path)
        print(f"\nSaved EDL: {output_path}")


if __name__ == "__main__":
    main()
