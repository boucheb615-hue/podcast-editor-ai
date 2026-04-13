#!/usr/bin/env python3
"""
Proxy video generator for fast editing workflow.

Creates low-resolution proxy files for fast experimentation,
then applies the same edits to original for HD output.

Workflow:
1. Generate proxy (720p, H.264, fast encode)
2. Run experiments on proxy (face detection, silence, cuts)
3. Export edit decision list (EDL)
4. Apply EDL to original → HD output (1080p)
"""

import subprocess
import json
from pathlib import Path
from typing import Tuple, Optional


class ProxyGenerator:
    """Generate proxy files for fast video editing."""
    
    def __init__(
        self,
        proxy_width: int = 1280,
        proxy_height: int = 720,
        output_format: str = "mp4",
        crf: int = 23,  # Lower = better quality, higher = faster
        preset: str = "fast",
    ):
        """
        Initialize proxy generator.
        
        Args:
            proxy_width: Proxy video width (default: 1280)
            proxy_height: Proxy video height (default: 720)
            output_format: Output format (default: mp4)
            crf: CRF quality (18-28, default: 23)
            preset: Encoding preset (ultrafast, fast, medium, slow)
        """
        self.proxy_width = proxy_width
        self.proxy_height = proxy_height
        self.output_format = output_format
        self.crf = crf
        self.preset = preset
    
    def generate_proxy(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Generate proxy file from input video.
        
        Args:
            input_path: Path to input video
            output_path: Optional output path (default: <input>_proxy.<format>)
        
        Returns:
            Path to generated proxy file
        """
        input_path = Path(input_path)
        
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_proxy.{self.output_format}"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # FFmpeg command for proxy generation
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"scale={self.proxy_width}:{self.proxy_height}:force_original_aspect_ratio=decrease,pad={self.proxy_width}:{self.proxy_height}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264",
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-c:a", "aac",
            "-b:a", "128k",
            str(output_path)
        ]
        
        print(f"Generating proxy: {input_path.name} → {output_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Proxy generation failed: {result.stderr}")
        
        # Get proxy info
        proxy_info = self.get_video_info(str(output_path))
        print(f"  Proxy created: {proxy_info['width']}x{proxy_info['height']}, {proxy_info['duration']:.1f}s")
        
        return str(output_path)
    
    def get_video_info(self, video_path: str) -> dict:
        """Get video metadata."""
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,duration",
             "-of", "json", video_path],
            capture_output=True, text=True
        )
        
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "duration": float(stream.get("duration", 0))
        }
    
    def calculate_scale_factor(self, original_path: str, proxy_path: str) -> float:
        """Calculate scale factor between original and proxy."""
        orig_info = self.get_video_info(original_path)
        proxy_info = self.get_video_info(proxy_path)
        
        return orig_info["width"] / proxy_info["width"]


class HDExporter:
    """Export final video in HD format (1080p)."""
    
    def __init__(
        self,
        output_width: int = 1920,
        output_height: int = 1080,
        output_format: str = "mp4",
        crf: int = 18,  # Higher quality for final output
        preset: str = "medium",
    ):
        """
        Initialize HD exporter.
        
        Args:
            output_width: Output width (default: 1920)
            output_height: Output height (default: 1080)
            output_format: Output format (default: mp4)
            crf: CRF quality (default: 18 for high quality)
            preset: Encoding preset (default: medium)
        """
        self.output_width = output_width
        self.output_height = output_height
        self.output_format = output_format
        self.crf = crf
        self.preset = preset
    
    def export_with_edl(self, input_path: str, edl: list, output_path: str) -> str:
        """
        Export video applying edit decision list.
        
        Args:
            input_path: Path to input video (original or proxy)
            edl: Edit decision list [{"start": X, "end": Y}, ...]
            output_path: Output path for final video
        
        Returns:
            Path to exported video
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build FFmpeg filter complex for cuts
        # Format: [0:v]trim=start=X:end=Y,setpts=PTS-STARTPTS[v0];[0:a]...
        video_filters = []
        audio_filters = []
        
        for i, segment in enumerate(edl):
            video_filters.append(
                f"[0:v]trim=start={segment['start']}:end={segment['end']},setpts=PTS-STARTPTS[v{i}]"
            )
            audio_filters.append(
                f"[0:a]atrim=start={segment['start']}:end={segment['end']},asetpts=PTS-STARTPTS[a{i}]"
            )
        
        # Concatenate all segments
        video_inputs = "".join([f"[v{i}]" for i in range(len(edl))])
        audio_inputs = "".join([f"[a{i}]" for i in range(len(edl))])
        
        video_filters.append(f"{video_inputs}concat=n={len(edl)}:v=1:a=0[outv]")
        audio_filters.append(f"{audio_inputs}concat=n={len(edl)}:v=0:a=1[outa]")
        
        filter_complex = ";".join(video_filters + audio_filters)
        
        # FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            "-vf", f"scale={self.output_width}:{self.output_height}:force_original_aspect_ratio=decrease,pad={self.output_width}:{self.output_height}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264",
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path)
        ]
        
        print(f"Exporting HD: {input_path.name} → {output_path.name}")
        print(f"  Segments: {len(edl)}")
        print(f"  Output: {self.output_width}x{self.output_height}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Export failed: {result.stderr}")
        
        print(f"  Export complete: {output_path.name}")
        
        return str(output_path)
    
    def export_with_zoom(self, input_path: str, zoom_timeline: list, output_path: str) -> str:
        """
        Export video with dynamic zoom to active speaker.
        
        Args:
            input_path: Path to input video
            zoom_timeline: List of [{"start": X, "end": Y, "zoom_x": X, "zoom_y": Y, "zoom_scale": Z}]
            output_path: Output path for final video
        
        Returns:
            Path to exported video
        """
        # TODO: Implement dynamic zoom with FFmpeg crop/zoom filters
        # This will use the zoom_timeline to apply pan-and-scan effects
        raise NotImplementedError("Dynamic zoom export coming in next iteration")


def main():
    """Test proxy generation workflow."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python proxy_video.py <input_video> [output_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    proxy_gen = ProxyGenerator()
    proxy_path = proxy_gen.generate_proxy(input_path, output_path)
    
    print(f"\nProxy ready: {proxy_path}")
    print("Now run experiments on proxy file for fast iteration.")


if __name__ == "__main__":
    main()
