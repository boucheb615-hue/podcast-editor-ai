# Architecture Plan

## Core Pipeline

```
Raw Video → Audio Analysis → Silence Map → Face Detection → Edit Decision List → FFmpeg → Final Video
```

## Phase 1: MVP (Silence Removal Only)

**Goal:** Prove value with single-camera silence cutting.

### Components

1. **Audio Processor** (`audio/`)
   - Extract audio track from video
   - Detect speech/silence segments using Whisper.cpp or pyannote.audio
   - Output: JSON timeline with timestamps

2. **Edit Decision List Generator** (`edit/`)
   - Convert silence map to cut list
   - Apply smoothing (minimum cut duration, fade handles)
   - Output: FFmpeg-compatible edit list

3. **Video Processor** (`video/`)
   - Apply cuts using FFmpeg
   - Preserve audio sync
   - Output: Final video file

### File Structure

```
podcast-editor-ai/
├── audio/
│   ├── __init__.py
│   ├── extractor.py      # Extract audio from video
│   └── silence_detector.py  # Detect silence segments
├── edit/
│   ├── __init__.py
│   └── decision_list.py  # Generate cut list
├── video/
│   ├── __init__.py
│   └── processor.py      # Apply edits with FFmpeg
├── config.py             # Configuration defaults
├── main.py               # CLI entry point
└── requirements.txt      # Python dependencies
```

## Phase 2: Speaker Focus

**Goal:** Auto-zoom to active speaker.

### Additions

1. **Face Detection** (`vision/`)
   - Detect faces per frame
   - Track active speaker (audio-visual correlation)
   - Calculate crop regions

2. **Zoom Engine** (`video/zoom.py`)
   - Apply dynamic crop/zoom
   - Smooth transitions between speakers

## Phase 3: Multi-Camera

**Goal:** Switch between camera angles.

### Additions

1. **Camera Sync** (`sync/`)
   - Align multi-camera timelines
   - Sync via audio waveform matching

2. **Switcher** (`edit/switcher.py`)
   - Choose best camera per segment
   - Apply cut/transition rules

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Processing speed | 0.5x realtime | 10min video → 5min processing |
| Silence detection accuracy | >95% | Manual review of test set |
| User satisfaction | 4/5 stars | Community feedback |

## Constraints

- **Local-first:** Run on consumer hardware (no cloud GPU required)
- **Batch-friendly:** Process multiple episodes unattended
- **Format support:** MP4, MOV, MKV input → MP4 output
