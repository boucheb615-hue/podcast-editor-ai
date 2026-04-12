# Podcast Editor AI

**AI agent that auto-edits video podcasts: silence removal + speaker focus zoom**

## Mission

Automate video podcast post-production for non-technical creators. Transform raw multi-camera recordings into polished episodes with:

- ✂️ **Silence removal** — Auto-cut dead air, pauses, filler words
- 🎯 **Speaker focus** — Zoom/crop to active speaker using face detection
- 🎬 **Smart cuts** — Maintain natural pacing, avoid jump cuts

## Tech Stack (Planned)

| Component | Technology | Purpose |
|-----------|------------|---------|
| Audio Analysis | Whisper.cpp | Speech detection, silence mapping |
| Face Detection | MediaPipe / OpenCV | Identify active speaker |
| Video Processing | FFmpeg | Cuts, zoom, transitions |
| Orchestration | Python | Pipeline coordination |

## Status

🚧 **Early stage** — Repository created, architecture planning in progress.

## Roadmap

- [ ] MVP: Silence detection + basic cuts
- [ ] Speaker detection (single camera)
- [ ] Multi-camera speaker switching
- [ ] Zoom/crop automation
- [ ] CLI interface
- [ ] Batch processing

## Community

Built by [WebModerne](https://webmoderne.com) — Teaching non-technical people to build apps with AI.

Join our Telegram: https://t.me/+Fon2ltdbEcc3N2Nh

---

*This is a community-first project. Contributions, feedback, and testing welcome.*
