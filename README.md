# Sentiment Observatory

## Overview

Sentiment Observatory is a cross-platform content analysis tool that scrapes, transcribes, and analyzes social media content from LinkedIn, Reddit, X/Twitter, YouTube, and TikTok. It uses Groq's LLMs for **multimodal sentiment, theme, and virality analysis**.

---

## Core Features

### Content Acquisition
- **Scraper** – Playwright with persistent sessions.
- **Audio Transcriber** – YouTube via Groq Whisper.
- **TikTok Transcriber** – yt_dlp + Groq Whisper.

### Analysis
- **Advanced Analyzer** – Text + image analysis (Groq LLMs).
- **Virality Agent** – Scores content (1–10 scale).
- **Trend Analyzer** – Tracks sentiment/themes over time.
- **Vector Engine** – Embeddings for similarity & clustering.

### Delivery
- **CLI** – Analyze URLs from the terminal.
- **Web Server** – FastAPI-based UI.
- **Reports** – JSON + human-readable text.

---

## Architecture

Pipeline flow:
1. Detect platform from URL.
2. Scrape/transcribe content.
3. Run multimodal analysis.
4. Generate insights (sentiment, themes, virality).
5. Output structured reports.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
playwright install chromium
```

Create `.env`:

```env
GROQ_API_KEY=your_key_here
```

## Usage

### CLI

```bash
python app.py <url>                 # Basic analysis
python app.py <url> --advanced      # Advanced analysis (default)
python app.py --system              # System report
```

Examples:

```bash
python app.py https://www.linkedin.com/posts/...
python app.py https://www.reddit.com/r/.../comments/...
python app.py https://x.com/.../status/...
python app.py https://www.youtube.com/watch?v=...
python app.py https://www.tiktok.com/@user/video/...
```

### Web Interface

```bash
python server.py
```
Open: `http://localhost:8000/web/`

## Dependencies

* **Groq API** – LLMs + Whisper for text & audio analysis.
* **Playwright** – Web scraping.
* **FFmpeg (pydub)** – Audio extraction & processing.
* **FastAPI** – Web server for the UI.

## Limitations

* **LinkedIn** – Requires a logged-in persistent browser session.
* **YouTube/TikTok** – Only first 3 minutes analyzed (coz of API limit)
* **Trend & vector modules** – Disabled by default.
* **Multimodal image analysis** – Only the first image is processed.

## File Structure

```bash
app.py                  # Main pipeline
server.py               # Web server
scraper.py              # Scraping logic
advanced_analyzer.py    # LLM content analysis
audio_transcriber.py    # YouTube transcription
tiktok_transcriber.py   # TikTok transcription
virality_agent.py       # Virality scoring
trend_analyzer.py       # Trend tracking
vector_engine.py        # Similarity & clustering
web/                    # Frontend files
```