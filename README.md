# Sentiment Observatory

## Overview
Sentiment Observatory is a tool for scraping and analyzing online content from social media platforms like LinkedIn, Reddit, and X/Twitter. It extracts text, images, and metadata, then performs sentiment analysis, theme extraction, virality scoring, and trend analysis using large language models.

## How It Works
1. **Scraping**: Uses Playwright to scrape content from URLs in a headless browser. Supports persistent sessions for authenticated platforms.
2. **Transcription**: For video/audio content (e.g., YouTube, TikTok), transcribes using Groq's Whisper model.
3. **Analysis**: Processes scraped/transcribed text with Groq LLMs for sentiment, themes, and virality scores. Optional vector-based similarity search and clustering.
4. **Output**: Results are saved to files and served via a FastAPI web server or CLI.

## APIs Used
- Groq API: For LLM analysis (e.g., meta-llama/llama-4-maverick-17b-128e-instruct) and audio transcription (Whisper).
- Playwright: For browser automation and scraping.

## Installation
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Set up environment variables in .env (e.g., GROQ_API_KEY).
4. Run the scraper: `python scraper.py <url>`.

## Usage
- CLI: `python scraper.py <url>` for scraping.
- Web: Run `python server.py` and access the UI at http://localhost:8000.

## License
This project is licensed under the MIT License. See LICENSE for details.
