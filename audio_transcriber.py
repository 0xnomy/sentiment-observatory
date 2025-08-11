import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
from groq import Groq
from pydub import AudioSegment
from pytubefix import YouTube
import importlib

def _lazy_import_yt_dlp():
    try:
        return importlib.import_module('yt_dlp')
    except Exception as e:
        raise RuntimeError('yt_dlp is required for TikTok download. Please `pip install yt_dlp`.') from e

# Optional TikTok downloader via Pyktok
try:
    from pyktok import pyk as pyktok_pyk  # type: ignore
except Exception:
    pyktok_pyk = None  # Will error at runtime if TikTok is used without dependency


# === CONFIG ===
THREE_MINUTES_MS = 3 * 60 * 1000

# Optional: set ffmpeg path manually if not in PATH
# AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
# AudioSegment.ffprobe   = r"C:\ffmpeg\bin\ffprobe.exe"


def _ensure_output_dir(directory_path: str) -> Path:
    output_dir = Path(directory_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(
            f"Missing environment variable: {var_name}. "
            f"Create a .env file with {var_name}=... or export it in your environment."
        )
    return value


# ----------------- YouTube -----------------

def _download_audio_from_youtube(url: str, tmp_dir: Path) -> Tuple[Path, str]:
    yt = YouTube(url)
    stream = (
        yt.streams.filter(only_audio=True)
        .order_by("abr")
        .desc()
        .first()
    )
    if stream is None:
        raise RuntimeError("No audio-only stream found for the provided YouTube URL.")

    video_id = getattr(yt, "video_id", None) or yt.video_id
    safe_filename = f"{video_id}.mp4"

    download_path_str = stream.download(output_path=str(tmp_dir), filename=safe_filename)
    download_path = Path(download_path_str)

    # Small delay to ensure Windows releases the file
    time.sleep(0.5)

    return download_path, video_id


def _trim_and_normalize_audio(input_audio_or_video_path: Path, output_audio_path: Path) -> None:
    audio = AudioSegment.from_file(input_audio_or_video_path)

    # Limit to first 3 minutes
    if len(audio) > THREE_MINUTES_MS:
        audio = audio[:THREE_MINUTES_MS]

    # Mono channel + 16kHz sample rate for Whisper
    audio = audio.set_channels(1).set_frame_rate(16000)

    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(output_audio_path, format="mp3", bitrate="64k")


def _transcribe_with_groq(audio_path: Path, api_key: str) -> str:
    client = Groq(api_key=api_key)
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3-turbo",
        )

    text: Optional[str] = getattr(transcription, "text", None)
    if text is None and isinstance(transcription, dict):
        text = transcription.get("text")
    if not text:
        raise RuntimeError("Transcription failed or returned empty text.")
    return text


def transcribe_youtube_audio(youtube_url: str, output_dir: str = "output") -> str:
    load_dotenv()
    api_key = _require_env("GROQ_API_KEY")

    output_base = _ensure_output_dir(output_dir)

    with tempfile.TemporaryDirectory(prefix="yt_audio_") as tmp:
        tmp_dir = Path(tmp)

        # Step 1: Download & safe filename
        downloaded_audio_path, video_id = _download_audio_from_youtube(youtube_url, tmp_dir)

        # Step 2: Trim & normalize
        trimmed_audio_path = tmp_dir / f"{video_id}_trimmed.mp3"
        _trim_and_normalize_audio(downloaded_audio_path, trimmed_audio_path)

        # Step 3: Transcribe
        transcript_text = _transcribe_with_groq(trimmed_audio_path, api_key)

        # Step 4: Save transcript
        output_path = output_base / f"{video_id}.txt"
        output_path.write_text(transcript_text, encoding="utf-8")

        return transcript_text


# ----------------- TikTok via Pyktok -----------------

def _download_tiktok_with_pyktok(url: str, tmp_dir: Path) -> Tuple[Path, str]:
    if pyktok_pyk is None:
        raise RuntimeError("Pyktok not installed. Please `pip install pyktok`." )
    # Call Pyktok into the temp dir to download a single video and metadata csv
    prev_cwd = os.getcwd()
    os.chdir(str(tmp_dir))
    try:
        meta_csv = tmp_dir / "video_data.csv"
        # download_video=True, metadata filename
        pyktok_pyk.save_tiktok(url, True, str(meta_csv))
        # pick the newest mp4 as the downloaded video
        vids = sorted(tmp_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not vids:
            raise RuntimeError("TikTok download did not produce an mp4 file")
        video_path = vids[0]
        video_id = video_path.stem
        return video_path, video_id
    finally:
        os.chdir(prev_cwd)


def _download_tiktok_with_ytdlp(url: str, tmp_dir: Path) -> Tuple[Path, str]:
    """Download TikTok video using yt_dlp per provided options snippet."""
    yt_dlp = _lazy_import_yt_dlp()
    outtmpl = str(tmp_dir / 'downloaded_video.%(ext)s')
    ydl_opts = {
        'outtmpl': outtmpl,
        'format': 'mp4',  # per snippet; could use 'mp4/best' for flexibility
        'quiet': True,
        'noprogress': True,
        'merge_output_format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        entry = (info.get('entries') or [info])[0] if info.get('_type') == 'playlist' else info
        video_id = entry.get('id') or str(int(time.time()))
    # Determine saved file path
    # Prefer declared outtmpl file
    # It will be downloaded_video.<ext>
    candidates = list(tmp_dir.glob('downloaded_video.*'))
    if candidates:
        video_path = candidates[0]
    else:
        # Fallback: newest mp4
        mp4s = sorted(tmp_dir.glob('*.mp4'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            raise RuntimeError('yt_dlp finished but no mp4 was found')
        video_path = mp4s[0]
    return video_path, video_id


def transcribe_tiktok_audio(tiktok_url: str, output_dir: str = "output") -> str:
    load_dotenv()
    api_key = _require_env("GROQ_API_KEY")

    output_base = _ensure_output_dir(output_dir)

    with tempfile.TemporaryDirectory(prefix="tt_video_") as tmp:
        tmp_dir = Path(tmp)

        # Step 1: Download TikTok video via yt_dlp (preferred);
        # fallback to Pyktok ONLY if available and yt_dlp fails.
        try:
            video_path, video_id = _download_tiktok_with_ytdlp(tiktok_url, tmp_dir)
        except Exception as e1:
            if pyktok_pyk is not None:
                try:
                    video_path, video_id = _download_tiktok_with_pyktok(tiktok_url, tmp_dir)
                except Exception as e2:
                    raise RuntimeError(
                        f"TikTok download failed. yt_dlp error: {e1}. Pyktok error: {e2}"
                    )
            else:
                raise RuntimeError(
                    f"TikTok download via yt_dlp failed: {e1}. Install or verify yt_dlp."
                )

        # Step 2: Extract audio and normalize
        audio_path = tmp_dir / f"{video_id}_audio.mp3"
        _trim_and_normalize_audio(video_path, audio_path)

        # Step 3: Transcribe
        transcript_text = _transcribe_with_groq(audio_path, api_key)

        # Step 4: Save transcript
        out_dir = output_base / "tiktok"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{video_id}.txt"
        output_path.write_text(transcript_text, encoding="utf-8")

        return transcript_text


def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python audio_transcriber.py <youtube_or_tiktok_url> [output_dir]", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) >= 3 else "output"

    try:
        if "tiktok.com" in url:
            text = transcribe_tiktok_audio(url, out_dir)
        else:
            text = transcribe_youtube_audio(url, out_dir)
        print(text)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    _main()
