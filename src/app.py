import asyncio
import os
import re
from functools import lru_cache
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from starlette.responses import JSONResponse, Response, HTMLResponse
import httpx
import yt_dlp
import uvicorn

load_dotenv()

app = FastAPI(title="ytx.ai", version="1.0.0")

class Chapter(BaseModel):
    """Represents one chapter/section of the video."""
    title: str = Field(description="Chapter title (2-6 words)")
    timestamp: str = Field(description="Timestamp in MM:SS format")


class VideoAnalysis(BaseModel):
    """Complete analysis result from LLM."""
    summary: str = Field(description="Bullet point summary with - prefix")
    chapters: list[Chapter] = Field(description="List of video chapters with timestamps")


@lru_cache(maxsize=1)
def get_llm():
    """Create LLM once and reuse it. Saves ~100ms per request."""
    return ChatOpenAI(
        model_name="xiaomi/mimo-v2-flash:free",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
    )

CACHE = {}

ALLOWED_NETLOCS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
}


def _extract_video_id(youtube_url: str) -> Optional[str]:
    """
    Extract video ID from YouTube URL.
    
    Supports:
    - https://www.youtube.com/watch?v=dQw4w9WgXcQ
    - https://youtu.be/dQw4w9WgXcQ
    - https://www.youtube.com/shorts/dQw4w9WgXcQ
    
    Returns: video_id (e.g., "dQw4w9WgXcQ") or None if invalid
    """
    parsed = urlparse(youtube_url)
    
    # Security: Only allow YouTube domains
    if parsed.netloc not in ALLOWED_NETLOCS:
        return None

    # Format 1: youtu.be/VIDEO_ID
    if parsed.netloc == "youtu.be" and parsed.path:
        return parsed.path.lstrip("/")

    # Format 2: youtube.com/watch?v=VIDEO_ID
    query_id = parse_qs(parsed.query).get("v")
    if query_id:
        return query_id[0]

    # Format 3: youtube.com/shorts/VIDEO_ID
    if parsed.path.startswith("/shorts/"):
        return parsed.path.split("/shorts/")[-1].split("/")[0]

    return None

def _pick_track(subs: dict):
    """Pick a caption track preferring English variants."""
    if not subs:
        return None
    for key in ("en", "en-US", "en-GB"):
        if key in subs:
            return subs[key]
    for key, val in subs.items():
        if key.startswith("en"):
            return val
    return next(iter(subs.values()), None)


def _pick_vtt_url(track: list):
    if not track:
        return None
    for fmt in track:
        if fmt.get("ext") == "vtt":
            return fmt.get("url")
    return track[0].get("url")


PROXY_URL = os.getenv("PROXY_URL")


def _fetch_vtt_url_sync(video_id: str):
    """Get a VTT caption URL using yt-dlp (manual or auto captions)."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "vtt",
        "subtitleslangs": ["en", "en.*", "en-US", "en-GB"],
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }
    if PROXY_URL:
        ydl_opts["proxy"] = PROXY_URL
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subs = info.get("subtitles") or {}
        autos = info.get("automatic_captions") or {}
        track = _pick_track(subs) or _pick_track(autos)
        if not track:
            return None, info
        return _pick_vtt_url(track), info


def _parse_vtt_to_text(vtt_content: str) -> str:
    """Convert VTT to plain text with [MM:SS] timestamps."""
    out = []
    for block in vtt_content.split("\n\n"):
        if "-->" not in block:
            continue
        parts = block.split("\n")
        if len(parts) < 2:
            continue
        ts_raw = parts[0].split(" --> ")[0].split(".")[0]  # HH:MM:SS
        text = " ".join(parts[1:]).strip()
        if not text:
            continue
        text = re.sub(r"<[^>]+>", "", text)
        try:
            hh, mm, ss = ts_raw.split(":")
            mm_total = int(hh) * 60 + int(mm)
            ts_fmt = f"{mm_total}:{int(ss):02d}"
        except Exception:
            ts_fmt = ts_raw
        out.append(f"[{ts_fmt}] {text}")
    return "\n".join(out)


def _fetch_transcript_sync(video_id: str):
    """Return transcript text with timestamps using yt-dlp captions only."""
    vtt_url, info = _fetch_vtt_url_sync(video_id)
    if not vtt_url:
        raise RuntimeError("No captions available for this video")
    resp = httpx.get(vtt_url, timeout=10.0)
    resp.raise_for_status()
    transcript_text = _parse_vtt_to_text(resp.text)
    return transcript_text, info



def _get_audio_url_sync(video_id: str) -> str:
    """Extract direct audio stream URL from YouTube video."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "format": "bestaudio/best"
    }
    if PROXY_URL:
        ydl_opts["proxy"] = PROXY_URL
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(
            f"https://www.youtube.com/watch?v={video_id}",
            download=False
        )
        return info.get("url", "")


async def get_video_data(video_id: str) -> dict:
    """
    Fetch video transcript and metadata.
    
    Returns: {
        "title": str,
        "description": str,
        "transcript_with_timestamps": str (formatted with [MM:SS] prefixes)
    }
    """
    transcript_with_timestamps, metadata = await asyncio.to_thread(
        _fetch_transcript_sync, video_id
    )
    
    return {
        "title": metadata.get("title", ""),
        "description": metadata.get("description", ""),
        "transcript_with_timestamps": transcript_with_timestamps,
    }


# ============================================================
# LLM SUMMARIZATION - Generate summary + chapters
# ============================================================
async def summarize_video_data(video_data: dict) -> dict:
    """Summarize video; handle long transcripts by chunking and merging."""

    llm = get_llm()
    parser = JsonOutputParser(pydantic_object=VideoAnalysis)

    transcript_text = video_data["transcript_with_timestamps"]
    title = video_data["title"]
    description = video_data["description"]

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",  # Token-accurate splitting
        chunk_size=6000,
        chunk_overlap=1200,
    )
    chunks = splitter.split_text(transcript_text)

    prompt = ChatPromptTemplate.from_template(
        """Analyze this YouTube video transcript and create a comprehensive summary.

Title: {title}
Description: {description}

Transcript (with timestamps):
{transcript_chunk}

Instructions:
1. Create a concise summary using bullet points (prefix each with "- ")
2. Extract key chapters with exact timestamps from the transcript in MM:SS format
3. Chapter titles should be 2-6 words, descriptive and action-oriented
4. Use actual timestamps that appear in the transcript

{format_instructions}

Respond ONLY with valid JSON."""
    )

    chain = prompt | llm | parser
    fmt = {"format_instructions": parser.get_format_instructions()}

    try:
        if len(chunks) == 1:
            result = await chain.ainvoke({
                "title": title,
                "description": description,
                "transcript_chunk": transcript_text,
                **fmt,
            })
            return {
                "summary": result.get("summary", ""),
                "chapters": result.get("chapters", []),
            }

        tasks = [
            chain.ainvoke({
                "title": title,
                "description": description,
                "transcript_chunk": chunk,
                **fmt,
            })
            for chunk in chunks
        ]
        results = await asyncio.gather(*tasks)

        bullets: list[str] = []
        for res in results:
            if res and isinstance(res, dict):
                bullets.extend(
                    line.strip() for line in res.get("summary", "").split("\n")
                    if line.strip().startswith("-")
                )

        chapters: list[dict] = []
        seen: set[str] = set()
        for res in results:
            if res and isinstance(res, dict):
                for ch in res.get("chapters", []):
                    ts = ch.get("timestamp", "")
                    if ts and ts not in seen:
                        seen.add(ts)
                        chapters.append(ch)

        def ts_to_seconds(ts: str) -> int:
            try:
                m, s = ts.split(":")
                return int(m) * 60 + int(s)
            except Exception:
                return 0

        chapters.sort(key=lambda ch: ts_to_seconds(ch.get("timestamp", "0:00")))

        return {
            "summary": "\n".join(bullets[:7]),
            "chapters": chapters[:10],
        }

    except Exception as e:
        print(f"LLM error: {e}")
        import traceback
        traceback.print_exc()
        return {"summary": f"Error: {e}", "chapters": []}


# ============================================================
# API ENDPOINTS

@app.get("/favicon.ico")
async def favicon():
    """Prevent 400 errors from browser favicon requests."""
    return Response(status_code=204)


@app.get("/api/audio/{video_id}")
async def get_audio(video_id: str):
    """
    Get audio stream URL for a YouTube video.
    Returns JSON with audio_url that can be used in <audio> tag.
    """
    try:
        audio_url = await asyncio.to_thread(_get_audio_url_sync, video_id)
        if not audio_url:
            raise HTTPException(status_code=404, detail="Audio stream not found")
        return JSONResponse({"audio_url": audio_url, "video_id": video_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/summarize")
async def api_summarize(url: str = Query(..., description="YouTube video URL")):
    """
    API endpoint for frontend: GET /api/summarize?url=YOUTUBE_URL
    """
    video_id = _extract_video_id(url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    cache_key = f"summary:{video_id}"
    if cache_key in CACHE:
        return JSONResponse(CACHE[cache_key])

    try:
        video_data = await get_video_data(video_id)
        result = await summarize_video_data(video_data)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_data = {
        "video_id": video_id,
        "title": video_data["title"],
        "description": video_data["description"],
        "summary": result["summary"],
        "chapters": result["chapters"],
    }
    CACHE[cache_key] = response_data
    return JSONResponse(response_data)


# Load HTML template
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "static", "results.html")
with open(TEMPLATE_PATH, "r") as f:
    HTML_TEMPLATE = f.read()


def _render_html(data: dict) -> str:
    """Render results using the HTML template."""
    bullets_html = ""
    for line in data.get("summary", "").split("\n"):
        line = line.strip()
        if line.startswith("-"):
            bullets_html += f"<li>{line[1:].strip()}</li>"
    
    chapters_html = ""
    for ch in data.get("chapters", []):
        chapters_html += f'''<li class="chapter-item">
            <span class="ts">{ch.get("timestamp", "0:00")}</span>
            <span>{ch.get("title", "")}</span>
        </li>'''
    
    return (
        HTML_TEMPLATE
        .replace("{{title}}", data.get("title", ""))
        .replace("{{description}}", data.get("description", "")[:300])
        .replace("{{bullets}}", bullets_html)
        .replace("{{chapters}}", chapters_html)
        .replace("{{video_id}}", data.get("video_id", ""))
    )


@app.get("/{target_path:path}")
async def summarize_endpoint(
    target_path: str,
    url: Optional[str] = Query(None),
    v: Optional[str] = Query(None), 
):
    """
    Main endpoint: GET /https://www.youtube.com/watch?v=VIDEO_ID
    Returns HTML page with video summary.
    """
    # Handle root path - return usage hint
    if not target_path and not url and not v:
        raise HTTPException(status_code=400, detail="Usage: ytx.ai/<youtube-url>")
    
    # Parse URL from various input formats
    if url:
        target_raw = url
    elif v:
        target_raw = f"{target_path}?v={v}"
    else:
        target_raw = target_path

    target_raw = unquote(target_raw).strip()
    if not target_raw:
        raise HTTPException(status_code=400, detail="Usage: ytx.ai/<youtube-url>")

    # Add https:// if missing
    target_url = target_raw
    if not target_url.startswith("http://") and not target_url.startswith("https://"):
        target_url = f"https://{target_url}"

    # Extract video ID
    video_id = _extract_video_id(target_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid or unsupported YouTube URL")

    # Check cache
    cache_key = f"summary:{video_id}"
    if cache_key in CACHE:
        return HTMLResponse(_render_html(CACHE[cache_key]))

    # Process the video
    try:
        video_data = await get_video_data(video_id)
        result = await summarize_video_data(video_data)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_data = {
        "video_id": video_id,
        "title": video_data["title"],
        "description": video_data["description"],
        "summary": result["summary"],
        "chapters": result["chapters"],
    }
    CACHE[cache_key] = response_data
    
    return HTMLResponse(_render_html(response_data))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)