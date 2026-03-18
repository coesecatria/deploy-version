"""
Streaming routes — Video feed and stream control.
"""

import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.stream_manager import streamer

router = APIRouter(tags=["Streaming"])


@router.get("/video-feed")
async def video_feed():
    """SSE endpoint for Real-Time AI Tracking Video Feed (Multi-subscriber ready)."""
    return StreamingResponse(
        streamer.get_frame_generator(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/stream/pause")
async def pause_stream():
    """Temporarily halts the background CCTV Stream to release hardware camera."""
    streamer.pause()
    return {"message": "Background stream paused. Camera released."}


@router.post("/stream/resume")
async def resume_stream():
    """Resumes the background CCTV Stream hardware camera."""
    streamer.resume()
    return {"message": "Background stream resumed."}
