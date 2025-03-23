"""
FastAPI web application for crypto trading pool.
"""
import asyncio
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import services
from services.app_service import get_app_service, AppService
from models.video import Video
from config.settings import VPH_THRESHOLD, MAX_VIDEOS

# Create the FastAPI application
app = FastAPI(
    title="Crypto Trading Pool API",
    description="API for cryptocurrency trading based on YouTube video analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for API requests and responses
class VideoModel(BaseModel):
    id: str
    title: str
    views: int
    publish_time: datetime
    vph: float = 0.0
    channel_id: Optional[str] = None
    channel_title: Optional[str] = None
    
    class Config:
        from_attributes = True

class TranscriptRequest(BaseModel):
    video_id: str

class TranscriptResponse(BaseModel):
    video_id: str
    transcript: Optional[str] = None
    cached: bool = False
    
class StatusResponse(BaseModel):
    status: str
    redis_available: bool
    is_running: bool
    last_cycle_time: str
    time_since_last_cycle: str
    next_cycle_in: str
    accumulated_videos_count: int
    
class VideoListResponse(BaseModel):
    videos: List[VideoModel]
    count: int

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    app_service = get_app_service()
    await app_service.initialize()
    
    # Register the background task as a lifespan event
    asyncio.create_task(app_service.cycle_loop())

# API endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "application": "Crypto Trading Pool API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/status", response_model=StatusResponse)
async def get_status(app_service: AppService = Depends(get_app_service)):
    """Get application status."""
    status_data = app_service.get_service_status()
    return StatusResponse(
        status="online",
        **status_data
    )

@app.post("/run_cycle", response_model=dict)
async def trigger_cycle(
    background_tasks: BackgroundTasks,
    app_service: AppService = Depends(get_app_service)
):
    """Manually trigger a processing cycle."""
    if app_service.is_running:
        raise HTTPException(status_code=409, detail="A cycle is already running")
    
    background_tasks.add_task(app_service.run_cycle)
    return {"status": "success", "message": "Processing cycle started"}

@app.get("/transcripts/{video_id}", response_model=TranscriptResponse)
async def get_transcript(
    video_id: str,
    app_service: AppService = Depends(get_app_service)
):
    """Get transcript for a video."""
    transcript = app_service.transcript_service.get_transcript(video_id)
    return TranscriptResponse(
        video_id=video_id,
        transcript=transcript,
        cached=transcript is not None
    )

@app.post("/transcripts", response_model=TranscriptResponse)
async def process_transcript(
    request: TranscriptRequest,
    app_service: AppService = Depends(get_app_service)
):
    """Process a video to get its transcript."""
    # Create a dummy video object
    video = Video(
        id=request.video_id,
        title="",
        views=0,
        publish_time=datetime.now(),
        vph=0.0
    )
    
    # Process the video
    video_id, transcript = await app_service.process_video(video)
    
    return TranscriptResponse(
        video_id=video_id,
        transcript=transcript,
        cached=False
    )

@app.get("/videos", response_model=VideoListResponse)
async def get_accumulated_videos(app_service: AppService = Depends(get_app_service)):
    """Get accumulated videos."""
    video_tuples = app_service.get_accumulated_videos()
    videos = []
    
    for video_tuple in video_tuples:
        video = Video.from_tuple(video_tuple)
        videos.append(VideoModel(
            id=video.id,
            title=video.title or "Unknown",
            views=video.views,
            publish_time=video.publish_time,
            vph=video.vph,
            channel_id=video.channel_id,
            channel_title=video.channel_title
        ))
    
    return VideoListResponse(
        videos=videos,
        count=len(videos)
    )

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, app_service: AppService = Depends(get_app_service)):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    
    try:
        while True:
            # Send current status every 5 seconds
            status_data = app_service.get_service_status()
            await websocket.send_json({
                "type": "status_update",
                "data": {
                    "status": "online",
                    **status_data
                }
            })
            
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        # Handle client disconnection
        pass

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")