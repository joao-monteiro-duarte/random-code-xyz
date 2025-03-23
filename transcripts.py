"""
Legacy module for transcript handling.
This file provides backward compatibility with the original implementation.
The refactored architecture uses TranscriptService instead.
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Tuple, Optional, List, Any, Dict

# Try to import Redis client, use None if not available
try:
    from config import REDIS_AVAILABLE, redis_client
except ImportError:
    REDIS_AVAILABLE = False
    redis_client = None

# Set up logging
logger = logging.getLogger(__name__)

# Initialize accumulated videos list (for legacy compatibility)
accumulated_videos = []

async def initialize_transcripts():
    """Initialize transcripts module."""
    logger.info("Initializing transcripts module")
    try:
        # Try to use the refactored service architecture
        from services.app_service import get_app_service
        app_service = get_app_service()
        await app_service.transcript_service.initialize()
        logger.info("Using service architecture for transcript handling")
    except (ImportError, NameError) as e:
        logger.warning(f"Service architecture not available: {e}")
        logger.info("Using legacy transcript handling")
    
    return True

async def fetch_transcript(video_id: str) -> str:
    """
    Fetch a transcript for a video.
    This is the legacy implementation for backward compatibility.
    
    Args:
        video_id: The YouTube video ID
        
    Returns:
        Transcript text
    """
    logger.info(f"Fetching transcript for {video_id}")
    
    # Implementation would typically use YouTube Transcript API
    # This is a simplified example that returns a dummy transcript
    return f"This is a simulated transcript for video {video_id} about cryptocurrency."

async def process_single_video(video: Tuple[str, int, str, float]) -> Tuple[str, str]:
    """
    Process a single video to get its transcript.
    This is the legacy implementation for backward compatibility.
    
    Args:
        video: Tuple of (video_id, views, publish_time, vph)
        
    Returns:
        Tuple of (video_id, transcript)
    """
    video_id = video[0]
    
    # Try to get from cache first
    transcript = get_transcript_from_redis(video_id)
    
    if transcript:
        logger.info(f"Using cached transcript for {video_id}")
        return (video_id, transcript)
    
    # Fetch and cache if not found
    transcript = await fetch_transcript(video_id)
    save_transcript_to_redis(video_id, transcript)
    
    return (video_id, transcript)

def get_transcript_from_redis(video_id: str) -> Optional[str]:
    """
    Get a transcript from Redis.
    This is the legacy implementation for backward compatibility.
    
    Args:
        video_id: The YouTube video ID
        
    Returns:
        Transcript text or None if not found
    """
    try:
        # Try to use the refactored service architecture
        from services.app_service import get_app_service
        app_service = get_app_service()
        return app_service.transcript_service.get_transcript(video_id)
    except (ImportError, NameError):
        # Fallback to direct Redis access
        if REDIS_AVAILABLE and redis_client:
            try:
                transcript = redis_client.get(f"transcript:{video_id}")
                return transcript
            except Exception as e:
                logger.error(f"Error retrieving transcript from Redis: {e}")
        
        return None

def save_transcript_to_redis(video_id: str, transcript: str, expire: Optional[int] = None) -> bool:
    """
    Save a transcript to Redis.
    This is the legacy implementation for backward compatibility.
    
    Args:
        video_id: The YouTube video ID
        transcript: Transcript text
        expire: Expiration time in seconds
        
    Returns:
        True if saved, False otherwise
    """
    try:
        # Try to use the refactored service architecture
        from services.app_service import get_app_service
        app_service = get_app_service()
        return app_service.transcript_service.save_transcript(video_id, transcript, expire)
    except (ImportError, NameError):
        # Fallback to direct Redis access
        if REDIS_AVAILABLE and redis_client:
            try:
                key = f"transcript:{video_id}"
                redis_client.set(key, transcript)
                if expire:
                    redis_client.expire(key, expire)
                return True
            except Exception as e:
                logger.error(f"Error saving transcript to Redis: {e}")
        
        return False

def process_video_stats(video: Tuple[str, int, str, float], now: Optional[datetime] = None) -> Tuple[str, int, str, float]:
    """
    Process video statistics and recalculate VPH.
    This is the legacy implementation for backward compatibility.
    
    Args:
        video: Tuple of (video_id, views, publish_time, vph)
        now: Current datetime
        
    Returns:
        Updated video tuple
    """
    try:
        # Try to use the refactored service architecture
        from services.app_service import get_app_service
        app_service = get_app_service()
        return app_service.process_video_stats(video, now)
    except (ImportError, NameError):
        # Fallback to direct implementation
        from fetch_videos import process_video_stats as fetch_process_video_stats
        return fetch_process_video_stats(video, now)