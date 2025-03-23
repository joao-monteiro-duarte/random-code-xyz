"""
Main application service for crypto trading pool.
Coordinates services and provides dependency injection for FastAPI.
"""

import logging
import asyncio
import redis
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from fastapi import Depends, FastAPI, WebSocket, BackgroundTasks

# Import configuration
from config.settings import (
    REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB, REDIS_EXPIRE,
    DB_PATH, VPH_THRESHOLD, MAX_VIDEOS, CYCLE_INTERVAL
)

# Import models
from models.video import Video

# Import services
from services.transcript_service import TranscriptService
from services.sentiment_service import SentimentAnalysisService
from utils.cache_manager import CacheManager

# Setup logging
logger = logging.getLogger(__name__)

class AppService:
    """
    Main application service that coordinates other services and manages the application lifecycle.
    """
    
    def __init__(self, redis_client=None, redis_available: bool = True, openrouter_api_key: Optional[str] = None):
        """
        Initialize the application service.
        
        Args:
            redis_client: Redis client instance or None to create a new one
            redis_available: Whether Redis is available
            openrouter_api_key: OpenRouter API key for sentiment analysis
        """
        # Initialize Redis client if not provided
        self.redis_client = redis_client or self._init_redis_client()
        self.redis_available = redis_available and self.redis_client is not None
        
        # Initialize services
        self.transcript_service = TranscriptService(
            redis_client=self.redis_client,
            redis_available=self.redis_available
        )
        
        # Initialize sentiment analysis service
        self.sentiment_service = SentimentAnalysisService(
            api_key=openrouter_api_key
        )
        
        # Initialize state
        self.accumulated_videos: List[Tuple[str, int, str, float]] = []
        self.last_cycle_time = datetime.now()
        self.is_running = False
        
    def _init_redis_client(self) -> Optional[redis.Redis]:
        """
        Initialize Redis client.
        
        Returns:
            Redis client or None if initialization fails
        """
        try:
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD or None,
                db=REDIS_DB,
                decode_responses=True
            )
            # Test connection
            client.ping()
            logger.info("Successfully connected to Redis")
            return client
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return None
            
    async def initialize(self):
        """
        Initialize all services.
        """
        logger.info("Initializing application services...")
        
        # Initialize transcript service
        await self.transcript_service.initialize()
        
        # Load accumulated videos from cache
        self.accumulated_videos = self.get_accumulated_videos()
        
        # We'll set the crypto_trading_service reference later to avoid circular dependencies
        self.crypto_trading_service = None
        logger.info("crypto_trading_service reference will be set later to avoid circular dependencies")
        
        logger.info("Application services initialized")
        
    def get_accumulated_videos(self) -> List[Tuple[str, int, str, float]]:
        """
        Get accumulated videos from Redis or local cache.
        
        Returns:
            List of video tuples
        """
        if self.redis_available and self.redis_client:
            try:
                serialized = self.redis_client.get('accumulated_videos')
                if serialized:
                    return json.loads(serialized)
            except Exception as e:
                logger.error(f"Error retrieving accumulated videos from Redis: {e}")
        
        # Return empty list if no cached data
        return []
        
    def set_accumulated_videos(self, videos: List[Tuple[str, int, str, float]]):
        """
        Store accumulated videos to Redis.
        
        Args:
            videos: List of video tuples
        """
        if self.redis_available and self.redis_client:
            try:
                self.redis_client.set('accumulated_videos', json.dumps(videos))
            except Exception as e:
                logger.error(f"Error storing accumulated videos to Redis: {e}")
        
        # Always update local state
        self.accumulated_videos = videos
    
    async def process_video(self, video: Union[Video, Tuple[str, int, str, float]]) -> Tuple[str, str]:
        """
        Process a single video to get its transcript.
        
        Args:
            video: Video object or tuple (video_id, views, publish_time, vph)
            
        Returns:
            Tuple of (video_id, transcript)
        """
        return await self.transcript_service.process_video(video)
        
    async def process_videos(self, videos: List[Union[Video, Tuple]], parallel: bool = True) -> List[Tuple[str, str]]:
        """
        Process multiple videos to get their transcripts.
        
        Args:
            videos: List of Video objects or tuples
            parallel: Whether to process videos in parallel
            
        Returns:
            List of (video_id, transcript) tuples
        """
        return await self.transcript_service.process_videos(videos, parallel)
    
    def process_video_stats(self, video: Union[Video, Tuple], now: Optional[datetime] = None) -> Union[Video, Tuple]:
        """
        Process video statistics and recalculate VPH.
        
        Args:
            video: Video object or tuple
            now: Current datetime
            
        Returns:
            Updated Video object or tuple
        """
        return self.transcript_service.process_video_stats(video, now)
    
    async def add_videos(self, videos: List[Union[Video, Tuple]]):
        """
        Add videos to the accumulated list.
        
        Args:
            videos: List of Video objects or tuples to add
        """
        # Get current accumulated videos
        accumulated = self.get_accumulated_videos()
        
        # Add new videos if they don't already exist
        for video in videos:
            # Convert Video objects to tuples if needed
            if isinstance(video, Video):
                video_tuple = video.to_tuple()
                video_id = video.id
            else:
                video_tuple = video
                video_id = video[0]
                
            # Check if video already exists in accumulated list
            if not any(v[0] == video_id for v in accumulated):
                accumulated.append(video_tuple)
                vph = video.vph if isinstance(video, Video) else video[3]
                logger.info(f"Added video {video_id} to accumulated videos (VPH: {vph:.2f})")
        
        # Store updated list
        self.set_accumulated_videos(accumulated)
        
    async def run_cycle(self):
        """
        Run a full processing cycle.
        This method coordinates video processing, sentiment analysis, and trading decisions.
        """
        if self.is_running:
            logger.warning("A cycle is already running, skipping")
            return
            
        try:
            self.is_running = True
            logger.info("Starting processing cycle")
            
            # Import the run_cycle function from the run_cycle_impl module
            from run_cycle_impl import run_cycle as run_cycle_impl
            
            # Execute the cycle implementation
            await run_cycle_impl(VPH_THRESHOLD)
            
            # Update last cycle time
            self.last_cycle_time = datetime.now()
            
            logger.info("Processing cycle completed")
        except Exception as e:
            logger.error(f"Error during processing cycle: {e}", exc_info=True)
        finally:
            self.is_running = False
            
    async def start_background_tasks(self, app: FastAPI):
        """
        Start background tasks when the FastAPI application starts.
        
        Args:
            app: FastAPI application instance
        """
        # Initialize services
        await self.initialize()
        
        # Start the cycle loop (30-minute cycle for comprehensive processing)
        cycle_task = asyncio.create_task(self.cycle_loop())
        
        # Start the video fetching loop (5-minute cycle for fresh videos and quick decisions)
        fetch_task = asyncio.create_task(self.fetch_videos_loop())
        
        # Register tasks with app state to prevent garbage collection
        if not hasattr(app.state, "background_tasks"):
            app.state.background_tasks = []
        
        app.state.background_tasks.extend([cycle_task, fetch_task])
        
        logger.info("Background tasks started: cycle_loop and fetch_videos_loop")
            
    async def cycle_loop(self):
        """
        Background task to run cycles at regular intervals.
        """
        while True:
            try:
                # Check if it's time to run a cycle
                now = datetime.now()
                time_since_last = (now - self.last_cycle_time).total_seconds()
                
                if time_since_last >= CYCLE_INTERVAL:
                    await self.run_cycle()
                
                # Wait before checking again
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in cycle loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait before retrying
                
    async def fetch_videos_loop(self):
        """
        Background task to fetch new videos at regular intervals and trigger quick decisions when new sentiment is available.
        """
        from fetch_videos import (
            fetch_initial_videos, fetch_crypto_news, 
            fetch_trending_videos, fetch_channel_videos, 
            CHANNELS, process_video_stats
        )
        from config.settings import FETCH_INTERVAL, MAX_VIDEOS_PER_FETCH, VPH_THRESHOLD
        
        logger.info("Starting video fetching background task")
        
        # On first run, get more videos to bootstrap the system
        if not self.accumulated_videos:
            logger.info("Initial video fetch (startup)")
            
            # Get initial batch of videos
            initial_videos = await fetch_initial_videos()
            if initial_videos:
                # Ensure all videos have proper VPH calculation
                now = datetime.now()
                processed_videos = [process_video_stats(video, now) for video in initial_videos]
                filtered_videos = [v for v in processed_videos if v[3] >= VPH_THRESHOLD * 0.5]  # Lower threshold for initial
                
                await self.add_videos(filtered_videos)
                logger.info(f"Added {len(filtered_videos)} initial videos on startup (filtered from {len(initial_videos)})")
            
            # Get trending videos on startup
            trending_videos = await fetch_trending_videos(startup=True)
            if trending_videos:
                await self.add_videos(trending_videos)
                logger.info(f"Added {len(trending_videos)} trending videos on startup")
                
            # Pre-fetch some transcripts as a bootstrap measure
            if self.accumulated_videos:
                bootstrap_videos = self.accumulated_videos[:5]  # Process first 5 videos
                logger.info(f"Pre-fetching transcripts for {len(bootstrap_videos)} videos")
                await self.process_videos(bootstrap_videos, parallel=True)
        
        # Regular fetch loop
        while True:
            try:
                # Wait for the configured interval
                await asyncio.sleep(FETCH_INTERVAL)
                
                # Determine current accumulated count
                current_videos = self.get_accumulated_videos()
                current_count = len(current_videos)
                logger.info(f"Currently have {current_count} accumulated videos")
                
                # Only fetch new videos if we're below the maximum
                from config.settings import MAX_VIDEOS
                if current_count >= MAX_VIDEOS:
                    logger.info(f"Skipping video fetch - already at max capacity ({MAX_VIDEOS})")
                    continue
                
                # Alternate between different fetch methods to get diverse content
                now = datetime.now()
                minute = now.minute
                
                if minute % 15 == 0:  # Every 15 minutes, fetch news
                    logger.info("Fetching crypto news videos")
                    videos = await fetch_crypto_news(max_results=MAX_VIDEOS_PER_FETCH)
                elif minute % 10 == 0:  # Every 10 minutes, fetch trending
                    logger.info("Fetching trending crypto videos")
                    videos = await fetch_trending_videos(max_results=MAX_VIDEOS_PER_FETCH)
                else:
                    # Rotate through channels
                    # Use a random selection of channels to avoid hitting quota limits
                    import random
                    selected_channels = random.sample(CHANNELS, min(3, len(CHANNELS)))
                    
                    logger.info(f"Fetching videos from {len(selected_channels)} random channels")
                    videos = []
                    for channel in selected_channels:
                        channel_videos = await fetch_channel_videos(
                            channel, 
                            max_results=MAX_VIDEOS_PER_FETCH // len(selected_channels)
                        )
                        videos.extend(channel_videos)
                
                if videos:
                    # Process videos to calculate accurate VPH
                    processed_videos = [process_video_stats(video, now) for video in videos]
                    
                    # Filter by VPH threshold (with a minimum for diversity)
                    high_vph_videos = [v for v in processed_videos if v[3] >= VPH_THRESHOLD]
                    
                    # Include some lower VPH videos with minimum 5% weight impact
                    from config.settings import MIN_WEIGHT
                    lower_vph_videos = [v for v in processed_videos if VPH_THRESHOLD * MIN_WEIGHT <= v[3] < VPH_THRESHOLD]
                    
                    # Combine and limit
                    videos_to_add = high_vph_videos + lower_vph_videos[:max(2, MAX_VIDEOS_PER_FETCH // 4)]
                    
                    # Add to accumulated videos
                    if videos_to_add:
                        await self.add_videos(videos_to_add)
                        logger.info(f"Added {len(videos_to_add)} new videos: {len(high_vph_videos)} high VPH, {len(lower_vph_videos[:max(2, MAX_VIDEOS_PER_FETCH // 4)])} lower VPH")
                        logger.info(f"New accumulated total: {len(self.get_accumulated_videos())} videos")
                        
                        # Process transcripts for the new videos using the batch function
                        video_ids = [video[0] if isinstance(video, tuple) else video.id for video in videos_to_add]
                        
                        # Use the transcript service's batch process function
                        processed_transcripts = []
                        try:
                            processed_transcripts = await self.transcript_service.process_videos(videos_to_add, parallel=True)
                        except Exception as e:
                            logger.error(f"Error processing transcripts: {e}")
                            # Proceed with whatever transcripts we have
                            pass
                        
                        # Filter valid transcripts
                        new_video_transcripts = [
                            (video_id, transcript) for video_id, transcript in processed_transcripts
                            if transcript and len(transcript) > 100
                        ]
                        
                        # Analyze sentiment for new videos
                        if new_video_transcripts and self.crypto_trading_service and hasattr(self.crypto_trading_service, 'sentiment_service'):
                            logger.info(f"Analyzing sentiment for {len(new_video_transcripts)} new videos for quick decisions")
                            
                            # Get sentiment for new videos
                            new_sentiments = await self.crypto_trading_service.sentiment_service.batch_analyze(new_video_transcripts)
                            
                            # Get current global scores for comparison
                            previous_scores = self.crypto_trading_service.sentiment_history if self.crypto_trading_service else {}
                            
                            # Update global scores incrementally
                            updated_scores, sentiment_changes = await self.crypto_trading_service.sentiment_service.update_global_scores_incremental(
                                new_sentiments, videos_to_add, previous_scores
                            )
                            
                            # Let master agent make decisions with updated scores if we have significant changes
                            has_significant_changes = any(change >= 1.0 for change in sentiment_changes.values())
                            
                            # Enhanced logging for quick decision cycle timing
                            now = datetime.now()
                            logger.info(f"Quick decision cycle triggered at {now.isoformat()}")
                            logger.info(f"Significant changes: {has_significant_changes}, Running: {self.is_running}")
                            
                            # Log the sentiment changes for debugging
                            for coin, change in sentiment_changes.items():
                                logger.info(f"  {coin}: change of {change:.2f} from previous")
                            
                            if has_significant_changes and not self.is_running:
                                logger.info("Significant sentiment changes detected, triggering quick decisions")
                                
                                # Make quick trading decisions
                                result = await self.crypto_trading_service.make_quick_decisions(
                                    updated_scores, previous_scores, sentiment_changes
                                )
                                
                                logger.info(f"Quick decision result: {result.get('message', 'No result')}")
                                logger.info(f"Trades executed: {result.get('trades_executed', 0)}")
                            else:
                                if self.is_running:
                                    logger.info("Skipping quick decisions - a cycle is already running")
                                else:
                                    logger.info("No significant sentiment changes - skipping quick decisions")
                                    # Log the highest changes for debugging
                                    if sentiment_changes:
                                        max_change = max(sentiment_changes.values())
                                        max_coin = [c for c, v in sentiment_changes.items() if v == max_change][0]
                                        logger.info(f"  Highest change was {max_change:.2f} for {max_coin} (threshold: 1.0)")
                        
                    else:
                        logger.info("No matching videos found after VPH filtering")
                else:
                    logger.info("No new videos found in this fetch cycle")
                    
            except Exception as e:
                logger.error(f"Error in video fetch loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait a minute before retrying on error
    
    def export_data(self):
        """
        Export cached data to persistent storage.
        """
        # Export transcripts
        transcript_count = self.transcript_service.export_transcripts()
        logger.info(f"Exported {transcript_count} transcripts")
        
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the status of all services.
        
        Returns:
            Dictionary containing service status information
        """
        now = datetime.now()
        time_since_last = (now - self.last_cycle_time).total_seconds()
        
        return {
            "redis_available": self.redis_available,
            "is_running": self.is_running,
            "last_cycle_time": self.last_cycle_time.isoformat(),
            "time_since_last_cycle": f"{time_since_last:.1f} seconds",
            "next_cycle_in": f"{max(0, CYCLE_INTERVAL - time_since_last):.1f} seconds",
            "accumulated_videos_count": len(self.accumulated_videos)
        }

# Import OpenRouter API key
from config.settings import OPENROUTER_API_KEY

# Dependency for FastAPI
app_service = AppService(openrouter_api_key=OPENROUTER_API_KEY)

def get_app_service() -> AppService:
    """
    Dependency provider for the AppService.
    
    Returns:
        AppService instance
    """
    return app_service

# For direct imports (e.g., in API endpoints)
async def get_transcripts(video_ids: List[str], app_service: AppService = Depends(get_app_service)) -> Dict[str, str]:
    """
    Get transcripts for a list of video IDs.
    
    Args:
        video_ids: List of video IDs
        app_service: AppService instance
        
    Returns:
        Dictionary mapping video IDs to transcripts
    """
    result = {}
    for video_id in video_ids:
        transcript = app_service.transcript_service.get_transcript(video_id)
        if transcript:
            result[video_id] = transcript
    return result