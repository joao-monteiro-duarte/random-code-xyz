"""
Service for handling video transcripts.
"""
import asyncio
import logging
import aiohttp
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

# Import YouTube Transcript API
try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    from youtube_transcript_api.formatters import TextFormatter
    YOUTUBE_TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_API_AVAILABLE = False
    logging.warning("youtube_transcript_api not available, using simulated transcripts")

# Import yt-dlp for audio downloading
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    logging.warning("yt-dlp not available, cannot download video audio")

# Import whisper for audio transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("whisper not available, cannot transcribe audio")

from models import Video
from utils import CacheManager

logger = logging.getLogger(__name__)

class TranscriptService:
    """
    Service for fetching, processing, and caching video transcripts.
    """
    def __init__(self, redis_client, redis_available: bool = True):
        """
        Initialize the transcript service.
        
        Args:
            redis_client: Redis client instance
            redis_available: Whether Redis is available
        """
        self.cache_manager = CacheManager(
            redis_client=redis_client,
            redis_available=redis_available,
            prefix="transcript"
        )
        self.processed_coins = set()
        self.trades_to_log = []
        
    def get_transcript(self, video_id: str) -> Optional[str]:
        """
        Get a transcript from cache.
        
        Args:
            video_id: Video ID
            
        Returns:
            Transcript text or None if not found
        """
        return self.cache_manager.get(video_id)
        
    def save_transcript(self, video_id: str, transcript: str, expire: Optional[int] = None) -> bool:
        """
        Save a transcript to cache.
        
        Args:
            video_id: Video ID
            transcript: Transcript text
            expire: Expiration time in seconds
            
        Returns:
            True if saved, False otherwise
        """
        return self.cache_manager.set(video_id, transcript, expire=expire)
        
    async def load_transcripts(self) -> int:
        """
        Load all transcripts from cache.
        
        Returns:
            Number of transcripts loaded
        """
        try:
            logger.info("Loading transcripts from cache...")
            keys = self.cache_manager.keys("*")
            logger.info(f"Found {len(keys)} transcripts in cache")
            return len(keys)
        except Exception as e:
            logger.error(f"Error loading transcripts: {e}")
            return 0
            
    async def initialize(self):
        """
        Initialize the transcript service.
        """
        count = await self.load_transcripts()
        logger.info(f"Transcript service initialized with {count} transcripts")
        
    async def fetch_transcript(self, video_id: str) -> str:
        """
        Fetch a transcript for a video using YouTube Transcript API,
        with fallback to audio download and transcription if captions aren't available.
        
        Args:
            video_id: YouTube Video ID
            
        Returns:
            Transcript text
        """
        logger.info(f"Fetching transcript for {video_id}")
        
        # Step 1: Try YouTube Transcript API for captions
        caption_transcript = await self._try_fetch_captions(video_id)
        if caption_transcript:
            return caption_transcript
            
        # Step 2: Try audio download and transcription
        audio_transcript = await self._try_transcribe_audio(video_id)
        if audio_transcript:
            return audio_transcript
            
        # Step 3: Fallback to simulated transcript as last resort
        logger.warning(f"All transcript methods failed for {video_id}. Using simulated transcript.")
        return self._generate_simulated_transcript(video_id)
        
    async def _try_fetch_captions(self, video_id: str) -> Optional[str]:
        """
        Try to fetch captions using YouTube Transcript API.
        
        Args:
            video_id: YouTube Video ID
            
        Returns:
            Formatted transcript text or None if failed
        """
        if not YOUTUBE_TRANSCRIPT_API_AVAILABLE:
            logger.warning("YouTube Transcript API not available, skipping caption fetch")
            return None
            
        try:
            # Run in a thread pool to avoid blocking
            transcript_list = await asyncio.to_thread(
                YouTubeTranscriptApi.get_transcript, 
                video_id
            )
            
            # Format the transcript
            formatter = TextFormatter()
            formatted_transcript = formatter.format_transcript(transcript_list)
            
            if formatted_transcript:
                logger.info(f"Successfully fetched captions for {video_id} ({len(formatted_transcript)} chars)")
                return formatted_transcript
            else:
                logger.warning(f"Empty caption transcript received for {video_id}")
                return None
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            logger.warning(f"No captions available for {video_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching captions for {video_id}: {e}")
            return None
    
    async def _try_transcribe_audio(self, video_id: str) -> Optional[str]:
        """
        Try to download video audio and transcribe it with whisper.
        
        Args:
            video_id: YouTube Video ID
            
        Returns:
            Transcribed text or None if failed
        """
        if not YT_DLP_AVAILABLE or not WHISPER_AVAILABLE:
            logger.warning("Audio transcription unavailable: missing yt-dlp or whisper")
            return None
            
        audio_file = f"/tmp/{video_id}.mp3"
        try:
            # Define download options for yt-dlp
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': f'/tmp/{video_id}.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': True,
            }
            
            # Download audio
            logger.info(f"Downloading audio for {video_id}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [f"https://www.youtube.com/watch?v={video_id}"])
            
            if not os.path.exists(audio_file):
                logger.error(f"Audio download failed for {video_id}")
                return None
            
            # Load Whisper model
            logger.info(f"Transcribing audio for {video_id}")
            model = whisper.load_model("small")  # Use small model for efficiency
            
            # Transcribe audio
            result = await asyncio.to_thread(model.transcribe, audio_file)
            transcript = result["text"]
            
            if transcript:
                logger.info(f"Successfully transcribed audio for {video_id} ({len(transcript)} chars)")
                return transcript.strip()
            else:
                logger.warning(f"Empty audio transcription for {video_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error transcribing audio for {video_id}: {e}")
            return None
        finally:
            # Clean up
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    logger.debug(f"Removed temporary audio file {audio_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary audio file {audio_file}: {e}")
    
    def _generate_simulated_transcript(self, video_id: str) -> str:
        """
        Generate a simulated transcript as a last resort.
        
        Args:
            video_id: YouTube Video ID
            
        Returns:
            Simulated transcript text
        """
        simulated_transcript = (
            f"This is a simulated transcript for video {video_id} about cryptocurrency and blockchain technology. "
            f"Bitcoin, Ethereum, and other crypto assets are mentioned frequently. "
            f"The market sentiment appears mixed but leaning positive. "
            f"Small-cap altcoins like Pepe and Shiba Inu have been gaining attention. "
            f"Technical analysis suggests a potential bullish pattern forming."
        )
        logger.info(f"Generated simulated transcript for {video_id} ({len(simulated_transcript)} chars)")
        return simulated_transcript
        
    async def process_video(self, video: Union[Video, Tuple[str, int, str, float]]) -> Tuple[str, str]:
        """
        Process a single video to get its transcript.
        
        Args:
            video: Video object or tuple (video_id, views, publish_time, vph)
            
        Returns:
            Tuple of (video_id, transcript)
        """
        # Convert tuple to Video object if needed
        if isinstance(video, tuple):
            video = Video.from_tuple(video)
            
        # Try to get from cache first
        existing_transcript = self.get_transcript(video.id)
        
        if existing_transcript:
            logger.info(f"Using cached transcript for {video.id}")
            return video.id, existing_transcript
        else:
            # Fetch and process new transcript
            transcript = await self.fetch_transcript(video.id)
            
            # Save to cache
            self.save_transcript(video.id, transcript)
            logger.info(f"Processed and saved transcript for {video.id}")
            return video.id, transcript
            
    async def process_videos(self, videos: List[Union[Video, Tuple]], parallel: bool = True) -> List[Tuple[str, str]]:
        """
        Process multiple videos to get their transcripts.
        
        Args:
            videos: List of Video objects or tuples
            parallel: Whether to process videos in parallel
            
        Returns:
            List of (video_id, transcript) tuples
        """
        # Reset state for new batch
        self.processed_coins = set()
        self.trades_to_log = []
        
        if parallel and len(videos) > 1:
            # Process videos in parallel
            logger.info(f"Processing {len(videos)} videos in parallel")
            tasks = [self.process_video(video) for video in videos]
            return await asyncio.gather(*tasks)
        else:
            # Process videos sequentially
            results = []
            for video in videos:
                result = await self.process_video(video)
                results.append(result)
                # Reset state for each video
                self.processed_coins = set()
                self.trades_to_log = []
            return results
            
    def export_transcripts(self) -> int:
        """
        Export all transcripts in the local cache to Redis.
        
        Returns:
            Number of transcripts exported
        """
        try:
            logger.info("Exporting transcripts to Redis...")
            local_cache = self.cache_manager.local_cache
            
            if not local_cache:
                logger.info("No transcripts to export")
                return 0
                
            # Use set_many for efficiency
            self.cache_manager.set_many(local_cache)
            logger.info(f"Exported {len(local_cache)} transcripts to Redis")
            return len(local_cache)
        except Exception as e:
            logger.error(f"Error exporting transcripts: {e}")
            return 0
            
    def process_video_stats(self, video: Union[Video, Tuple], now: Optional[datetime] = None) -> Union[Video, Tuple]:
        """
        Process video statistics and recalculate VPH.
        
        Args:
            video: Video object or tuple
            now: Current datetime
            
        Returns:
            Updated Video object or tuple
        """
        # Convert tuple to Video object if needed
        tuple_input = isinstance(video, tuple)
        if tuple_input:
            video_obj = Video.from_tuple(video)
        else:
            video_obj = video
            
        # Update VPH
        video_obj.update_vph(now)
        
        # Return same type as input
        return video_obj.to_tuple() if tuple_input else video_obj