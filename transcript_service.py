import asyncio
import logging
from typing import Optional, List, Tuple
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

logger = logging.getLogger(__name__)

class TranscriptService:
    def __init__(self):
        self.missing_transcripts = 0

    async def _try_fetch_captions(self, video_id: str) -> Optional[str]:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'es', 'pt', 'en-US', 'en-GB'])
            transcript = " ".join([entry["text"] for entry in transcript_list])
            logger.info(f"Fetched captions for {video_id} ({len(transcript)} chars)")
            return transcript
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            logger.debug(f"No captions for {video_id}: {str(e)}")
            self.missing_transcripts += 1
            return None
        except Exception as e:
            logger.error(f"Error fetching captions for {video_id}: {e}")
            self.missing_transcripts += 1
            return None

    async def fetch_transcript(self, video_id: str, is_live: str) -> Optional[str]:
        logger.info(f"Fetching transcript for {video_id} (live_status: {is_live})")
        # Only try captions; skip if not available
        transcript = await self._try_fetch_captions(video_id)
        if not transcript:
            logger.info(f"Skipping {video_id} due to no available captions")
        return transcript

    async def process_videos(self, videos: List[Tuple]) -> List[Tuple[str, str]]:
        logger.info(f"Processing {len(videos)} videos")
        self.missing_transcripts = 0
        transcripts = []
        for video in videos:
            video_id = video[0]
            is_live = video[5]  # New element in the tuple
            transcript = await self.fetch_transcript(video_id, is_live)
            if transcript:
                transcripts.append((video_id, transcript))
        logger.info(f"Processed {len(transcripts)} videos with transcripts, {self.missing_transcripts} missing")
        return transcripts