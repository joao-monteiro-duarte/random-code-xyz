# transcript_service.py
import asyncio
import logging
import os
from typing import Optional, List, Tuple
import yt_dlp
import whisper
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

logger = logging.getLogger(__name__)

class TranscriptService:
    def __init__(self):
        self.transcription_timeout = 300
        self.transcription_queue = asyncio.Queue()
        self.missing_transcripts = 0
        self.live_segment_duration = 300  # 10 minutes in seconds (adjust to 300 for 5 minutes if preferred)

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

    async def extract_last_segment_audio(self, video_id: str) -> Optional[str]:
        """
        Extract the last 5-10 minutes of audio from a live stream using yt-dlp and ffmpeg.
        """
        audio_file = f"/tmp/{video_id}_last_segment.mp3"
        temp_file = f"/tmp/{video_id}_temp.mp3"
        logger.info(f"Extracting last {self.live_segment_duration} seconds of audio for live stream {video_id}")
        try:
            # Step 1: Download the full audio stream (yt-dlp will handle live streams)
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_file.replace('.mp3', '.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': False,
                'verbose': True,
                'no_warnings': False,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            }

            logger.info(f"Downloading audio for {video_id} with options: {ydl_opts}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [f"https://www.youtube.com/watch?v={video_id}"])

            if not os.path.exists(temp_file):
                logger.error(f"Audio file not found after download for {video_id}: {temp_file}")
                return None

            # Step 2: Use ffmpeg to extract the last 5-10 minutes
            import subprocess
            ffmpeg_cmd = [
                'ffmpeg', '-i', temp_file, '-ss', f'-{self.live_segment_duration}',
                '-c', 'copy', audio_file
            ]
            logger.info(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
            process = await asyncio.create_subprocess_exec(*ffmpeg_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"ffmpeg failed for {video_id}: {stderr.decode()}")
                return None

            if not os.path.exists(audio_file):
                logger.error(f"Segmented audio file not found for {video_id}: {audio_file}")
                return None

            # Step 3: Transcribe the segmented audio
            logger.info(f"Transcribing last segment audio for {video_id}")
            model = whisper.load_model("small")
            task = asyncio.to_thread(model.transcribe, audio_file)
            transcript = await asyncio.wait_for(task, timeout=self.transcription_timeout)

            if transcript["text"]:
                logger.info(f"Successfully transcribed last segment for {video_id} ({len(transcript['text'])} chars)")
                return transcript["text"]
            else:
                logger.warning(f"Empty transcription for last segment of {video_id}")
                return None
        except asyncio.TimeoutError:
            logger.warning(f"Transcription timeout for last segment of {video_id}")
            return None
        except Exception as e:
            logger.error(f"Error extracting/transcribing last segment for {video_id}: {e}")
            return None
        finally:
            for file in [temp_file, audio_file]:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                        logger.debug(f"Removed temporary file {file}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {file}: {e}")

    async def fetch_transcript(self, video_id: str, is_live: str) -> Optional[str]:
        logger.info(f"Fetching transcript for {video_id} (live_status: {is_live})")
        # Try captions first for all videos
        transcript = await self._try_fetch_captions(video_id)
        if transcript:
            return transcript

        # If no captions and it's a live stream, extract the last 5-10 minutes
        if is_live == "live":
            return await self.extract_last_segment_audio(video_id)
        
        # For non-live videos or ended live streams, transcribe the full video (as before)
        return await self.transcribe_in_background(video_id)

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

    async def transcribe_in_background(self, video_id: str) -> Optional[str]:
        audio_file = f"/tmp/{video_id}.mp3"
        logger.info(f"Starting background transcription for {video_id}")
        try:
            # transcript_service.py
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': f'/tmp/{video_id}.%(ext)s',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'quiet': False,
                'verbose': True,
                'no_warnings': False,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'cookies': 'cookies.txt',  # Add a cookies file (optional, see below)
                'geo_bypass': True,  # Bypass geographic restrictions
            }

            logger.info(f"Downloading audio for {video_id} with options: {ydl_opts}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                await asyncio.to_thread(ydl.download, [f"https://www.youtube.com/watch?v={video_id}"])

            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found after download for {video_id}: {audio_file}")
                return None

            logger.info(f"Transcribing audio for {video_id}")
            model = whisper.load_model("small")
            task = asyncio.to_thread(model.transcribe, audio_file)
            transcript = await asyncio.wait_for(task, timeout=self.transcription_timeout)

            if transcript["text"]:
                logger.info(f"Successfully transcribed audio for {video_id} ({len(transcript['text'])} chars)")
                return transcript["text"]
            else:
                logger.warning(f"Empty audio transcription for {video_id}")
                return None
        except asyncio.TimeoutError:
            logger.warning(f"Transcription timeout for {video_id}")
            return None
        except Exception as e:
            logger.error(f"Error transcribing audio for {video_id}: {e}")
            return None
        finally:
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    logger.debug(f"Removed temporary audio file {audio_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary audio file {audio_file}: {e}")