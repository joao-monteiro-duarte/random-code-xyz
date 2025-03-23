#!/usr/bin/env python3
"""
Simple test script for audio-based transcript retrieval.
This script tests the transcript service's ability to download audio and transcribe it 
when captions are not available.
"""
import asyncio
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Import the necessary components directly without circular dependencies
import yt_dlp
import whisper
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

async def download_and_transcribe_audio(video_id: str):
    """
    Download video audio and transcribe it with Whisper.
    This is a simplified version of the transcript service implementation.
    """
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
        model = whisper.load_model("tiny")  # Use tiny model for faster processing in this test
        
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

async def try_get_captions(video_id: str):
    """Try to get captions using YouTube Transcript API"""
    try:
        transcript_list = await asyncio.to_thread(
            YouTubeTranscriptApi.get_transcript, 
            video_id
        )
        return "Captions available"
    except (NoTranscriptFound, TranscriptsDisabled) as e:
        return f"No captions: {e}"
    except Exception as e:
        return f"Error: {e}"

async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_transcript_audio.py <youtube_video_id>")
        return
    
    video_id = sys.argv[1]
    print(f"Testing transcript acquisition for video ID: {video_id}")
    
    # First check if captions are available
    caption_status = await try_get_captions(video_id)
    print(f"Caption status: {caption_status}")
    
    # Then try to transcribe the audio
    transcript = await download_and_transcribe_audio(video_id)
    
    if transcript:
        print(f"\nSuccessfully transcribed audio! ({len(transcript)} characters)")
        print("\nSample transcript:")
        print("-----------------")
        print(transcript[:500] + "..." if len(transcript) > 500 else transcript)
        print("-----------------")
    else:
        print("\nFailed to transcribe audio.")

if __name__ == "__main__":
    asyncio.run(main())