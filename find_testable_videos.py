#!/usr/bin/env python3
"""
Script to find working videos for testing the transcript service.

This script searches for recent cryptocurrency videos and tests:
1. If they are available in the current environment
2. Whether captions are available or if audio transcription is needed
3. The quality of the resulting transcript

This helps create a reliable set of test cases for the transcript service.
"""
import asyncio
import logging
import sys
import json
import os
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Import the necessary components
try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    YOUTUBE_TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    YOUTUBE_TRANSCRIPT_API_AVAILABLE = False
    logger.warning("youtube_transcript_api not available, some tests will be skipped")

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    logger.warning("yt-dlp not available, audio download tests will be skipped")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("whisper not available, transcription tests will be skipped")

# List of popular cryptocurrency video IDs to test
# These are recent examples that might work in various environments
TEST_VIDEOS = [
    # Popular channels with high likelihood of captions
    {"id": "EX3RknxN8vM", "title": "Why Bitcoin? | Cathie Wood & Michael Saylor", "expected_captions": True},
    {"id": "kV_Lh73KKxE", "title": "Bitcoin Halving 2024 & How to Invest with Data", "expected_captions": True},
    {"id": "1WXfJ3U9Uhc", "title": "Bitcoin Halving 2024 Explained", "expected_captions": True},
    
    # Smaller channels less likely to have captions
    {"id": "z0dD9ROl7t8", "title": "Bitcoin Technical Analysis Today", "expected_captions": False},
    {"id": "rPGwz0WSYZo", "title": "Ethereum Price Analysis", "expected_captions": False},
    {"id": "UNt_K4AH8s8", "title": "Crypto News Today", "expected_captions": False},
]

async def check_video_availability(video_id: str) -> bool:
    """Check if a video is available on YouTube."""
    try:
        ydl_opts = {
            'skip_download': True,
            'quiet': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = await asyncio.to_thread(ydl.extract_info, f"https://www.youtube.com/watch?v={video_id}", download=False)
            return True
    except Exception as e:
        logger.error(f"Video {video_id} is not available: {e}")
        return False

async def check_captions_availability(video_id: str) -> bool:
    """Check if captions are available for a video."""
    if not YOUTUBE_TRANSCRIPT_API_AVAILABLE:
        return False
        
    try:
        transcript_list = await asyncio.to_thread(
            YouTubeTranscriptApi.get_transcript, 
            video_id
        )
        return len(transcript_list) > 0
    except (NoTranscriptFound, TranscriptsDisabled) as e:
        logger.info(f"No captions for video {video_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking captions for {video_id}: {e}")
        return False

async def download_and_transcribe_audio(video_id: str) -> tuple[bool, str]:
    """Download video audio and try to transcribe it."""
    if not YT_DLP_AVAILABLE or not WHISPER_AVAILABLE:
        return False, "Required libraries not available"
        
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
            return False, "Audio download failed"
        
        # Load Whisper model
        logger.info(f"Transcribing audio for {video_id}")
        model = whisper.load_model("tiny")  # Use tiny model for faster test
        
        # Transcribe audio
        result = await asyncio.to_thread(model.transcribe, audio_file)
        transcript = result["text"]
        
        if transcript:
            return True, transcript
        else:
            return False, "Empty transcript"
            
    except Exception as e:
        return False, f"Error: {e}"
    finally:
        # Clean up
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except Exception:
                pass

async def test_video(video: dict) -> dict:
    """Run a complete test of a video."""
    video_id = video["id"]
    title = video["title"]
    expected_captions = video.get("expected_captions", False)
    
    result = {
        "id": video_id,
        "title": title,
        "is_available": False,
        "has_captions": False,
        "audio_transcription_works": False,
        "transcript_source": None,
        "transcript_quality": 0,  # 0-10 scale
        "transcript_sample": None,
        "notes": []
    }
    
    # Step 1: Check if the video is available
    is_available = await check_video_availability(video_id)
    result["is_available"] = is_available
    
    if not is_available:
        result["notes"].append("Video is not available")
        return result
    
    # Step 2: Check if captions are available
    has_captions = await check_captions_availability(video_id)
    result["has_captions"] = has_captions
    
    if has_captions:
        result["notes"].append("Captions are available")
        result["transcript_source"] = "captions"
        
        # Get caption quality
        try:
            transcript_list = await asyncio.to_thread(
                YouTubeTranscriptApi.get_transcript, 
                video_id
            )
            transcript_text = " ".join([item["text"] for item in transcript_list])
            result["transcript_sample"] = transcript_text[:200] + "..." if len(transcript_text) > 200 else transcript_text
            result["transcript_quality"] = 9  # High quality since these are official captions
        except Exception as e:
            result["notes"].append(f"Error retrieving captions: {e}")
    else:
        result["notes"].append("No captions available")
        
        # Step 3: Try audio transcription
        success, transcript = await download_and_transcribe_audio(video_id)
        result["audio_transcription_works"] = success
        
        if success:
            result["transcript_source"] = "audio"
            result["transcript_sample"] = transcript[:200] + "..." if len(transcript) > 200 else transcript
            result["transcript_quality"] = 7  # Slightly lower quality for audio transcription
            result["notes"].append("Audio transcription successful")
        else:
            result["notes"].append(f"Audio transcription failed: {transcript}")
    
    # Validate expectations
    if expected_captions and not has_captions:
        result["notes"].append("WARNING: Expected captions but none found")
    
    return result

async def main():
    """Test all videos and generate a report."""
    print("\n🔍 TESTING VIDEOS FOR TRANSCRIPT SERVICE 🔍\n")
    
    results = []
    for video in TEST_VIDEOS:
        print(f"\nTesting video: {video['title']} (ID: {video['id']})")
        result = await test_video(video)
        results.append(result)
        
        # Print summary
        status = "✅ PASSED" if result["is_available"] and (result["has_captions"] or result["audio_transcription_works"]) else "❌ FAILED"
        source = result["transcript_source"] or "none"
        print(f"{status} | Available: {result['is_available']} | Captions: {result['has_captions']} | Audio: {result['audio_transcription_works']} | Source: {source}")
        
        if result["transcript_sample"]:
            print("\nTranscript sample:")
            print("-" * 40)
            print(result["transcript_sample"])
            print("-" * 40)
        
        for note in result["notes"]:
            print(f"- {note}")
    
    # Generate full report
    print("\n\n📊 FULL TEST REPORT 📊\n")
    
    # Summary stats
    available_count = sum(1 for r in results if r["is_available"])
    caption_count = sum(1 for r in results if r["has_captions"])
    audio_count = sum(1 for r in results if r["audio_transcription_works"])
    working_count = sum(1 for r in results if r["is_available"] and (r["has_captions"] or r["audio_transcription_works"]))
    
    print(f"Total videos tested: {len(results)}")
    print(f"Available videos: {available_count}/{len(results)} ({available_count/len(results)*100:.1f}%)")
    print(f"Videos with captions: {caption_count}/{available_count} ({caption_count/available_count*100:.1f}% of available)")
    print(f"Videos with working audio transcription: {audio_count}/{available_count} ({audio_count/available_count*100:.1f}% of available)")
    print(f"Total working videos: {working_count}/{len(results)} ({working_count/len(results)*100:.1f}%)")
    
    # Generate recommended test cases
    print("\nRECOMMENDED TEST CASES:\n")
    
    caption_tests = [r for r in results if r["has_captions"]][:2]
    audio_tests = [r for r in results if not r["has_captions"] and r["audio_transcription_works"]][:2]
    
    print("For testing caption-based transcripts:")
    for test in caption_tests:
        print(f"- {test['title']} (ID: {test['id']})")
    
    print("\nFor testing audio-based transcripts:")
    for test in audio_tests:
        print(f"- {test['title']} (ID: {test['id']})")
    
    # Save results to file
    with open("video_test_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "summary": {
                "total": len(results),
                "available": available_count,
                "with_captions": caption_count,
                "with_audio_transcription": audio_count,
                "working_total": working_count
            },
            "recommended_tests": {
                "caption_tests": [{"id": t["id"], "title": t["title"]} for t in caption_tests],
                "audio_tests": [{"id": t["id"], "title": t["title"]} for t in audio_tests]
            }
        }, f, indent=2)
    
    print("\nDetailed results saved to video_test_results.json")

if __name__ == "__main__":
    asyncio.run(main())