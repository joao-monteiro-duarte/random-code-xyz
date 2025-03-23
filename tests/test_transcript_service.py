#!/usr/bin/env python3
"""
Test script for the enhanced transcript service with audio fallback
"""
import asyncio
import logging
import sys
from services.transcript_service import TranscriptService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

async def test_transcript_service():
    """Test the transcript service with different scenarios"""
    
    # Initialize the transcript service without Redis for simplicity
    transcript_service = TranscriptService(redis_client=None, redis_available=False)
    
    # Test case 1: Video with captions available
    # Example: Recent crypto video from a popular channel
    video_with_captions = "rPGwz0WSYZo"  # "Ethereum Price Analysis $3,000 ?"
    
    # Test case 2: Video without captions (will need audio fallback)
    # Example: Small crypto channel video that might not have captions
    video_without_captions = "AHtaxBV3mLc"  # "Crypto updates for March"
    
    # Test the service
    print("\n----- Testing transcript service with multi-stage fallback -----")
    
    # Test case 1: Video with captions
    print(f"\nTEST CASE 1: Video with captions (ID: {video_with_captions})")
    transcript1 = await transcript_service.fetch_transcript(video_with_captions)
    
    if transcript1:
        print(f"✅ Successfully retrieved transcript ({len(transcript1)} chars)")
        print(f"Sample: {transcript1[:200]}...")
    else:
        print("❌ Failed to retrieve transcript")
    
    # Test case 2: Video without captions (should use audio fallback)
    print(f"\nTEST CASE 2: Video without captions (ID: {video_without_captions})")
    transcript2 = await transcript_service.fetch_transcript(video_without_captions)
    
    if transcript2:
        print(f"✅ Successfully retrieved transcript ({len(transcript2)} chars)")
        print(f"Sample: {transcript2[:200]}...")
    else:
        print("❌ Failed to retrieve transcript")
    
    # Custom video ID input
    if len(sys.argv) > 1:
        custom_video_id = sys.argv[1]
        print(f"\nTEST CASE 3: Custom video ID (ID: {custom_video_id})")
        transcript3 = await transcript_service.fetch_transcript(custom_video_id)
        
        if transcript3:
            print(f"✅ Successfully retrieved transcript ({len(transcript3)} chars)")
            print(f"Sample: {transcript3[:200]}...")
        else:
            print("❌ Failed to retrieve transcript")

if __name__ == "__main__":
    asyncio.run(test_transcript_service())