#!/usr/bin/env python3
"""
Simplified test for just the comprehensive cycle aspect of the dual-cycle framework.
Focuses on identifying and fixing the specific issue with accumulated_videos handling.
"""
import asyncio
import logging
import json
import os
import sys
from datetime import datetime, timedelta

# Set up logging to console only for clarity
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("test_comprehensive")
logger.setLevel(logging.INFO)

async def test_comprehensive_cycle():
    """Test the comprehensive cycle with proper Video objects"""
    
    # Import services and models
    print("Importing necessary modules...")
    from services.app_service import AppService, get_app_service
    from services.crypto_trading_service import CryptoTradingService
    from models.video import Video
    
    # Create services
    print("Creating services...")
    app_service = get_app_service()
    trading_service = CryptoTradingService(test_mode=True)
    
    # Initialize services
    print("Initializing services...")
    await app_service.initialize()
    await trading_service.initialize()
    
    # Test comprehensive cycle
    print("\n=== Testing comprehensive cycle ===")
    
    # Create test videos as proper Video objects
    print("Creating test video objects...")
    test_videos = [
        Video.from_tuple(("dQw4w9WgXcQ", 50000, "2023-01-01T00:00:00Z", 1000.0)),  # High VPH video
        Video.from_tuple(("xvFZjo5PgG0", 25000, "2023-01-02T00:00:00Z", 750.0))    # Medium VPH video
    ]
    
    # Verify the Video objects are created correctly
    for i, video in enumerate(test_videos):
        print(f"Video {i+1}: Type={type(video).__name__}, ID={video.id}, VPH={video.vph}")
    
    # Set accumulated videos in app service
    print("\nSetting accumulated_videos in app_service...")
    app_service.accumulated_videos = test_videos
    
    # Verify the accumulated_videos were set properly
    print(f"Checking accumulated_videos in app_service: {len(app_service.accumulated_videos)} videos")
    for i, video in enumerate(app_service.accumulated_videos):
        print(f"Video {i+1} in accumulated_videos: Type={type(video).__name__}, ID={video.id if hasattr(video, 'id') else 'unknown'}")
    
    # Create mock transcripts
    print("\nCreating mock transcripts...")
    for video in test_videos:
        mock_transcript = f"""
        Today we're discussing cryptocurrencies. Bitcoin is showing strong performance,
        with potential for significant growth this year. Ethereum is also performing well
        with its recent updates. Among smaller coins, Solana has impressive technology and
        adoption. I'm particularly bullish on Pepe coin, which has massive social media presence
        and could see significant price action in the coming weeks.
        """
        app_service.transcript_service.save_transcript(video.id, mock_transcript)
        print(f"Saved mock transcript for video {video.id}")
    
    # Run comprehensive cycle with explicit debugging at critical points
    from config.settings import VPH_THRESHOLD
    print(f"\nRunning comprehensive cycle with VPH threshold: {VPH_THRESHOLD}")
    
    try:
        # Add debugging for crypto_trading_service.run_cycle
        print("Calling trading_service.run_cycle()...")
        comprehensive_result = await trading_service.run_cycle(vph_threshold=VPH_THRESHOLD, background=False)
        print(f"Comprehensive cycle result: {json.dumps(comprehensive_result, indent=2)}")
        return True
    except Exception as e:
        print(f"ERROR in comprehensive cycle: {str(e)}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    print("Starting comprehensive cycle test...\n")
    success = asyncio.run(test_comprehensive_cycle())
    
    # Print final result
    print("\n=== Test Result ===")
    if success:
        print("Comprehensive cycle test: SUCCESS")
        sys.exit(0)
    else:
        print("Comprehensive cycle test: FAILED")
        sys.exit(1)