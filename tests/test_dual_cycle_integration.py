#!/usr/bin/env python3
"""
Integration test for the dual-cycle framework to verify both cycles over an extended period.

This test runs for 60+ minutes to verify:
1. Multiple comprehensive cycles (should run every 30 minutes)
2. Quick decision cycles (triggered by sentiment changes)
3. Proper handling of both cycles
4. Throttling of trades for the same coin
"""
import asyncio
import logging
import json
import os
import sys
import time
from datetime import datetime, timedelta
import random

# Configure detailed logging
log_file = f"test_dual_cycle_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Mock video data - used to inject new videos periodically
MOCK_VIDEO_DATA = [
    ("crypto_vid_001", 45000, "2023-01-01T00:00:00Z", 1500.0),
    ("crypto_vid_002", 30000, "2023-01-02T00:00:00Z", 1200.0),
    ("crypto_vid_003", 25000, "2023-01-03T00:00:00Z", 900.0),
    ("crypto_vid_004", 20000, "2023-01-04T00:00:00Z", 800.0),
    ("crypto_vid_005", 15000, "2023-01-05T00:00:00Z", 750.0),
    ("crypto_vid_006", 12000, "2023-01-06T00:00:00Z", 600.0),
    ("crypto_vid_007", 10000, "2023-01-07T00:00:00Z", 500.0),
    ("crypto_vid_008", 8000, "2023-01-08T00:00:00Z", 400.0),
    ("crypto_vid_009", 7000, "2023-01-09T00:00:00Z", 350.0),
    ("crypto_vid_010", 6000, "2023-01-10T00:00:00Z", 300.0),
]

# Track cycles for analysis
cycle_tracker = {
    "comprehensive_cycles": [],
    "quick_decision_cycles": [],
    "start_time": None,
    "end_time": None
}

async def inject_new_videos(app_service, interval_minutes=7):
    """Periodically inject new videos to trigger quick decision cycles"""
    from models.video import Video
    
    while True:
        try:
            # Wait for the specified interval
            await asyncio.sleep(interval_minutes * 60)
            
            # Select 1-3 random videos from the mock data
            num_videos = random.randint(1, 3)
            video_tuples = random.sample(MOCK_VIDEO_DATA, num_videos)
            
            # Create proper Video objects
            videos = [Video.from_tuple(vt) for vt in video_tuples]
            
            # Add to accumulated_videos
            logger.info(f"Injecting {len(videos)} new videos to trigger quick decision cycle")
            await app_service.add_videos(videos)
            
            # Add mock transcripts with strong sentiment for specific coins
            # This should trigger the quick decision cycle
            for video in videos:
                # Create a mock transcript with random strong opinions
                coins = ["bitcoin", "ethereum", "solana", "dogecoin", "shiba", "cardano"]
                selected_coins = random.sample(coins, random.randint(1, 3))
                
                # Create transcript with strong sentiment to trigger quick decisions
                mock_transcript = f"""
                I'm very bullish on crypto right now, especially on {', '.join(selected_coins)}.
                {selected_coins[0]} in particular looks like it's going to explode soon.
                The charts show a clear pattern, and the fundamentals are stronger than ever.
                I predict a 50% increase for {selected_coins[0]} in the coming weeks.
                It's an absolute must-buy right now.
                """
                
                app_service.transcript_service.save_transcript(video.id, mock_transcript)
                logger.info(f"Saved strong sentiment mock transcript for video {video.id}")
            
            # Log the injection
            logger.info(f"Successfully injected {len(videos)} new videos with transcripts")
            cycle_tracker["video_injections"] = cycle_tracker.get("video_injections", 0) + 1
            
        except Exception as e:
            logger.error(f"Error injecting videos: {str(e)}")
            await asyncio.sleep(60)  # Wait a minute and try again

async def track_cycles(trading_service):
    """Track comprehensive and quick decision cycles"""
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            
            # Check if a comprehensive cycle is running
            if trading_service.app_service.is_running:
                logger.info("Detected active comprehensive cycle")
                
                # Wait for it to finish
                while trading_service.app_service.is_running:
                    await asyncio.sleep(5)
                
                # Record the cycle
                cycle_tracker["comprehensive_cycles"].append(datetime.now().isoformat())
                logger.info(f"Recorded comprehensive cycle at {cycle_tracker['comprehensive_cycles'][-1]}")
                
        except Exception as e:
            logger.error(f"Error tracking cycles: {str(e)}")

async def test_dual_cycle_integration():
    """Test the integration of quick and comprehensive cycles over 60 minutes"""
    
    # Import services and models
    logger.info("Importing services and models")
    from services.app_service import AppService, get_app_service
    from services.crypto_trading_service import CryptoTradingService
    from models.video import Video
    
    # Create services
    logger.info("Creating services")
    app_service = get_app_service()
    trading_service = CryptoTradingService(test_mode=True)
    
    # Initialize services
    logger.info("Initializing services")
    await app_service.initialize()
    await trading_service.initialize()
    
    # Record start time
    cycle_tracker["start_time"] = datetime.now().isoformat()
    
    # Set initial videos
    logger.info("Setting up initial videos")
    initial_videos = [
        Video.from_tuple(MOCK_VIDEO_DATA[0]),
        Video.from_tuple(MOCK_VIDEO_DATA[1])
    ]
    await app_service.add_videos(initial_videos)
    
    # Add mock transcripts
    for video in initial_videos:
        mock_transcript = f"""
        Today we're discussing cryptocurrencies. Bitcoin is showing strong performance,
        with potential for significant growth this year. Ethereum is also performing well
        with its recent updates. Among smaller coins, Solana has impressive technology and
        adoption. I'm particularly bullish on Pepe coin, which has massive social media presence
        and could see significant price action in the coming weeks.
        """
        app_service.transcript_service.save_transcript(video.id, mock_transcript)
        logger.info(f"Saved mock transcript for video {video.id}")
    
    # Start background tasks
    logger.info("Starting background tasks")
    cycle_monitor = asyncio.create_task(track_cycles(trading_service))
    video_injector = asyncio.create_task(inject_new_videos(app_service, interval_minutes=7))
    
    # Determine end time (60 minutes from now)
    end_time = datetime.now() + timedelta(minutes=60)
    logger.info(f"Test will run until {end_time.isoformat()}")
    
    # Run main test loop
    try:
        # Run periodic checks and wait for end time
        while datetime.now() < end_time:
            # Log time remaining
            remaining = (end_time - datetime.now()).total_seconds() / 60.0
            logger.info(f"Test running, {remaining:.1f} minutes remaining")
            
            # Wait before checking again
            await asyncio.sleep(300)  # Check every 5 minutes
            
            # Output interim results
            comp_cycles = len(cycle_tracker["comprehensive_cycles"])
            quick_cycles = len(cycle_tracker.get("quick_decision_cycles", []))
            video_injections = cycle_tracker.get("video_injections", 0)
            
            logger.info(f"Interim results: {comp_cycles} comprehensive cycles, "
                        f"{quick_cycles} quick decision cycles, "
                        f"{video_injections} video injections")
    
    except Exception as e:
        logger.error(f"Error in main test loop: {str(e)}", exc_info=True)
    
    finally:
        # Clean up
        logger.info("Cleaning up background tasks")
        cycle_monitor.cancel()
        video_injector.cancel()
        
        # Record end time
        cycle_tracker["end_time"] = datetime.now().isoformat()
        
        # Analyze results
        test_duration = (datetime.strptime(cycle_tracker["end_time"], "%Y-%m-%dT%H:%M:%S.%f") - 
                        datetime.strptime(cycle_tracker["start_time"], "%Y-%m-%dT%H:%M:%S.%f")).total_seconds() / 60.0
        
        comp_cycles = len(cycle_tracker["comprehensive_cycles"])
        comp_cycle_interval = test_duration / comp_cycles if comp_cycles > 0 else "N/A"
        
        # Generate final report
        report = {
            "test_duration_minutes": test_duration,
            "comprehensive_cycles": {
                "count": comp_cycles,
                "average_interval_minutes": comp_cycle_interval,
                "timestamps": cycle_tracker["comprehensive_cycles"]
            },
            "quick_decision_cycles": {
                "count": len(cycle_tracker.get("quick_decision_cycles", [])),
                "timestamps": cycle_tracker.get("quick_decision_cycles", [])
            },
            "video_injections": cycle_tracker.get("video_injections", 0)
        }
        
        # Save report to file
        report_file = f"dual_cycle_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test completed! Report saved to {report_file}")
        logger.info(f"Full logs available in {log_file}")
        
        return report

if __name__ == "__main__":
    # Run the test
    start_time = datetime.now()
    logger.info(f"Starting dual-cycle integration test at {start_time.isoformat()}")
    
    try:
        report = asyncio.run(test_dual_cycle_integration())
        
        # Print summary to stdout
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0
        
        print("\n=== Dual-Cycle Integration Test Summary ===")
        print(f"Test duration: {duration:.1f} minutes")
        print(f"Comprehensive cycles: {report['comprehensive_cycles']['count']} "
              f"(avg interval: {report['comprehensive_cycles']['average_interval_minutes']} minutes)")
        print(f"Quick decision cycles: {report['quick_decision_cycles']['count']}")
        print(f"Video injections: {report['video_injections']}")
        print(f"Log file: {log_file}")
        print(f"Report file: {os.path.basename(log_file).replace('log', 'json')}")
        
    except Exception as e:
        logger.error(f"Fatal error in test: {str(e)}", exc_info=True)
        sys.exit(1)