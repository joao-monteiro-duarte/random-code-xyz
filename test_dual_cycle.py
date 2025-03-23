#!/usr/bin/env python3
"""
Test script for the dual-cycle framework
This verifies that both the 5-minute quick decision cycle and 30-minute comprehensive cycle
are executing correctly at the expected intervals.
"""
import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"dual_cycle_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

async def run_dual_cycle_test(duration_minutes=30):
    """
    Run a test of the dual-cycle framework for a specified duration
    
    Args:
        duration_minutes: How long to run the test, in minutes
    """
    # Import necessary services
    from services.app_service import AppService
    from services.crypto_trading_service import CryptoTradingService
    
    # Create AppService and CryptoTradingService
    logger.info("Initializing services for dual-cycle test")
    app_service = AppService()
    crypto_service = CryptoTradingService(test_mode=True)
    
    # Set cross-references (required to avoid circular dependency issues)
    app_service.crypto_trading_service = crypto_service
    
    # Initialize services
    await app_service.initialize()
    await crypto_service.initialize()
    
    logger.info("Services initialized successfully")
    
    # Record start time
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    logger.info(f"Starting dual-cycle test at {start_time.isoformat()}")
    logger.info(f"Test will run for {duration_minutes} minutes (until {end_time.isoformat()})")
    
    # Track cycle executions
    comprehensive_cycles = 0
    quick_decision_cycles = 0
    
    # Manually add some test data for cycles to process
    from config.settings import VPH_THRESHOLD
    
    # Add some fake accumulated videos for the cycles to process
    test_videos = [
        ("dQw4w9WgXcQ", 50000, "2023-01-01T00:00:00Z", 1000.0),  # High VPH video
        ("xvFZjo5PgG0", 25000, "2023-01-02T00:00:00Z", 750.0),   # Medium VPH video
        ("xvFZjo5PgG0", 10000, "2023-01-03T00:00:00Z", 300.0)    # Below threshold video
    ]
    app_service.set_accumulated_videos(test_videos)
    logger.info(f"Added {len(test_videos)} test videos to accumulated videos")
    
    # Create mock transcripts for the test videos
    for video_id, _, _, _ in test_videos:
        # Create a mock transcript with mentions of various cryptocurrencies
        mock_transcript = f"""
        Today we're talking about cryptocurrency investments.
        Bitcoin has been showing strong momentum lately, with many analysts predicting it could reach $100,000.
        Ethereum is also looking bullish due to upcoming protocol upgrades.
        
        For small-cap coins, I'm particularly interested in Pepe coin which has massive social media attention.
        Solana is another project with impressive technology.
        
        Overall, the market sentiment is positive, but always remember to do your own research.
        """
        
        # Store the transcript in the service
        app_service.transcript_service.save_transcript(video_id, mock_transcript)
        logger.info(f"Created mock transcript for video {video_id}")
    
    # Mock sentiment data for the quick decision cycle to use
    mock_sentiment_data = {
        "bitcoin": {
            "score": 8.0,
            "is_small_cap": False,
            "videos_mentioned": 2,
            "reasons": ["Institutional adoption", "Technical breakout"],
            "urgency": "medium"
        },
        "ethereum": {
            "score": 7.5,
            "is_small_cap": False,
            "videos_mentioned": 1,
            "reasons": ["Protocol upgrades", "DeFi growth"],
            "urgency": "medium"
        },
        "pepe": {
            "score": 9.0,
            "is_small_cap": True,
            "videos_mentioned": 1,
            "reasons": ["Massive social media attention", "Viral trend"],
            "urgency": "high"
        }
    }
    
    # Set previous sentiment for comparison
    previous_sentiment = {
        "bitcoin": {
            "score": 7.0,
            "is_small_cap": False,
            "videos_mentioned": 1
        },
        "ethereum": {
            "score": 6.5,
            "is_small_cap": False,
            "videos_mentioned": 1
        }
    }
    
    # Set initial sentiment history
    crypto_service.sentiment_history = previous_sentiment
    
    # Calculate sentiment changes (pepe is new, bitcoin and ethereum changed)
    sentiment_changes = {
        "bitcoin": 1.0,  # Changed by 1.0
        "ethereum": 1.0,  # Changed by 1.0
        "pepe": 9.0      # New coin
    }
    
    # Run tests until the duration expires
    try:
        while datetime.now() < end_time:
            current_time = datetime.now()
            elapsed_seconds = (current_time - start_time).total_seconds()
            elapsed_minutes = elapsed_seconds / 60
            
            # Log time check with detailed information
            logger.info(f"Time check - Elapsed: {elapsed_minutes:.1f} minutes")
            logger.info(f"Last cycle time: {app_service.last_cycle_time.isoformat()}")
            logger.info(f"Now: {current_time.isoformat()}")
            time_since_last = (current_time - app_service.last_cycle_time).total_seconds()
            logger.info(f"Time since last cycle: {time_since_last:.1f}s")
            
            # Very detailed logging for cycle timing
            from config.settings import CYCLE_INTERVAL
            logger.info(f"CYCLE_INTERVAL setting: {CYCLE_INTERVAL}s")
            logger.info(f"Next comprehensive cycle in: {max(0, CYCLE_INTERVAL - time_since_last):.1f}s")
            
            # 1. Test quick decision cycle (every 5 minutes)
            if int(elapsed_minutes) % 5 == 0 and int(elapsed_seconds) % 300 < 10:  # Within 10s of 5-min mark
                logger.info("\n=== TESTING QUICK DECISION CYCLE ===")
                quick_decision_result = await crypto_service.make_quick_decisions(
                    updated_scores=mock_sentiment_data,
                    previous_scores=previous_sentiment,
                    sentiment_changes=sentiment_changes,
                    significance_threshold=1.0
                )
                
                quick_decision_cycles += 1
                logger.info(f"Quick decision cycle complete: {json.dumps(quick_decision_result)}")
                logger.info(f"Portfolio after quick decisions: {json.dumps(crypto_service.get_portfolio())}")
            
            # 2. Test if comprehensive cycle would run
            from config.settings import CYCLE_INTERVAL
            if time_since_last >= CYCLE_INTERVAL:
                logger.info("\n=== TESTING COMPREHENSIVE CYCLE ===")
                
                # Get accumulated videos
                accumulated_videos = app_service.get_accumulated_videos()
                logger.info(f"Running cycle with {len(accumulated_videos)} accumulated videos")
                
                # Run the cycle
                comprehensive_result = await crypto_service.run_cycle(
                    vph_threshold=VPH_THRESHOLD, 
                    background=False
                )
                
                comprehensive_cycles += 1
                logger.info(f"Comprehensive cycle complete: {json.dumps(comprehensive_result)}")
                logger.info(f"Portfolio after comprehensive cycle: {json.dumps(crypto_service.get_portfolio())}")
                
                # Update timestamp
                app_service.last_cycle_time = datetime.now()
            
            # Wait before checking again
            await asyncio.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error during dual cycle test: {e}", exc_info=True)
    finally:
        # Final summary
        logger.info("\n=== DUAL CYCLE TEST SUMMARY ===")
        logger.info(f"Test duration: {(datetime.now() - start_time).total_seconds() / 60:.1f} minutes")
        logger.info(f"Quick decision cycles executed: {quick_decision_cycles}")
        logger.info(f"Comprehensive cycles executed: {comprehensive_cycles}")
        logger.info(f"Final portfolio: {json.dumps(crypto_service.get_portfolio())}")
        logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(run_dual_cycle_test(30))  # Run for 30 minutes by default