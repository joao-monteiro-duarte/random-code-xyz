#!/usr/bin/env python3
"""
Quick test for dual-cycle framework to verify both 5-minute quick decisions and 30-minute comprehensive cycles
Focuses on checking the mechanics rather than creating a full simulation
"""
import asyncio
import logging
import json
import os
import sys
from datetime import datetime, timedelta

# Configure logging with enhanced setup for better visibility
log_file = f"test_quick_dual_cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

# Ensure our logger's output is visible
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

print(f"Starting test with log file: {log_file}")

async def test_quick_dual_cycle():
    """Test the quick decision and comprehensive cycles"""
    
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
    
    # Print initial state
    logger.info("Initial portfolio state:")
    portfolio_before = trading_service.get_portfolio()
    logger.info(f"Total value: ${portfolio_before['total_value']:.2f}")
    
    # 1. Test quick decision cycle
    logger.info("== Testing quick decision cycle ==")
    
    # Mock sentiment scores (previous and updated)
    previous_scores = {}
    updated_scores = {
        "bitcoin": {
            "score": 7.5,
            "is_small_cap": False,
            "videos_mentioned": 3,
            "reasons": ["Institutional adoption", "Technical breakout"],
            "price_predictions": ["$100,000 by end of year"],
            "urgency": "medium"
        },
        "ethereum": {
            "score": 6.5,
            "is_small_cap": False,
            "videos_mentioned": 2,
            "reasons": ["Protocol upgrades", "DeFi growth"],
            "price_predictions": [],
            "urgency": "medium"
        },
        "solana": {
            "score": 8.5,
            "is_small_cap": False,
            "videos_mentioned": 2,
            "reasons": ["Strong ecosystem growth", "Developer adoption increasing"],
            "price_predictions": ["$150 target mentioned"],
            "urgency": "medium"
        },
        "pepe": {
            "score": 9.0,
            "is_small_cap": True,
            "videos_mentioned": 1,
            "reasons": ["Massive social media adoption", "Whale accumulation observed"],
            "price_predictions": ["10x potential in coming weeks"],
            "urgency": "high"
        }
    }
    
    # Calculate sentiment changes
    sentiment_changes = {}
    for coin, data in updated_scores.items():
        sentiment_changes[coin] = data["score"]  # For new coins, change is full score
    
    # Run quick decision cycle
    logger.info("Running quick decision cycle...")
    quick_result = await trading_service.make_quick_decisions(
        updated_scores=updated_scores,
        previous_scores=previous_scores,
        sentiment_changes=sentiment_changes,
        significance_threshold=1.0
    )
    
    logger.info(f"Quick decision result: {json.dumps(quick_result, indent=2)}")
    
    # 2. Test comprehensive cycle
    logger.info("== Testing comprehensive cycle ==")
    
    # Create test videos as proper Video objects
    logger.info("Creating test videos as Video objects")
    test_videos = [
        Video.from_tuple(("dQw4w9WgXcQ", 50000, "2023-01-01T00:00:00Z", 1000.0)),  # High VPH video
        Video.from_tuple(("xvFZjo5PgG0", 25000, "2023-01-02T00:00:00Z", 750.0))    # Medium VPH video
    ]
    
    # Set accumulated videos in app service
    logger.info(f"Setting accumulated_videos with {len(test_videos)} Video objects")
    app_service.accumulated_videos = test_videos
    
    # Verify the accumulated_videos were set properly
    logger.info(f"Checking accumulated_videos: {len(app_service.accumulated_videos)} videos")
    for i, video in enumerate(app_service.accumulated_videos):
        logger.info(f"Video {i+1}: {type(video).__name__}, id={video.id if hasattr(video, 'id') else 'unknown'}")
    
    # Create mock transcripts
    logger.info("Creating mock transcripts")
    for video in test_videos:
        mock_transcript = f"""
        Today we're discussing cryptocurrencies. Bitcoin is showing strong performance,
        with potential for significant growth this year. Ethereum is also performing well
        with its recent updates. Among smaller coins, Solana has impressive technology and
        adoption. I'm particularly bullish on Pepe coin, which has massive social media presence
        and could see significant price action in the coming weeks.
        """
        app_service.transcript_service.save_transcript(video.id, mock_transcript)
        logger.info(f"Saved mock transcript for video {video.id}")
    
    # Run comprehensive cycle
    from config.settings import VPH_THRESHOLD
    logger.info(f"Running comprehensive cycle with VPH threshold: {VPH_THRESHOLD}")
    try:
        comprehensive_result = await trading_service.run_cycle(vph_threshold=VPH_THRESHOLD, background=False)
        logger.info(f"Comprehensive cycle result: {json.dumps(comprehensive_result, indent=2)}")
    except Exception as e:
        logger.error(f"Error in comprehensive cycle: {str(e)}", exc_info=True)
        comprehensive_result = {"status": "error", "message": str(e)}
    
    # Print final portfolio state
    portfolio_after = trading_service.get_portfolio()
    logger.info("Final portfolio state:")
    logger.info(f"Total value: ${portfolio_after['total_value']:.2f}")
    logger.info(f"Unallocated: ${portfolio_after['unallocated_value']:.2f}")
    logger.info(f"Coins: {list(portfolio_after.get('coins', {}).keys())}")
    
    # 3. Test throttling in quick decisions
    logger.info("== Testing throttling in quick decisions ==")
    
    # Record the throttling test decision in history to ensure it's throttled
    now = datetime.now().isoformat()
    trading_service.decision_history = [
        {"symbol": "sol", "action": "buy", "timestamp": now}
    ]
    logger.info(f"Added decision to history: SOL buy at {now}")
    
    # Run another quick decision with small change - should be throttled
    updated_scores_2 = updated_scores.copy()
    updated_scores_2["solana"]["score"] = 9.0  # Increased score
    sentiment_changes_2 = {"solana": 0.5}  # Small change
    
    logger.info("Running throttled quick decision test...")
    throttle_result = await trading_service.make_quick_decisions(
        updated_scores=updated_scores_2,
        previous_scores=updated_scores,
        sentiment_changes=sentiment_changes_2,
        significance_threshold=0.5  # Lower threshold to ensure it's considered significant
    )
    
    logger.info(f"Throttled decision result: {json.dumps(throttle_result, indent=2)}")
    
    # Show final portfolio state
    portfolio_final = trading_service.get_portfolio()
    logger.info("Final portfolio state:")
    logger.info(f"Total value: ${portfolio_final['total_value']:.2f}")
    logger.info(f"Unallocated: ${portfolio_final['unallocated_value']:.2f}")
    logger.info(f"Coins: {list(portfolio_final.get('coins', {}).keys())}")
    
    # Test successful
    logger.info("Dual-cycle test completed successfully!")
    
    # Return test results summary
    return {
        "quick_decision": quick_result,
        "comprehensive_cycle": comprehensive_result,
        "throttling_test": throttle_result
    }

if __name__ == "__main__":
    # Run the test and get the results
    test_results = asyncio.run(test_quick_dual_cycle())
    
    # Print summary to stdout
    print("\n=== Test Summary ===")
    print(f"Quick Decision: {test_results['quick_decision']['status']}")
    print(f"Comprehensive Cycle: {test_results['comprehensive_cycle']['status']}")
    print(f"Throttling Test: {test_results['throttling_test']['status']}")
    print(f"Log file: {log_file}")
    
    # Exit with non-zero code if any test failed
    if (test_results['quick_decision']['status'] != 'completed' or
        test_results['comprehensive_cycle']['status'] != 'completed' or
        test_results['throttling_test']['status'] != 'completed'):
        sys.exit(1)