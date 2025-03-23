#!/usr/bin/env python3
"""
Quick test script for the dual-cycle framework that runs for 5 minutes
This tests the coordination between quick decisions and comprehensive analysis
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
        logging.FileHandler(f"quick_dual_cycle_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)

logger = logging.getLogger(__name__)

async def run_quick_dual_cycle_test():
    """
    Run a quick test of the dual-cycle framework
    """
    # Import necessary services
    from services.crypto_trading_service import CryptoTradingService
    
    # Initialize the trading service
    trading_service = CryptoTradingService(test_mode=True)
    await trading_service.initialize()
    
    # Get current portfolio state before cycles
    logger.info("Initial portfolio state:")
    portfolio_before = trading_service.get_portfolio()
    logger.info(f"Total value: ${portfolio_before['total_value']:.2f}")
    logger.info(f"Unallocated: ${portfolio_before['unallocated_value']:.2f}")
    logger.info(f"Coins: {list(portfolio_before.get('coins', {}).keys())}")
    
    # 1. Test quick decision cycle
    logger.info("\n== Testing quick decision cycle ==")
    
    # Get previous scores for comparison
    previous_scores = trading_service.sentiment_history or {}
    logger.info(f"Previous sentiment scores: {json.dumps(previous_scores, indent=2)}")
    
    # Mock new sentiment data with significant changes for testing
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
    
    # Calculate sentiment changes (difference between old and new)
    sentiment_changes = {}
    for coin, data in updated_scores.items():
        old_score = 0
        if coin in previous_scores:
            old_score_data = previous_scores[coin]
            old_score = old_score_data["score"] if isinstance(old_score_data, dict) else old_score_data
        sentiment_changes[coin] = abs(data["score"] - old_score)
    
    logger.info(f"Updated sentiment scores: {json.dumps(updated_scores, indent=2)}")
    logger.info(f"Sentiment changes: {json.dumps(sentiment_changes, indent=2)}")
    
    # Run quick decision cycle
    quick_result = await trading_service.make_quick_decisions(
        updated_scores=updated_scores,
        previous_scores=previous_scores,
        sentiment_changes=sentiment_changes,
        significance_threshold=1.0
    )
    
    logger.info(f"Quick decision result: {json.dumps(quick_result, indent=2)}")
    
    # Get portfolio after quick decisions
    portfolio_after_quick = trading_service.get_portfolio()
    logger.info("\nPortfolio after quick decisions:")
    logger.info(f"Total value: ${portfolio_after_quick['total_value']:.2f}")
    logger.info(f"Unallocated: ${portfolio_after_quick['unallocated_value']:.2f}")
    logger.info(f"Coins: {list(portfolio_after_quick.get('coins', {}).keys())}")
    
    # 2. Test comprehensive cycle
    logger.info("\n== Testing comprehensive cycle ==")
    
    # Add our test videos for comprehensive analysis
    test_videos = [
        ("dQw4w9WgXcQ", 50000, "2023-01-01T00:00:00Z", 1000.0),  # High VPH video
        ("xvFZjo5PgG0", 25000, "2023-01-02T00:00:00Z", 750.0),   # Medium VPH video
    ]
    
    # Run the comprehensive cycle
    from config.settings import VPH_THRESHOLD
    
    # Set test videos for app service
    from services.app_service import get_app_service
    app_service = get_app_service()
    await app_service.set_accumulated_videos(test_videos)
    
    # Now run the comprehensive cycle
    comprehensive_result = await trading_service.run_cycle(
        vph_threshold=VPH_THRESHOLD, 
        background=False
    )
    
    logger.info(f"Comprehensive cycle result: {json.dumps(comprehensive_result, indent=2)}")
    
    # Get final portfolio state
    portfolio_final = trading_service.get_portfolio()
    logger.info("\nFinal portfolio state:")
    logger.info(f"Total value: ${portfolio_final['total_value']:.2f}")
    logger.info(f"Unallocated: ${portfolio_final['unallocated_value']:.2f}")
    logger.info(f"Coins: {list(portfolio_final.get('coins', {}).keys())}")
    
    # Show trading decisions
    logger.info("\nTrading decisions made:")
    for decision in trading_service.decision_history[-5:]:
        logger.info(f"- {decision.get('symbol', '?')}: {decision.get('action', '?')} with {decision.get('confidence', 0):.2f} confidence")
    
    # 3. Test a second quick decision to check throttling
    logger.info("\n== Testing second quick decision (throttling) ==")
    
    # Make a small change to see if throttling works
    updated_scores_2 = updated_scores.copy()
    updated_scores_2["solana"]["score"] = 9.0
    sentiment_changes_2 = {"solana": 0.5}
    
    # This should be throttled
    quick_result_2 = await trading_service.make_quick_decisions(
        updated_scores=updated_scores_2,
        previous_scores=updated_scores,
        sentiment_changes=sentiment_changes_2,
        significance_threshold=1.0
    )
    
    logger.info(f"Second quick decision result: {json.dumps(quick_result_2, indent=2)}")
    
    logger.info("\nQuick dual-cycle test completed!")

if __name__ == "__main__":
    asyncio.run(run_quick_dual_cycle_test())