#!/usr/bin/env python3
"""
Test script to run a dual-cycle test with the accumulated videos in Redis
"""
import asyncio
import logging
import sys
import json
from datetime import datetime
from services.crypto_trading_service import CryptoTradingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

async def run_dual_cycle_test():
    """Run a test of both the quick decision and comprehensive cycles"""
    
    # Initialize the trading service
    trading_service = CryptoTradingService(test_mode=True)
    await trading_service.initialize()
    
    # Get current portfolio state before cycles
    logger.info("Initial portfolio state:")
    portfolio_before = trading_service.get_portfolio()
    logger.info(f"Total value: ${portfolio_before['total_value']:.2f}")
    logger.info(f"Unallocated: ${portfolio_before['unallocated_value']:.2f}")
    logger.info(f"Coins: {list(portfolio_before.get('coins', {}).keys())}")
    
    # Run a quick decision cycle (5-minute)
    logger.info("\n== Running quick decision cycle (5-minute) ==")
    
    # Get previous scores for comparison
    previous_scores = trading_service.sentiment_history
    logger.info(f"Previous sentiment scores: {json.dumps(previous_scores, indent=2)}")
    
    # Mock new sentiment data with significant changes for testing
    updated_scores = previous_scores.copy() if previous_scores else {}
    
    # Add a new coin with high sentiment as test
    updated_scores["solana"] = {
        "score": 8.5,
        "is_small_cap": False,
        "videos_mentioned": 2,
        "reasons": ["Strong ecosystem growth", "Developer adoption increasing"],
        "price_predictions": ["$150 target mentioned"],
        "urgency": "medium"
    }
    
    # Add a new small-cap coin with very high sentiment
    updated_scores["pepe"] = {
        "score": 9.0,
        "is_small_cap": True,
        "videos_mentioned": 1,
        "reasons": ["Massive social media adoption", "Whale accumulation observed"],
        "price_predictions": ["10x potential in coming weeks"],
        "urgency": "high"
    }
    
    # Calculate sentiment changes (difference between old and new)
    sentiment_changes = {}
    for coin, data in updated_scores.items():
        old_score = 0
        if coin in previous_scores:
            old_score = previous_scores[coin]["score"] if isinstance(previous_scores[coin], dict) else previous_scores[coin]
        sentiment_changes[coin] = abs(data["score"] - old_score)
    
    logger.info(f"Updated sentiment scores: {json.dumps(updated_scores, indent=2)}")
    logger.info(f"Sentiment changes: {json.dumps(sentiment_changes, indent=2)}")
    
    # Run quick decision cycle
    result = await trading_service.make_quick_decisions(
        updated_scores=updated_scores,
        previous_scores=previous_scores,
        sentiment_changes=sentiment_changes,
        significance_threshold=1.0
    )
    
    logger.info(f"Quick decision result: {json.dumps(result, indent=2)}")
    
    # Get portfolio after quick decisions
    portfolio_after_quick = trading_service.get_portfolio()
    logger.info("\nPortfolio after quick decisions:")
    logger.info(f"Total value: ${portfolio_after_quick['total_value']:.2f}")
    logger.info(f"Unallocated: ${portfolio_after_quick['unallocated_value']:.2f}")
    logger.info(f"Coins: {list(portfolio_after_quick.get('coins', {}).keys())}")
    
    # Run comprehensive cycle (30-minute)
    logger.info("\n== Running comprehensive cycle (30-minute) ==")
    comprehensive_result = await trading_service.run_cycle(vph_threshold=500.0, background=False)
    
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
    
    logger.info("\nDual-cycle test completed!")

if __name__ == "__main__":
    asyncio.run(run_dual_cycle_test())