#!/usr/bin/env python3
"""
Test script for quick decision cycle with incremental sentiment updates
"""
import asyncio
import logging
import sys
import json
from services.sentiment_service import SentimentAnalysisService
from services.crypto_trading_service import CryptoTradingService
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

async def test_quick_decisions():
    """Test the quick decision cycle with incremental sentiment updates"""
    
    # Initialize the services
    trading_service = CryptoTradingService(test_mode=True)
    await trading_service.initialize()
    
    # Create mock previous sentiment data
    previous_scores = {
        "bitcoin": {
            "score": 6.0,
            "is_small_cap": False,
            "videos_mentioned": 5,
            "reasons": ["Strong institutional adoption", "Halving event approaching"],
            "price_predictions": ["$100,000 in 2025"],
            "urgency": "medium"
        },
        "ethereum": {
            "score": 5.0,
            "is_small_cap": False,
            "videos_mentioned": 4,
            "reasons": ["ETH 2.0 progress", "Growing DeFi ecosystem"],
            "price_predictions": [],
            "urgency": "low"
        },
        "solana": {
            "score": 7.0,
            "is_small_cap": False,
            "videos_mentioned": 3,
            "reasons": ["Fast growing ecosystem", "Performance improvements"],
            "price_predictions": ["$300 target mentioned"],
            "urgency": "medium"
        }
    }
    
    # Create mock updated sentiment data with significant changes
    updated_scores = {
        "bitcoin": {
            "score": 6.5,  # Small change (+0.5)
            "is_small_cap": False,
            "videos_mentioned": 6,
            "reasons": ["Strong institutional adoption", "Halving event approaching", "ETF approval rumors"],
            "price_predictions": ["$100,000 in 2025"],
            "urgency": "medium"
        },
        "ethereum": {
            "score": 5.2,  # Small change (+0.2)
            "is_small_cap": False,
            "videos_mentioned": 5,
            "reasons": ["ETH 2.0 progress", "Growing DeFi ecosystem"],
            "price_predictions": [],
            "urgency": "low"
        },
        "solana": {
            "score": 9.0,  # Big change (+2.0) - should trigger quick decision
            "is_small_cap": False,
            "videos_mentioned": 4,
            "reasons": ["Fast growing ecosystem", "Performance improvements", "Major partnership announcement"],
            "price_predictions": ["$300 target mentioned", "$500 possible soon"],
            "urgency": "high"  # Increased urgency
        },
        "pepe": {  # New coin appeared - should trigger quick decision
            "score": 8.5,
            "is_small_cap": True,
            "videos_mentioned": 2,
            "reasons": ["Massive social media attention", "Whale accumulation"],
            "price_predictions": ["10x potential mentioned"],
            "urgency": "high",
            "is_newly_discovered": True
        }
    }
    
    # Create mock sentiment changes for quick decisions
    sentiment_changes = {
        "bitcoin": 0.5,  # Small change
        "ethereum": 0.2,  # Small change
        "solana": 2.0,    # Significant change
        "pepe": 8.5       # New coin (very significant)
    }
    
    # Test the quick decision cycle
    print("\n----- Testing Quick Decision Cycle -----")
    print("\n1. Previous sentiment scores:")
    print(json.dumps(previous_scores, indent=2))
    
    print("\n2. Updated sentiment scores:")
    print(json.dumps(updated_scores, indent=2))
    
    print("\n3. Sentiment changes:")
    print(json.dumps(sentiment_changes, indent=2))
    
    print("\n4. Running quick decision cycle with significance threshold of 1.0...")
    result = await trading_service.make_quick_decisions(
        updated_scores=updated_scores,
        previous_scores=previous_scores,
        sentiment_changes=sentiment_changes,
        significance_threshold=1.0
    )
    
    print("\n5. Quick decision result:")
    print(json.dumps(result, indent=2))
    
    # Check current opportunities
    print("\n6. Trading opportunities identified:")
    opportunities = trading_service.current_opportunities
    for opp in opportunities:
        print(f"- {opp['symbol'].upper()}: Action={opp['action']}, Sentiment={opp['sentiment_score']}, Change={opp['sentiment_change']}")
    
    # Check decision history
    print(f"\n7. Decision history (last {min(5, len(trading_service.decision_history))} entries):")
    for decision in trading_service.decision_history[-5:]:
        print(f"- {decision.get('symbol', 'Unknown')}: {decision.get('action', 'unknown')} with {decision.get('confidence', 0):.2f} confidence")

if __name__ == "__main__":
    asyncio.run(test_quick_decisions())