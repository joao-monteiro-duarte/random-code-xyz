"""
Module for tracking trades and global scores.
This file provides basic trade tracking functionality.
"""
import logging
from datetime import datetime

# Global state for crypto sentiment scores
global_scores = {
    "bitcoin": 0,
    "ethereum": 0,
    "solana": 0
}

logger = logging.getLogger(__name__)

def get_portfolio_state():
    """Get current portfolio state"""
    # Mock implementation
    return {
        'bitcoin_allocation': 100,
        'ethereum_allocation': 100,
        'solana_allocation': 100,
        'allocated_ada': 300,
        'unallocated_ada': 700
    }

async def prune_score_history():
    """Prune old scores from history"""
    # Mock implementation
    logger.info("Score history pruned")
    return True

def update_allocation(coin, sentiment_score):
    """
    Update allocation based on sentiment score.
    
    Args:
        coin: Cryptocurrency name
        sentiment_score: Sentiment score (-10 to +10)
    """
    # Update global score
    global_scores[coin] = sentiment_score
    
    # Log update
    logger.info(f"Updated allocation for {coin} based on sentiment: {sentiment_score:.2f}")
    
    return True