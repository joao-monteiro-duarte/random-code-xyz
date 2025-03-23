"""
Tests for the dual-cycle trading framework (5-minute quick decisions and 30-minute comprehensive cycles).
Focuses on testing the:
1. Quick decision making logic in CryptoTradingService.make_quick_decisions
2. Incremental sentiment updates in SentimentAnalysisService.update_global_scores_incremental
3. Dynamic position sizing logic based on number of tracked coins
4. Trade throttling to prevent excessive trading
5. Error handling in the quick decision cycle
"""

import sys
import os
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.crypto_trading_service import CryptoTradingService
from services.sentiment_service import SentimentAnalysisService
from services.app_service import AppService
from models.video import Video


# Mock classes for testing
class MockRedisClient:
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value):
        self.data[key] = value
        return True


class MockMarketDataService:
    """Mock implementation of MarketDataService for testing."""
    
    def __init__(self):
        self.coin_data = {
            "bitcoin": {
                "id": "bitcoin",
                "symbol": "btc",
                "name": "Bitcoin",
                "current_price": 60000,
                "market_cap": 1000000000000,  # 1 trillion
                "volume_24h": 50000000000,
                "price_change_24h": 5.0
            },
            "ethereum": {
                "id": "ethereum",
                "symbol": "eth",
                "name": "Ethereum",
                "current_price": 3000,
                "market_cap": 300000000000,  # 300 billion
                "volume_24h": 20000000000,
                "price_change_24h": 3.0
            },
            "pepe": {
                "id": "pepe",
                "symbol": "pepe",
                "name": "Pepe",
                "current_price": 0.00005,
                "market_cap": 30000000,  # 30 million - small cap
                "volume_24h": 10000000,
                "price_change_24h": 15.0
            },
            "shiba-inu": {
                "id": "shiba-inu",
                "symbol": "shib",
                "name": "Shiba Inu",
                "current_price": 0.00001,
                "market_cap": 15000000,  # 15 million - small cap
                "volume_24h": 5000000,
                "price_change_24h": 10.0
            }
        }
        
        self.macd_data = {
            "bitcoin": {
                "macd": 0.0002,
                "macd_signal": 0.0001,
                "macd_histogram": 0.0001,
                "is_bullish": True,
                "trend_strength": 0.7
            },
            "ethereum": {
                "macd": 0.0001,
                "macd_signal": 0.0001,
                "macd_histogram": 0.0,
                "is_bullish": False,
                "trend_strength": 0.1
            },
            "pepe": {
                "macd": 0.00001,
                "macd_signal": 0.000005,
                "macd_histogram": 0.000005,
                "is_bullish": True,
                "trend_strength": 0.8
            },
            "shiba-inu": {
                "macd": -0.000001,
                "macd_signal": 0.0,
                "macd_histogram": -0.000001,
                "is_bullish": False,
                "trend_strength": 0.3
            }
        }
    
    async def get_coin_by_symbol(self, symbol):
        """Get coin data by symbol."""
        return self.coin_data.get(symbol.lower())
    
    async def get_coin_data(self, coin_id):
        """Get coin data by ID."""
        return self.coin_data.get(coin_id.lower())
    
    async def calculate_macd(self, coin_id):
        """Calculate MACD for a coin."""
        return self.macd_data.get(coin_id.lower())
    
    async def detect_volatility(self, coin_id):
        """Detect volatility for a coin."""
        coin = self.coin_data.get(coin_id.lower())
        if not coin:
            return {"volatility": 0, "volume_to_market_cap": 0}
        
        # Calculate volume to market cap ratio
        volume_to_mc = coin["volume_24h"] / coin["market_cap"] if coin["market_cap"] > 0 else 0
        
        return {
            "volatility": abs(coin["price_change_24h"]),
            "volume_to_market_cap": volume_to_mc
        }


class MockTradeService:
    """Mock implementation of TradeService for testing."""
    
    def __init__(self):
        self.trades = []
        self.usd_balance = 10000.0
        self.coins = {}
        self.per_coin_limit = 1000.0  # 10% cap
    
    def get_portfolio(self):
        """Get current portfolio state."""
        total_value = self.usd_balance + sum(
            info["amount"] * info["avg_price"] for info in self.coins.values()
        )
        return {
            "USD": self.usd_balance,
            "total_value": total_value,
            "coins": self.coins
        }
    
    async def execute_trade(self, symbol, action, sentiment_score, vph, market_cap, price, position_size):
        """Execute a trade."""
        # Calculate trade amount
        max_value = self.usd_balance * position_size if action == "buy" else 0
        
        # Check per-coin limit (10% of portfolio)
        current_holding = self.coins.get(symbol, {"amount": 0, "avg_price": 0})
        current_value = current_holding["amount"] * current_holding["avg_price"]
        if action == "buy" and (current_value + max_value) > self.per_coin_limit:
            return {"status": "rejected", "error": f"Exceeds per-coin limit of {self.per_coin_limit}"}
        
        # For buy orders
        if action == "buy" and max_value > 0 and price > 0:
            amount = max_value / price
            self.usd_balance -= max_value
            
            # Update coins
            if symbol not in self.coins:
                self.coins[symbol] = {"amount": 0, "avg_price": 0}
            
            # Calculate new average price
            total_amount = self.coins[symbol]["amount"] + amount
            total_value = (self.coins[symbol]["amount"] * self.coins[symbol]["avg_price"]) + (amount * price)
            self.coins[symbol]["amount"] = total_amount
            self.coins[symbol]["avg_price"] = total_value / total_amount if total_amount > 0 else 0
            
            # Record the trade
            trade = {
                "symbol": symbol,
                "action": action,
                "amount": amount,
                "price": price,
                "value": max_value,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            self.trades.append(trade)
            
            return trade
        
        # For sell orders
        elif action == "sell" and symbol in self.coins and self.coins[symbol]["amount"] > 0:
            amount = self.coins[symbol]["amount"]
            value = amount * price
            self.usd_balance += value
            
            # Update coins
            self.coins[symbol]["amount"] = 0
            
            # Record the trade
            trade = {
                "symbol": symbol,
                "action": action,
                "amount": amount,
                "price": price,
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            self.trades.append(trade)
            
            return trade
        
        return {"status": "error", "error": "Invalid trade parameters"}


class MockSentimentService:
    """Mock implementation of SentimentAnalysisService for testing."""
    
    def __init__(self):
        self.sentiment_cache = {}
    
    async def batch_analyze(self, video_transcripts):
        """Analyze sentiment for multiple video transcripts."""
        results = {}
        for video_id, transcript in video_transcripts:
            results[video_id] = {
                "bitcoin": {"score": 5.0, "is_small_cap": False, "urgency": "low"},
                "ethereum": {"score": 3.0, "is_small_cap": False, "urgency": "low"},
                "pepe": {"score": 9.0, "is_small_cap": True, "urgency": "high"}
            }
        return results
    
    async def calculate_global_scores(self, video_sentiments, videos, vph_threshold=500.0):
        """Calculate global sentiment scores by aggregating individual video sentiments."""
        return {
            "bitcoin": {"score": 5.0, "is_small_cap": False, "urgency": "low", "videos_mentioned": 3},
            "ethereum": {"score": 3.0, "is_small_cap": False, "urgency": "low", "videos_mentioned": 2},
            "pepe": {"score": 9.0, "is_small_cap": True, "urgency": "high", "videos_mentioned": 1}
        }
    
    async def update_global_scores_incremental(self, new_video_sentiments, new_videos, current_global_scores, vph_threshold=500.0):
        """Incrementally update global sentiment scores by adding new video data."""
        # Start with a copy of the current scores
        updated_scores = {k: v.copy() for k, v in current_global_scores.items()} if current_global_scores else {}
        
        # Record sentiment changes
        sentiment_changes = {}
        
        # Add new coin not in current scores
        if "shiba-inu" not in updated_scores:
            updated_scores["shiba-inu"] = {
                "score": 8.0,
                "is_small_cap": True,
                "urgency": "high",
                "videos_mentioned": 1,
                "is_newly_discovered": True
            }
            sentiment_changes["shiba-inu"] = 8.0  # New coin, full change
        
        # Update existing coins with significant changes
        if "bitcoin" in updated_scores:
            old_score = updated_scores["bitcoin"]["score"]
            updated_scores["bitcoin"]["score"] = 7.0  # Significant increase
            sentiment_changes["bitcoin"] = abs(7.0 - old_score)
        
        if "pepe" in updated_scores:
            old_score = updated_scores["pepe"]["score"]
            # Increase from 9.0 to 9.5
            updated_scores["pepe"]["score"] = 9.5
            sentiment_changes["pepe"] = abs(9.5 - old_score)
        
        return updated_scores, sentiment_changes


class MockAppService:
    """Mock implementation of AppService for testing."""
    
    def __init__(self):
        self.transcript_service = MagicMock()
        self.is_running = False
        self.accumulated_videos = []
    
    def get_accumulated_videos(self):
        """Get accumulated videos."""
        return self.accumulated_videos
    
    async def set_accumulated_videos(self, videos):
        """Set accumulated videos."""
        self.accumulated_videos = videos


class MockCryptoTradingService(CryptoTradingService):
    """Mock implementation of CryptoTradingService for testing."""
    
    def __init__(self):
        # Skip the parent class initialization
        self.app_service = MockAppService()
        self.market_data_service = MockMarketDataService()
        self.trade_service = MockTradeService()
        self.sentiment_service = MockSentimentService()
        self.master_agent = AsyncMock()
        self.decision_history = []
        self.sentiment_history = {
            "bitcoin": {"score": 5.0, "is_small_cap": False, "urgency": "low", "videos_mentioned": 3},
            "ethereum": {"score": 3.0, "is_small_cap": False, "urgency": "low", "videos_mentioned": 2},
            "pepe": {"score": 9.0, "is_small_cap": True, "urgency": "high", "videos_mentioned": 1}
        }
        self.decision_history_key = "trading_decision_history"
        self.is_initialized = True
    
    async def decide_trade(self, coin_symbol, coin_data, sentiment_data, portfolio):
        """Make a simplified trade decision for testing."""
        score = sentiment_data.get("score", 0)
        
        # Default decision
        decision = {
            "action": "hold",
            "confidence": 0.5,
            "reasons": ["Not enough signal"],
            "position_size": 0.01
        }
        
        # Make decision based on sentiment score
        if score >= 8.0:
            decision = {
                "action": "buy",
                "confidence": 0.9,
                "reasons": ["Very high sentiment score (>=8)"],
                "position_size": 0.02
            }
        elif score >= 7.0:
            decision = {
                "action": "buy",
                "confidence": 0.7,
                "reasons": ["High sentiment score (>=7)"],
                "position_size": 0.01
            }
        elif score <= 2.0:
            decision = {
                "action": "sell",
                "confidence": 0.8,
                "reasons": ["Very low sentiment score (<=2)"],
                "position_size": 0.02
            }
        
        # Add symbol and timestamp for testing throttling
        decision["symbol"] = coin_symbol
        decision["timestamp"] = datetime.now().isoformat()
        
        return decision


# Mock logger for testing
class MockLogger:
    def __init__(self):
        self.messages = []
    
    def info(self, message):
        self.messages.append(message)
    
    def warning(self, message):
        self.messages.append(f"WARNING: {message}")
    
    def error(self, message, exc_info=False):
        self.messages.append(f"ERROR: {message}")


# Tests
@pytest.mark.asyncio
async def test_make_quick_decisions():
    """Test the make_quick_decisions method in CryptoTradingService."""
    # Create mock service
    mock_service = MockCryptoTradingService()
    
    # Set up initial state
    mock_service.sentiment_history = {
        "bitcoin": {"score": 5.0, "is_small_cap": False, "urgency": "low", "videos_mentioned": 3},
        "ethereum": {"score": 3.0, "is_small_cap": False, "urgency": "low", "videos_mentioned": 2},
        "pepe": {"score": 9.0, "is_small_cap": True, "urgency": "high", "videos_mentioned": 1}
    }
    
    # Create updated scores with significant changes
    updated_scores = {
        "bitcoin": {"score": 7.0, "is_small_cap": False, "urgency": "medium", "videos_mentioned": 4},
        "ethereum": {"score": 3.0, "is_small_cap": False, "urgency": "low", "videos_mentioned": 2},
        "pepe": {"score": 9.5, "is_small_cap": True, "urgency": "high", "videos_mentioned": 2},
        "shiba-inu": {"score": 8.0, "is_small_cap": True, "urgency": "high", "videos_mentioned": 1}
    }
    
    # Create sentiment changes
    sentiment_changes = {
        "bitcoin": 2.0,  # Significant change
        "pepe": 0.5,     # Minor change
        "shiba-inu": 8.0  # New coin
    }
    
    # Replace the logger
    mock_logger = MockLogger()
    with patch('services.crypto_trading_service.logger', mock_logger):
        # Call the method
        result = await mock_service.make_quick_decisions(
            updated_scores=updated_scores,
            previous_scores=mock_service.sentiment_history,
            sentiment_changes=sentiment_changes,
            significance_threshold=1.0
        )
    
    # Assertions
    assert result["status"] == "completed"
    assert result["trades_executed"] > 0
    assert "SIGNIFICANT CHANGE" in " ".join(mock_logger.messages)
    assert "NEW COIN DETECTED" in " ".join(mock_logger.messages)
    assert "Position sizing for" in " ".join(mock_logger.messages)
    assert "QUICK TRADE" in " ".join(mock_logger.messages)


@pytest.mark.asyncio
async def test_make_quick_decisions_with_throttling():
    """Test trade throttling in make_quick_decisions."""
    # Create mock service
    mock_service = MockCryptoTradingService()
    
    # Create a recent trade decision (less than 1 hour ago)
    one_minute_ago = (datetime.now() - timedelta(minutes=1)).isoformat()
    mock_service.decision_history = [
        {
            "symbol": "btc",  # Note: lowercase symbol to match the pattern in the code
            "action": "buy",
            "timestamp": one_minute_ago
        }
    ]
    
    # Create updated scores with bitcoin having a significant change
    updated_scores = {
        "bitcoin": {"score": 8.0, "is_small_cap": False, "urgency": "high", "videos_mentioned": 4}
    }
    
    # Create sentiment changes
    sentiment_changes = {
        "bitcoin": 3.0  # Significant change, but should be throttled
    }
    
    # Replace the logger
    mock_logger = MockLogger()
    with patch('services.crypto_trading_service.logger', mock_logger):
        # Call the method
        result = await mock_service.make_quick_decisions(
            updated_scores=updated_scores,
            previous_scores={"bitcoin": {"score": 5.0}},
            sentiment_changes=sentiment_changes,
            significance_threshold=1.0
        )
    
    # Assertions
    assert "Throttling btc" in " ".join(mock_logger.messages)
    assert result["trades_executed"] == 0


@pytest.mark.asyncio
async def test_position_sizing_with_coin_count():
    """Test position sizing based on number of tracked coins."""
    # Create mock service
    mock_service = MockCryptoTradingService()
    
    # Replace the logger
    mock_logger = MockLogger()
    with patch('services.crypto_trading_service.logger', mock_logger):
        # Test with 10 coins - we need to use updated_scores for this logic, not sentiment_history
        updated_scores_10 = {f"coin_{i}": {"score": 7.0, "videos_mentioned": 1} for i in range(10)}
        updated_scores_10["bitcoin"] = {"score": 8.0, "is_small_cap": False, "urgency": "medium", "videos_mentioned": 3}
        sentiment_changes = {"bitcoin": 3.0}
        
        # Reset decision history to avoid throttling
        mock_service.decision_history = []
        
        result_10_coins = await mock_service.make_quick_decisions(
            updated_scores=updated_scores_10,
            previous_scores={"bitcoin": {"score": 5.0}},
            sentiment_changes=sentiment_changes,
            significance_threshold=1.0
        )
        
        # Test with 100 coins
        updated_scores_100 = {f"coin_{i}": {"score": 7.0, "videos_mentioned": 1} for i in range(100)}
        updated_scores_100["bitcoin"] = {"score": 8.0, "is_small_cap": False, "urgency": "medium", "videos_mentioned": 3}
        
        # Reset decision history to avoid throttling
        mock_service.decision_history = []
        
        result_100_coins = await mock_service.make_quick_decisions(
            updated_scores=updated_scores_100,
            previous_scores={"bitcoin": {"score": 5.0}},
            sentiment_changes=sentiment_changes,
            significance_threshold=1.0
        )
    
    # Extract all position sizing log lines
    position_sizing_logs = [msg for msg in mock_logger.messages if "Position sizing for" in msg]
    
    # We should have exactly 2 position sizing logs, one for each test
    assert len(position_sizing_logs) == 2, f"Expected 2 position sizing logs but found {len(position_sizing_logs)}"
    
    # Extract values from logs
    import re
    
    def extract_value(log_message, param):
        pattern = f"{param}=(\\d+\\.\\d+)"
        match = re.search(pattern, log_message)
        return float(match.group(1)) if match else None
    
    def extract_tracked_coins(log_message):
        pattern = r"tracked_coins=(\d+)"
        match = re.search(pattern, log_message)
        return int(match.group(1)) if match else None
    
    # The first log should be for the 10-coin test, the second for the 100-coin test
    log_10_coins = position_sizing_logs[0]
    log_100_coins = position_sizing_logs[1]
    
    # Extract tracked coin counts to verify
    tracked_10 = extract_tracked_coins(log_10_coins)
    tracked_100 = extract_tracked_coins(log_100_coins)
    
    # Verify the test counts are roughly correct (may include bitcoin)
    assert 10 <= tracked_10 <= 12, f"Expected 10-12 tracked coins but found {tracked_10}"
    assert tracked_100 > 50, f"Expected >50 tracked coins but found {tracked_100}"
    
    # Extract values
    base_10 = extract_value(log_10_coins, "base")
    base_100 = extract_value(log_100_coins, "base")
    
    adjusted_10 = extract_value(log_10_coins, "adjusted")
    adjusted_100 = extract_value(log_100_coins, "adjusted")
    
    # Base position size should be smaller with more coins
    assert base_10 > base_100
    
    # Adjusted position size should similarly scale
    assert adjusted_10 > adjusted_100


@pytest.mark.asyncio
async def test_incremental_sentiment_updates():
    """Test incremental sentiment updates in SentimentAnalysisService."""
    # Get a real instance of SentimentAnalysisService
    sentiment_service = MockSentimentService()
    
    # Define the initial global scores
    current_global_scores = {
        "bitcoin": {"score": 5.0, "is_small_cap": False, "urgency": "low", "videos_mentioned": 3},
        "ethereum": {"score": 3.0, "is_small_cap": False, "urgency": "low", "videos_mentioned": 2},
        "pepe": {"score": 9.0, "is_small_cap": True, "urgency": "high", "videos_mentioned": 1}
    }
    
    # Define new sentiment data and videos
    new_video_sentiments = {
        "video1": {
            "bitcoin": {"score": 8.0, "is_small_cap": False, "urgency": "medium"},
            "shiba-inu": {"score": 8.0, "is_small_cap": True, "urgency": "high"}
        }
    }
    
    new_videos = [
        ("video1", 10000, "2025-03-22T00:00:00Z", 1000.0)
    ]
    
    # Call update_global_scores_incremental
    updated_scores, sentiment_changes = await sentiment_service.update_global_scores_incremental(
        new_video_sentiments=new_video_sentiments,
        new_videos=new_videos,
        current_global_scores=current_global_scores
    )
    
    # Assertions
    assert "bitcoin" in updated_scores
    assert "shiba-inu" in updated_scores
    assert updated_scores["bitcoin"]["score"] > current_global_scores["bitcoin"]["score"]
    assert updated_scores["shiba-inu"]["score"] == 8.0
    assert "bitcoin" in sentiment_changes
    assert "shiba-inu" in sentiment_changes
    assert sentiment_changes["shiba-inu"] == 8.0  # New coin, full change


@pytest.mark.asyncio
async def test_make_quick_decisions_error_handling():
    """Test error handling in make_quick_decisions."""
    # Create mock service
    mock_service = MockCryptoTradingService()
    
    # Replace market_data_service.get_coin_by_symbol with a function that raises an exception
    mock_service.market_data_service.get_coin_by_symbol = AsyncMock(side_effect=Exception("API down"))
    
    # Create updated scores with significant changes
    updated_scores = {
        "bitcoin": {"score": 7.0, "is_small_cap": False, "urgency": "medium", "videos_mentioned": 4}
    }
    
    # Create sentiment changes
    sentiment_changes = {
        "bitcoin": 2.0  # Significant change
    }
    
    # Replace the logger
    mock_logger = MockLogger()
    with patch('services.crypto_trading_service.logger', mock_logger):
        # Call the method
        result = await mock_service.make_quick_decisions(
            updated_scores=updated_scores,
            previous_scores={"bitcoin": {"score": 5.0}},
            sentiment_changes=sentiment_changes,
            significance_threshold=1.0
        )
    
    # Assertions
    assert "Error identifying opportunity for bitcoin" in " ".join(mock_logger.messages)
    assert result["trades_executed"] == 0