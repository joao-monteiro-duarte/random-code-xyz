"""
Service for fetching and analyzing cryptocurrency market data.
Focuses on smaller coins with high volatility for trading opportunities.
"""
import aiohttp
import logging
import json
import os
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class MarketDataService:
    """
    Service for fetching cryptocurrency market data from external APIs.
    Specialized in identifying small-cap coins with high trading potential.
    """
    
    def __init__(self, api_key: Optional[str] = None, test_mode: bool = False):
        """
        Initialize the market data service.
        
        Args:
            api_key: API key for CoinGecko Pro (optional)
            test_mode: Parameter for compatibility with CryptoTradingService (not used)
        """
        self.api_key = api_key or os.getenv("COINGECKO_API_KEY")
        self.base_url = "https://api.coingecko.com/api/v3"
        self.session = None
        self.market_data_cache = {}
        self.small_cap_coins_cache = []
        self.last_cache_update = None
        self.cache_ttl = 3600  # 1 hour
        # test_mode parameter ignored - just for compatibility
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.5  # seconds between requests (free tier = 30 req/min)
        self.rate_limit_remaining = 30  # Free tier default
        self.rate_limit_reset_time = 0
        self.backoff_time = 0
    
    async def _respect_rate_limits(self):
        """Respect API rate limits with exponential backoff on errors"""
        now = time.time()
        
        # Check if we need to wait for rate limit reset
        if self.rate_limit_remaining <= 1 and now < self.rate_limit_reset_time:
            wait_time = self.rate_limit_reset_time - now + 1  # Add 1 second buffer
            logger.warning(f"Rate limit almost reached. Waiting {wait_time:.1f}s for reset")
            await asyncio.sleep(wait_time)
            return
        
        # Apply backoff if there was a previous error
        if self.backoff_time > 0 and now < self.backoff_time:
            wait_time = self.backoff_time - now + 0.5  # Add small buffer
            logger.warning(f"In backoff period. Waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            return
        
        # Normal rate limiting - ensure minimum interval between requests
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        # Update last request time
        self.last_request_time = time.time()
    
    async def _handle_api_response(self, response):
        """Handle API response and update rate limit tracking"""
        # Check for rate limit headers
        if 'x-ratelimit-remaining' in response.headers:
            self.rate_limit_remaining = int(response.headers['x-ratelimit-remaining'])
            
        if 'x-ratelimit-reset' in response.headers:
            self.rate_limit_reset_time = int(response.headers['x-ratelimit-reset'])
        
        # Handle rate limiting errors
        if response.status == 429:  # Too Many Requests
            # Exponential backoff - wait 10 seconds by default, more if we're close to limit
            backoff_seconds = 10 + (30 - self.rate_limit_remaining) * 2 if self.rate_limit_remaining > 0 else 30
            self.backoff_time = time.time() + backoff_seconds
            logger.warning(f"Rate limit exceeded. Backing off for {backoff_seconds}s")
            return None
            
        # Handle other errors
        if response.status != 200:
            if response.status == 400:  # Bad Request
                logger.error(f"Bad request error: {response.status}")
                # Short backoff for bad requests
                self.backoff_time = time.time() + 5
            elif response.status >= 500:  # Server Error
                # Longer backoff for server errors
                self.backoff_time = time.time() + 10
                logger.error(f"Server error: {response.status}")
            return None
            
        # Return the response data
        return await response.json()
            
    async def initialize(self):
        """Initialize the market data service."""
        logger.info("Initializing MarketDataService")
        self.session = aiohttp.ClientSession()
        
        # Initial fetch of small cap coins
        await self.refresh_small_cap_coins()
        logger.info(f"MarketDataService initialized with {len(self.small_cap_coins_cache)} small-cap coins")
        
    async def close(self):
        """Close resources."""
        if self.session:
            await self.session.close()
        logger.info("MarketDataService closed")
        
    async def refresh_small_cap_coins(self, market_cap_threshold: float = 50_000_000):
        """
        Fetch and cache a list of coins with market cap below threshold.
        
        Args:
            market_cap_threshold: Maximum market cap in USD (default: $50M)
        """
        try:
            # Respect rate limits
            await self._respect_rate_limits()
            
            headers = {}
            if self.api_key:
                # Use demo header for keys starting with CG-
                if self.api_key.startswith("CG-"):
                    headers["x-cg-demo-api-key"] = self.api_key
                else:
                    headers["x-cg-pro-api-key"] = self.api_key
            
            # Fetch coins sorted by market cap (ascending)
            async with self.session.get(
                f"{self.base_url}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_asc", 
                    "per_page": 250,
                    "page": 1,
                    "sparkline": "false"
                },
                headers=headers
            ) as response:
                data = await self._handle_api_response(response)
                
                if not data:
                    logger.warning("Failed to fetch small cap coins, keeping existing cache")
                    return
                
                # Filter coins below the threshold
                small_caps = [
                    {
                        "id": coin["id"],
                        "symbol": coin["symbol"].upper(),
                        "name": coin["name"],
                        "market_cap": coin.get("market_cap", 0),
                        "current_price": coin.get("current_price", 0),
                        "price_change_24h": coin.get("price_change_percentage_24h", 0),
                        "volume_24h": coin.get("total_volume", 0)
                    }
                    for coin in data
                    if coin.get("market_cap", 0) is not None and coin.get("market_cap", 0) <= market_cap_threshold
                ]
                
                # Only update cache if we got valid data
                if small_caps:
                    # Sort by volatility (24h price change, absolute value)
                    small_caps.sort(key=lambda x: abs(x.get("price_change_24h", 0) or 0), reverse=True)
                    self.small_cap_coins_cache = small_caps
                    self.last_cache_update = datetime.now()
                    
                    logger.info(f"Refreshed small cap coins cache with {len(small_caps)} coins")
                    
        except Exception as e:
            logger.error(f"Error refreshing small cap coins: {e}")
            
    async def get_small_cap_coins(self, market_cap_threshold: float = 50_000_000) -> List[Dict]:
        """
        Get a list of small-cap coins with market cap below threshold.
        
        Args:
            market_cap_threshold: Maximum market cap in USD (default: $50M)
            
        Returns:
            List of coin dictionaries
        """
        # Refresh cache if needed
        now = datetime.now()
        if not self.last_cache_update or (now - self.last_cache_update).total_seconds() > self.cache_ttl:
            await self.refresh_small_cap_coins(market_cap_threshold)
            
        return self.small_cap_coins_cache
        
    async def get_coin_data(self, coin_id: str) -> Optional[Dict]:
        """
        Get detailed data for a specific coin.
        
        Args:
            coin_id: CoinGecko coin ID
            
        Returns:
            Dictionary with coin data or None if not found
        """
        try:
            # Check cache first
            if coin_id in self.market_data_cache:
                cache_entry = self.market_data_cache[coin_id]
                cache_age = (datetime.now() - cache_entry["timestamp"]).total_seconds()
                if cache_age < 300:  # 5 minutes TTL for coin data
                    return cache_entry["data"]
            
            # Respect rate limits for API calls
            await self._respect_rate_limits()
            
            # Fetch fresh data
            headers = {}
            if self.api_key:
                # Use demo header for keys starting with CG-
                if self.api_key.startswith("CG-"):
                    headers["x-cg-demo-api-key"] = self.api_key
                else:
                    headers["x-cg-pro-api-key"] = self.api_key
                
            async with self.session.get(
                f"{self.base_url}/coins/{coin_id}",
                params={
                    "localization": "false",
                    "tickers": "false", 
                    "market_data": "true",
                    "community_data": "false",
                    "developer_data": "false"
                },
                headers=headers
            ) as response:
                data = await self._handle_api_response(response)
                
                if not data:
                    logger.warning(f"Failed to fetch data for coin {coin_id}")
                    return None
                
                # Extract relevant data
                result = {
                    "id": data["id"],
                    "symbol": data["symbol"].upper(),
                    "name": data["name"],
                    "market_cap": data.get("market_data", {}).get("market_cap", {}).get("usd"),
                    "current_price": data.get("market_data", {}).get("current_price", {}).get("usd"),
                    "price_change_24h": data.get("market_data", {}).get("price_change_percentage_24h"),
                    "volume_24h": data.get("market_data", {}).get("total_volume", {}).get("usd"),
                    "high_24h": data.get("market_data", {}).get("high_24h", {}).get("usd"),
                    "low_24h": data.get("market_data", {}).get("low_24h", {}).get("usd")
                }
                
                # Cache the result
                self.market_data_cache[coin_id] = {
                    "data": result,
                    "timestamp": datetime.now()
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Error fetching data for coin {coin_id}: {e}")
            return None
            
    async def get_market_cap(self, coin_id: str) -> Optional[float]:
        """
        Get market cap for a specific coin.
        
        Args:
            coin_id: CoinGecko coin ID
            
        Returns:
            Market cap in USD or None if not found
        """
        coin_data = await self.get_coin_data(coin_id)
        if coin_data:
            return coin_data.get("market_cap")
        return None
        
    async def get_coin_by_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Find a coin by its symbol.
        
        Args:
            symbol: Coin symbol (e.g., BTC)
            
        Returns:
            Coin data or None if not found
        """
        # Make sure small cap coins are loaded
        if not self.small_cap_coins_cache:
            await self.refresh_small_cap_coins()
            
        # First look in small cap coins
        normalized_symbol = symbol.upper()
        for coin in self.small_cap_coins_cache:
            if coin["symbol"] == normalized_symbol:
                return coin
                
        # If not found, try searching all coins with proper rate limiting
        try:
            # Respect rate limits
            await self._respect_rate_limits()
            
            headers = {}
            if self.api_key:
                # Use demo header for keys starting with CG-
                if self.api_key.startswith("CG-"):
                    headers["x-cg-demo-api-key"] = self.api_key
                else:
                    headers["x-cg-pro-api-key"] = self.api_key
                
            async with self.session.get(
                f"{self.base_url}/search",
                params={"query": symbol},
                headers=headers
            ) as response:
                data = await self._handle_api_response(response)
                
                if not data:
                    logger.warning(f"Failed to search for coin {symbol}")
                    return None
                
                coins = data.get("coins", [])
                
                if not coins:
                    return None
                    
                # Find exact symbol match
                for coin in coins:
                    if coin.get("symbol", "").upper() == normalized_symbol:
                        return await self.get_coin_data(coin["id"])
                        
                # If no exact match, return the first result
                return await self.get_coin_data(coins[0]["id"])
                
        except Exception as e:
            logger.error(f"Error searching for coin {symbol}: {e}")
            return None
            
    async def detect_volatility(self, coin_id: str) -> Dict:
        """
        Calculate volatility metrics for a coin.
        
        Args:
            coin_id: CoinGecko coin ID
            
        Returns:
            Dictionary with volatility metrics
        """
        coin_data = await self.get_coin_data(coin_id)
        if not coin_data:
            return {"volatility": 0, "is_volatile": False}
            
        # Calculate volatility ratio (high-low range relative to price)
        high = coin_data.get("high_24h", 0) or 0
        low = coin_data.get("low_24h", 0) or 0
        current = coin_data.get("current_price", 0) or 0
        
        # If high/low not available, estimate from price_change_24h
        if (high == 0 or low == 0) and current > 0 and coin_data.get("price_change_24h"):
            change_pct = abs(coin_data.get("price_change_24h", 0))
            high = current * (1 + change_pct/100)
            low = current * (1 - change_pct/100)
        
        if current == 0 or low == 0:
            volatility = 0
        else:
            volatility = (high - low) / current * 100  # Percentage range
            
        # Volume to market cap ratio (higher means more trading relative to size)
        volume = coin_data.get("volume_24h", 0) or 0
        market_cap = coin_data.get("market_cap", 0) or 0
        
        if market_cap == 0:
            volume_to_mc = 0
        else:
            volume_to_mc = volume / market_cap
            
        # Determine if coin is highly volatile
        is_volatile = volatility > 10 or volume_to_mc > 0.2
        
        return {
            "volatility": volatility,
            "volume_to_market_cap": volume_to_mc,
            "is_volatile": is_volatile
        }
        
    async def identify_potential_trades(self, sentiment_data: Dict[str, float]) -> List[Dict]:
        """
        Identify potential trade opportunities by combining sentiment data with market data.
        
        Args:
            sentiment_data: Dictionary mapping coin symbols to sentiment scores (-10 to +10)
            
        Returns:
            List of trade opportunities with combined metrics
        """
        opportunities = []
        
        for symbol, sentiment in sentiment_data.items():
            # Skip coins with neutral or negative sentiment
            if sentiment <= 7:
                continue
                
            # Find coin data
            coin_data = await self.get_coin_by_symbol(symbol)
            if not coin_data:
                continue
                
            # Get volatility metrics
            volatility = await self.detect_volatility(coin_data["id"])
            market_cap = coin_data.get("market_cap", 0)
            
            # Score this opportunity (higher is better)
            # Weight small cap coins with high sentiment and volatility
            if market_cap > 0 and market_cap <= 50_000_000:
                # Calculate a score from 0-100
                score = min(100, (
                    (sentiment - 7) / 3 * 40 +  # Sentiment contribution (40%)
                    min(volatility["volatility"], 20) / 20 * 30 +  # Volatility contribution (30%)
                    min(volatility["volume_to_market_cap"], 0.5) / 0.5 * 20 +  # Volume/MC contribution (20%)
                    (1 - min(market_cap, 50_000_000) / 50_000_000) * 10  # Small cap bonus (10%)
                ))
                
                # Create opportunity entry
                opportunities.append({
                    "id": coin_data["id"],
                    "symbol": coin_data["symbol"],
                    "name": coin_data["name"],
                    "market_cap": market_cap,
                    "current_price": coin_data.get("current_price", 0),
                    "sentiment_score": sentiment,
                    "volatility": volatility["volatility"],
                    "volume_to_market_cap": volatility["volume_to_market_cap"],
                    "is_volatile": volatility["is_volatile"],
                    "opportunity_score": score
                })
        
        # Sort by opportunity score (highest first)
        opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
        return opportunities

    async def calculate_macd(self, coin_id: str, days: int = 30) -> Dict:
        """
        Calculate the MACD (Moving Average Convergence Divergence) for a coin.
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of historical data to use
            
        Returns:
            Dictionary with MACD metrics and signals
        """
        try:
            # Respect rate limits
            await self._respect_rate_limits()
            
            # Get historical price data from CoinGecko
            headers = {}
            if self.api_key:
                # Use demo header for keys starting with CG-
                if self.api_key.startswith("CG-"):
                    headers["x-cg-demo-api-key"] = self.api_key
                else:
                    headers["x-cg-pro-api-key"] = self.api_key
                
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for API
            from_timestamp = int(start_date.timestamp())
            to_timestamp = int(end_date.timestamp())
            
            async with self.session.get(
                f"{self.base_url}/coins/{coin_id}/market_chart/range",
                params={
                    "vs_currency": "usd",
                    "from": from_timestamp,
                    "to": to_timestamp
                },
                headers=headers
            ) as response:
                data = await self._handle_api_response(response)
                
                if not data:
                    logger.error(f"Failed to fetch historical data for {coin_id}")
                    return {
                        "macd": 0,
                        "macd_signal": 0,
                        "macd_histogram": 0,
                        "is_bullish": False,
                        "trend_strength": 0,
                        "historical_prices": []
                    }
                
                prices = data.get("prices", [])
                
                if not prices or len(prices) < 15:  # Need at least 15 data points for meaningful MACD
                    logger.warning(f"Insufficient price data for {coin_id}")
                    return {
                        "macd": 0,
                        "macd_signal": 0,
                        "macd_histogram": 0,
                        "is_bullish": False,
                        "trend_strength": 0,
                        "historical_prices": []
                    }
                
                # Convert to pandas DataFrame
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                
                # Calculate EMAs
                df['ema12'] = df['price'].ewm(span=12, adjust=False).mean()
                df['ema26'] = df['price'].ewm(span=26, adjust=False).mean()
                
                # Calculate MACD line
                df['macd'] = df['ema12'] - df['ema26']
                
                # Calculate signal line
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                
                # Calculate MACD histogram
                df['macd_histogram'] = df['macd'] - df['macd_signal']
                
                # Get last values
                last_macd = float(df['macd'].iloc[-1])
                last_signal = float(df['macd_signal'].iloc[-1])
                last_histogram = float(df['macd_histogram'].iloc[-1])
                
                # Calculate trend strength (0-1)
                max_histogram = max(abs(df['macd_histogram'].max()), abs(df['macd_histogram'].min()))
                trend_strength = abs(last_histogram) / max_histogram if max_histogram > 0 else 0
                
                # Determine if bullish
                is_bullish = last_macd > last_signal and last_histogram > 0
                
                # Extract recent prices
                recent_prices = [float(p) for _, p in prices[-6:]]
                
                return {
                    "macd": last_macd,
                    "macd_signal": last_signal,
                    "macd_histogram": last_histogram,
                    "is_bullish": is_bullish,
                    "trend_strength": float(trend_strength),
                    "historical_prices": recent_prices
                }
                
        except Exception as e:
            logger.error(f"Error calculating MACD for {coin_id}: {e}")
            return {
                "macd": 0,
                "macd_signal": 0,
                "macd_histogram": 0,
                "is_bullish": False,
                "trend_strength": 0,
                "historical_prices": []
            }