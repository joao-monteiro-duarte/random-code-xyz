"""
Service for executing and tracking cryptocurrency trades based on sentiment analysis.
Handles order placement, stop losses, and risk management.
"""
import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time
import uuid
import hashlib

logger = logging.getLogger(__name__)

# Check if ccxt is available (optional dependency)
try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    logger.warning("ccxt not available, trade execution will be simulated")
    CCXT_AVAILABLE = False

class TradeService:
    """
    Service for executing trades based on sentiment scores and market data.
    Supports real trading via ccxt or simulated trading for testing.
    """
    
    def __init__(self, exchange_id: str = "binance", api_key: Optional[str] = None, api_secret: Optional[str] = None, test_mode: bool = True):
        """
        Initialize the trade service.
        
        Args:
            exchange_id: Exchange ID for ccxt (e.g., binance, kraken)
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            test_mode: Whether to use test mode (simulated trading)
        """
        self.exchange_id = exchange_id
        self.api_key = api_key or os.getenv("EXCHANGE_API_KEY")
        self.api_secret = api_secret or os.getenv("EXCHANGE_API_SECRET")
        self.test_mode = test_mode
        self.exchange = None
        
        # Portfolio and trade tracking
        self.portfolio = {
            "USD": 10000.0,  # Start with $10,000 in simulated mode
            "total_value": 10000.0,
            "coins": {}
        }
        self.trade_history = []
        self.open_orders = []
        self.stop_losses = {}
        
        # Risk management settings
        self.max_position_pct = 0.02  # 2% max position size
        self.stop_loss_pct = 0.10     # 10% stop loss
        self.take_profit_pct = 0.20   # 20% take profit
        self.sentiment_thresholds = {
            "buy": 7.0,       # Min sentiment score for buy
            "strong_buy": 9.0,  # Strong buy signal
            "sell": 0.0,      # Sell if sentiment drops below this
            "strong_sell": -5.0  # Strong sell signal
        }
        
    async def initialize(self):
        """Initialize the trade service."""
        logger.info(f"Initializing TradeService (exchange: {self.exchange_id}, test_mode: {self.test_mode})")
        
        if CCXT_AVAILABLE and not self.test_mode:
            try:
                # Initialize the exchange
                exchange_class = getattr(ccxt, self.exchange_id)
                self.exchange = exchange_class({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True,
                })
                
                # Test connection
                await self.exchange.load_markets()
                logger.info(f"Connected to {self.exchange_id} exchange")
            except Exception as e:
                logger.error(f"Error initializing exchange: {e}")
                logger.warning("Falling back to test mode")
                self.test_mode = True
        elif not CCXT_AVAILABLE:
            logger.warning("ccxt not available, running in test mode")
            self.test_mode = True
            
        # In test mode, load saved portfolio if available
        if self.test_mode:
            try:
                if os.path.exists("data/portfolio.json"):
                    with open("data/portfolio.json", "r") as f:
                        self.portfolio = json.load(f)
                if os.path.exists("data/trade_history.json"):
                    with open("data/trade_history.json", "r") as f:
                        self.trade_history = json.load(f)
                logger.info("Loaded portfolio and trade history from file")
            except Exception as e:
                logger.error(f"Error loading portfolio: {e}")
                # Use default portfolio
                
        logger.info(f"TradeService initialized with portfolio value: ${self.portfolio['total_value']:.2f}")
        
    async def close(self):
        """Close resources and save state."""
        if CCXT_AVAILABLE and self.exchange and not self.test_mode:
            await self.exchange.close()
            
        # Save portfolio and trade history in test mode
        if self.test_mode:
            os.makedirs("data", exist_ok=True)
            with open("data/portfolio.json", "w") as f:
                json.dump(self.portfolio, f, indent=2)
            with open("data/trade_history.json", "w") as f:
                json.dump(self.trade_history, f, indent=2)
            logger.info("Saved portfolio and trade history to file")
            
        logger.info("TradeService closed")
        
    async def execute_trade(self, symbol: str, action: str, sentiment_score: float, 
                           vph: float, market_cap: float, price: Optional[float] = None,
                           position_size: Optional[float] = None) -> Dict:
        """
        Execute a trade based on sentiment score and VPH.
        
        Args:
            symbol: Coin symbol (e.g., BTC)
            action: Trade action (buy, sell)
            sentiment_score: Sentiment score from -10 to +10
            vph: Views per hour (engagement metric)
            market_cap: Market cap in USD
            price: Price override (optional)
            position_size: Position size as a percentage of portfolio (optional, overrides calculation)
            
        Returns:
            Dictionary with trade details
        """
        normalized_symbol = symbol.upper()
        trade_id = self._generate_trade_id(normalized_symbol, action)
        timestamp = datetime.now().isoformat()
        
        try:
            # Calculate position size based on portfolio value, sentiment and VPH
            if position_size is not None:
                # Use provided position size (as a percentage of portfolio)
                position_size_usd = self.portfolio["total_value"] * position_size
                logger.info(f"Using provided position size: {position_size:.2%} of portfolio (${position_size_usd:.2f})")
            else:
                # Calculate position size based on sentiment, VPH, and market cap
                position_size_usd = self._calculate_position_size(normalized_symbol, sentiment_score, vph, market_cap)
            
            # Ensure minimum position size ($10)
            position_size_usd = max(10, position_size_usd)
            
            if self.test_mode:
                # Simulated trade
                return await self._simulated_trade(normalized_symbol, action, position_size_usd, sentiment_score, price, trade_id, timestamp)
            else:
                # Real trade via ccxt
                return await self._real_trade(normalized_symbol, action, position_size_usd, price, trade_id, timestamp)
                
        except Exception as e:
            logger.error(f"Error executing {action} trade for {normalized_symbol}: {e}")
            return {
                "id": trade_id,
                "symbol": normalized_symbol,
                "action": action,
                "status": "error",
                "error": str(e),
                "timestamp": timestamp
            }
            
    async def _simulated_trade(self, symbol: str, action: str, position_size_usd: float, 
                              sentiment_score: float, price: Optional[float], trade_id: str, timestamp: str) -> Dict:
        """Execute a simulated trade."""
        if price is None:
            # Generate a realistic price in test mode
            price = 100.0  # Default price
            seed = int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 10000
            price = seed / 100.0  # Price between 0.01 and 100.00
            
            # For small caps, use lower prices
            if "is_small_cap" in symbol or len(symbol) > 4:
                price /= 10  # Smaller price for small caps
                
        # Check if we have enough USD for buying
        if action == "buy" and position_size_usd > self.portfolio["USD"]:
            position_size_usd = self.portfolio["USD"]  # Limit to available USD
            if position_size_usd <= 0:
                return {
                    "id": trade_id,
                    "symbol": symbol,
                    "action": action,
                    "status": "error",
                    "error": "Insufficient USD balance",
                    "timestamp": timestamp
                }
                
        # Calculate coin amount
        amount = position_size_usd / price if action == "buy" else self._get_coin_balance(symbol)
        
        # For sell, check if we have enough coins
        if action == "sell" and (symbol not in self.portfolio["coins"] or amount <= 0):
            return {
                "id": trade_id,
                "symbol": symbol,
                "action": action,
                "status": "error",
                "error": f"Insufficient {symbol} balance",
                "timestamp": timestamp
            }
            
        # Execute the simulated trade
        if action == "buy":
            # Deduct USD
            self.portfolio["USD"] -= position_size_usd
            
            # Add coins
            if symbol in self.portfolio["coins"]:
                self.portfolio["coins"][symbol]["amount"] += amount
                # Calculate new average price
                old_amount = self.portfolio["coins"][symbol]["amount"] - amount
                old_price = self.portfolio["coins"][symbol]["avg_price"]
                new_amount = self.portfolio["coins"][symbol]["amount"]
                self.portfolio["coins"][symbol]["avg_price"] = (old_amount * old_price + amount * price) / new_amount
            else:
                self.portfolio["coins"][symbol] = {
                    "amount": amount,
                    "avg_price": price
                }
                
            # Set stop loss
            stop_loss_price = price * (1 - self.stop_loss_pct)
            take_profit_price = price * (1 + self.take_profit_pct)
            self.stop_losses[symbol] = {
                "price": stop_loss_price,
                "amount": amount
            }
            
            logger.info(f"Simulated BUY: {amount:.6f} {symbol} @ ${price:.2f} = ${position_size_usd:.2f}")
            
        elif action == "sell":
            # Add USD
            self.portfolio["USD"] += amount * price
            
            # Remove coins
            if symbol in self.portfolio["coins"]:
                self.portfolio["coins"][symbol]["amount"] -= amount
                if self.portfolio["coins"][symbol]["amount"] <= 0:
                    del self.portfolio["coins"][symbol]
                    # Remove stop loss
                    if symbol in self.stop_losses:
                        del self.stop_losses[symbol]
                        
            logger.info(f"Simulated SELL: {amount:.6f} {symbol} @ ${price:.2f} = ${amount * price:.2f}")
            
        # Update portfolio value
        self._update_portfolio_value()
        
        # Add to trade history
        trade = {
            "id": trade_id,
            "symbol": symbol,
            "action": action,
            "amount": amount,
            "price": price,
            "value_usd": amount * price,
            "sentiment_score": sentiment_score,
            "status": "completed",
            "timestamp": timestamp
        }
        
        if action == "buy":
            trade["stop_loss"] = stop_loss_price
            trade["take_profit"] = take_profit_price
            
        self.trade_history.append(trade)
        return trade
        
    async def _real_trade(self, symbol: str, action: str, position_size_usd: float, 
                         price: Optional[float], trade_id: str, timestamp: str) -> Dict:
        """Execute a real trade via ccxt."""
        if not CCXT_AVAILABLE or not self.exchange:
            logger.error("Cannot execute real trade: exchange not available")
            return {
                "id": trade_id,
                "symbol": symbol,
                "action": action,
                "status": "error",
                "error": "Exchange not available",
                "timestamp": timestamp
            }
            
        try:
            # Convert the symbol to exchange format
            market_symbol = f"{symbol}/USDT"
            
            # Get current market data
            ticker = await self.exchange.fetch_ticker(market_symbol)
            current_price = ticker["last"] if price is None else price
            
            # Calculate amount to buy/sell
            amount = position_size_usd / current_price
            
            # Create the order
            order_type = "market"
            side = "buy" if action == "buy" else "sell"
            
            # Execute the order
            order = await self.exchange.create_order(market_symbol, order_type, side, amount)
            
            # Set stop loss for buys
            stop_loss_order = None
            take_profit_order = None
            
            if action == "buy" and self.exchange.has["createOrder"]:
                # Set stop loss
                stop_loss_price = current_price * (1 - self.stop_loss_pct)
                try:
                    stop_loss_order = await self.exchange.create_order(
                        market_symbol, "stop_loss", "sell", amount, 
                        {"stopPrice": stop_loss_price}
                    )
                except Exception as e:
                    logger.error(f"Error setting stop loss: {e}")
                    
                # Set take profit
                take_profit_price = current_price * (1 + self.take_profit_pct)
                try:
                    take_profit_order = await self.exchange.create_order(
                        market_symbol, "take_profit", "sell", amount,
                        {"stopPrice": take_profit_price}
                    )
                except Exception as e:
                    logger.error(f"Error setting take profit: {e}")
                    
            # Create trade record
            trade = {
                "id": trade_id,
                "symbol": symbol,
                "action": action,
                "amount": amount,
                "price": current_price,
                "value_usd": amount * current_price,
                "order_id": order["id"],
                "status": "completed",
                "timestamp": timestamp
            }
            
            if stop_loss_order:
                trade["stop_loss_order"] = stop_loss_order["id"]
                trade["stop_loss_price"] = stop_loss_price
                
            if take_profit_order:
                trade["take_profit_order"] = take_profit_order["id"]
                trade["take_profit_price"] = take_profit_price
                
            # Add to trade history
            self.trade_history.append(trade)
            
            logger.info(f"Executed {action.upper()}: {amount:.6f} {symbol} @ ${current_price:.2f}")
            return trade
            
        except Exception as e:
            logger.error(f"Error executing real trade: {e}")
            return {
                "id": trade_id,
                "symbol": symbol,
                "action": action,
                "status": "error",
                "error": str(e),
                "timestamp": timestamp
            }
            
    def _calculate_position_size(self, symbol: str, sentiment_score: float, vph: float, market_cap: float) -> float:
        """
        Calculate position size based on portfolio value, sentiment and VPH.
        Uses a risk-based approach that scales with sentiment and VPH but is limited by max position size.
        
        Args:
            symbol: Coin symbol
            sentiment_score: Sentiment score from -10 to +10
            vph: Views per hour (engagement metric)
            market_cap: Market cap in USD
            
        Returns:
            Position size in USD
        """
        # Base position is a percentage of portfolio
        base_position_pct = self.max_position_pct / 2  # Start at half the max
        
        # Adjust based on sentiment (normalize from 7-10 to 0-1)
        sentiment_weight = max(0, min(1, (sentiment_score - 7) / 3))
        
        # Adjust based on VPH (normalize with diminishing returns)
        vph_weight = min(1, vph / 2000)  # Cap at 2000 VPH
        
        # Adjust based on market cap (smaller caps get higher weight)
        if market_cap <= 0:
            market_cap_weight = 1  # Default if market cap unknown
        else:
            # Scale from 0-1, higher for smaller caps
            market_cap_weight = max(0, min(1, 1 - (market_cap / 50_000_000)))
            
        # Calculate final position percentage
        position_pct = base_position_pct + (self.max_position_pct - base_position_pct) * (
            0.4 * sentiment_weight +  # 40% weight to sentiment
            0.3 * vph_weight +       # 30% weight to VPH
            0.3 * market_cap_weight  # 30% weight to market cap
        )
        
        # Cap at max position size
        position_pct = min(position_pct, self.max_position_pct)
        
        # Calculate dollar amount
        position_size_usd = self.portfolio["total_value"] * position_pct
        
        # Ensure minimum position size ($10)
        position_size_usd = max(10, position_size_usd)
        
        return position_size_usd
        
    def _get_coin_balance(self, symbol: str) -> float:
        """Get coin balance from portfolio."""
        if symbol in self.portfolio["coins"]:
            return self.portfolio["coins"][symbol]["amount"]
        return 0
        
    def _update_portfolio_value(self):
        """Update the total portfolio value."""
        # Sum USD and all coin values
        total = self.portfolio["USD"]
        
        # Add value of all coins
        for symbol, data in self.portfolio["coins"].items():
            total += data["amount"] * data["avg_price"]
            
        self.portfolio["total_value"] = total
        
    def _generate_trade_id(self, symbol: str, action: str) -> str:
        """Generate a unique trade ID."""
        timestamp = int(time.time() * 1000)
        random_id = str(uuid.uuid4())[:8]
        return f"{symbol}-{action}-{timestamp}-{random_id}"
        
    async def check_stop_losses(self, current_prices: Dict[str, float]):
        """
        Check if any stop losses have been triggered.
        
        Args:
            current_prices: Dictionary mapping coin symbols to current prices
        
        Returns:
            List of executed stop loss trades
        """
        triggered_trades = []
        
        for symbol, stop_loss in list(self.stop_losses.items()):
            if symbol in current_prices and current_prices[symbol] <= stop_loss["price"]:
                # Stop loss triggered
                logger.info(f"Stop loss triggered for {symbol} at ${current_prices[symbol]:.2f}")
                
                # Execute the sell
                trade = await self.execute_trade(
                    symbol=symbol,
                    action="sell",
                    sentiment_score=0,  # Not relevant for stop loss
                    vph=0,  # Not relevant for stop loss
                    market_cap=0,  # Not relevant for stop loss
                    price=current_prices[symbol]
                )
                
                trade["reason"] = "stop_loss"
                triggered_trades.append(trade)
                
                # Remove stop loss
                del self.stop_losses[symbol]
                
        return triggered_trades
        
    def get_portfolio(self) -> Dict:
        """Get the current portfolio state."""
        return self.portfolio
        
    def get_trade_history(self) -> List[Dict]:
        """Get the trade history."""
        return self.trade_history
        
    def get_recent_trades(self, symbol: str, limit: int = 5) -> List[Dict]:
        """
        Get recent trades for a specific coin.
        
        Args:
            symbol: Coin symbol
            limit: Maximum number of trades to return
            
        Returns:
            List of recent trades for the coin
        """
        # Filter trades for the symbol
        symbol_trades = [
            trade for trade in self.trade_history 
            if trade.get("symbol") == symbol and trade.get("status") == "completed"
        ]
        
        # Sort by timestamp (newest first)
        symbol_trades.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Return limited number of trades
        return symbol_trades[:limit]
        
    def evaluate_sentiment_change(self, symbol: str, old_score: float, new_score: float) -> Optional[str]:
        """
        Evaluate sentiment change and recommend action.
        
        Args:
            symbol: Coin symbol
            old_score: Previous sentiment score
            new_score: New sentiment score
            
        Returns:
            Recommended action ('buy', 'sell', 'hold', or None)
        """
        # Check if we hold this coin
        holding = symbol in self.portfolio["coins"] and self.portfolio["coins"][symbol]["amount"] > 0
        
        # Significant sentiment drop for a coin we hold
        if holding and old_score >= self.sentiment_thresholds["buy"] and new_score <= self.sentiment_thresholds["sell"]:
            return "sell"
            
        # Significant sentiment increase for a coin we don't hold
        if not holding and old_score < self.sentiment_thresholds["buy"] and new_score >= self.sentiment_thresholds["buy"]:
            return "buy"
            
        # Strong sentiment signals
        if not holding and new_score >= self.sentiment_thresholds["strong_buy"]:
            return "buy"
            
        if holding and new_score <= self.sentiment_thresholds["strong_sell"]:
            return "sell"
            
        # Default is to hold
        return "hold"
        
    def make_decision(self, coin_data: Dict, sentiment_score: float, vph: float) -> Dict:
        """
        Make a trading decision based on coin data, sentiment and VPH.
        
        Args:
            coin_data: Dictionary with coin data
            sentiment_score: Sentiment score from -10 to +10
            vph: Views per hour
            
        Returns:
            Decision dictionary with action and reasoning
        """
        symbol = coin_data["symbol"]
        market_cap = coin_data.get("market_cap", 0)
        
        # Check if we already hold this coin
        holding = symbol in self.portfolio["coins"] and self.portfolio["coins"][symbol]["amount"] > 0
        
        # Determine action based on sentiment thresholds
        action = "hold"
        reasons = []
        
        if holding:
            # Sell decisions
            if sentiment_score <= self.sentiment_thresholds["strong_sell"]:
                action = "sell"
                reasons.append(f"Strong negative sentiment ({sentiment_score:.1f})")
            elif sentiment_score <= self.sentiment_thresholds["sell"]:
                action = "sell"
                reasons.append(f"Sentiment turned negative ({sentiment_score:.1f})")
        else:
            # Buy decisions
            if sentiment_score >= self.sentiment_thresholds["strong_buy"]:
                action = "buy"
                reasons.append(f"Very strong positive sentiment ({sentiment_score:.1f})")
            elif sentiment_score >= self.sentiment_thresholds["buy"]:
                # Consider buying, but check other factors
                if vph >= 500:
                    action = "buy"
                    reasons.append(f"Positive sentiment ({sentiment_score:.1f}) with high engagement (VPH: {vph:.1f})")
                    
                    # Check market cap for small coins
                    if market_cap > 0 and market_cap <= 50_000_000:
                        reasons.append(f"Small market cap (${market_cap/1000000:.1f}M)")
                else:
                    reasons.append(f"Positive sentiment ({sentiment_score:.1f}) but low engagement (VPH: {vph:.1f})")
        
        # Ensure USD balance for buys
        if action == "buy" and self.portfolio["USD"] < 10:
            action = "hold"
            reasons.append("Insufficient USD balance")
            
        return {
            "symbol": symbol,
            "action": action,
            "sentiment_score": sentiment_score,
            "vph": vph,
            "market_cap": market_cap,
            "reasons": reasons
        }