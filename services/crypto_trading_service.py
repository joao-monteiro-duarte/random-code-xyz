"""
CryptoTradingService - Top-level coordinator service for crypto trading application.
Provides dependency injection for the FastAPI endpoints and coordinates all services.
Specialized in identifying and trading small-cap cryptocurrencies based on sentiment analysis.
Implements a Langroid master agent with Claude 3 Sonnet for sophisticated trade decisions.
"""
import logging
import asyncio
import os
import json
import re
import redis
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

# Setup logging first to avoid reference errors
logger = logging.getLogger(__name__)

# Import services
from services.app_service import get_app_service, AppService
from services.transcript_service import TranscriptService
from services.sentiment_service import SentimentAnalysisService
from services.market_data_service import MarketDataService
from services.trade_service import TradeService
from config.settings import VPH_THRESHOLD, CYCLE_INTERVAL

# Import Langroid for Claude 3 Sonnet integration
try:
    import langroid as lr
    from langroid.language_models.anthropic_models import Claude3Sonnet
    LANGROID_AVAILABLE = True
except ImportError:
    LANGROID_AVAILABLE = False
    logger.warning("Langroid not available. Master agent will not be used.")

# Setup Redis for decision history
try:
    from fakeredis import FakeRedis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_TEST_MODE = os.getenv("REDIS_TEST_MODE", "true").lower() == "true"
    
    if REDIS_TEST_MODE:
        redis_client = FakeRedis()
        logger.info("Using FakeRedis for testing")
    else:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        
    REDIS_AVAILABLE = True
except (ImportError, redis.exceptions.ConnectionError) as e:
    REDIS_AVAILABLE = False
    redis_client = None
    logger.warning(f"Redis not available: {e}")
    logger.warning("Decision history will not be persistent")

# Import models
from models.video import Video

class CryptoTradingService:
    """
    Top-level coordinator service for the crypto trading application.
    Specializes in maximizing profits through YouTube-based sentiment analysis
    with a focus on smaller, more volatile cryptocurrencies.
    """
    
    def __init__(self, 
                 openrouter_api_key: Optional[str] = None,
                 coingecko_api_key: Optional[str] = None,
                 exchange_api_key: Optional[str] = None,
                 exchange_api_secret: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 test_mode: bool = True):
        """
        Initialize the CryptoTradingService.
        
        Args:
            openrouter_api_key: API key for OpenRouter (Mixtral access)
            coingecko_api_key: API key for CoinGecko (market data)
            exchange_api_key: API key for trading exchange
            exchange_api_secret: API secret for trading exchange
            anthropic_api_key: API key for Anthropic Claude 3 Sonnet
            test_mode: Whether to run in test mode (simulated trading)
        """
        # Get base services
        self.app_service = get_app_service()
        self.transcript_service = self.app_service.transcript_service
        
        # API keys from params or env vars
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.coingecko_api_key = coingecko_api_key or os.getenv("COINGECKO_API_KEY")
        self.exchange_api_key = exchange_api_key or os.getenv("EXCHANGE_API_KEY")
        self.exchange_api_secret = exchange_api_secret or os.getenv("EXCHANGE_API_SECRET")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        
        # Set up specialized services
        self.sentiment_service = SentimentAnalysisService(api_key=self.openrouter_api_key)
        self.market_data_service = MarketDataService(api_key=self.coingecko_api_key)
        self.trade_service = TradeService(
            exchange_id="binance",
            api_key=self.exchange_api_key,
            api_secret=self.exchange_api_secret,
            test_mode=test_mode
        )
        
        # Set up Langroid master agent with Claude 3 Sonnet if available
        self.master_agent = None
        if LANGROID_AVAILABLE and self.anthropic_api_key:
            try:
                # Create Claude 3 Sonnet LLM
                llm = Claude3Sonnet(api_key=self.anthropic_api_key)
                
                # Define system prompt for the master agent
                system_prompt = """
                You are an expert cryptocurrency trading advisor with deep knowledge of:
                1. Technical analysis and indicators (particularly MACD)
                2. Sentiment analysis interpretation
                3. Small-cap cryptocurrency opportunities
                4. Risk management principles
                
                Your role is to make sophisticated trading decisions combining:
                - Sentiment data from YouTube video analysis
                - Market data (price, market cap, volume)
                - Technical indicators (MACD)
                - Current portfolio allocation
                
                Focus especially on finding opportunities in small-cap cryptocurrencies 
                (market cap < $50M) that show positive sentiment on YouTube.
                
                Be cautious but decisive. Recommend strong buy signals when multiple 
                indicators align favorably. Always include your reasoning.
                """
                
                # Create agent config
                agent_config = lr.AgentConfig(
                    llm=llm,
                    system_prompt=system_prompt,
                    max_tokens=1000
                )
                
                # Create the agent
                self.master_agent = lr.Agent(agent_config)
                logger.info("Claude 3 Sonnet master agent initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Langroid master agent: {e}")
                self.master_agent = None
        
        # Initialize decision history storage
        self.decision_history = []
        self.decision_history_key = "trading_decision_history"
        
        # Trading state
        self.sentiment_history = {}  # Tracks sentiment over time
        self.current_opportunities = []  # Current trading opportunities
        self.is_initialized = False
        self.last_small_cap_refresh = None
        self.small_cap_coins = []  # List of small cap coins to monitor
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.portfolio_history = []
        
        # Load previous decision history from Redis if available
        if REDIS_AVAILABLE and redis_client:
            try:
                history_json = redis_client.get(self.decision_history_key)
                if history_json:
                    self.decision_history = json.loads(history_json)
                    logger.info(f"Loaded {len(self.decision_history)} previous trading decisions from Redis")
            except Exception as e:
                logger.error(f"Error loading decision history from Redis: {e}")
                self.decision_history = []
        
    async def initialize(self):
        """Initialize all services and load initial state."""
        if self.is_initialized:
            return
            
        logger.info("Initializing CryptoTradingService...")
        
        # Initialize all underlying services
        try:
            # First initialize the app service
            await self.app_service.initialize()
            
            # Now set the circular reference to avoid dependency loop
            self.app_service.crypto_trading_service = self
            logger.info("Set app_service.crypto_trading_service reference to break circular dependency")
            
            # Initialize other services
            await self.market_data_service.initialize()
            await self.trade_service.initialize()
            
            # Get initial list of small cap coins
            self.small_cap_coins = await self.market_data_service.get_small_cap_coins(50_000_000)
            self.last_small_cap_refresh = datetime.now()
            
            logger.info(f"Found {len(self.small_cap_coins)} small cap coins to monitor")
            
            # Save initial portfolio state
            portfolio = self.trade_service.get_portfolio()
            self.portfolio_history.append({
                "timestamp": datetime.now().isoformat(),
                "total_value": portfolio["total_value"],
                "allocation": {
                    coin: data["amount"] * data["avg_price"]
                    for coin, data in portfolio.get("coins", {}).items()
                }
            })
            
            self.is_initialized = True
            logger.info("CryptoTradingService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing CryptoTradingService: {e}", exc_info=True)
            raise
        
    async def run_cycle(self, vph_threshold: float = VPH_THRESHOLD, background: bool = True):
        """
        Run a full trading cycle that includes:
        1. Processing video transcripts
        2. Analyzing sentiment
        3. Identifying trading opportunities
        4. Executing trades
        
        Args:
            vph_threshold: Threshold for VPH filtering
            background: Whether to run in the background
            
        Returns:
            Dictionary with cycle results
        """
        if not self.is_initialized:
            await self.initialize()
            
        if background:
            # Run in background
            asyncio.create_task(self._run_trading_cycle(vph_threshold))
            return {"status": "started", "message": "Trading cycle started in background"}
        else:
            # Run synchronously and return result
            return await self._run_trading_cycle(vph_threshold)
    
    async def _run_trading_cycle(self, vph_threshold: float):
        """
        Implementation of the enhanced trading cycle focused on small-cap opportunities.
        
        Args:
            vph_threshold: Threshold for VPH filtering
            
        Returns:
            Dictionary with cycle results
        """
        if self.app_service.is_running:
            logger.warning("A cycle is already running, skipping")
            return {"status": "skipped", "message": "A cycle is already running"}
            
        try:
            # Set running flag
            self.app_service.is_running = True
            logger.info(f"Starting trading cycle with VPH threshold: {vph_threshold}")
            
            # 1. Refresh small cap coins list (once a day)
            now = datetime.now()
            if not self.last_small_cap_refresh or (now - self.last_small_cap_refresh).total_seconds() > 86400:
                self.small_cap_coins = await self.market_data_service.get_small_cap_coins(50_000_000)
                self.last_small_cap_refresh = now
                logger.info(f"Refreshed small cap coins list, monitoring {len(self.small_cap_coins)} coins")
            
            # 2. Process accumulated videos
            logger.info("Processing accumulated videos")
            accumulated_videos = self.app_service.get_accumulated_videos()
            
            if not accumulated_videos:
                logger.info("No accumulated videos to process")
                return {"status": "completed", "message": "No videos to process", "trades_executed": 0}
            
            logger.info(f"Processing {len(accumulated_videos)} accumulated videos")
            
            # 3. Update VPH for all videos
            updated_videos = []
            for video in accumulated_videos:
                if isinstance(video, Video):
                    video_id = video.id
                else:
                    video_id = video[0]
                    
                # Only process high VPH videos
                processed_video = self.app_service.process_video_stats(video)
                
                # Extract VPH for filtering
                if isinstance(processed_video, Video):
                    vph = processed_video.vph
                else:
                    vph = processed_video[3]
                    
                if vph >= vph_threshold:
                    updated_videos.append(processed_video)
            
            if not updated_videos:
                logger.info(f"No videos with VPH >= {vph_threshold}")
                return {"status": "completed", "message": "No high-VPH videos found", "trades_executed": 0}
                
            logger.info(f"Found {len(updated_videos)} videos with VPH >= {vph_threshold}")
            
            # 4. Get transcripts for filtered videos
            video_transcripts = []
            for video in updated_videos:
                if isinstance(video, Video):
                    video_id = video.id
                else:
                    video_id = video[0]
                
                transcript = self.transcript_service.get_transcript(video_id)
                if transcript and len(transcript) > 100:  # Ensure meaningful transcripts
                    video_transcripts.append((video_id, transcript))
            
            if not video_transcripts:
                logger.info("No valid transcripts found for analysis")
                return {"status": "completed", "message": "No valid transcripts found", "trades_executed": 0}
                
            logger.info(f"Analyzing {len(video_transcripts)} video transcripts")
            
            # 5. Analyze sentiment in transcripts
            all_sentiments = await self.sentiment_service.batch_analyze(video_transcripts)
            
            # 6. Calculate global sentiment scores with video weighting
            global_sentiment = await self.sentiment_service.calculate_global_scores(
                all_sentiments, updated_videos, vph_threshold
            )
            
            # 7. Store previous sentiment scores for comparison
            prev_sentiment = self.sentiment_history.copy()
            self.sentiment_history = global_sentiment
            
            # 8. Identify trading opportunities
            opportunities = await self._identify_opportunities(global_sentiment, prev_sentiment)
            
            if not opportunities:
                logger.info("No trading opportunities identified")
                return {"status": "completed", "message": "No trading opportunities found", "trades_executed": 0}
                
            logger.info(f"Identified {len(opportunities)} trading opportunities")
            self.current_opportunities = opportunities
            
            # 9. Execute trades
            executed_trades = await self._execute_trades(opportunities)
            
            # 10. Update portfolio tracking
            portfolio = self.trade_service.get_portfolio()
            self.portfolio_history.append({
                "timestamp": datetime.now().isoformat(),
                "total_value": portfolio["total_value"],
                "allocation": {
                    coin: data["amount"] * data["avg_price"]
                    for coin, data in portfolio.get("coins", {}).items()
                }
            })
            
            # Keep only last 30 portfolio snapshots
            if len(self.portfolio_history) > 30:
                self.portfolio_history = self.portfolio_history[-30:]
            
            # 11. Check for stop losses and take profits
            current_prices = {}
            for opportunity in opportunities:
                coin_id = opportunity["id"]
                coin_data = await self.market_data_service.get_coin_data(coin_id)
                if coin_data and coin_data.get("current_price"):
                    current_prices[coin_data["symbol"]] = coin_data["current_price"]
            
            if current_prices:
                triggered_trades = await self.trade_service.check_stop_losses(current_prices)
                if triggered_trades:
                    logger.info(f"Executed {len(triggered_trades)} stop-loss trades")
                    executed_trades.extend(triggered_trades)
            
            # Calculate cycle duration
            self.app_service.last_cycle_time = datetime.now()
            
            # Summarize results
            trades_summary = f"Executed {len(executed_trades)} trades"
            if executed_trades:
                trades_summary += ": " + ", ".join([f"{t['action'].upper()} {t['symbol']}" for t in executed_trades[:5]])
                if len(executed_trades) > 5:
                    trades_summary += f" and {len(executed_trades) - 5} more"
            
            result = {
                "status": "completed", 
                "message": f"Trading cycle completed successfully. {trades_summary}",
                "trades_executed": len(executed_trades),
                "opportunities_found": len(opportunities),
                "portfolio_value": portfolio["total_value"]
            }
            
            logger.info(f"Trading cycle completed with {len(executed_trades)} trades")
            return result
            
        except Exception as e:
            logger.error(f"Error during trading cycle: {e}", exc_info=True)
            return {"status": "error", "message": f"Error during trading cycle: {str(e)}", "trades_executed": 0}
        finally:
            self.app_service.is_running = False
    
    async def decide_trade(self, 
                      coin_symbol: str, 
                      coin_data: Dict, 
                      sentiment_data: Dict, 
                      portfolio: Dict) -> Dict:
        """
        Make a sophisticated trade decision using the Claude 3 Sonnet master agent.
        
        Args:
            coin_symbol: Coin symbol (e.g., BTC)
            coin_data: Market data for the coin
            sentiment_data: Sentiment analysis data
            portfolio: Current portfolio state
            
        Returns:
            Decision dictionary with action and reasoning
        """
        # Default decision if master agent not available
        default_decision = {
            "action": "hold",
            "reasons": ["Master agent not available, using default decision logic"],
            "confidence": 0.5
        }
        
        # Simple decision logic based on sentiment score
        sentiment_score = sentiment_data.get("score", 0)
        if sentiment_score > 8:
            default_decision["action"] = "buy"
            default_decision["reasons"] = ["High sentiment score (> 8)"]
            default_decision["confidence"] = min(sentiment_score / 10, 0.9)
        
        # Use master agent if available
        if not self.master_agent:
            logger.warning("Master agent not available, using simplified decision logic")
            return default_decision
            
        try:
            # Get MACD data
            macd_data = await self.market_data_service.calculate_macd(coin_data["id"])
            
            # Format recent portfolio history for context
            recent_trades = self.trade_service.get_recent_trades(coin_symbol, limit=5)
            
            # Format previous decisions for the coin
            previous_decisions = [
                decision for decision in self.decision_history[-10:] 
                if decision.get("symbol") == coin_symbol
            ]
            
            # Prepare the prompt for Claude
            prompt = f"""
            I need to make a trade decision for {coin_symbol} ({coin_data.get('name', '')}).
            
            MARKET DATA:
            - Current Price: ${coin_data.get('current_price', 0):.6f}
            - Market Cap: ${coin_data.get('market_cap', 0):,}
            - 24h Volume: ${coin_data.get('volume_24h', 0):,}
            - 24h Price Change: {coin_data.get('price_change_24h', 0):.2f}%
            - Volatility: {coin_data.get('volatility', {}).get('volatility', 0):.2f}%
            - Volume/Market Cap Ratio: {coin_data.get('volatility', {}).get('volume_to_market_cap', 0):.4f}
            
            SENTIMENT DATA:
            - Overall Score: {sentiment_data.get('score', 0):.1f}/10
            - Is Small Cap: {sentiment_data.get('is_small_cap', False)}
            - Urgency Level: {sentiment_data.get('urgency', 'low')}
            - Videos Mentioning: {sentiment_data.get('videos_mentioned', 0)}
            - Price Predictions: {sentiment_data.get('price_predictions', [])}
            - Reasons: {sentiment_data.get('reasons', [])}
            
            TECHNICAL INDICATORS:
            - MACD: {macd_data.get('macd', 0):.8f}
            - MACD Signal: {macd_data.get('macd_signal', 0):.8f}
            - MACD Histogram: {macd_data.get('macd_histogram', 0):.8f}
            - Is Bullish: {macd_data.get('is_bullish', False)}
            - Trend Strength: {macd_data.get('trend_strength', 0):.2f}
            - Recent Prices: {macd_data.get('historical_prices', [])}
            
            PORTFOLIO CONTEXT:
            - Current Allocation: {portfolio.get('coins', {}).get(coin_symbol, {}).get('amount', 0)} {coin_symbol}
            - Portfolio Value: ${portfolio.get('total_value', 0):,.2f}
            - Available Funds: ${portfolio.get('unallocated_value', 0):,.2f}
            
            PREVIOUS TRADES:
            {recent_trades}
            
            PREVIOUS DECISIONS:
            {previous_decisions}
            
            Based on ALL of this information, make a trade decision (buy, sell, or hold).
            Provide your reasoning and a confidence level (0.0-1.0).
            
            FORMAT YOUR RESPONSE AS:
            ACTION: [buy/sell/hold]
            CONFIDENCE: [0.0-1.0]
            REASONING: [detailed explanation]
            POSITION SIZE: [percentage of available funds, only for buy]
            """
            
            # Send to Claude 3 Sonnet via Langroid
            response = await self.master_agent.achat(prompt)
            response_text = response.msg.content
            
            # Parse response
            action_match = re.search(r"ACTION:\s*(\w+)", response_text)
            confidence_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response_text)
            reasoning_match = re.search(r"REASONING:\s*(.*?)(?:\n\w+:|$)", response_text, re.DOTALL)
            position_match = re.search(r"POSITION SIZE:\s*([\d.]+%|[\d.]+)", response_text)
            
            action = action_match.group(1).lower() if action_match else "hold"
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            position_size = position_match.group(1) if position_match else "2%"  # Default position size
            
            # Convert position_size to a number
            if isinstance(position_size, str):
                position_size = position_size.strip("%")
                try:
                    position_size = float(position_size) / 100.0
                except ValueError:
                    position_size = 0.02  # Default to 2%
            
            # Ensure position size is reasonable
            position_size = min(max(position_size, 0.005), 0.05)  # Between 0.5% and 5%
            
            # Create decision
            decision = {
                "symbol": coin_symbol,
                "action": action,
                "confidence": confidence,
                "reasons": [reasoning],
                "timestamp": datetime.now().isoformat(),
                "position_size": position_size,
                "data": {
                    "sentiment_score": sentiment_data.get("score", 0),
                    "price": coin_data.get("current_price", 0),
                    "market_cap": coin_data.get("market_cap", 0),
                    "macd_bullish": macd_data.get("is_bullish", False)
                }
            }
            
            # Save decision to history
            self.decision_history.append(decision)
            
            # Save to Redis
            if REDIS_AVAILABLE and redis_client:
                try:
                    redis_client.set(
                        self.decision_history_key, 
                        json.dumps(self.decision_history[-50:])  # Keep only last 50 decisions
                    )
                except Exception as e:
                    logger.error(f"Error saving decision history to Redis: {e}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Error using master agent for {coin_symbol}: {e}")
            return default_decision
    
    async def _identify_opportunities(self, global_sentiment: Dict[str, Dict], prev_sentiment: Dict[str, Dict]) -> List[Dict]:
        """
        Identify trading opportunities based on sentiment analysis and market data.
        Prioritizes small cap coins with high sentiment scores and urgency.
        Enhanced with Claude 3 Sonnet master agent for sophisticated decision making.
        
        Args:
            global_sentiment: Dictionary mapping cryptocurrencies to sentiment data
            prev_sentiment: Previous sentiment data for comparison
            
        Returns:
            List of trading opportunity dictionaries
        """
        opportunities = []
        
        # Get current portfolio for context
        portfolio = self.trade_service.get_portfolio()
        
        for crypto, data in global_sentiment.items():
            # Get basic sentiment data
            sentiment_score = data.get("score", 0)
            is_small_cap = data.get("is_small_cap", False)
            urgency = data.get("urgency", "low")
            
            # Skip low sentiment coins
            if sentiment_score < 7:
                continue
                
            # Get market data
            try:
                coin_data = await self.market_data_service.get_coin_by_symbol(crypto)
                
                if not coin_data:
                    logger.warning(f"Could not find market data for {crypto}")
                    continue
                    
                market_cap = coin_data.get("market_cap", 0)
                volatility = await self.market_data_service.detect_volatility(coin_data["id"])
                
                # Check for sentiment change
                old_score = 0
                if crypto in prev_sentiment:
                    if isinstance(prev_sentiment[crypto], dict):
                        old_score = prev_sentiment[crypto].get("score", 0)
                    else:
                        old_score = prev_sentiment[crypto]
                
                sentiment_change = sentiment_score - old_score
                
                # Make trade decision using master agent
                decision = await self.decide_trade(
                    coin_symbol=coin_data["symbol"],
                    coin_data=coin_data,
                    sentiment_data=data,
                    portfolio=portfolio
                )
                
                # Create opportunity
                opportunity = {
                    "id": coin_data["id"],
                    "symbol": coin_data["symbol"],
                    "name": coin_data.get("name", coin_data["symbol"]),
                    "sentiment_score": sentiment_score,
                    "sentiment_change": sentiment_change,
                    "is_small_cap": is_small_cap,
                    "market_cap": market_cap,
                    "current_price": coin_data.get("current_price", 0),
                    "volatility": volatility.get("volatility", 0),
                    "volume_to_mc": volatility.get("volume_to_market_cap", 0),
                    "urgency": urgency,
                    "price_predictions": data.get("price_predictions", []),
                    "reasons": data.get("reasons", []),
                    "action": decision["action"],
                    "decision_reasons": decision.get("reasons", []),
                    "confidence": decision.get("confidence", 0.5),
                    "position_size": decision.get("position_size", 0.02)
                }
                
                # Add to opportunities if actionable
                if decision["action"] in ["buy", "sell"]:
                    opportunities.append(opportunity)
                    
                    # Log detailed decision information
                    logger.info(f"Master agent decision for {coin_data['symbol']}: {decision['action'].upper()} with {decision['confidence']:.2f} confidence")
                    for reason in decision.get("reasons", []):
                        logger.info(f"  - {reason}")
                
            except Exception as e:
                logger.error(f"Error identifying opportunity for {crypto}: {e}")
                continue
        
        # Sort opportunities by priority
        opportunities.sort(key=lambda x: (
            # First by action (buys first)
            0 if x["action"] == "buy" else 1,
            # Then by sentiment score (higher first)
            -x["sentiment_score"],
            # Then by small cap status (small caps first)
            0 if x["is_small_cap"] else 1,
            # Then by urgency (high urgency first)
            0 if x["urgency"] == "high" else 1 if x["urgency"] == "medium" else 2
        ))
        
        return opportunities
    
    async def _execute_trades(self, opportunities: List[Dict]) -> List[Dict]:
        """
        Execute trades based on identified opportunities.
        Enhanced with position sizing from the Claude 3 Sonnet master agent.
        
        Args:
            opportunities: List of trading opportunity dictionaries
            
        Returns:
            List of executed trade dictionaries
        """
        executed_trades = []
        
        # Process all buy opportunities first, then sells
        for action in ["buy", "sell"]:
            action_opportunities = [op for op in opportunities if op["action"] == action]
            
            for opportunity in action_opportunities:
                # Extract data
                symbol = opportunity["symbol"]
                sentiment_score = opportunity["sentiment_score"]
                market_cap = opportunity["market_cap"]
                confidence = opportunity.get("confidence", 0.5)
                position_size = opportunity.get("position_size", 0.02)  # Default to 2% if not specified
                
                # Get the current sentiment history to calculate tracked coins
                tracked_coins = len(self.sentiment_history)
                
                # Dynamic position sizing based on number of tracked coins and confidence
                base_position_size = min(0.02, 1.0 / max(tracked_coins, 10))  # Default to 1% for 100 coins, capped at 2%
                
                # Apply confidence adjustment
                adjusted_position_size = base_position_size * confidence
                
                # Dynamic bounds based on tracked coins
                min_size = max(0.001, 0.1 / max(tracked_coins, 10))  # Minimum 0.1% for 100 coins
                max_size = min(0.02, 2.0 / max(tracked_coins, 10))   # Maximum 2% for 100 coins
                
                # Ensure position size stays within bounds
                adjusted_position_size = max(min(adjusted_position_size, max_size), min_size)
                
                # Log the position sizing calculation
                logger.info(f"Position sizing for {symbol}: base={base_position_size:.4f}, adjusted={adjusted_position_size:.4f} " +
                          f"(min={min_size:.4f}, max={max_size:.4f}, tracked_coins={tracked_coins})")
                
                # Execute the trade
                try:
                    trade = await self.trade_service.execute_trade(
                        symbol=symbol,
                        action=action,
                        sentiment_score=sentiment_score,
                        vph=1000,  # Default VPH when not directly from a video
                        market_cap=market_cap,
                        price=opportunity.get("current_price"),
                        position_size=adjusted_position_size  # Use adjusted position size from master agent
                    )
                    
                    if trade and trade.get("status") == "completed":
                        # Track trade performance
                        self.total_trades += 1
                        executed_trades.append(trade)
                        
                        # Log the trade
                        price = trade.get("price", 0)
                        amount = trade.get("amount", 0)
                        value = price * amount
                        
                        log_message = f"Executed {action.upper()} for {symbol}: {amount:.6f} @ ${price:.4f} = ${value:.2f} (Position size: {adjusted_position_size*100:.2f}%)"
                        if action == "buy":
                            if "stop_loss" in trade:
                                log_message += f" (Stop loss: ${trade['stop_loss']:.4f})"
                            if "take_profit" in trade:
                                log_message += f" (Take profit: ${trade['take_profit']:.4f})"
                        
                        # Add confidence level to log
                        log_message += f" (Confidence: {confidence:.2f})"
                        
                        logger.info(log_message)
                        
                        # Log decision reasons
                        for reason in opportunity.get("decision_reasons", []):
                            if reason:
                                logger.info(f"  Decision rationale: {reason}")
                    else:
                        # Log failed trade
                        error = trade.get("error", "Unknown error") if trade else "Trade failed"
                        logger.warning(f"Failed to execute {action} for {symbol}: {error}")
                
                except Exception as e:
                    logger.error(f"Error executing {action} for {symbol}: {e}")
        
        return executed_trades
    
    async def get_transcript(self, video_id: str) -> Optional[str]:
        """
        Get a transcript for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Transcript text or None if not found
        """
        if not self.is_initialized:
            await self.initialize()
            
        return self.transcript_service.get_transcript(video_id)
    
    async def process_video(self, video_id: str) -> Tuple[str, Optional[str]]:
        """
        Process a video to get its transcript.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Tuple of (video_id, transcript or None)
        """
        if not self.is_initialized:
            await self.initialize()
            
        # Create a dummy video object for processing
        video = Video(
            id=video_id,
            title="",  # Will be populated during processing
            views=0,   # Will be populated during processing
            publish_time=datetime.now(),
            vph=0.0    # Will be calculated during processing
        )
        
        return await self.app_service.process_video(video)
    
    async def analyze_sentiment(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze sentiment in a transcript with enhanced detection of trading opportunities.
        
        Args:
            transcript: Transcript text
            
        Returns:
            Dictionary of detailed sentiment analysis with trading recommendations
        """
        if not self.is_initialized:
            await self.initialize()
            
        # Analyze the transcript
        sentiment_data = await self.sentiment_service.analyze_transcript("manual_analysis", transcript)
        
        # Find high sentiment coins
        opportunities = []
        for crypto, data in sentiment_data.items():
            # Extract score and check if it's high enough
            if isinstance(data, dict):
                score = data.get("score", 0)
                is_small_cap = data.get("is_small_cap", False)
            else:
                # Handle old format where data is just the score
                score = data
                is_small_cap = False
                
            if score >= 7:
                # Look up market data
                try:
                    coin_data = await self.market_data_service.get_coin_by_symbol(crypto)
                    if coin_data:
                        volatility = await self.market_data_service.detect_volatility(coin_data["id"])
                        opportunities.append({
                            "symbol": crypto,
                            "sentiment_score": score,
                            "is_small_cap": is_small_cap,
                            "current_price": coin_data.get("current_price", 0),
                            "market_cap": coin_data.get("market_cap", 0),
                            "volatility": volatility.get("volatility", 0),
                            "recommendation": "Potential buy opportunity" if score >= 8 else "Monitor closely"
                        })
                except Exception as e:
                    logger.error(f"Error getting market data for {crypto}: {e}")
        
        return {
            "sentiment": sentiment_data,
            "opportunities": sorted(opportunities, key=lambda x: x["sentiment_score"], reverse=True)
        }
    
    async def get_small_cap_coins(self, limit: int = 20) -> List[Dict]:
        """
        Get a list of small cap coins with market data.
        
        Args:
            limit: Maximum number of coins to return
            
        Returns:
            List of coin dictionaries with market data
        """
        if not self.is_initialized:
            await self.initialize()
            
        small_caps = await self.market_data_service.get_small_cap_coins()
        
        # Get detailed data for top coins
        detailed_coins = []
        for coin_id in small_caps[:min(limit, len(small_caps))]:
            coin_data = await self.market_data_service.get_coin_data(coin_id)
            if coin_data:
                volatility = await self.market_data_service.detect_volatility(coin_id)
                coin_data["volatility"] = volatility.get("volatility", 0)
                coin_data["volume_to_market_cap"] = volatility.get("volume_to_market_cap", 0)
                detailed_coins.append(coin_data)
                
        return detailed_coins
    
    def get_portfolio(self) -> Dict:
        """
        Get the current portfolio state.
        
        Returns:
            Dictionary with portfolio information
        """
        return self.trade_service.get_portfolio()
    
    def get_trade_history(self) -> List[Dict]:
        """
        Get the trade history.
        
        Returns:
            List of trade dictionaries
        """
        return self.trade_service.get_trade_history()
    
    def get_portfolio_history(self) -> List[Dict]:
        """
        Get the portfolio history.
        
        Returns:
            List of portfolio snapshots
        """
        return self.portfolio_history
    
    async def get_trading_opportunities(self) -> List[Dict]:
        """
        Get the current trading opportunities.
        
        Returns:
            List of opportunity dictionaries
        """
        if not self.current_opportunities:
            # Refresh opportunities if none available
            global_sentiment = self.sentiment_history
            prev_sentiment = {}
            self.current_opportunities = await self._identify_opportunities(global_sentiment, prev_sentiment)
            
        return self.current_opportunities
    
    async def make_quick_decisions(self, updated_scores: Dict[str, Dict], 
                                 previous_scores: Dict[str, Dict],
                                 sentiment_changes: Dict[str, float],
                                 significance_threshold: float = 1.0) -> Dict:
        """
        Make quick trading decisions based on incremental score updates from new videos.
        This is called during the 5-minute cycle to respond faster to sentiment changes.
        
        Args:
            updated_scores: Updated global sentiment scores
            previous_scores: Previous global sentiment scores
            sentiment_changes: Magnitude of sentiment changes by coin
            significance_threshold: Minimum change required to trigger a decision
            
        Returns:
            Dictionary with results of the quick decision cycle
        """
        if self.app_service.is_running:
            logger.warning("A cycle is already running, skipping quick decision")
            return {"status": "skipped", "message": "A cycle is already running"}
            
        try:
            # Set running flag
            self.app_service.is_running = True
            logger.info("Making quick decisions based on new videos")
            
            # Find significantly changed coins
            significant_changes = {
                coin: change for coin, change in sentiment_changes.items() 
                if change >= significance_threshold or coin not in previous_scores
            }
            
            if not significant_changes:
                logger.info("No significant sentiment changes detected")
                return {
                    "status": "completed", 
                    "message": "No significant sentiment changes", 
                    "trades_executed": 0
                }
                
            # Log the significant changes
            for coin, change in significant_changes.items():
                new_score = updated_scores[coin]["score"]
                old_score = previous_scores.get(coin, {}).get("score", 0) if coin in previous_scores else 0
                
                if coin not in previous_scores:
                    logger.info(f"⭐ NEW COIN DETECTED: {coin.upper()} with sentiment score {new_score:.2f}")
                else:
                    logger.info(f"📈 SIGNIFICANT CHANGE: {coin.upper()} sentiment: {old_score:.2f} → {new_score:.2f} (Δ{change:.2f})")
            
            # Update sentiment history with new scores
            self.sentiment_history = updated_scores
            
            # Get current portfolio for context
            portfolio = self.trade_service.get_portfolio()
            
            # Identify opportunities for coins with significant changes
            opportunities = []
            for coin, change in significant_changes.items():
                # Skip low sentiment coins
                sentiment_score = updated_scores[coin].get("score", 0)
                if sentiment_score < 7:
                    continue
                    
                # Get market data
                try:
                    coin_data = await self.market_data_service.get_coin_by_symbol(coin)
                    
                    if not coin_data:
                        logger.warning(f"Could not find market data for {coin}")
                        continue
                        
                    # Make trade decision using master agent
                    decision = await self.decide_trade(
                        coin_symbol=coin_data["symbol"],
                        coin_data=coin_data,
                        sentiment_data=updated_scores[coin],
                        portfolio=portfolio
                    )
                    
                    # Create opportunity if actionable
                    if decision["action"] in ["buy", "sell"]:
                        opportunity = {
                            "id": coin_data["id"],
                            "symbol": coin_data["symbol"],
                            "name": coin_data.get("name", coin_data["symbol"]),
                            "sentiment_score": sentiment_score,
                            "sentiment_change": change,
                            "is_small_cap": updated_scores[coin].get("is_small_cap", False),
                            "market_cap": coin_data.get("market_cap", 0),
                            "current_price": coin_data.get("current_price", 0),
                            "action": decision["action"],
                            "decision_reasons": decision.get("reasons", []),
                            "confidence": decision.get("confidence", 0.5),
                            "position_size": decision.get("position_size", 0.02),
                            "triggered_by": "quick_decision"
                        }
                        
                        opportunities.append(opportunity)
                        
                        # Log detailed decision information
                        logger.info(f"Quick decision for {coin_data['symbol']}: {decision['action'].upper()} with {decision['confidence']:.2f} confidence")
                        for reason in decision.get("reasons", []):
                            logger.info(f"  - {reason}")
                except Exception as e:
                    logger.error(f"Error identifying opportunity for {coin}: {e}")
                    continue
            
            if not opportunities:
                logger.info("No actionable trading opportunities from sentiment changes")
                return {
                    "status": "completed", 
                    "message": "No actionable opportunities from sentiment changes", 
                    "trades_executed": 0
                }
                
            logger.info(f"Found {len(opportunities)} actionable opportunities from sentiment changes")
            
            # Implement throttling - don't trade same coin again too soon
            current_time = datetime.now()
            throttled_opportunities = []
            
            for opportunity in opportunities:
                symbol = opportunity["symbol"]
                
                # Check if we've traded this coin recently (within last hour)
                recent_decisions = [
                    d for d in self.decision_history 
                    if d.get("symbol") == symbol and d.get("action") in ["buy", "sell"]
                ]
                
                if recent_decisions:
                    # Get most recent decision
                    last_decision = recent_decisions[-1]
                    last_time = datetime.fromisoformat(last_decision.get("timestamp", "2000-01-01T00:00:00"))
                    hours_since = (current_time - last_time).total_seconds() / 3600
                    
                    # Throttle if traded within last hour
                    if hours_since < 1.0:
                        logger.info(f"Throttling {symbol} - last traded {hours_since:.2f} hours ago")
                        continue
                
                # Add to throttled opportunities if passes filter
                throttled_opportunities.append(opportunity)
            
            if not throttled_opportunities:
                logger.info("All opportunities throttled due to recent trades")
                return {
                    "status": "completed", 
                    "message": "All opportunities throttled", 
                    "trades_executed": 0
                }
            
            # Execute trades for non-throttled opportunities
            executed_trades = []
            for opportunity in throttled_opportunities:
                symbol = opportunity["symbol"]
                action = opportunity["action"]
                sentiment_score = opportunity["sentiment_score"]
                confidence = opportunity.get("confidence", 0.5)
                position_size = opportunity.get("position_size", 0.02)
                
                # Dynamic position sizing based on number of tracked coins and confidence
                tracked_coins = len(updated_scores)  # Number of coins we're tracking
                base_position_size = min(0.02, 1.0 / max(tracked_coins, 10))  # Default to 1% for 100 coins, capped at 2%
                
                # Apply confidence adjustment
                adjusted_position_size = base_position_size * confidence
                
                # Dynamic bounds based on tracked coins
                min_size = max(0.001, 0.1 / max(tracked_coins, 10))  # Minimum 0.1% for 100 coins
                max_size = min(0.02, 2.0 / max(tracked_coins, 10))   # Maximum 2% for 100 coins
                
                # Ensure position size stays within bounds
                adjusted_position_size = max(min(adjusted_position_size, max_size), min_size)
                
                # Log the position sizing calculation
                logger.info(f"Position sizing for {symbol}: base={base_position_size:.4f}, adjusted={adjusted_position_size:.4f} " +
                          f"(min={min_size:.4f}, max={max_size:.4f}, tracked_coins={tracked_coins})")
                
                try:
                    trade = await self.trade_service.execute_trade(
                        symbol=symbol,
                        action=action,
                        sentiment_score=sentiment_score,
                        vph=1000,  # Default VPH when not directly from a video
                        market_cap=opportunity.get("market_cap", 0),
                        price=opportunity.get("current_price", 0),
                        position_size=adjusted_position_size
                    )
                    
                    if trade and trade.get("status") == "completed":
                        # Add relevant fields to trade record
                        trade.update({
                            "symbol": symbol,
                            "action": action,
                            "confidence": confidence,
                            "triggered_by": "quick_decision"
                        })
                        executed_trades.append(trade)
                        
                        # Log the trade
                        price = trade.get("price", 0)
                        amount = trade.get("amount", 0)
                        value = price * amount
                        
                        logger.info(f"📊 QUICK TRADE: {action.upper()} {amount:.6f} {symbol} @ ${price:.6f} = ${value:.2f} (conf: {confidence:.2f})")
                        logger.info(f"  Triggered by significant sentiment change: {sentiment_changes.get(symbol.lower(), 0):.2f}")
                except Exception as e:
                    logger.error(f"Error executing trade for {symbol}: {e}")
            
            # Update current opportunities
            self.current_opportunities = throttled_opportunities
            
            # Return result
            return {
                "status": "completed",
                "message": f"Quick decision cycle executed {len(executed_trades)} trades",
                "trades_executed": len(executed_trades),
                "opportunities": len(throttled_opportunities)
            }
            
        except Exception as e:
            logger.error(f"Error in quick decision cycle: {e}", exc_info=True)
            return {"status": "error", "message": f"Error in quick decision cycle: {str(e)}", "trades_executed": 0}
        finally:
            self.app_service.is_running = False
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get status of all services.
        
        Returns:
            Dictionary with service status information
        """
        return {
            "initialized": self.is_initialized,
            "app_service": self.app_service.get_service_status(),
            "sentiment_service_available": hasattr(self.sentiment_service, "agent") and self.sentiment_service.agent is not None,
            "market_data_service_available": hasattr(self.market_data_service, "session") and self.market_data_service.session is not None,
            "trade_service_mode": "test" if getattr(self.trade_service, "test_mode", True) else "live",
            "master_agent_available": self.master_agent is not None,
            "decision_history_count": len(self.decision_history),
            "small_cap_coins_monitored": len(self.small_cap_coins),
            "portfolio_value": self.trade_service.get_portfolio().get("total_value", 0),
            "total_trades": self.total_trades,
            "current_opportunities": len(self.current_opportunities),
            "langroid_available": LANGROID_AVAILABLE
        }

# Get API keys from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY")
EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY")
EXCHANGE_API_SECRET = os.getenv("EXCHANGE_API_SECRET")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Dependency for FastAPI
crypto_trading_service = CryptoTradingService(
    openrouter_api_key=OPENROUTER_API_KEY,
    coingecko_api_key=COINGECKO_API_KEY,
    exchange_api_key=EXCHANGE_API_KEY,
    exchange_api_secret=EXCHANGE_API_SECRET,
    anthropic_api_key=ANTHROPIC_API_KEY,
    test_mode=True  # Start in test mode for safety
)

async def get_crypto_trading_service() -> CryptoTradingService:
    """
    Dependency provider for the CryptoTradingService.
    
    Returns:
        CryptoTradingService instance
    """
    # Use global flag to prevent circular initialization
    global _initializing_crypto_trading_service
    
    if not hasattr(get_crypto_trading_service, '_initializing_crypto_trading_service'):
        get_crypto_trading_service._initializing_crypto_trading_service = False
    
    if not crypto_trading_service.is_initialized and not get_crypto_trading_service._initializing_crypto_trading_service:
        get_crypto_trading_service._initializing_crypto_trading_service = True
        try:
            await crypto_trading_service.initialize()
        finally:
            get_crypto_trading_service._initializing_crypto_trading_service = False
    
    return crypto_trading_service