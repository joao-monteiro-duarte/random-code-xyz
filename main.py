"""
FastAPI main application for crypto trading pool using the enhanced CryptoTradingService.
Specializes in identifying and trading small-cap cryptocurrencies based on YouTube sentiment analysis.
"""
import asyncio
import logging
import logging.config
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import fastapi
from pydantic import BaseModel

# Import services
from services.crypto_trading_service import get_crypto_trading_service, CryptoTradingService
from models.video import Video
from config.settings import VPH_THRESHOLD

# Configure logging
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": os.getenv("LOG_PATH", f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
})

logger = logging.getLogger(__name__)

# Setup FastAPI lifecycle events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and shutdown services with the application lifecycle."""
    # Initialize services
    service = await get_crypto_trading_service()
    logger.info("CryptoTradingService initialized")
    
    # Register the background cycle loop
    cycle_task = asyncio.create_task(cycle_loop(service))
    
    # Signal ready
    yield
    
    # Shutdown
    cycle_task.cancel()
    try:
        await cycle_task
    except asyncio.CancelledError:
        pass
    
    # Close services
    await service.market_data_service.close()
    await service.trade_service.close()
    logger.info("Services closed gracefully")

# Create the FastAPI application
app = FastAPI(
    title="Crypto Trading Pool API",
    description="API for cryptocurrency trading based on YouTube video sentiment analysis",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models for API requests and responses
class VideoModel(BaseModel):
    id: str
    title: str
    views: int
    publish_time: datetime
    vph: float = 0.0
    channel_id: Optional[str] = None
    channel_title: Optional[str] = None
    
    class Config:
        from_attributes = True

class TranscriptRequest(BaseModel):
    video_id: str

class TranscriptResponse(BaseModel):
    video_id: str
    transcript: Optional[str] = None
    cached: bool = False
    
class StatusResponse(BaseModel):
    status: str
    service_status: Dict[str, Any]
    
class VideoListResponse(BaseModel):
    videos: List[VideoModel]
    count: int

class SentimentRequest(BaseModel):
    transcript: str
    
class SentimentResponse(BaseModel):
    sentiment: Dict[str, Any]
    opportunities: List[Dict[str, Any]]
    
class CoinModel(BaseModel):
    id: str
    symbol: str
    name: str
    market_cap: Optional[float] = None
    current_price: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    volatility: Optional[float] = None
    volume_to_market_cap: Optional[float] = None
    
class SmallCapCoinsResponse(BaseModel):
    coins: List[CoinModel]
    count: int
    
class PortfolioModel(BaseModel):
    total_value: float
    usd_balance: float
    coins: Dict[str, Dict[str, Any]]
    
class TradeModel(BaseModel):
    id: str
    symbol: str
    action: str
    amount: float
    price: float
    value_usd: float
    timestamp: str
    status: str
    
class TradeRequest(BaseModel):
    symbol: str
    action: str
    amount: Optional[float] = None
    
class OpportunityModel(BaseModel):
    id: str
    symbol: str
    name: str
    sentiment_score: float
    sentiment_change: float
    is_small_cap: bool
    market_cap: float
    current_price: float
    volatility: float
    urgency: str
    action: str
    
class OpportunitiesResponse(BaseModel):
    opportunities: List[OpportunityModel]
    count: int
    
class TradeHistoryResponse(BaseModel):
    trades: List[TradeModel]
    count: int

async def cycle_loop(service: CryptoTradingService):
    """Background task to run trading cycles periodically."""
    from config.settings import CYCLE_INTERVAL
    
    logger.info(f"Starting background cycle loop (interval: {CYCLE_INTERVAL}s)")
    
    while True:
        try:
            # Check if it's time to run a cycle
            app_service = service.app_service
            now = datetime.now()
            
            # Handle potential mocked objects or test environment
            try:
                time_since_last = (now - app_service.last_cycle_time).total_seconds()
                # Add detailed logging for cycle timing
                logger.info(f"Checking cycle timing - Last cycle: {app_service.last_cycle_time.isoformat()}, Now: {now.isoformat()}, Time since: {time_since_last:.1f}s")
            except (TypeError, AttributeError) as e:
                logger.warning(f"Error calculating time since last cycle: {e}. Setting to CYCLE_INTERVAL + 1.")
                time_since_last = CYCLE_INTERVAL + 1  # Force a cycle to run once
            
            if time_since_last >= CYCLE_INTERVAL and not app_service.is_running:
                logger.info(f"Auto-triggering trading cycle after {time_since_last:.1f}s")
                await service.run_cycle(VPH_THRESHOLD)
            else:
                # Log why cycle was not triggered
                if app_service.is_running:
                    logger.info(f"Cycle not triggered - A cycle is already running")
                else:
                    logger.info(f"Cycle not triggered - Next cycle in {max(0, CYCLE_INTERVAL - time_since_last):.1f}s")
            
            # Wait before checking again
            await asyncio.sleep(60)  # Check every minute
        except asyncio.CancelledError:
            logger.info("Cycle loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in cycle loop: {e}", exc_info=True)
            await asyncio.sleep(60)  # Wait before retrying

# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "application": "Crypto Trading Pool API",
        "version": "2.0.0",
        "status": "active",
        "description": "Cryptocurrency trading based on YouTube video sentiment analysis"
    }

@app.get("/status", response_model=StatusResponse)
async def get_status(service: CryptoTradingService = Depends(get_crypto_trading_service)):
    """Get application status and service information."""
    return StatusResponse(
        status="online",
        service_status=service.get_service_status()
    )

@app.post("/run-cycle", response_model=Dict[str, str])
async def trigger_cycle(
    background_tasks: BackgroundTasks,
    vph_threshold: float = VPH_THRESHOLD,
    service: CryptoTradingService = Depends(get_crypto_trading_service)
):
    """
    Manually trigger a trading cycle.
    
    - Processes accumulated videos
    - Analyzes sentiment in transcripts
    - Identifies trading opportunities
    - Executes trades based on sentiment
    """
    if service.app_service.is_running:
        raise HTTPException(status_code=409, detail="A cycle is already running")
    
    background_tasks.add_task(service.run_cycle, vph_threshold)
    return {"message": "Trading cycle started in the background"}

@app.get("/transcripts/{video_id}", response_model=TranscriptResponse)
async def get_transcript(
    video_id: str,
    service: CryptoTradingService = Depends(get_crypto_trading_service)
):
    """Get transcript for a specific YouTube video by ID."""
    transcript = await service.get_transcript(video_id)
    return TranscriptResponse(
        video_id=video_id,
        transcript=transcript,
        cached=transcript is not None
    )

@app.post("/transcripts", response_model=TranscriptResponse)
async def process_transcript(
    request: TranscriptRequest,
    service: CryptoTradingService = Depends(get_crypto_trading_service)
):
    """Process a YouTube video to extract and store its transcript."""
    video_id, transcript = await service.process_video(request.video_id)
    
    return TranscriptResponse(
        video_id=video_id,
        transcript=transcript,
        cached=False
    )

@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    service: CryptoTradingService = Depends(get_crypto_trading_service)
):
    """
    Analyze cryptocurrency sentiment in a transcript using Mixtral.
    Identifies potential trading opportunities based on sentiment scores.
    """
    result = await service.analyze_sentiment(request.transcript)
    return SentimentResponse(
        sentiment=result["sentiment"],
        opportunities=result["opportunities"]
    )

@app.get("/videos", response_model=VideoListResponse)
async def get_accumulated_videos(service: CryptoTradingService = Depends(get_crypto_trading_service)):
    """Get list of accumulated YouTube videos waiting to be processed."""
    video_tuples = service.app_service.get_accumulated_videos()
    videos = []
    
    for video_tuple in video_tuples:
        video = Video.from_tuple(video_tuple)
        videos.append(VideoModel(
            id=video.id,
            title=video.title or "Unknown",
            views=video.views,
            publish_time=video.publish_time,
            vph=video.vph,
            channel_id=video.channel_id,
            channel_title=video.channel_title
        ))
    
    return VideoListResponse(
        videos=videos,
        count=len(videos)
    )

@app.get("/small-cap-coins", response_model=SmallCapCoinsResponse)
async def get_small_cap_coins(
    limit: int = Query(20, description="Maximum number of coins to return"),
    service: CryptoTradingService = Depends(get_crypto_trading_service)
):
    """
    Get a list of small-cap cryptocurrency coins with market data.
    These are coins with market cap under $50M that are monitored for trading.
    """
    try:
        coins = await service.get_small_cap_coins(limit)
        
        if not coins:
            logger.warning("No small-cap coins found")
            return SmallCapCoinsResponse(coins=[], count=0)
            
        return SmallCapCoinsResponse(
            coins=[CoinModel(**coin) for coin in coins],
            count=len(coins)
        )
    except Exception as e:
        logger.error(f"Error fetching small-cap coins: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching small-cap coins: {str(e)}")

@app.get("/portfolio", response_model=PortfolioModel)
async def get_portfolio(service: CryptoTradingService = Depends(get_crypto_trading_service)):
    """Get the current portfolio state with asset allocation."""
    portfolio = service.get_portfolio()
    
    return PortfolioModel(
        total_value=portfolio["total_value"],
        usd_balance=portfolio["USD"],
        coins=portfolio.get("coins", {})
    )

@app.post("/trade", response_model=TradeModel)
async def execute_trade(
    request: TradeRequest,
    service: CryptoTradingService = Depends(get_crypto_trading_service)
):
    """
    Manually execute a cryptocurrency trade.
    
    - Buy or sell a specific coin
    - Set amount in USD or coin units
    - Automatically sets stop-loss for buy orders
    """
    # Validate action
    if request.action.lower() not in ["buy", "sell"]:
        raise HTTPException(status_code=400, detail="Action must be 'buy' or 'sell'")
    
    # Get market data for the coin
    try:
        coin_data = await service.market_data_service.get_coin_by_symbol(request.symbol)
        if not coin_data:
            raise HTTPException(status_code=404, detail=f"Coin {request.symbol} not found")
        
        # Use default sentiment score for manual trades
        sentiment_score = 7.0  # Neutral-positive
        
        # Execute trade
        trade = await service.trade_service.execute_trade(
            symbol=request.symbol,
            action=request.action.lower(),
            sentiment_score=sentiment_score,
            vph=1000,  # Default VPH for manual trades
            market_cap=coin_data.get("market_cap", 0),
            price=coin_data.get("current_price")
        )
        
        if trade.get("status") == "error":
            raise HTTPException(status_code=400, detail=trade.get("error", "Trade execution failed"))
        
        return TradeModel(**trade)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error executing trade: {str(e)}")

@app.get("/trade-history", response_model=TradeHistoryResponse)
async def get_trade_history(
    limit: int = Query(20, description="Maximum number of trades to return"),
    service: CryptoTradingService = Depends(get_crypto_trading_service)
):
    """Get history of executed trades with details."""
    all_trades = service.get_trade_history()
    
    # Sort by timestamp (newest first) and limit
    sorted_trades = sorted(all_trades, key=lambda x: x.get("timestamp", ""), reverse=True)
    limited_trades = sorted_trades[:min(limit, len(sorted_trades))]
    
    return TradeHistoryResponse(
        trades=limited_trades,
        count=len(limited_trades)
    )

@app.get("/opportunities", response_model=OpportunitiesResponse)
async def get_opportunities(service: CryptoTradingService = Depends(get_crypto_trading_service)):
    """
    Get current trading opportunities based on sentiment analysis.
    Lists coins with high sentiment scores and recommended actions.
    """
    opportunities = await service.get_trading_opportunities()
    
    return OpportunitiesResponse(
        opportunities=opportunities,
        count=len(opportunities)
    )

# WebSocket for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, 
    service: CryptoTradingService = Depends(get_crypto_trading_service)
):
    """
    WebSocket endpoint for real-time trading updates.
    Provides status, portfolio and trading opportunity updates.
    """
    await websocket.accept()
    
    try:
        counter = 0
        while True:
            # Send different types of data at different intervals
            status_data = service.get_service_status()
            
            if counter % 1 == 0:  # Every 5 seconds
                # Send status update
                await websocket.send_json({
                    "type": "status_update",
                    "data": {
                        "status": "online",
                        "service_status": status_data,
                        "timestamp": datetime.now().isoformat()
                    }
                })
            
            if counter % 3 == 0:  # Every 15 seconds
                # Send portfolio update
                portfolio = service.get_portfolio()
                await websocket.send_json({
                    "type": "portfolio_update",
                    "data": {
                        "total_value": portfolio["total_value"],
                        "usd_balance": portfolio["USD"],
                        "coin_count": len(portfolio.get("coins", {})),
                        "timestamp": datetime.now().isoformat()
                    }
                })
            
            if counter % 12 == 0:  # Every 60 seconds
                # Send opportunities update
                opportunities = await service.get_trading_opportunities()
                if opportunities:
                    await websocket.send_json({
                        "type": "opportunities_update",
                        "data": {
                            "count": len(opportunities),
                            "top_opportunity": opportunities[0] if opportunities else None,
                            "timestamp": datetime.now().isoformat()
                        }
                    })
            
            # Increment counter and sleep
            counter = (counter + 1) % 60
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        # Handle client disconnection
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)

# Import prometheus client for metrics
try:
    from prometheus_client import Counter, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    logger.warning("prometheus_client not installed, metrics endpoint will return JSON only")
    generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"

# Define Prometheus metrics (if available)
try:
    TRADES_TOTAL = Counter('trades_total', 'Total number of trades executed', ['action'])
    PORTFOLIO_VALUE = Gauge('portfolio_value_usd', 'Current portfolio value in USD')
    PORTFOLIO_PROFIT_LOSS = Gauge('portfolio_profit_loss_usd', 'Current profit/loss in USD')
    PORTFOLIO_PROFIT_LOSS_PCT = Gauge('portfolio_profit_loss_pct', 'Current profit/loss percentage')
    COINS_HELD = Gauge('coins_held_total', 'Total number of coins held in portfolio')
    PORTFOLIO_VOLATILITY = Gauge('portfolio_volatility_pct', 'Portfolio volatility percentage')
    TRADE_EXECUTION_TIME = Summary('trade_execution_seconds', 'Time spent executing trades')
    SENTIMENT_SCORES = Gauge('sentiment_score', 'Sentiment score for a coin', ['coin'])
except NameError:
    # prometheus_client not installed
    pass

# Metrics endpoint for monitoring
@app.get("/metrics", response_class=fastapi.responses.Response)
async def metrics(service: CryptoTradingService = Depends(get_crypto_trading_service),
                  format: str = Query("json", description="Format: json or prometheus")):
    """Get performance metrics for monitoring."""
    portfolio = service.get_portfolio()
    
    # Calculate basic metrics
    initial_value = 10000.0  # Starting portfolio value
    current_value = portfolio["total_value"]
    profit_loss = current_value - initial_value
    profit_loss_pct = (profit_loss / initial_value) * 100 if initial_value > 0 else 0
    
    # Get historical data
    portfolio_history = service.get_portfolio_history()
    history_values = [entry["total_value"] for entry in portfolio_history] if portfolio_history else []
    
    # Calculate volatility if enough data points
    volatility = 0
    if len(history_values) > 1:
        import numpy as np
        try:
            volatility = float(np.std(history_values) / np.mean(history_values) * 100)
        except:
            volatility = 0
    
    # If prometheus-client is available, update metrics
    if generate_latest is not None and format.lower() == "prometheus":
        # Update Prometheus metrics
        PORTFOLIO_VALUE.set(current_value)
        PORTFOLIO_PROFIT_LOSS.set(profit_loss)
        PORTFOLIO_PROFIT_LOSS_PCT.set(profit_loss_pct)
        COINS_HELD.set(len(portfolio.get("coins", {})))
        PORTFOLIO_VOLATILITY.set(volatility)
        
        # Update sentiment scores (from latest sentiment history)
        for coin, data in service.sentiment_history.items():
            if isinstance(data, dict) and "score" in data:
                SENTIMENT_SCORES.labels(coin=coin).set(data["score"])
            elif isinstance(data, (int, float)):
                SENTIMENT_SCORES.labels(coin=coin).set(data)
                
        # Return Prometheus format
        return fastapi.responses.Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    # Return JSON format (default or fallback)
    return {
        "current_value": current_value,
        "profit_loss": profit_loss,
        "profit_loss_pct": profit_loss_pct,
        "total_trades": service.total_trades,
        "volatility": volatility,
        "coins_held": len(portfolio.get("coins", {})),
        "timestamp": datetime.now().isoformat()
    }

# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        reload=True
    )