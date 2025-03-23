#!/usr/bin/env python
"""
Script to test trade execution directly.
This helps diagnose why trades aren't being executed in the regular cycle.
"""
import asyncio
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("trade_test.log")
    ]
)
logger = logging.getLogger("trade_test")

async def test_trade_execution():
    """Test trade execution directly."""
    try:
        # Import services
        from services.crypto_trading_service import get_crypto_trading_service

        # Get service instance
        service = await get_crypto_trading_service()
        
        # Log service status
        logger.info("Service initialized")
        logger.info(f"Service status: {service.get_service_status()}")
        
        # Create a test coin with high sentiment
        test_sentiment = {
            "pepe": {
                "score": 9.0,
                "is_small_cap": True,
                "urgency": "high",
                "videos_mentioned": 3,
                "reasons": ["Strong momentum", "High social interest"],
                "price_predictions": ["10x potential"]
            }
        }
        
        # Store sentiment for testing
        service.sentiment_history = test_sentiment
        
        # Identify opportunities based on sentiment
        logger.info("Identifying opportunities based on high sentiment")
        opportunities = await service._identify_opportunities(test_sentiment, {})
        
        if not opportunities:
            logger.error("No opportunities identified despite high sentiment!")
            # Try to debug why
            logger.info("Debugging opportunity identification:")
            for crypto, data in test_sentiment.items():
                logger.info(f"Coin {crypto}: {data}")
                try:
                    coin_data = await service.market_data_service.get_coin_by_symbol(crypto)
                    logger.info(f"Market data for {crypto}: {coin_data}")
                    if not coin_data:
                        logger.error(f"Could not find market data for {crypto}")
                        logger.info("Adding mock market data for testing")
                        # Create mock market data for testing
                        mock_market_data = {
                            "id": "pepe",
                            "symbol": "PEPE",
                            "name": "Pepe",
                            "market_cap": 30000000,  # $30M market cap
                            "current_price": 0.00005,
                            "volume_24h": 5000000,
                        }
                        # Monkey patch the market data service
                        original_get_coin = service.market_data_service.get_coin_by_symbol
                        async def mock_get_coin(symbol):
                            if symbol.lower() == "pepe":
                                return mock_market_data
                            return await original_get_coin(symbol)
                        service.market_data_service.get_coin_by_symbol = mock_get_coin
                        
                        # Try identifying opportunities again
                        logger.info("Retrying opportunity identification with mock data")
                        opportunities = await service._identify_opportunities(test_sentiment, {})
                except Exception as e:
                    logger.error(f"Error getting market data for {crypto}: {e}")
        
        if opportunities:
            logger.info(f"Found {len(opportunities)} opportunities:")
            for i, opportunity in enumerate(opportunities):
                logger.info(f"Opportunity {i+1}: {opportunity}")
                
            # Execute trades based on opportunities
            logger.info("Executing test trades")
            executed_trades = await service._execute_trades(opportunities)
            
            if executed_trades:
                logger.info(f"Successfully executed {len(executed_trades)} trades:")
                for i, trade in enumerate(executed_trades):
                    logger.info(f"Trade {i+1}: {trade}")
            else:
                logger.error("No trades were executed despite opportunities!")
                # Check the trade service portfolio
                portfolio = service.trade_service.get_portfolio()
                logger.info(f"Trade service portfolio: {portfolio}")
        else:
            logger.error("No opportunities identified even with mock market data!")
            
        # Check portfolio and trade history
        portfolio = service.trade_service.get_portfolio()
        trades = service.trade_service.get_trade_history()
        
        logger.info(f"Final portfolio: {portfolio}")
        logger.info(f"Trade history: {trades}")
        
    except Exception as e:
        logger.error(f"Error executing test: {e}", exc_info=True)
    finally:
        # Clean up
        logger.info("Test completed")

if __name__ == "__main__":
    asyncio.run(test_trade_execution())