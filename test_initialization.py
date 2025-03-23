#!/usr/bin/env python3
"""
Test script to verify that the circular dependency issue is fixed
"""
import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

async def test_initialization():
    """Test that the circular dependency is fixed by initializing services"""
    
    logger.info("=== Testing service initialization ===")
    
    # Import services
    from services.crypto_trading_service import get_crypto_trading_service
    
    # Initialize trading service
    logger.info("Getting crypto trading service...")
    trading_service = await get_crypto_trading_service()
    
    # Check if initialization was successful
    if trading_service.is_initialized:
        logger.info("SUCCESS: CryptoTradingService initialized successfully!")
        
        # Check if the cross-reference is set up
        if trading_service.app_service.crypto_trading_service is trading_service:
            logger.info("SUCCESS: Cross-reference correctly set up!")
        else:
            logger.error("ERROR: Cross-reference is not correctly set up")
            
        # Log information about initialized services
        logger.info(f"AppService status: initialized={trading_service.app_service is not None}")
        logger.info(f"TranscriptService status: initialized={trading_service.transcript_service is not None}")
        logger.info(f"SentimentService status: initialized={trading_service.sentiment_service is not None}")
        logger.info(f"MarketDataService status: initialized={trading_service.market_data_service is not None}")
        logger.info(f"TradeService status: initialized={trading_service.trade_service is not None}")
    else:
        logger.error("ERROR: CryptoTradingService failed to initialize")
    
    logger.info("=== Initialization test completed ===")

if __name__ == "__main__":
    asyncio.run(test_initialization())