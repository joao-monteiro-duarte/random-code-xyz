import asyncio
import logging

logger = logging.getLogger(__name__)

async def fetch_market_data():
    # Mock data for now; replace with real API (e.g., CoinGecko) later
    logger.info("Fetching market data...")
    await asyncio.sleep(1)  # Simulate async fetch
    data = {
        "bitcoin": {"price": 88000, "market_cap": 1740000000000, "volume": 28000000000},
        "ethereum": {"price": 2100, "market_cap": 252000000000, "volume": 15000000000},
        "solana": {"price": 142, "market_cap": 72000000000, "volume": 4600000000}
    }
    logger.info("Successfully fetched market data")
    return data