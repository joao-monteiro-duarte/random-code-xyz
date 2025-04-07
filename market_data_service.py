import aiohttp
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class MarketDataService:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3/coins/markets"
        self.coins = ["bitcoin", "ethereum", "solana"]

    async def get_market_data(self) -> Dict[str, Dict]:
        market_data = {}
        params = {
            "vs_currency": "usd",
            "ids": ",".join(self.coins),
            "order": "market_cap_desc",
            "per_page": "3",
            "page": "1",
            "sparkline": "false"  # String, not bool
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for coin in data:
                            coin_id = coin["id"]
                            market_data[coin_id] = {
                                "price": coin["current_price"],
                                "market_cap": coin["market_cap"],
                                "volume": coin["total_volume"]
                            }
                        logger.info("Successfully fetched market data")
                    else:
                        logger.error(f"Failed to fetch market data: {response.status}")
            except Exception as e:
                logger.error(f"Error fetching market data: {e}")
        return market_data