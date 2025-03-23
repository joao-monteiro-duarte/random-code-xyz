# Using Mock Market Data

This document explains how to use mock market data for testing the crypto-trading-pool system without hitting API rate limits.

## Overview

The `MarketDataService` now includes a comprehensive mock data system that provides simulated market data for testing purposes. This allows you to run tests and simulations without worrying about API rate limits or connectivity issues with external services like CoinGecko.

## Features of the Mock Data System

- **Predefined meme coins**: Mock data for popular small-cap meme coins like PEPE, FLOKI, BONK, etc.
- **Major cryptocurrencies**: Mock data for Bitcoin, Ethereum, and Solana
- **Dynamic data generation**: Automatic generation of reasonable mock data for any unknown coin
- **Smart fallbacks**: Graceful degradation from API → cache → mock data
- **Rate limiting**: Smart rate limit handling when using real APIs
- **Test mode**: Complete isolation from external APIs when in test mode

## How to Enable Mock Data

### Environment Variables

Set the `USE_MOCK_DATA` environment variable to `true`:

```bash
export USE_MOCK_DATA=true
```

### In Python Code

When initializing the `MarketDataService`, set `test_mode=True`:

```python
market_data_service = MarketDataService(api_key="your_api_key", test_mode=True)
```

### For the CryptoTradingService

The `CryptoTradingService` will automatically use mock data for market data when initialized with `test_mode=True`:

```python
trading_service = CryptoTradingService(test_mode=True)
```

## Running Tests with Mock Data

The `run_dual_cycle_test.sh` script automatically enables mock data:

```bash
./run_dual_cycle_test.sh
```

For other test scripts, you can enable mock data:

```bash
USE_MOCK_DATA=true python your_test_script.py
```

## Mock Data Details

### Small Cap Coins

The system includes mock data for these small-cap coins:

- PEPE: Pepe ($30M market cap)
- FLOKI: Floki Inu ($45M market cap)
- BONK: Bonk ($35M market cap)
- MOG: Mog Coin ($12M market cap)
- WIF: Dogwifhat ($40M market cap)

### Major Cryptocurrencies

The system includes mock data for these major cryptocurrencies:

- BTC: Bitcoin ($1.2T market cap)
- ETH: Ethereum ($340B market cap)
- SOL: Solana ($50B market cap)

### Volatility Data

Mock volatility data is provided for:

- PEPE: 25% volatility
- DOGE: 15% volatility
- SHIB: 20% volatility

### MACD Data

Mock MACD data is provided for technical analysis testing:

- PEPE: Bullish MACD with 0.75 trend strength
- DOGE: Bullish MACD with 0.6 trend strength
- SHIB: Bearish MACD with 0.3 trend strength

## Benefits of Using Mock Data

1. **Improved testing stability**: Tests don't fail due to external API issues or rate limits
2. **Faster test execution**: No network delays or API throttling
3. **Reproducible results**: Same data every time for consistent testing
4. **No API costs**: Avoid using up API quota during testing
5. **Offline development**: Work on the project without internet connectivity

## When to Use Real Data

While mock data is great for development and testing, you should use real data for:

1. Final production validation
2. Benchmarking actual market performance
3. When testing specific market conditions that require real-world data

## Implementation Details

The mock data system is implemented in `market_data_service.py` with several key components:

- `_initialize_mock_data()`: Sets up the predefined mock data
- `_respect_rate_limits()`: Handles rate limiting for real API calls
- `_handle_api_response()`: Processes API responses and falls back to mock data when needed

Each data retrieval method includes fallback logic to use mock data when necessary, ensuring the system never fails due to external API issues.