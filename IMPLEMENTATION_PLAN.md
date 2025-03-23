# Implementation Plan for Dual-Cycle Trading Framework

## Status Update - March 23, 2025

### Completed Work

1. **Enhanced Transcript Service**
   - Implemented multi-stage transcript acquisition with 3 fallback methods:
     1. YouTube Transcript API for captions
     2. Audio download with yt-dlp + transcription with Whisper
     3. Simulated transcript as last resort
   - Added required dependencies (yt-dlp, whisper) to requirements.txt
   - Created comprehensive error handling and logging

2. **Dual-Cycle Trading Framework**
   - Implemented 30-minute comprehensive cycle in `run_cycle_impl.py`
   - Added 5-minute quick decision cycle in `crypto_trading_service.py`
   - Created incremental sentiment updates in `sentiment_service.py`
   - Added test cases for the dual-cycle framework

3. **Environment Configuration**
   - Fixed environment variable parsing in `run.sh`
   - Enhanced error handling for malformed entries in .env
   - Verified ffmpeg installation for audio processing

4. **Trading Enhancements**
   - Implemented dynamic position sizing based on number of tracked coins
   - Added throttling mechanism to prevent excessive trading
   - Created framework for Claude 3 Sonnet as master agent

### Verification

- Enhanced transcript service verified with a dedicated test script
- Fixed circular dependency issues between services
- Verified ffmpeg installation for audio transcription

### Issues Discovered

1. **Circular Dependency**: There is a circular dependency between `app_service.py` and `crypto_trading_service.py` that causes initialization loops in test scripts
2. **YouTube API Issues**: Some YouTube videos may be region-restricted or unavailable in the testing environment
3. **Master Agent Fallback**: The system gracefully falls back to simplified decision logic when Claude 3 Sonnet is unavailable

## Next Steps (Pre-Simulation)

1. **Break Circular Dependency**
   - Refactor `app_service.py` and `crypto_trading_service.py` to eliminate the circular initialization
   - Use a deferred initialization pattern or dependency injection

2. **Create a .env File**
   ```
   # API Keys
   OPENROUTER_API_KEY=your_openrouter_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here
   COINGECKO_API_KEY=your_coingecko_key_here
   
   # Trading Configuration
   EXCHANGE_API_KEY=your_exchange_key_here
   EXCHANGE_API_SECRET=your_exchange_secret_here
   TEST_MODE=true
   
   # System Settings
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_TEST_MODE=true
   
   # Cycle Configuration
   CYCLE_INTERVAL=30  # Minutes
   QUICK_DECISION_INTERVAL=5  # Minutes
   VPH_THRESHOLD=500.0
   
   # Transcript Settings
   WHISPER_MODEL=small  # Options: tiny, base, small, medium, large
   ```

3. **Test with Known Working Videos**
   - Create a list of known working cryptocurrency videos with captions
   - Create a list of videos without captions for testing audio transcription
   - Document these video IDs for future testing

4. **Create a Quick Start Script**
   ```bash
   #!/bin/bash
   # Quick start script for the dual-cycle framework
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install ffmpeg if not already available
   if ! command -v ffmpeg &> /dev/null; then
     echo "Installing ffmpeg..."
     sudo apt-get update && sudo apt-get install -y ffmpeg
   fi
   
   # Set up environment
   if [ ! -f .env ]; then
     cp .env.example .env
     echo "Created .env file. Please edit it to add your API keys."
     exit 1
   fi
   
   # Start the system
   ./run.sh --api
   ```

5. **Create a Monitoring Dashboard**
   - Set up a Prometheus and Grafana dashboard to monitor:
     - Sentiment scores by cryptocurrency
     - Trade decisions (quick vs. comprehensive)
     - Portfolio performance
     - System metrics (error rates, latency)

## Simulation Plan (March 22 - April 5, 2025)

### Initial Setup (Day 1)

1. **Configure API Keys**
   - Set up OpenRouter API key for Mixtral sentiment analysis
   - Configure Anthropic API key for Claude 3 Sonnet
   - Set up CoinGecko API key for market data
   - Set exchange API keys in test mode

2. **Set Initial Parameters**
   - Start with VPH threshold of 500.0
   - Set sentiment significance threshold at 1.0 for quick decisions
   - Use dynamic position sizing with 1.0/N baseline
   - Set throttling period to 1 hour

### Daily Monitoring

1. **Monitor Sentiment Analysis**
   - Check the quality of transcripts (captions vs. audio vs. simulation)
   - Analyze sentiment distribution (-10 to +10) for major and small-cap coins
   - Verify incremental sentiment updates in the quick decision cycle

2. **Track Trading Performance**
   - Compare performance of quick decisions vs. comprehensive decisions
   - Analyze position sizing effect with different numbers of tracked coins
   - Monitor throttling effectiveness in preventing excessive trading

### Parameter Tuning (Day 5)

1. **Adjust Sentiment Thresholds**
   - Fine-tune based on initial performance:
     - Sentiment threshold for actionable scores (currently >7)
     - Significance threshold for quick decisions (currently >1.0)

2. **Fine-Tune Position Sizing**
   - Adjust dynamic position sizing parameters:
     - Base size (currently 1.0/N)
     - Minimum bound (currently 0.1/N)
     - Maximum bound (currently 2.0/N)

### Final Analysis (Day 14)

1. **Performance Metrics**
   - Total return on investment
   - Win/loss ratio
   - Performance attribution (quick vs. comprehensive decisions)
   - Effect of transcript quality on trade performance

2. **System Improvements**
   - Recommendations for enhanced sentiment analysis
   - Optimized parameters for position sizing and thresholds
   - Suggestions for improved audio transcription quality

## Post-Simulation Work

1. **Implement Database Integration**
   - Store historical sentiment scores
   - Track trading decisions and performance
   - Implement SQLAlchemy models and repositories

2. **Extend AI Capabilities**
   - Improve sentiment analysis prompts
   - Add support for analyzing news articles
   - Enhance decision-making logic in the master agent

3. **Create Web Dashboard**
   - Build a React frontend for visualization
   - Add real-time updates with WebSockets
   - Create charts for sentiment trends and portfolio performance

4. **Prepare for Live Trading**
   - Implement additional safeguards
   - Add stricter position sizing
   - Configure monitoring alerts