# Crypto Trading Pool

Automated cryptocurrency trading based on YouTube video sentiment analysis using a dual-cycle framework with AI agents for decision-making.

## Overview

This application analyzes cryptocurrency-related YouTube videos, processes their transcripts, and makes trading decisions based on sentiment analysis and views per hour (VPH) metrics. It uses two specialized AI agents:

- **Scout Agent (Mixtral-8x7b-instruct)**: Analyzes YouTube transcripts for sentiment and opportunities
- **Master Agent (Claude 3 Sonnet)**: Makes sophisticated trading decisions with market context

The system implements a dual-cycle framework with two distinct processing frequencies:
- **5-minute quick decisions**: Responds rapidly to significant sentiment changes
- **30-minute comprehensive analysis**: Performs thorough market and sentiment evaluation

## Key Features

- **Multi-stage transcript acquisition**:
  1. YouTube Transcript API for captions
  2. Audio download via yt-dlp + transcription via Whisper
  3. Simulated transcript as last resort

- **Dual-cycle trading framework**:
  - 5-minute quick decisions with incremental sentiment updates
  - 30-minute comprehensive analysis with full video processing

- **Sentiment enhancement**:
  - Freshness boost (20%) for new content in quick decisions
  - New coin boost (10%) for newly discovered cryptocurrencies
  - Weighted by VPH (Views-Per-Hour) for influence measurement

- **Dynamic position sizing**:
  - Base position size of 1.0/N (where N is number of tracked coins)
  - Minimum bound of 0.1/N
  - Maximum bound of 2.0/N
  - Confidence-based adjustment from Claude 3 Sonnet

- **Throttling mechanism**:
  - 1-hour cooldown period to prevent excessive trading
  - Prioritization based on sentiment score and urgency

## Architecture

The application is built using a modular, service-based architecture:

- **Models**: Data models representing domain objects (like videos)
- **Services**: Core business logic organized as services
- **Utils**: Utility classes and functions (cache management, etc.)
- **Config**: Configuration settings and environment variables
- **API**: FastAPI web interface for interacting with the application

### Key Components

- **CryptoTradingService**: Top-level coordinator that handles dual-cycle framework
- **SentimentService**: Analyzes cryptocurrency sentiment using Mixtral-8x7b-instruct
- **TranscriptService**: Multi-stage transcript acquisition with fallbacks
- **MarketDataService**: Retrieves cryptocurrency market data
- **TradeService**: Executes trades with position sizing and throttling

## Getting Started

### Prerequisites

- Python 3.11+
- Redis (optional, will use local fallback if not available)
- ffmpeg (required for audio extraction)
- OpenRouter API key (for Mixtral-8x7b-instruct)
- Anthropic API key (for Claude 3 Sonnet)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/crypto-trading-pool.git
cd crypto-trading-pool
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Install ffmpeg (if not already installed):

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download and install from https://ffmpeg.org/download.html
```

4. Create a `.env` file with your configuration:

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

### Running the Application

#### API Server

```bash
./run.sh --api
```

The API will be available at http://localhost:8000

#### Single Processing Cycle

```bash
./run.sh --cycle --vph 500.0
```

## Dual-Cycle Framework

### 5-Minute Quick Decision Cycle

The quick decision cycle responds rapidly to significant sentiment changes without waiting for a full analysis:

1. Process any new videos that have arrived since the last cycle
2. Calculate incremental sentiment updates with a 20% freshness boost
3. Identify significant sentiment changes (threshold of 1.0)
4. Make quick trading decisions for coins with significant changes
5. Apply throttling to prevent excessive trading (1-hour cooldown)

### 30-Minute Comprehensive Cycle

The comprehensive cycle performs a thorough analysis of all accumulated videos:

1. Process all accumulated videos and update VPH metrics
2. Analyze complete transcripts using Mixtral-8x7b-instruct
3. Calculate global sentiment scores weighted by VPH
4. Identify trading opportunities based on comprehensive analysis
5. Execute trades with position sizing based on tracked coins

## Multi-Stage Transcript Acquisition

The system acquires transcripts through a three-stage process:

1. **Caption Retrieval**: First attempts to fetch official captions using YouTube Transcript API
2. **Audio Transcription**: If captions aren't available, downloads audio using yt-dlp and transcribes with Whisper
3. **Simulation Fallback**: As a last resort, generates a simulated transcript to avoid complete failure

## Testing

Run tests using pytest:

```bash
python run_tests.py
```

Test the transcript service with audio transcription:

```bash
python test_transcript_audio.py <video_id>
```

Find testable videos:

```bash
python find_testable_videos.py
```

## Architecture Diagram

```
┌────────────────┐         ┌────────────────┐
│    FastAPI     │◄───────►│CryptoTradingService
└────────────────┘         └───────┬────────┘
                                  │
                          ┌───────┴────────┐
                          │                │
┌─────────────────┐       ▼                ▼
│  CacheManager   │◄────►┌─────────────┐   ┌─────────────────┐
└─────────────────┘      │ AppService  │   │SentimentService │
        │                └─────┬───────┘   └────────┬────────┘
        ▼                      │                    │
┌─────────────────┐           ▼                    ▼
│      Redis      │    ┌────────────────┐   ┌─────────────┐
└─────────────────┘    │TranscriptService│   │  Langroid   │
                       └───────┬────────┘   └─────┬───────┘
                               │                  │
                       ┌───────┴────────┐   ┌────┴────────┐
                       │    yt-dlp      │   │   Mixtral   │
                       └───────┬────────┘   └─────────────┘
                               │
                       ┌───────┴────────┐
                       │    Whisper     │
                       └────────────────┘
```

## Configuration

The application uses environment variables for configuration. See `config/settings.py` for all available options.

Key configuration options:
- `OPENROUTER_API_KEY`: API key for accessing the Mixtral model via OpenRouter
- `ANTHROPIC_API_KEY`: API key for accessing Claude 3 Sonnet via Anthropic
- `VPH_THRESHOLD`: Threshold for filtering videos based on views per hour (default: 500.0)
- `CYCLE_INTERVAL`: Interval between comprehensive cycles in minutes (default: 30)
- `QUICK_DECISION_INTERVAL`: Interval between quick decision cycles in minutes (default: 5)
- `WHISPER_MODEL`: Whisper model size for audio transcription (default: small)

## Sentiment Analysis

The application uses Langroid with the Mixtral-8x7b-instruct model to analyze sentiment in video transcripts:

1. Transcripts are retrieved through the multi-stage acquisition process
2. The Mixtral model analyzes the transcript for cryptocurrency sentiment
3. Sentiment is rated on a scale of -10 to +10 for each cryptocurrency mentioned
4. Special attention is given to small-cap coins and urgent opportunities
5. Scores are weighted by the video's VPH (views per hour)
6. New content receives a 20% freshness boost in quick decisions
7. Newly discovered coins receive a 10% boost for visibility

## Trading Logic

Trading decisions are made by the Claude 3 Sonnet master agent with sophisticated reasoning:

1. Sentiment data is combined with market data (price, volume, market cap)
2. Technical indicators (MACD) provide additional context
3. The master agent evaluates the overall situation and provides a confidence score
4. Position sizing is dynamically adjusted based on:
   - Number of coins being tracked (1.0/N base)
   - Confidence level from the master agent
   - Min/max bounds to prevent extreme allocations
5. Throttling prevents trading the same coin more than once per hour

## Simulation Implementation

The system is designed to run in simulation mode with $10,000 in simulated capital:

1. Start with all funds in cash
2. Trade based on sentiment analysis and master agent decisions
3. Dynamic position sizing allocates funds across opportunities
4. Track performance metrics and compare quick vs. comprehensive decisions
5. Generate reports on trading performance and sentiment accuracy

## Roadmap

1. **Database Integration**: Store historical sentiment scores and trading decisions
2. **UI Dashboard**: Create a React-based frontend for visualization
3. **Monitoring**: Add Prometheus and Grafana for observability
4. **Backtesting**: Implement a proper backtesting framework for strategy evaluation
5. **Extended AI**: Enhance sentiment analysis with news articles and social media
6. **Live Trading**: Add support for live trading with additional safeguards