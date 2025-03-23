"""
Application settings and configuration.
"""
import os
import logging.config
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys and credentials
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
YOUTUBE_API_KEYS = os.getenv("YOUTUBE_API_KEYS", "AIzaSyD8LfVIXxQ9IDCKKTAxruyRTWOowcGRikE").split(",")
COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "CG-9QJjJ8s1jHGcVBU3qXB9imY3")

# Database settings
DB_PATH = os.getenv("DB_PATH", "trades.db")
DB_CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))

# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_EXPIRE = int(os.getenv("REDIS_EXPIRE", "86400"))  # 24 hours in seconds

# Application settings
VPH_THRESHOLD = float(os.getenv("VPH_THRESHOLD", "500"))
MIN_WEIGHT = float(os.getenv("MIN_WEIGHT", "0.05"))  # 5% minimum weight for low-VPH videos
MAX_VIDEOS = int(os.getenv("MAX_VIDEOS", "15"))
FETCH_INTERVAL = int(os.getenv("FETCH_INTERVAL", "300"))  # 5 minutes in seconds
CYCLE_INTERVAL = int(os.getenv("CYCLE_INTERVAL", "1800"))  # 30 minutes in seconds
API_THROTTLE_INTERVAL = int(os.getenv("API_THROTTLE_INTERVAL", "600"))  # 10 minutes in seconds
MAX_VIDEOS_PER_FETCH = int(os.getenv("MAX_VIDEOS_PER_FETCH", "5"))

# Crypto settings
INITIAL_BUDGET = float(os.getenv("INITIAL_BUDGET", "1000"))  # ADA
CRYPTOCURRENCIES = ["bitcoin", "ethereum", "solana"]
TIME_DECAY_HOURS = float(os.getenv("TIME_DECAY_HOURS", "12"))  # Half-life in hours

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "process_log.txt")

# Channel IDs for crypto content
CHANNEL_IDS = [
    "UCbLhGKVY-bJPcawebgtNfbw",  # Coinbureau
    "UCL1v8HMN_H6NQhK9vW_Hc8Q",  # Altcoin Daily
    "UCJgHxpqfhWEEjYH9cLXqhIQ",  # Benjamin Cowen
    "UCCatR7nWbYrkVXdxXb4cGXw",  # Anthony Pompliano
    "UCRvqjQPSeaWn-uEx-w0XOIg",  # Coin Market Cap
    "UCWH7q2MxnpD-pMGF-yZNvhQ",  # Finematics
    "UCiDbqzdpj6L_DKsHvHY8n0w",  # Lark Davis
    "UCMtJYS0PrtiUwlk6zjGDEMA",  # DataDash
    "UCl2oCaw8hdR_kbqyqd2klIA",  # Crypto Casey
    "UCNZb8eUomqPYgrdVeOn4eZA",  # Chico Crypto
    "UCEFJVYNiPp8xeIUyfaPCPQw",  # Crypto Zombie
    "UCqK_GSMbpiV8spgD3ZGloSw",  # BitBoy Crypto
    "UCvBqzQOhDOPm5A7W_XlpmCQ",  # The Moon
    "UCtOV5M-T3GcsJAq8QKaf0lg",  # Coin Bureau
    "UC4nXWTi1Mpgdwi-L_ESC5wQ",  # The Crypto Lark
    "UCofTOFX9_Fc7Hh2HV-rXhJA",  # MMCrypto
    "UCEuZmIQh6S-AiMqasfVmGBw",  # Crypto Michael
    "UCDUJvtXmmF7-NZm57XbHvWQ",  # Crypto Jebb
    "UC-5HLi3buMzdxjdTdic3Aig",  # JRNY Crypto
    "UC5sjcN5ZDKSQKm000LGZc5g"   # Crypto Tips
]

# Configure logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': LOG_LEVEL,
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': LOG_LEVEL,
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': LOG_FILE,
            'mode': 'a',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': LOG_LEVEL,
            'propagate': True
        },
    }
}

# Initialize logging
logging.config.dictConfig(LOGGING_CONFIG)