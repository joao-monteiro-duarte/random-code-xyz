"""
Configuration module for the crypto-trading-pool application.
"""
import redis
from config.settings import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB

# Initialize Redis client
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD or None,
        db=REDIS_DB,
        decode_responses=True
    )
    redis_client.ping()
    REDIS_AVAILABLE = True
except Exception:
    redis_client = None
    REDIS_AVAILABLE = False