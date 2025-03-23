"""
Module for managing cache operations using Redis.
"""
import json
import logging
from typing import Optional, Dict, List, Any, TypeVar, Generic, Union
import time
from functools import wraps
from redis.exceptions import RedisError

# Setup logging
logger = logging.getLogger(__name__)

T = TypeVar('T')

class CacheManager:
    """
    Manages Redis caching operations with fallback to local storage.
    """
    def __init__(self, redis_client, redis_available: bool = True, prefix: str = ""):
        """
        Initialize the cache manager.
        
        Args:
            redis_client: Redis client instance
            redis_available: Whether Redis is available
            prefix: Prefix for all Redis keys
        """
        self.redis_client = redis_client
        self.redis_available = redis_available
        self.prefix = prefix
        self.local_cache: Dict[str, Any] = {}
        self.retry_attempts = 3
        self.retry_delay = 0.5  # seconds
        
    def _get_key(self, key: str) -> str:
        """
        Get the full Redis key with prefix.
        
        Args:
            key: Base key
        
        Returns:
            Full key with prefix
        """
        return f"{self.prefix}:{key}" if self.prefix else key
    
    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get a value from Redis with fallback to local cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
        
        Returns:
            Cached value or default if not found
        """
        if not key:
            return default
            
        full_key = self._get_key(key)
        
        # Try Redis first if available
        if self.redis_available and self.redis_client:
            for attempt in range(self.retry_attempts):
                try:
                    value = self.redis_client.get(full_key)
                    if value:
                        try:
                            # Try to decode JSON
                            return json.loads(value)
                        except (TypeError, json.JSONDecodeError):
                            # Return as is if not JSON
                            return value
                    break
                except RedisError as e:
                    logger.warning(f"Redis error on attempt {attempt+1}/{self.retry_attempts}: {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Redis get operation failed after {self.retry_attempts} attempts")
        
        # Fallback to local cache
        return self.local_cache.get(key, default)
    
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """
        Set a value in Redis and local cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Expiration time in seconds
        
        Returns:
            True if set in Redis, False otherwise
        """
        if not key:
            return False
            
        full_key = self._get_key(key)
        
        # Always update local cache
        self.local_cache[key] = value
        
        # Try Redis if available
        if self.redis_available and self.redis_client:
            for attempt in range(self.retry_attempts):
                try:
                    # Serialize value if it's not a string
                    if not isinstance(value, (str, bytes)):
                        serialized = json.dumps(value)
                    else:
                        serialized = value
                        
                    if expire:
                        self.redis_client.setex(full_key, expire, serialized)
                    else:
                        self.redis_client.set(full_key, serialized)
                    return True
                except RedisError as e:
                    logger.warning(f"Redis error on attempt {attempt+1}/{self.retry_attempts}: {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Redis set operation failed after {self.retry_attempts} attempts")
                except (TypeError, ValueError) as e:
                    logger.error(f"Error serializing value: {e}")
                    return False
        
        return False
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from Redis and local cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted, False otherwise
        """
        if not key:
            return False
        
        full_key = self._get_key(key)
        
        # Remove from local cache
        if key in self.local_cache:
            del self.local_cache[key]
        
        # Try Redis if available
        if self.redis_available and self.redis_client:
            for attempt in range(self.retry_attempts):
                try:
                    self.redis_client.delete(full_key)
                    return True
                except RedisError as e:
                    logger.warning(f"Redis error on attempt {attempt+1}/{self.retry_attempts}: {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Redis delete operation failed after {self.retry_attempts} attempts")
        
        return False
    
    def keys(self, pattern: str) -> List[str]:
        """
        Get keys matching a pattern from Redis.
        
        Args:
            pattern: Key pattern
        
        Returns:
            List of matching keys
        """
        full_pattern = self._get_key(pattern)
        
        if self.redis_available and self.redis_client:
            for attempt in range(self.retry_attempts):
                try:
                    keys = self.redis_client.keys(full_pattern)
                    # Remove prefix from keys
                    if self.prefix:
                        return [key.replace(f"{self.prefix}:", "", 1) for key in keys]
                    return keys
                except RedisError as e:
                    logger.warning(f"Redis error on attempt {attempt+1}/{self.retry_attempts}: {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Redis keys operation failed after {self.retry_attempts} attempts")
        
        # Fallback: for local cache, just return keys that match the pattern
        prefix_len = len(self.prefix) + 1 if self.prefix else 0
        pattern = pattern.replace("*", "")
        return [k for k in self.local_cache.keys() if pattern in k[prefix_len:]]
    
    def clear_cache(self, pattern: str = "*") -> int:
        """
        Clear cache entries matching a pattern.
        
        Args:
            pattern: Key pattern to match
        
        Returns:
            Number of entries cleared
        """
        full_pattern = self._get_key(pattern)
        count = 0
        
        # Clear matching entries from local cache
        keys_to_delete = []
        for key in self.local_cache:
            if pattern == "*" or pattern in key:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.local_cache[key]
            count += 1
        
        # Clear from Redis if available
        if self.redis_available and self.redis_client:
            for attempt in range(self.retry_attempts):
                try:
                    keys = self.redis_client.keys(full_pattern)
                    if keys:
                        self.redis_client.delete(*keys)
                        count += len(keys) - count  # Avoid double counting local cache entries
                    break
                except RedisError as e:
                    logger.warning(f"Redis error on attempt {attempt+1}/{self.retry_attempts}: {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Redis clear operation failed after {self.retry_attempts} attempts")
        
        return count
    
    def get_many(self, keys: List[str], default: Optional[T] = None) -> Dict[str, Any]:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            default: Default value for missing keys
        
        Returns:
            Dictionary of key-value pairs
        """
        result = {}
        
        # Get from Redis if available
        if self.redis_available and self.redis_client and keys:
            full_keys = [self._get_key(key) for key in keys]
            for attempt in range(self.retry_attempts):
                try:
                    values = self.redis_client.mget(full_keys)
                    for key, value in zip(keys, values):
                        if value:
                            try:
                                result[key] = json.loads(value)
                            except (TypeError, json.JSONDecodeError):
                                result[key] = value
                        else:
                            result[key] = self.local_cache.get(key, default)
                    return result
                except RedisError as e:
                    logger.warning(f"Redis error on attempt {attempt+1}/{self.retry_attempts}: {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Redis mget operation failed after {self.retry_attempts} attempts")
        
        # Fallback to local cache
        for key in keys:
            result[key] = self.local_cache.get(key, default)
        
        return result
    
    def set_many(self, mapping: Dict[str, Any], expire: Optional[int] = None) -> bool:
        """
        Set multiple values in cache.
        
        Args:
            mapping: Dictionary of key-value pairs
            expire: Expiration time in seconds
        
        Returns:
            True if all values were set, False otherwise
        """
        if not mapping:
            return False
            
        # Always update local cache
        self.local_cache.update(mapping)
        
        # Try Redis if available
        if self.redis_available and self.redis_client:
            for attempt in range(self.retry_attempts):
                try:
                    # Convert values to JSON if needed
                    redis_mapping = {}
                    for key, value in mapping.items():
                        full_key = self._get_key(key)
                        if not isinstance(value, (str, bytes)):
                            try:
                                redis_mapping[full_key] = json.dumps(value)
                            except (TypeError, ValueError) as e:
                                logger.error(f"Error serializing value for key {key}: {e}")
                                redis_mapping[full_key] = str(value)
                        else:
                            redis_mapping[full_key] = value
                    
                    # Use pipeline for efficiency
                    pipe = self.redis_client.pipeline()
                    for key, value in redis_mapping.items():
                        if expire:
                            pipe.setex(key, expire, value)
                        else:
                            pipe.set(key, value)
                    pipe.execute()
                    return True
                except RedisError as e:
                    logger.warning(f"Redis error on attempt {attempt+1}/{self.retry_attempts}: {e}")
                    if attempt < self.retry_attempts - 1:
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"Redis mset operation failed after {self.retry_attempts} attempts")
        
        return False


def cached(prefix: str, expire: Optional[int] = None):
    """
    Decorator to cache function results.
    
    Args:
        prefix: Key prefix
        expire: Expiration time in seconds
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key from function name and arguments
            key = f"{prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try to get from cache
            cache_manager = getattr(self, 'cache_manager', None)
            if not cache_manager:
                return func(self, *args, **kwargs)
                
            cached_result = cache_manager.get(key)
            if cached_result is not None:
                return cached_result
                
            # Call function and cache result
            result = func(self, *args, **kwargs)
            cache_manager.set(key, result, expire=expire)
            return result
        return wrapper
    return decorator