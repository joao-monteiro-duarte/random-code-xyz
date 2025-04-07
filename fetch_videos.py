from datetime import datetime, timezone, timedelta
import aiohttp
import asyncio
import logging
import json
import os
import redis as real_redis
import time
from config import VPH_THRESHOLD, VPH_VIRAL_THRESHOLD, CACHE_FILE, YOUTUBE_API_KEYS,REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)  # Keep this!

class MockRedis:
    def __init__(self, *args, **kwargs):
        pass
    def ping(self): pass
    def get(self, key): return None
    def set(self, key, value, ex=None): pass
    def hget(self, key, field): return None
    def hset(self, key, field, value): pass

def connect_redis_with_retry(max_retries=10, delay=2):
    pool = real_redis.ConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        decode_responses=True,
        max_connections=10
    )
    for attempt in range(max_retries):
        try:
            client = real_redis.Redis(connection_pool=pool)
            client.ping()
            logger.info(f"Redis connected successfully on attempt {attempt + 1}")
            return client
        except real_redis.ConnectionError as e:
            logger.warning(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
    logger.error("Redis unavailable after all retries, using mock client")
    return MockRedis()

def safe_redis_op(redis_client, operation, *args, **kwargs):
    if isinstance(redis_client, MockRedis):
        return getattr(redis_client, operation)(*args, **kwargs)
    for _ in range(3):  # Retry 3 times
        try:
            return getattr(redis_client, operation)(*args, **kwargs)
        except real_redis.ConnectionError as e:
            logger.warning(f"Redis {operation} failed: {e}, retrying...")
            time.sleep(1)  # Brief delay before retry
    logger.warning(f"Redis {operation} failed after retries, using mock behavior")
    return MockRedis().__getattribute__(operation)(*args, **kwargs)

async def fetch_with_key_rotation(session, url, params, key_index=0):
    if key_index >= len(YOUTUBE_API_KEYS):
        raise Exception("All YouTube API keys exceeded quota")
    params["key"] = YOUTUBE_API_KEYS[key_index]
    async with session.get(url, params=params) as response:
        if response.status == 200:
            return await response.json()
        elif response.status == 403:
            logger.warning(f"Quota exceeded for key {key_index}, switching to key {key_index + 1}")
            return await fetch_with_key_rotation(session, url, params, key_index + 1)
        else:
            raise Exception(f"API error: {response.status}")

async def get_video_stats(session, video_id, cache, new_videos):
    if video_id in cache:
        logger.debug(f"Using cached stats for {video_id}")
        return cache[video_id]
    if len(new_videos) >= 20:
        logger.debug(f"Skipping stats for {video_id} - daily limit reached")
        return {"views": 0, "is_live": "not_live"}
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {"part": "statistics,liveStreamingDetails", "id": video_id}
    data = await fetch_with_key_rotation(session, url, params)
    item = data["items"][0]
    stats = item["statistics"]
    views = int(stats.get("viewCount", 0))
    is_live = "liveStreamingDetails" in item
    live_status = "live" if is_live and item["liveStreamingDetails"].get("actualEndTime") is None else "ended" if is_live else "not_live"
    result = {"views": views, "is_live": live_status}
    cache[video_id] = result
    new_videos.add(video_id)
    logger.debug(f"Video {video_id} stats: views={views}, live_status={live_status}")
    return result

async def fetch_videos(full_search=False):
    videos = []
    seen_ids = set()
    max_vph = 0
    current_time = datetime.now(timezone.utc)
    published_after = (current_time - timedelta(days=2 if full_search else 1)).isoformat().replace("+00:00", "Z")
    queries = ["bitcoin crypto news", "ethereum crypto news", "solana crypto news", "crypto market analysis"]
    short_queries = ["bitcoin news", "crypto update"]
    deep_search_queries = [
        "bitcoin price prediction", "ethereum price prediction", "solana price prediction",
        "crypto trading strategy", "blockchain news", "crypto market trends",
        "altcoin analysis", "defi news", "nft market update"
    ]
    redis_client = connect_redis_with_retry()
    if isinstance(redis_client, MockRedis):
        logger.warning("Redis unavailable, proceeding without caching")

    cache = {}
    new_videos = set()
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        logger.info(f"Loaded video stats cache from {CACHE_FILE}")

    async with aiohttp.ClientSession() as session:
        search_queries = deep_search_queries if full_search else short_queries
        for query in search_queries:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": query,
                "maxResults": 50 if full_search else 15,
                "order": "relevance",
                "publishedAfter": published_after,
                "type": "video"
            }
            try:
                data = await fetch_with_key_rotation(session, url, params)
                for item in data.get("items", []):
                    video_id = item["id"]["videoId"]
                    if video_id in seen_ids:
                        logger.debug(f"Skipping duplicate video {video_id}")
                        continue
                    seen_ids.add(video_id)
                    title = item["snippet"]["title"]
                    description = item["snippet"]["description"]
                    publish_time = item["snippet"]["publishedAt"]
                    stats = await get_video_stats(session, video_id, cache, new_videos)
                    views = stats["views"]
                    is_live = stats["is_live"]
                    publish_dt = datetime.fromisoformat(publish_time.replace("Z", "+00:00"))
                    hours_since = max((current_time - publish_dt).total_seconds() / 3600, 1)
                    safe_redis_op(redis_client, "hset", f"video:{video_id}", "hours_since", hours_since)
                    vph = views / hours_since
                    if vph >= VPH_THRESHOLD:
                        videos.append((video_id, title, description, vph, title + " " + description, is_live))
                        logger.info(f"Added video {video_id}: {title} (VPH: {vph:.2f})")
                        max_vph = max(max_vph, vph)
            except Exception as e:
                logger.error(f"Error fetching videos for {query}: {e}")

    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
    logger.info(f"Saved video stats cache to {CACHE_FILE}")

    logger.info(f"Total videos with VPH > {VPH_THRESHOLD}: {len(videos)}")
    trigger_deep_search = max_vph > VPH_VIRAL_THRESHOLD
    if trigger_deep_search:
        logger.info("High VPH detected, triggering deep search in next full sweep")
    return videos, trigger_deep_search

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(fetch_videos(full_search=False))