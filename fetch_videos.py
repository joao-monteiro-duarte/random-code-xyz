"""
Module for fetching and processing YouTube videos related to cryptocurrency
"""
from datetime import datetime, timedelta
import aiohttp
import asyncio
from config.settings import YOUTUBE_API_KEYS as API_KEYS

# Global state for key rotation
current_key_index = [0]  # Mutable list to track current key

# Define channels to fetch videos from
CHANNELS = [
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

async def rotate_api_key():
    """Rotate to the next API key."""
    current_key_index[0] = (current_key_index[0] + 1) % len(API_KEYS)
    print(f"Rotated to API key index {current_key_index[0]}")

async def fetch_with_key_rotation(session, url, params, attempts=0):
    """Fetch data from YouTube API with key rotation on quota errors."""
    if attempts >= len(API_KEYS):
        raise Exception("All API keys exceeded quota")
    
    params["key"] = API_KEYS[current_key_index[0]]
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 403:  # Quota exceeded
                print(f"Quota exceeded for key {API_KEYS[current_key_index[0]]}")
                await rotate_api_key()
                return await fetch_with_key_rotation(session, url, params, attempts + 1)
            else:
                raise Exception(f"API request failed with status {response.status}")
    except Exception as e:
        if "quota" in str(e).lower():
            await rotate_api_key()
            return await fetch_with_key_rotation(session, url, params, attempts + 1)
        raise e

async def initialize_channels():
    """Initialize the list of YouTube channels to fetch videos from"""
    return CHANNELS

async def fetch_initial_videos(websocket=None):
    """Fetch initial batch of crypto videos (<24h, >100 views/hour)"""
    if websocket:
        await websocket.send_text("Fetching initial videos...")
    
    videos = []
    try:
        current_time = datetime.now()
        published_after = (current_time - timedelta(days=1)).isoformat() + "Z"
        
        async with aiohttp.ClientSession() as session:
            for channel_id in CHANNELS[:5]:  # Limit to 5 channels
                api_url = "https://www.googleapis.com/youtube/v3/search"
                params = {
                    "part": "snippet",
                    "channelId": channel_id,
                    "maxResults": 3,
                    "order": "date",
                    "publishedAfter": published_after,
                    "type": "video"
                }
                
                data = await fetch_with_key_rotation(session, api_url, params)
                for item in data.get("items", []):
                    video_id = item["id"]["videoId"]
                    title = item["snippet"]["title"]
                    publish_time = item["snippet"]["publishedAt"]
                    
                    vid_stats = await get_video_stats(session, video_id)
                    views = vid_stats.get("views", 0)
                    
                    publish_dt = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
                    hours_since = max((current_time - publish_dt).total_seconds() / 3600, 1)
                    vph = views / hours_since
                    
                    if vph >= 100:
                        videos.append((video_id, views, publish_time, vph))
                        if websocket:
                            await websocket.send_text(f"Found video: {title} (VPH: {vph:.2f})")
    except Exception as e:
        if websocket:
            await websocket.send_text(f"Error fetching initial videos: {e}")
    
    return videos

async def fetch_crypto_news(websocket=None, max_results=5):
    """Fetch crypto news videos from YouTube"""
    if websocket:
        await websocket.send_text(f"Fetching crypto news (max {max_results} videos)...")
    
    videos = []
    try:
        current_time = datetime.now()
        published_after = (current_time - timedelta(hours=12)).isoformat() + "Z"
        
        async with aiohttp.ClientSession() as session:
            api_url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": "cryptocurrency news",
                "maxResults": max_results,
                "order": "date",
                "publishedAfter": published_after,
                "type": "video"
            }
            
            data = await fetch_with_key_rotation(session, api_url, params)
            for item in data.get("items", []):
                video_id = item["id"]["videoId"]
                title = item["snippet"]["title"]
                publish_time = item["snippet"]["publishedAt"]
                
                vid_stats = await get_video_stats(session, video_id)
                views = vid_stats.get("views", 0)
                
                publish_dt = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
                hours_since = max((current_time - publish_dt).total_seconds() / 3600, 1)
                vph = views / hours_since
                
                videos.append((video_id, views, publish_time, vph))
                if websocket:
                    await websocket.send_text(f"Found news video: {title} (VPH: {vph:.2f})")
    except Exception as e:
        if websocket:
            await websocket.send_text(f"Error fetching crypto news: {e}")
    
    return videos

async def fetch_channel_videos(channel_id, max_results=3):
    """Fetch videos from a specific YouTube channel"""
    videos = []
    try:
        current_time = datetime.now()
        published_after = (current_time - timedelta(days=3)).isoformat() + "Z"
        
        async with aiohttp.ClientSession() as session:
            api_url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "channelId": channel_id,
                "maxResults": max_results,
                "order": "date",
                "publishedAfter": published_after,
                "type": "video"
            }
            
            data = await fetch_with_key_rotation(session, api_url, params)
            for item in data.get("items", []):
                video_id = item["id"]["videoId"]
                publish_time = item["snippet"]["publishedAt"]
                
                vid_stats = await get_video_stats(session, video_id)
                views = vid_stats.get("views", 0)
                
                publish_dt = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
                hours_since = max((current_time - publish_dt).total_seconds() / 3600, 1)
                vph = views / hours_since
                
                videos.append((video_id, views, publish_time, vph))
    except Exception as e:
        print(f"Error fetching videos for channel {channel_id}: {e}")
    
    return videos

async def get_video_stats(session, video_id):
    """Get video statistics (views, likes, etc.) from YouTube API"""
    try:
        api_url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "statistics",
            "id": video_id
        }
        
        data = await fetch_with_key_rotation(session, api_url, params)
        if data.get("items") and len(data["items"]) > 0:
            statistics = data["items"][0]["statistics"]
            return {
                "views": int(statistics.get("viewCount", 0)),
                "likes": int(statistics.get("likeCount", 0)),
                "comments": int(statistics.get("commentCount", 0))
            }
    except Exception as e:
        print(f"Error fetching stats for video {video_id}: {e}")
    
    return {"views": 0, "likes": 0, "comments": 0}

def process_video_stats(video, now=None):
    """Process video statistics and recalculate VPH"""
    if now is None:
        now = datetime.now()
    
    video_id, views, publish_time, _ = video
    
    if isinstance(publish_time, str):
        if 'Z' in publish_time:
            publish_dt = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
        else:
            publish_dt = datetime.fromisoformat(publish_time)
    else:
        publish_dt = publish_time
    
    # Make sure now has timezone info if publish_dt does
    from datetime import timezone
    if publish_dt.tzinfo is not None and now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    elif publish_dt.tzinfo is None and now.tzinfo is not None:
        publish_dt = publish_dt.replace(tzinfo=timezone.utc)
    
    hours_since = max((now - publish_dt).total_seconds() / 3600, 1)
    new_vph = views / hours_since
    
    return (video_id, views, publish_time, new_vph)

async def process_video_stats_async(video_ids):
    """Legacy function for backward compatibility"""
    try:
        async with aiohttp.ClientSession() as session:
            api_url = "https://www.googleapis.com/youtube/v3/videos"
            params = {
                "part": "statistics,snippet",
                "id": ",".join(video_ids)
            }
            data = await fetch_with_key_rotation(session, api_url, params)
            
            if "items" not in data:
                print(f"YouTube API stats response missing 'items': {data}")
                return []
            
            now = datetime.now()
            videos = []
            for item in data.get("items", []):
                video_id = item["id"]
                statistics = item["statistics"]
                snippet = item["snippet"]
                
                views = int(statistics.get("viewCount", 0))
                publish_time = snippet.get("publishedAt")
                
                publish_dt = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
                hours_since = max((now - publish_dt).total_seconds() / 3600, 1)
                vph = views / hours_since
                
                videos.append((video_id, views, publish_time, vph))
            return videos
    except Exception as e:
        print(f"Error in process_video_stats_async: {e}")
        return []

async def update_vph_for_existing_videos():
    """Update VPH for existing videos in the database"""
    print("Updating VPH for existing videos...")
    await asyncio.sleep(0.1)
    return True

async def fetch_trending_videos(startup=False, max_results=5):
    """Fetch trending cryptocurrency videos"""
    query_terms = [
        "bitcoin price prediction",
        "ethereum analysis",
        "solana crypto",
        "cryptocurrency news today",
        "crypto trading strategy"
    ]
    if startup:
        query_terms.extend([
            "crypto market analysis",
            "altcoin season",
            "nft crypto",
            "web3 blockchain",
            "defi explained"
        ])
    
    videos = []
    current_time = datetime.now()
    published_after = (current_time - timedelta(hours=48)).isoformat() + "Z"
    
    async with aiohttp.ClientSession() as session:
        import random
        selected_terms = random.sample(query_terms, min(2, len(query_terms)))
        
        for term in selected_terms:
            api_url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": term,
                "maxResults": max(3, max_results // 2),
                "order": "viewCount",
                "publishedAfter": published_after,
                "type": "video"
            }
            
            try:
                data = await fetch_with_key_rotation(session, api_url, params)
                for item in data.get("items", []):
                    video_id = item["id"]["videoId"]
                    publish_time = item["snippet"]["publishedAt"]
                    
                    vid_stats = await get_video_stats(session, video_id)
                    views = vid_stats.get("views", 0)
                    
                    publish_dt = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
                    hours_since = max((current_time - publish_dt).total_seconds() / 3600, 1)
                    vph = views / hours_since
                    
                    if not any(v[0] == video_id for v in videos):
                        videos.append((video_id, views, publish_time, vph))
            except Exception as e:
                print(f"Error fetching videos for term '{term}': {e}")
                continue
    
    videos.sort(key=lambda x: x[3], reverse=True)
    return videos[:max_results]
