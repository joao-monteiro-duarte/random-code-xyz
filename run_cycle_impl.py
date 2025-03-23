# Implementation of run_cycle for crypto_trading_app.py
# This file provides the run_cycle function for integration with Redis transcript caching

from datetime import datetime, timedelta
import json
import aiohttp
import asyncio
from typing import List, Tuple, Optional, Union

# Import from the refactored architecture
from services.app_service import get_app_service, AppService
from models.video import Video
from config.settings import YOUTUBE_API_KEYS, VPH_THRESHOLD

# Keep the original implementation accessible
from transcripts import process_video_stats as original_process_video_stats

# Define get_accumulated_videos and set_accumulated_videos functions
def get_accumulated_videos():
    """Get accumulated videos from service or fallback to Redis"""
    try:
        # Use the app service if available
        app_service = get_app_service()
        return app_service.get_accumulated_videos()
    except (ImportError, NameError):
        # Fallback to original implementation
        from config import REDIS_AVAILABLE, redis_client
        
        if REDIS_AVAILABLE and redis_client:
            try:
                serialized = redis_client.get('accumulated_videos')
                if serialized:
                    return json.loads(serialized)
            except Exception as e:
                print(f"Error retrieving accumulated videos from Redis: {e}")
        
        # If Redis is not available or retrieval failed, use the fallback
        global accumulated_videos_fallback
        if 'accumulated_videos_fallback' not in globals():
            accumulated_videos_fallback = []
        return accumulated_videos_fallback

def set_accumulated_videos(videos):
    """Store accumulated videos using service or fallback to Redis"""
    try:
        # Use the app service if available
        app_service = get_app_service()
        app_service.set_accumulated_videos(videos)
    except (ImportError, NameError):
        # Fallback to original implementation
        from config import REDIS_AVAILABLE, redis_client
        
        if REDIS_AVAILABLE and redis_client:
            try:
                redis_client.set('accumulated_videos', json.dumps(videos))
            except Exception as e:
                print(f"Error storing accumulated videos to Redis: {e}")
        
        # Always update the fallback
        global accumulated_videos_fallback
        accumulated_videos_fallback = videos

# Initialize with empty list
accumulated_videos_fallback = []  # Fallback to local storage when Redis is unavailable

async def log_step(message):
    """Log a message with timestamp"""
    timestamp = datetime.now().isoformat()
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)

async def update_market_data():
    """Update market data for cryptocurrencies"""
    # Mock implementation
    await log_step("Market data updated")

async def prune_score_history():
    """Prune old scores from history"""
    # Mock implementation
    await log_step("Score history pruned")

async def update_vph_for_existing_videos():
    """Update views per hour for existing videos"""
    # Mock implementation
    await log_step("VPH updated for existing videos")

async def analyze_sentiment(videos, vph_threshold, make_decision=True):
    """
    Analyze sentiment for videos using Mixtral via Langroid
    
    Args:
        videos: List of videos to analyze
        vph_threshold: Threshold for VPH weighting
        make_decision: Whether to make trading decisions based on sentiment
        
    Returns:
        Dictionary of global sentiment scores for cryptocurrencies
    """
    try:
        # Try to get the app service to use its sentiment service
        app_service = get_app_service()
        has_app_service = True
    except (ImportError, NameError):
        has_app_service = False
        
    if has_app_service:
        await log_step(f"Analyzing sentiment for {len(videos)} videos using Mixtral")
        
        # Prepare video transcripts for analysis
        video_transcripts = []
        for video in videos:
            if isinstance(video, Video):
                vid_id = video.id
            else:
                vid_id = video[0]
                
            # Get transcript from the service
            transcript = app_service.transcript_service.get_transcript(vid_id)
            if transcript and len(transcript) > 50:
                video_transcripts.append((vid_id, transcript))
        
        if not video_transcripts:
            await log_step("No valid transcripts found for sentiment analysis")
            return {}
        
        # Process the transcripts in batch
        video_sentiments = await app_service.sentiment_service.batch_analyze(video_transcripts)
        
        # Calculate global scores weighted by VPH
        global_scores = await app_service.sentiment_service.calculate_global_scores(
            video_sentiments, videos, vph_threshold
        )
        
        # Log the results
        await log_step(f"Analyzed sentiment for {len(video_transcripts)} videos using Mixtral")
        
        # Enhanced logging for debugging trade decisions
        for crypto, data in global_scores.items():
            if isinstance(data, dict):
                score_value = data.get("score", 0)
                is_small_cap = data.get("is_small_cap", False)
                urgency = data.get("urgency", "low")
                videos_mentioned = data.get("videos_mentioned", 0)
                
                await log_step(f"Global sentiment for {crypto}: score={score_value:.2f}, small_cap={is_small_cap}, urgency={urgency}, mentions={videos_mentioned}")
                
                # Check if this coin meets trading criteria
                if score_value >= 7:
                    await log_step(f"  ✅ {crypto} meets criteria for potential trade (score >= 7)")
                else:
                    await log_step(f"  ❌ {crypto} does not meet criteria for trade (score < 7)")
            else:
                # Handle legacy format where data is just the score
                await log_step(f"Global sentiment for {crypto}: {data:.2f}")
                
                if data >= 7:
                    await log_step(f"  ✅ {crypto} meets criteria for potential trade (score >= 7)")
                else:
                    await log_step(f"  ❌ {crypto} does not meet criteria for trade (score < 7)")
            
        return global_scores
    else:
        # Mock implementation if service isn't available
        await log_step(f"Mocked sentiment analysis for {len(videos)} videos (service unavailable)")
        # Include a high sentiment coin (pepe) to test trading logic
        return {
            "bitcoin": {
                "score": 5.0,
                "is_small_cap": False,
                "urgency": "low",
                "videos_mentioned": 2
            },
            "ethereum": {
                "score": 3.0,
                "is_small_cap": False,
                "urgency": "low",
                "videos_mentioned": 1
            },
            "solana": {
                "score": 6.0,
                "is_small_cap": False,
                "urgency": "medium",
                "videos_mentioned": 1
            },
            "pepe": {
                "score": 9.0,
                "is_small_cap": True,
                "urgency": "high",
                "videos_mentioned": 3
            }
        }

def get_portfolio_state():
    """Get current portfolio state"""
    # Mock implementation
    return {
        'bitcoin_allocation': 100,
        'ethereum_allocation': 100,
        'solana_allocation': 100,
        'allocated_ada': 300,
        'unallocated_ada': 700
    }

# Modified process_video_stats function that integrates with our service architecture
def process_video_stats(video_data, now=None):
    """
    Process video stats using the service architecture when available.
    Falls back to the original implementation when not.
    
    Args:
        video_data: Video tuple or Video object
        now: Current datetime
        
    Returns:
        Updated video tuple or Video object
    """
    try:
        # Use the app service if available
        app_service = get_app_service()
        return app_service.process_video_stats(video_data, now)
    except (ImportError, NameError):
        # Fallback to original implementation
        return original_process_video_stats(video_data, now)

# Main run_cycle function that integrates with Redis transcript caching
async def run_cycle(vph_threshold):
    """Process accumulated videos and update global scores every 30 minutes"""
    # Get accumulated videos from Redis or fallback
    accumulated_videos = get_accumulated_videos()
    
    # Import necessary functions
    from trades import get_portfolio_state, prune_score_history
    from fetch_videos import update_vph_for_existing_videos
    import aiohttp
    from config.settings import YOUTUBE_API_KEYS
    from transcripts import get_transcript_from_redis, save_transcript_to_redis
    
    # Try to get the app service
    try:
        app_service = get_app_service()
        has_app_service = True
    except (ImportError, NameError):
        has_app_service = False
    
    # Prune score history and update VPH for existing videos
    await prune_score_history()
    await update_vph_for_existing_videos()
    
    # Process accumulated videos
    if accumulated_videos:
        await log_step(f"Processing {len(accumulated_videos)} accumulated videos from 5-minute fetches")
        
        # Update VPH for accumulated videos using process_video_stats
        new_videos_with_updated_vph = []
        now = datetime.now()
        current_key_index = [0]  # Variable to track the current API key index
        
        for video in accumulated_videos:
            vid_id = video[0]
            try:
                # Fetch updated views using the YouTube API
                stats_url = "https://www.googleapis.com/youtube/v3/videos"
                stats_params = {
                    "part": "statistics", 
                    "id": vid_id, 
                    "key": YOUTUBE_API_KEYS[current_key_index[0] % len(YOUTUBE_API_KEYS)]
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(stats_url, params=stats_params) as resp:
                        stats = await resp.json()
                        
                        for item in stats.get("items", []):
                            # Get updated view count
                            new_views = int(item["statistics"].get("viewCount", 0))
                            
                            # Create updated video tuple with new view count
                            updated_video = (vid_id, new_views, video[2], video[3])
                            
                            # Process video stats to calculate updated VPH
                            processed_video = process_video_stats(updated_video, now)
                            
                            # Add to list of videos with updated VPH
                            new_videos_with_updated_vph.append(processed_video)
                            
                            # Log the update
                            vph = processed_video[3] if isinstance(processed_video, tuple) else processed_video.vph
                            await log_step(f"Updated VPH for accumulated video {vid_id}: {vph:.2f} (Views: {new_views})")
            except Exception as e:
                await log_step(f"Error updating VPH for accumulated video {vid_id}: {e}")
                # Rotate API key on error
                current_key_index[0] = (current_key_index[0] + 1) % len(YOUTUBE_API_KEYS)
                continue
        
        # Analyze sentiment with updated VPH values
        if new_videos_with_updated_vph:
            # Process videos that haven't been processed yet
            videos_to_process = []
            for video in new_videos_with_updated_vph:
                if isinstance(video, Video):
                    vid_id = video.id
                else:
                    vid_id = video[0]
                
                # Use service architecture or fallback to check transcript availability
                if has_app_service:
                    transcript = app_service.transcript_service.get_transcript(vid_id)
                else:
                    transcript = get_transcript_from_redis(vid_id)
                
                # Only add if transcript is available and not empty
                if transcript and not any(x in transcript for x in ["Transcript unavailable", "No transcript", "Error fetching transcript"]):
                    videos_to_process.append(video)
                else:
                    await log_step(f"Skipping video {vid_id} - transcript not available or contains errors")
            
            if videos_to_process:
                for video in videos_to_process:
                    if isinstance(video, Video):
                        vid_id = video.id
                        vph = video.vph
                        publish_time = video.publish_time
                    else:
                        vid_id, _, publish_time, vph = video
                    await log_step(f"Processing video {vid_id}: Updated VPH={vph:.2f}, Publish Time={publish_time}")
                
                # Update market data before making decisions
                await log_step("Refreshing market data before making trading decisions...")
                await update_market_data()
                
                # Store previous global scores for comparison
                from trades import global_scores
                previous_scores = {coin: score for coin, score in global_scores.items()}
                
                # Make trading decisions based on accumulated videos
                new_scores = await analyze_sentiment(videos_to_process, vph_threshold, make_decision=True)
                
                # Update global scores with new sentiment scores
                for coin, score in new_scores.items():
                    global_scores[coin] = score
                
                # Log final global scores and detect significant changes
                await log_step(f"Final global scores after cycle: {json.dumps(global_scores)}")
                
                # Log information about low-VPH videos' contribution
                await log_step(f"Videos with VPH < 500 are now guaranteed a minimum 5% weight impact")
                # Count low-VPH videos that contributed
                if isinstance(videos_to_process[0], Video):
                    low_vph_count = sum(1 for v in videos_to_process if v.vph < 500)
                else:
                    low_vph_count = sum(1 for v in videos_to_process if v[3] < 500)
                
                if low_vph_count > 0:
                    await log_step(f"This cycle included {low_vph_count} videos with VPH < 500 that received a minimum 5% weight boost")
                
                # Monitor for significant sentiment shifts
                for coin, new_score_data in global_scores.items():
                    old_score_data = previous_scores.get(coin, 0)
                    
                    # Extract score values based on data type
                    new_score_value = new_score_data["score"] if isinstance(new_score_data, dict) else new_score_data
                    old_score_value = old_score_data["score"] if isinstance(old_score_data, dict) else old_score_data
                    
                    if abs(new_score_value - old_score_value) > 3:  # Threshold for significant change
                        await log_step(f"⚠️ SIGNIFICANT SCORE CHANGE for {coin}: {old_score_value:.2f} → {new_score_value:.2f}")
                        
                        # If change is extremely large, flag for potential manipulation
                        if abs(new_score_value - old_score_value) > 5:
                            await log_step(f"🚨 POSSIBLE MANIPULATION ALERT: Extreme score change for {coin}")
            else:
                await log_step("No valid videos with transcripts to analyze in this cycle")
        else:
            await log_step("No accumulated videos to process after updating VPH")
        
        # Clear accumulated videos after processing
        accumulated_videos = []
        set_accumulated_videos(accumulated_videos)
    else:
        await log_step("No accumulated videos to process in this cycle")
    
    # Update market data and log portfolio summary
    await update_market_data()
    portfolio = get_portfolio_state()
    await log_step(f"\n=== CYCLE SUMMARY ===")
    await log_step(f"Bitcoin: {portfolio['bitcoin_allocation']:.2f} ADA")
    await log_step(f"Ethereum: {portfolio['ethereum_allocation']:.2f} ADA")
    await log_step(f"Solana: {portfolio['solana_allocation']:.2f} ADA")
    await log_step(f"Total Allocated: {portfolio['allocated_ada']:.2f} ADA ({(portfolio['allocated_ada']/1000)*100:.2f}%)")
    await log_step(f"Unallocated: {portfolio['unallocated_ada']:.2f} ADA ({(portfolio['unallocated_ada']/1000)*100:.2f}%)")