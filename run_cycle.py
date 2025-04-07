import asyncio
import logging
import aiohttp
import os
import json
from datetime import datetime, timedelta, timezone
from fetch_videos import fetch_videos
from transcript_service import TranscriptService
from sentiment_service import SentimentService
from trade_service import TradeService
from config import VPH_THRESHOLD, OPENROUTER_API_KEY
from market_data_service import MarketDataService

logger = logging.getLogger(__name__)

class TradingCycle:
    def __init__(self):
        self.transcript_service = TranscriptService()
        self.sentiment_service = SentimentService(api_key=OPENROUTER_API_KEY)
        self.trade_service = TradeService(api_key=OPENROUTER_API_KEY)
        self.tracked_videos = []
        self.video_scores = []
        self.global_scores = {"BTC": 0.0, "ETH": 0.0, "SOL": 0.0}
        self.last_full_sweep = None
        self.last_long_cycle = None
        self.full_sweep_count = 0
        self.last_long_cycle = datetime.now() - timedelta(minutes=31)

    async def short_cycle(self, full_search=False):
        if (datetime.now() - self.sentiment_service.last_analysis_time).total_seconds() < 10:
            logger.info("Rate limit buffer: sleeping 10s")
            await asyncio.sleep(10)
        logger.info(f"Starting short cycle (full_search={full_search})...")
        videos, trigger_full_sweep = await fetch_videos(full_search=full_search)
        logger.info(f"Fetched {len(videos)} videos with VPH > {VPH_THRESHOLD}: {[v[0] for v in videos]}")
        if videos:
            highest_vph_video = max(videos, key=lambda x: x[3])
            lowest_vph_video = min(videos, key=lambda x: x[3])
            logger.info(f"Highest VPH video: {highest_vph_video[0]} - {highest_vph_video[1]} (VPH: {highest_vph_video[3]:.2f})")
            logger.info(f"Lowest VPH video: {lowest_vph_video[0]} - {lowest_vph_video[1]} (VPH: {lowest_vph_video[3]:.2f})")

        transcripts = await self.transcript_service.process_videos(videos)
        new_transcripts = [(vid, t) for vid, t in transcripts if vid not in self.sentiment_service.sentiment_cache]
        logger.info(f"Processing {len(new_transcripts)} new transcripts, skipping {len(transcripts) - len(new_transcripts)} cached")
        videos_with_transcripts = {v[0]: {"vph": v[3]} for v in videos if v[0] in [t[0] for t in transcripts]}  # Dict instead of list
        
        video_scores = []
        for video_id, transcript in new_transcripts:
            scores = await self.sentiment_service.analyze_transcript(video_id, transcript)
            video_scores.append(scores)

        for video_id in videos_with_transcripts:
            if video_id not in [t[0] for t in new_transcripts]:
                cached_scores = self.sentiment_service.sentiment_cache.get(video_id, {coin: {"sign": 0, "I": 0, "E": 0, "S": 0} for coin in self.trade_service.fixed_coins})
                video_scores.append(cached_scores)
        
        # Unindented: Calculate global scores, market data, decisions, and execute trades
        global_scores = await self.sentiment_service.calculate_global_scores(video_scores, videos)
        market_service = MarketDataService()
        market_data = await market_service.get_market_data()
        decisions = await self.trade_service.make_decisions(video_scores, videos_with_transcripts, market_data)
        logger.info(f"Trading decisions: {decisions}")
        self.trade_service.execute_trade(decisions, market_data)
        portfolio_value = self.trade_service.calculate_portfolio_value(market_data)
        logger.info(f"Portfolio value: ${portfolio_value:.2f}")
        self.log_mock_trade(decisions)

        if full_search or not self.tracked_videos:
            self.tracked_videos = videos[:50]  # Keep as list of tuples
            self.video_scores = video_scores[:50]
        else:
            self.tracked_videos.extend(videos)
            self.video_scores.extend(video_scores)
            self.tracked_videos = self.tracked_videos[:50]
            self.video_scores = self.video_scores[:50]
        self.global_scores = global_scores
        self.sentiment_service.last_analysis_time = datetime.now()
        return trigger_full_sweep

    async def long_cycle(self):
        if not self.tracked_videos:
            logger.info("No tracked videos for long cycle update")
            return

        logger.info("Starting 30-min long cycle: Updating VPH and scores...")
        updated_videos = []
        async with aiohttp.ClientSession() as session:
            for video in self.tracked_videos:
                video_id = video[0]
                stats = await self.get_video_stats(session, video_id)
                vph = stats["vph"]
                updated_videos.append((video_id, video[1], video[2], vph, video[4], video[5]))
                logger.info(f"Updated video {video_id}: VPH={vph:.2f}")

        new_global_scores = await self.sentiment_service.calculate_global_scores(self.video_scores, updated_videos)
        logger.info(f"New global scores after long cycle: {new_global_scores}")

        for coin in ["BTC", "ETH", "SOL"]:
            shift = abs(new_global_scores[coin] - self.global_scores[coin])
            if shift > 1:
                logger.info(f"Significant score shift for {coin}: {self.global_scores[coin]:.2f} -> {new_global_scores[coin]:.2f}")
                market_service = MarketDataService()
                market_data = await market_service.get_market_data()
                videos_with_transcripts = {v[0]: {"vph": v[3]} for v in updated_videos}  # Dict for make_decisions
                decisions = await self.trade_service.make_decisions(self.video_scores, videos_with_transcripts, market_data)
                logger.info(f"Re-evaluated decisions due to {coin} shift: {decisions}")
                self.log_mock_trade(decisions)

        self.global_scores = new_global_scores
        self.last_long_cycle = datetime.now()

    async def get_video_stats(self, session, video_id):
        from fetch_videos import CACHE_FILE, fetch_with_key_rotation
        cache = {}
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)

        if video_id in cache:
            views = cache[video_id]["views"]
            is_live = cache[video_id]["is_live"]
        else:
            url = "https://www.googleapis.com/youtube/v3/videos"
            params = {"part": "statistics,liveStreamingDetails,snippet", "id": video_id}
            data = await fetch_with_key_rotation(session, url, params)
            item = data["items"][0]
            stats = item["statistics"]
            views = int(stats.get("viewCount", 0))
            is_live = "liveStreamingDetails" in item and item["liveStreamingDetails"].get("actualEndTime") is None
            cache[video_id] = {"views": views, "is_live": "live" if is_live else "not_live"}
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f)

        video = next(v for v in self.tracked_videos if v[0] == video_id)
        publish_time = await self.get_publish_time(session, video_id)
        publish_dt = datetime.fromisoformat(publish_time.replace("Z", "+00:00"))
        hours_since = max((datetime.now(timezone.utc) - publish_dt).total_seconds() / 3600, 1)
        vph = views / hours_since
        return {"vph": vph}

    async def get_publish_time(self, session, video_id):
        from fetch_videos import CACHE_FILE, fetch_with_key_rotation
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {"part": "snippet", "id": video_id}
        data = await fetch_with_key_rotation(session, url, params)
        return data["items"][0]["snippet"]["publishedAt"]

    def log_mock_trade(self, decisions):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("mock_trades.log", "a") as f:
            for coin, decision in decisions.items():
                if decision["decision"] in ["BUY", "SELL"]:
                    f.write(f"{timestamp} - {coin}: {decision}\n")

async def main():
    cycle = TradingCycle()
    while True:
        now = datetime.now()
        do_full_sweep = (now.hour == 0 and now.minute < 5 and 
                         (cycle.last_full_sweep is None or cycle.last_full_sweep.date() != now.date()))
        if do_full_sweep and cycle.full_sweep_count < 1:
            trigger_full = await cycle.short_cycle(full_search=True)
            cycle.full_sweep_count += 1
            cycle.last_full_sweep = now
        else:
            trigger_full = await cycle.short_cycle(full_search=False)
        if (not cycle.last_long_cycle or (now - cycle.last_long_cycle) >= timedelta(minutes=30)):
            await cycle.long_cycle()
            cycle.last_long_cycle = now
        if trigger_full and cycle.full_sweep_count < 2:
            logger.info("High VPH detected, triggering extra full sweep...")
            await cycle.short_cycle(full_search=True)
            cycle.full_sweep_count += 1
            cycle.last_full_sweep = now
        await asyncio.sleep(300)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())