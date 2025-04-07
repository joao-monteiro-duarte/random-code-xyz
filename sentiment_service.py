import logging
import os
import importlib.util
import asyncio
import json
import re
import time
import redis as real_redis
import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from utils.openrouter_utils import configure_openrouter_env
from config import REDIS_HOST, REDIS_PORT, OPENROUTER_API_KEY

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

LANGROID_AVAILABLE = False
try:
    if importlib.util.find_spec("langroid") is not None:
        import langroid as lr
        lr.redis = type('MockModule', (), {'Redis': MockRedis})
        lm = lr.language_models
        if hasattr(lm, "OpenAIGPT"):
            LANGROID_AVAILABLE = True
            logger.info("Successfully imported Langroid components")
except Exception as e:
    logger.warning(f"Error checking for Langroid: {e}")

class SentimentService:
    def __init__(self, api_key: Optional[str] = None):
        self.redis_client = connect_redis_with_retry()
        self.run_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("Scout Logs", exist_ok=True)
        self.log_file = f"Scout Logs/scout_run_{self.run_start}.txt"
        with open(self.log_file, "a") as f:
            f.write(f"Run started: {self.run_start}\n")
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("No OpenRouter API key provided. Set OPENROUTER_API_KEY in .env.")
        self.last_analysis_time = datetime.now()
        self.sentiment_cache = {}
        self.fixed_coins = ["BTC", "ETH", "SOL"]
        self.agent = None
        self.model_ready = False

        if LANGROID_AVAILABLE:
            try:
                from langroid.language_models import OpenAIGPTConfig, OpenAIGPT
                from langroid.utils.configuration import set_global, Settings

                set_global(Settings(cache=False))  # Disable caching
                logger.info("Configuring Langroid with OpenRouter for Llama-3.1-70B")
                os.environ["OPENAI_API_KEY"] = self.api_key
                os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
                configure_openrouter_env(self.api_key, "SentimentService", debug=True)

                openai_config = OpenAIGPTConfig(
                    chat_model="meta-llama/llama-3.1-70b-instruct",
                    api_key=self.api_key,
                    api_base="https://openrouter.ai/api/v1",
                    temperature=0.0,
                    max_output_tokens=1000,
                    chat_context_length=128000,
                )

                self.llm = OpenAIGPT(config=openai_config)
                self.agent = lr.ChatAgent(
                    config=lr.ChatAgentConfig(
                        llm=openai_config,
                        system_message="""Analyze sentiment for BTC, ETH, SOL ONLY in video transcripts (any language):
1. Sentiment: +1 (bullish: price up or good news), -1 (bearish: price down or bad news), 0 (not mentioned). Lean bullish if growth implied, even if vague, across languages.
2. Impact (I): 1-10 (1=minor mention, 10=major news like ETF/legislation).
3. Specificity (E): 1-10 (1=vague, 10=precise targets/timeframes).
Output ONLY this JSON: {"BTC": {"sign": n, "I": n, "E": n}, "ETH": {...}, "SOL": {...}}
"""
                    )
                )

                test_prompt = "Respond with 'OK' if working."
                response = self.agent.llm_response(test_prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                if "OK" in response_text:
                    self.model_ready = True
                    logger.info("Langroid agent test successful")
            except Exception as e:
                logger.error(f"Error initializing Langroid agent: {e}")
                self.agent = None

    async def analyze_transcript(self, video_id: str, transcript: str) -> Dict[str, Dict]:
        logger.info(f"Processing {video_id}: transcript length={len(transcript)}, sample={transcript[:100]}...")
        max_chars = 400000
        if not transcript or len(transcript.strip()) < 10:
            logger.warning(f"Empty or invalid transcript for {video_id}")
            return {coin: {"sign": 0, "I": 0, "E": 0, "S": 0} for coin in self.fixed_coins}
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + " [TRUNCATED]"
            logger.info(f"Truncated transcript for {video_id} to {max_chars} chars")
        if not self.model_ready:
            logger.warning("Model not ready, returning mock sentiments")
            return {coin: {"sign": 0, "I": 0, "E": 0, "S": 0} for coin in self.fixed_coins}

        prompt = f"{self.agent.config.system_message}\n\nVideo {video_id}:\n{transcript}\n\n"
        eval_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.log_file, "a") as f:
            f.write(f"\nEvaluation at {eval_time} for {video_id}:\nInput:\n{prompt}\n")

        try:
            response = await asyncio.to_thread(self.agent.llm_response, prompt)
            response_text = response.content.strip() if response.content else ""
            logger.info(f"Scout Agent response for {video_id}: {response_text[:100] or 'EMPTY'}...")
            if not response_text:
                logger.error(f"No response from LLM for {video_id}, resetting agent")
                self.reset_agent()
                return {coin: {"sign": 0, "I": 0, "E": 0, "S": 0} for coin in self.fixed_coins}
            json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', response_text, re.DOTALL)
            if not json_match:
                logger.error(f"Invalid JSON for {video_id}: {response_text[:200]}")
                raise json.JSONDecodeError("No valid JSON found", response_text, 0)
            scores = json.loads(json_match.group(0))

            with open(self.log_file, "a") as f:
                f.write(f"Output:\n{response_text}\nParsed JSON:\n{json.dumps(scores)}\n")

            for coin in self.fixed_coins:
                if coin not in scores or not all(k in scores[coin] for k in ["sign", "I", "E"]):
                    scores[coin] = {"sign": 0, "I": 0, "E": 0}
                scores[coin]["S"] = scores[coin]["sign"] * (0.5 * scores[coin]["I"] + 0.5 * scores[coin]["E"])
            self.sentiment_cache[video_id] = scores
            return scores
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {video_id}: {e}")
            self.reset_agent()
            return {coin: {"sign": 0, "I": 0, "E": 0, "S": 0} for coin in self.fixed_coins}

    def reset_agent(self):
        logger.info("Resetting SentimentService agent")
        try:
            from langroid import ChatAgent, ChatAgentConfig
            self.agent = ChatAgent(config=self.agent.config)
            test_response = self.agent.llm_response("Respond with 'OK' if working.")
            if "OK" in test_response.content:
                self.model_ready = True
                logger.info("Agent reset successful")
            else:
                self.model_ready = False
                logger.error("Agent reset failed")
        except Exception as e:
            logger.error(f"Failed to reset agent: {e}")
            self.model_ready = False    

    async def calculate_global_scores(self, video_scores: List[Dict[str, Dict]], videos: List[Tuple]) -> Dict[str, float]:
        global_scores = {coin: 0.0 for coin in self.fixed_coins}
        weighted_scores = {coin: 0.0 for coin in self.fixed_coins}
        total_weights = {coin: 0.0 for coin in self.fixed_coins}

        for video, scores in zip(videos, video_scores):
            video_id = video[0]
            vph = video[3]
            redis_key = f"video:{video_id}"
            hours_since = float(safe_redis_op(self.redis_client, "hget", redis_key, "hours_since") or 1.0)
            if hours_since < 0.1:
                logger.warning(f"Invalid hours_since for {video_id}: {hours_since}, resetting to 1.0")
                hours_since = 1.0
            age_days = hours_since / 24.0
            age_factor = -(math.exp(age_days) / 8) + (9 / 8)
            logger.info(f"Video {video_id}: VPH={vph:.2f}, hours_since={hours_since:.2f}, age_days={age_days:.2f}, age_factor={age_factor:.2f}")
            for coin in self.fixed_coins:
                s = scores[coin]["S"]
                weight = vph * 0.6 * age_factor
                weighted_scores[coin] += s * weight
                total_weights[coin] += weight
                logger.debug(f"  {coin}: S={s:.2f}, weight={weight:.2f}, weighted_score_contrib={s * weight:.2f}, total_weight_so_far={total_weights[coin]:.2f}")

        for coin in self.fixed_coins:
            if total_weights[coin] > 0:
                global_scores[coin] = weighted_scores[coin] / total_weights[coin]
                global_scores[coin] = min(max(global_scores[coin], -10), 10)
            logger.info(f"Global score computation for {coin}: weighted_score={weighted_scores[coin]:.2f}, total_weight={total_weights[coin]:.2f}, final_score={global_scores[coin]:.2f}")
        return global_scores