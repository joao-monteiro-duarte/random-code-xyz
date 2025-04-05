import logging
import os
import importlib.util
import asyncio
import json
import aiohttp
import time
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import redis as real_redis
from config import YOUTUBE_API_KEYS, REDIS_HOST, REDIS_PORT, OPENROUTER_API_KEY
from utils.openrouter_utils import configure_openrouter_env
from decimal import Decimal, ROUND_DOWN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MockRedis:
    def __init__(self, *args, **kwargs): pass
    def ping(self): pass
    def get(self, key): return None
    def set(self, key, value, ex=None): pass
    def hget(self, key, field): return None
    def hset(self, key, field, value): pass

def connect_redis_with_retry(max_retries=10, delay=2):
    pool = real_redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True, max_connections=10)
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
    for _ in range(3):
        try:
            return getattr(redis_client, operation)(*args, **kwargs)
        except real_redis.ConnectionError as e:
            logger.warning(f"Redis {operation} failed: {e}, retrying...")
            time.sleep(1)
    logger.warning(f"Redis {operation} failed after retries, using mock behavior")
    return MockRedis().__getattribute__(operation)(*args, **kwargs)

LANGROID_AVAILABLE = importlib.util.find_spec("langroid") is not None and hasattr(importlib.import_module("langroid.language_models"), "OpenAIGPT")

class TradeService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError("No OpenRouter API key provided. Set OPENROUTER_API_KEY in .env.")
        self.redis_client = connect_redis_with_retry()
        self.portfolio = {"USD": Decimal('10000.00'), "coins": {}}
        self.trade_history = []
        self.agent = None
        self.model_ready = False
        self.fixed_coins = ["BTC", "ETH", "SOL", "ADA", "DOT", "LINK"]
        self.run_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("Master Logs", exist_ok=True)
        self.log_file = f"Master Logs/master_run_{self.run_start}.txt"
        with open(self.log_file, "a") as f:
            f.write(f"Run started: {self.run_start}\n")
        self.logger = logging.getLogger('trade_service')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        # Define level caps and max coins as instance attributes
        self.max_coins_per_level = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
        self.level_max_p = {1: Decimal('0.10'), 2: Decimal('0.20'), 3: Decimal('0.30'), 4: Decimal('0.40'), 5: Decimal('0.50')}

    def some_method(self):
        self.logger.info("This log should now appear in the terminal")
        if LANGROID_AVAILABLE:
            try:
                import langroid as lr
                from langroid.language_models import OpenAIGPTConfig, OpenAIGPT
                from langroid.utils.configuration import set_global, Settings

                set_global(Settings(cache=False))
                logger.info("Configuring Langroid with OpenRouter for Claude-3-Sonnet")
                configure_openrouter_env(self.api_key, "TradeService", debug=True)

                openai_config = OpenAIGPTConfig(
                    chat_model="anthropic/claude-3-sonnet",
                    api_key=self.api_key,
                    api_base="https://openrouter.ai/api/v1",
                    temperature=0.1,
                    max_output_tokens=1000,
                )

                self.llm = OpenAIGPT(config=openai_config)
                self.agent = lr.ChatAgent(
                    config=lr.ChatAgentConfig(
                        llm=openai_config,
                        system_message=f"""You are a trading decision agent for {', '.join(self.fixed_coins)}.
Based on sentiment scores, VPH, credibility, age, market_data, and portfolio state, decide whether to BUY, SELL, or HOLD each coin for short-term trades (hours to days). Reason holistically, prioritizing:
- Short-term Sentiment (S_short): +1 (bullish), -1 (bearish), 0 (neutral). Primary driver.
- Long-term Sentiment (S_long): +1 (bullish), -1 (bearish), 0 (neutral). Secondary signal for confidence.
- VPH: Higher VPH indicates relevance; use as tiebreaker.
- Credibility (C): Weight reliability (0.6=unknown, 1.0=expert).
- Impact (I): 1-10 (1=minor, 10=major news). Boosts confidence.
- Specificity (E): 1-10 (1=vague, 10=precise). Boosts confidence.
- Age: Recent videos (lower days) take precedence.
- Market Data: Price, volume, market cap inform feasibility.
- Portfolio: USD and coin holdings guide allocation.

Allocation Rules:
- Total portfolio starts at $10,000, aim for 100% allocation.
- Confidence to Percentage: 1=0.02, 2=0.05, 3=0.10, 4=0.20, 5=0.50.
- Level Caps: 1=10% ($1,000), 2=20%, 3=30%, 4=40%, 5=50%.
- BUY: Buy target percentage. If cap exceeded, sell underperformers (daily profit < half the level's percentage) from oldest to newest to fit cap.
- SELL: Sells entire holding; reinvest proceeds into new BUY if specified.
- HOLD: Maintain current position.

Output ONLY this JSON:
{{
  {', '.join(f'"{coin}": {{"decision": "BUY"/"SELL"/"HOLD", "confidence": 1-5, "reasoning": "explanation"}}' for coin in self.fixed_coins)}
}} or "OK" for confirmation."""
                    )
                )
            except Exception as e:
                logger.error(f"Error initializing Langroid agent: {e}")

    async def confirm_trade(self, trade_type, coin, level, amount, price, usd_value, sell_details=None, diff_message=None):
        """
        Asynchronously confirm a trade with the master agent.
        
        Args:
            trade_type (str): "SELL" or "BUY"
            coin (str): The coin being traded
            level (int): Confidence level of the trade
            amount (Decimal): Amount of the coin to trade
            price (Decimal): Price per unit of the coin
            usd_value (Decimal): Total USD value of the trade
            sell_details (list, optional): List of strings detailing coins sold to fund a buy
            diff_message (str, optional): Message about price difference for buys at higher levels
        
        Returns:
            bool: True if confirmed with "OK", False otherwise
        """
        # If we're in a test environment and model_ready is False, auto-confirm
        if not self.model_ready or self.agent is None:
            logger.info(f"Auto-confirming {trade_type} for {coin} (model not ready)")
            return True
            
        if trade_type == "SELL":
            prompt = f"Proposed SELL for {coin}: Selling {amount} at ${price} for ${usd_value} from level {level}. Confirm with 'OK' or adjust."
        elif trade_type == "BUY":
            sell_message = f" Funded by selling: {', '.join(sell_details)}" if sell_details else ""
            diff_message = f" {diff_message}" if diff_message else ""
            prompt = f"Proposed BUY for {coin}: Buying {amount} at ${price} for ${usd_value} at level {level}{diff_message}.{sell_message} Confirm with 'OK' or adjust."
        else:
            return False

        try:
            response = await self.agent.llm_response(prompt)
            response_text = str(response.content).strip()
            if response_text == "OK":
                logger.info(f"Trade for {coin} confirmed")
                return True
            else:
                logger.info(f"Trade for {coin} not confirmed: {response_text}")
                return False
        except Exception as e:
            logger.error(f"Error confirming trade for {coin}: {e}")
            return False
    
    async def check_model_ready(self):
        if self.agent is not None:
            test_prompt = "Respond with a short 'OK' if you're working properly."
            response = await self.agent.llm_response(test_prompt)
            if "OK" in response.content:
                self.model_ready = True
                logger.info("Langroid agent test successful")

    def calculate_p(self, confidence: int, level_5_active: bool = False) -> Decimal:
        """
        Calculate the target percentage for a confidence level.
        
        When level_5_active is True, all non-level 5 percentages are halved.
        
        Args:
            confidence: The confidence level (1-5)
            level_5_active: Whether a level 5 coin is active in the portfolio
            
        Returns:
            The target percentage as a decimal (e.g., 0.02 for 2%)
        """
        if level_5_active:
            # Exactly half the normal percentages when level 5 is active
            p_map = {
                1: Decimal('0.01'),  # 2% halved
                2: Decimal('0.025'), # 5% halved
                3: Decimal('0.05'),  # 10% halved
                4: Decimal('0.10'),  # 20% halved
                5: Decimal('0.50')   # Level 5 remains at 50%
            }
        else:
            p_map = {
                1: Decimal('0.02'),
                2: Decimal('0.05'),
                3: Decimal('0.10'),
                4: Decimal('0.20'),
                5: Decimal('0.50')
            }
        return p_map.get(confidence, Decimal('0.0'))

    def is_level_5_active(self, decisions: Dict[str, Dict]) -> bool:
        for trade in self.trade_history:
            if trade["confidence"] == 5 and trade["action"] == "BUY" and trade["coin"] in self.portfolio["coins"]:
                return True
        for decision in decisions.values():
            if decision["decision"] == "BUY" and decision["confidence"] == 5:
                return True
        return False

    def enforce_level_caps(self, decisions, market_data):
        total_value = self.calculate_portfolio_value(market_data)
        logger.info(f"Enforcing caps with total_value: {total_value}")
        
        level_max_p = {1: Decimal('0.10'), 2: Decimal('0.20'), 3: Decimal('0.30'), 4: Decimal('0.40'), 5: Decimal('0.50')}
        level_5_active = self.is_level_5_active(decisions)
        if level_5_active:
            level_max_p.update({1: Decimal('0.05'), 2: Decimal('0.10'), 3: Decimal('0.15'), 4: Decimal('0.20')})
        
        for level in range(1, 6):
            coins_at_level = [t for t in self.trade_history if t["confidence"] == level and t["coin"] in self.portfolio["coins"]]
            if not coins_at_level:
                continue
            level_value = sum((self.portfolio["coins"][c["coin"]] * Decimal(market_data.get(c["coin"].lower(), {}).get("price", 1))).quantize(Decimal('0.01')) for c in coins_at_level)
            level_cap = (level_max_p[level] * total_value).quantize(Decimal('0.01'))
            tolerance = Decimal('0.01')
            logger.info(f"Level {level}: value={level_value}, cap={level_cap}")
            
            while level_value > level_cap + tolerance:
                previous_level_value = level_value
                excess = level_value - level_cap
                sell_fraction = (excess / level_value).quantize(Decimal('0.000001'))
                logger.info(f"Excess: {excess}, sell_fraction: {sell_fraction}")
                for trade in coins_at_level:
                    coin = trade["coin"]
                    current_amount = self.portfolio["coins"][coin]
                    current_price = Decimal(market_data.get(coin.lower(), {}).get("price", 1))
                    # Increase precision to 8 decimal places
                    sell_amount = (current_amount * sell_fraction).quantize(Decimal('0.00000001'))
                    sell_value = (sell_amount * current_price).quantize(Decimal('0.01'))
                    self.portfolio["coins"][coin] -= sell_amount
                    self.portfolio["USD"] += sell_value
                    logger.info(f"Sold {sell_amount} of {coin} for ${sell_value}")
                    if self.portfolio["coins"][coin] < Decimal('0.000001'):
                        del self.portfolio["coins"][coin]
                        self.trade_history = [t for t in self.trade_history if t["coin"] != coin or t["action"] != "BUY"]
                    
                    # Recalculate after each sale
                    total_value = self.calculate_portfolio_value(market_data)
                    level_value = sum((self.portfolio["coins"].get(c["coin"], Decimal('0')) * Decimal(market_data.get(c["coin"].lower(), {}).get("price", 1))).quantize(Decimal('0.01')) for c in coins_at_level)
                    logger.info(f"After sale: total_value={total_value}, level_value={level_value}, cap={level_cap}")
                    if level_value <= level_cap + tolerance:
                        break
                # Check if reduction is too small to continue
                if previous_level_value - level_value < Decimal('0.01'):
                    logger.warning(f"Stopping cap enforcement for level {level}: reduction too small (current excess: {level_value - level_cap})")
                    break

    async def make_decisions(self, video_scores: List[Dict[str, Dict]], videos_with_transcripts: Dict[str, Dict], market_data: Dict[str, Dict]) -> Dict[str, Dict]:

        if not self.model_ready or not video_scores or not videos_with_transcripts:
            return {coin: {"decision": "HOLD", "confidence": 1, "reasoning": "No data", "usd_allocation": 0, "percentage": 0.0, "target": "N/A", "stop_loss": "N/A"} for coin in self.fixed_coins}

        prompt = f"Given the following data, decide whether to buy, sell, or hold for {', '.join(self.fixed_coins)}:\n\nVideo Data:\n"
        for idx, scores in enumerate(video_scores):
            video_id = list(videos_with_transcripts.keys())[idx]
            vph = videos_with_transcripts[video_id].get("vph", 0)
            channel_id = await self.get_channel_id(video_id)
            c_score = safe_redis_op(self.redis_client, "get", f"channel:{channel_id}:credibility")
            if not c_score:
                c_score = await self.evaluate_channel_credibility(channel_id)
                safe_redis_op(self.redis_client, "set", f"channel:{channel_id}:credibility", c_score, ex=86400)
            hours_since = float(safe_redis_op(self.redis_client, "hget", f"video:{video_id}", "hours_since") or 1.0)
            age_days = hours_since / 24.0
            prompt += f"Video {video_id} (VPH: {vph:.2f}, Age: {age_days:.2f} days, Credibility: {c_score}):\n"
            for coin in self.fixed_coins:
                scores[coin]["C"] = c_score
                s = scores[coin]["sign"] * (0.5 * scores[coin]["I"] + 0.5 * scores[coin]["E"])
                prompt += f"  {coin}: sign={scores[coin]['sign']}, I={scores[coin]['I']}, E={scores[coin]['E']}, C={c_score}, S={s:.2f}\n"

        prompt += "\nMarket Data:\n" + "\n".join(f"{coin}: {data}" for coin, data in market_data.items()) + "\n\n"
        prompt += "Portfolio: " + str(self.portfolio) + "\n\n"

        decision_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.log_file, "a") as f:
            f.write(f"\nDecision at {decision_time}:\nInput:\n{prompt}\n")

        try:
            prompt = "Generate trade decisions based on video scores and market data"  # Example
            response = await self.agent.llm_response(prompt)
            response_text = str(response.content).strip().replace("None", '"N/A"')
            logger.info(f"LLM response: {response_text}")
            logger.info("Master Agent response: %s...", response_text[:100])
            with open(self.log_file, "a") as f:
                f.write(f"Output:\n{response_text}\n")
            initial_decisions = json.loads(response_text) if response_text != "OK" else {}
            decisions = {}  # Populate this based on your actual logic
            final_decisions = {}
            for coin in self.fixed_coins:
                if response_text == "OK" or coin not in initial_decisions or initial_decisions[coin]["decision"] == "HOLD":
                    final_decisions[coin] = {
                        "decision": "HOLD",
                        "confidence": initial_decisions.get(coin, {}).get("confidence", 1),
                        "reasoning": initial_decisions.get(coin, {}).get("reasoning", "Neutral"),
                        "usd_allocation": 0,
                        "percentage": 0.0,
                        "target": "N/A",
                        "stop_loss": "N/A"
                    }
                else:
                    confirmed = await self.confirm_decision(coin, initial_decisions[coin], market_data, video_scores, initial_decisions)
                    if confirmed == "OK":
                        final_decisions[coin] = {
                            "decision": initial_decisions[coin]["decision"],
                            "confidence": initial_decisions[coin]["confidence"],
                            "reasoning": initial_decisions[coin]["reasoning"],
                            "usd_allocation": 0,
                            "percentage": 0.0,
                            "target": "N/A",
                            "stop_loss": "N/A"
                        }
                    elif isinstance(confirmed, dict) and confirmed.get("coin") == coin:
                        final_decisions[coin] = {
                            "decision": confirmed["decision"],
                            "confidence": confirmed["confidence"],
                            "reasoning": confirmed["reasoning"],
                            "usd_allocation": 0,
                            "percentage": 0.0,
                            "target": "N/A",
                            "stop_loss": "N/A"
                        }
                    else:
                        final_decisions[coin] = {
                            "decision": "HOLD",
                            "confidence": 1,
                            "reasoning": "Confirmation failed or adjusted",
                            "usd_allocation": 0,
                            "percentage": 0.0,
                            "target": "N/A",
                            "stop_loss": "N/A"
                        }
            return final_decisions
        except Exception as e:
            logger.error(f"Error in make_decisions: {str(e)}")
            decisions = {
                coin: {"decision": "HOLD", "confidence": 1, "reasoning": f"Error: {str(e)}"}
                for coin in self.fixed_coins
            }
            logger.info(f"Returning default decisions due to error: {decisions}")
            return decisions

    async def confirm_decision(self, coin: str, initial_decision: Dict, market_data: Dict, video_scores: List[Dict], all_decisions: Dict) -> str | Dict:
        if initial_decision["confidence"] == 5 and initial_decision["decision"] == "BUY":
            total_value = self.calculate_portfolio_value(market_data)
            usd_to_use = self.calculate_p(5) * total_value
            sell_details = []
            fraction = Decimal('0.75')  # Adjusted to reflect realistic distribution
            for level in range(1, 5):  # Now includes level 1
                level_value = sum(self.portfolio["coins"][c] * market_data.get(c.lower(), {}).get("price", 1)
                                for t in self.trade_history if t["confidence"] == level and t["coin"] in self.portfolio["coins"])
                sell_usd = min(level_value * fraction, usd_to_use / 4)  # Even split across 4 levels
                for trade in [t for t in self.trade_history if t["confidence"] == level and t["coin"] in self.portfolio["coins"]]:
                    coin_value = self.portfolio["coins"][trade["coin"]] * market_data.get(trade["coin"].lower(), {}).get("price", 1)
                    sell_fraction = sell_usd / level_value if level_value > 0 else 0
                    usd_value = coin_value * sell_fraction
                    sell_details.append(f"Sell ${usd_value:.2f} of {trade['coin']} from level {level}")
            prompt = f"""Proposed BUY for {coin} at level 5 (50% = ${usd_to_use:.2f}).
    This will sell:
    {'; '.join(sell_details)}.
    Portfolio: {self.portfolio}.
    Market Data: {market_data.get(coin.lower(), {})}.
    Confirm with "OK" or adjust with JSON:
    {{"coin": "{coin}", "decision": "BUY"/"HOLD", "confidence": 1-5, "reasoning": "explanation"}}"""
            response = await self.agent.llm_response(prompt)
            response_text = str(response.content).strip()
            return response_text if response_text == "OK" else json.loads(response_text)

        existing_trade = next((t for t in self.trade_history if t["coin"] == coin and t["action"] == "BUY" and t["confidence"] == initial_decision["confidence"]), None)
        if existing_trade and initial_decision["decision"] == "BUY":
            prompt = f"""Proposed trade for {coin}: {initial_decision["decision"]} at confidence {initial_decision["confidence"]}.
    However, {coin} already exists at confidence level {initial_decision["confidence"]} in the portfolio. Buying again at the same level is invalid.
    Please choose: HOLD to keep the existing trade, or specify a different confidence level (1-5) to buy instead.
    Respond with a JSON object: {{"coin": "{coin}", "decision": "BUY"/"HOLD", "confidence": 1-5, "reasoning": "explanation"}}"""
        else:
            prompt = f"""Proposed trade for {coin}: {initial_decision["decision"]} at confidence {initial_decision["confidence"]}.
    Reassess based on current market data: {market_data.get(coin.lower(), {})}
    and video scores: {video_scores}.
    Respond with a JSON object: {{"coin": "{coin}", "decision": "BUY"/"SELL"/"HOLD", "confidence": 1-5, "reasoning": "explanation"}}"""

        try:
            response = await self.agent.llm_response(prompt)
            response_text = str(response.content).strip()
            logger.info("Confirmation response for %s: %s...", coin, response_text[:100])
            with open(self.log_file, "a") as f:
                f.write(f"Confirmation for {coin}:\n{prompt}\nOutput:\n{response_text}\n")
            return response_text if response_text == "OK" else json.loads(response_text)
        except Exception as e:
            logger.error("Error in confirmation for %s: %s", coin, str(e))
            return {"coin": coin, "decision": "HOLD", "confidence": 1, "reasoning": "Error in confirmation"}

    
    async def execute_trade(self, decisions, market_data):
        total_value = self.calculate_portfolio_value(market_data)
        logger.info(f"Initial total portfolio value: {total_value}")
        
        level_5_active = self.is_level_5_active(decisions)
        level_max_p = self.level_max_p.copy()
        if level_5_active:
            for lvl in range(1, 5):
                level_max_p[lvl] /= Decimal('2')
        logger.info(f"Level 5 active: {level_5_active}, adjusted level_max_p: {level_max_p}")
        
        # Process sells
        sells_to_execute = []
        for coin, decision in decisions.items():
            if decision["decision"] == "SELL" and coin in self.portfolio["coins"]:
                amount = self.portfolio["coins"][coin]
                price = Decimal(market_data.get(coin.lower(), {}).get("price", 1))
                usd = (amount * price).quantize(Decimal('0.01'))
                level = next((t["confidence"] for t in self.trade_history if t["coin"] == coin and t["action"] == "BUY"), 0)
                confirmed = await self.confirm_trade("SELL", coin, level, amount, price, usd)
                if confirmed:
                    sells_to_execute.append({"coin": coin, "amount": amount, "price": price, "usd": usd, "level": level})
        
        for sell in sells_to_execute:
            coin = sell["coin"]
            level = sell["level"]
            usd = sell["usd"]
            self.portfolio["USD"] += usd
            if coin in self.portfolio["coins"]:
                del self.portfolio["coins"][coin]
            self.trade_history = [t for t in self.trade_history if t["coin"] != coin or t["action"] != "BUY"]
            if level == 5:
                self.redistribute_level_5_proceeds(coin, usd, market_data, decisions)
            elif level in [1, 2, 3, 4]:
                self.redistribute_level_proceeds(level, coin, usd, market_data, decisions)
        total_value = self.calculate_portfolio_value(market_data)
        logger.info(f"Portfolio value after sells: {total_value}")
        
        # Process buys (use self.max_coins_per_level and level_max_p)
        for level in range(1, 6):
            buy_coins = [coin for coin, d in decisions.items() if d["decision"] == "BUY" and d["confidence"] == level]
            if not buy_coins:
                continue
            if level == 5:
                for coin in buy_coins:
                    self.rebalance_for_level_5(coin, market_data, decisions)
                continue
            
            current_coins = [t["coin"] for t in self.trade_history if t["confidence"] == level and t["coin"] in self.portfolio["coins"]]
            n_current = len(current_coins)
            n_new = sum(1 for c in buy_coins if c not in current_coins)
            n_total = n_current + n_new
            
            if n_total <= self.max_coins_per_level[level]:
                per_coin_percentage = level_max_p[level] / Decimal(self.max_coins_per_level[level])
            else:
                per_coin_percentage = level_max_p[level] / Decimal(n_total)
            
            level_cap_value = (level_max_p[level] * total_value).quantize(Decimal('0.01'))
            current_value = sum((self.portfolio["coins"].get(c, Decimal('0')) * Decimal(market_data.get(c.lower(), {}).get("price", 1))).quantize(Decimal('0.01'))
                            for c in current_coins)
            available_space = max(Decimal('0'), level_cap_value - current_value)
            available_usd = self.portfolio["USD"]
            logger.info(f"Level {level}: buy_coins={buy_coins}, n_total={n_total}, per_coin_percentage={per_coin_percentage}, available_space={available_space}, available_usd={available_usd}")
            
            for coin in buy_coins:
                existing_amount = self.portfolio["coins"].get(coin, Decimal('0'))
                price = Decimal(market_data.get(coin.lower(), {}).get("price", 1))
                existing_value = (existing_amount * price).quantize(Decimal('0.01'))
                target_usd = (per_coin_percentage * total_value).quantize(Decimal('0.01'))
                funding_needed = max(Decimal('0'), target_usd - existing_value)
                logger.info(f"{coin}: target_usd={target_usd}, existing_value={existing_value}, funding_needed={funding_needed}")
                
                if funding_needed <= 0:
                    continue
                
                total_funded = Decimal('0')
                if available_usd > 0 and available_space > 0:
                    usable_usd = min(available_usd, available_space, funding_needed)
                    total_funded += usable_usd
                    self.portfolio["USD"] -= usable_usd
                    available_usd -= usable_usd
                    available_space -= usable_usd
                    logger.info(f"Funded {coin} with ${usable_usd} from USD")
                
                remaining_needed = funding_needed - total_funded
                if remaining_needed > 0 and current_coins:
                    current_value = sum((self.portfolio["coins"].get(c, Decimal('0')) * Decimal(market_data.get(c.lower(), {}).get("price", 1)))
                                    for c in current_coins if c != coin)
                    if current_value > 0:
                        sell_fraction = min(Decimal('1'), remaining_needed / current_value)
                        for existing_coin in current_coins[:]:
                            if existing_coin == coin:
                                continue
                            amount = self.portfolio["coins"][existing_coin]
                            price_c = Decimal(market_data.get(existing_coin.lower(), {}).get("price", 1))
                            sell_amount = (amount * sell_fraction).quantize(Decimal('0.000001'))
                            sell_value = (sell_amount * price_c).quantize(Decimal('0.01'))
                            if await self.confirm_trade("SELL", existing_coin, level, sell_amount, price_c, sell_value):
                                self.portfolio["coins"][existing_coin] -= sell_amount
                                total_funded += sell_value
                                if self.portfolio["coins"][existing_coin] < Decimal('0.000001'):
                                    del self.portfolio["coins"][existing_coin]
                                    self.trade_history = [t for t in self.trade_history if t["coin"] != existing_coin or t["action"] != "BUY"]
                
                remaining_needed = funding_needed - total_funded
                if remaining_needed > 0:
                    for search_level in list(range(level + 1, 6)) + list(range(level - 1, 0, -1)):
                        source_coins = [t["coin"] for t in self.trade_history if t["confidence"] == search_level and t["coin"] in self.portfolio["coins"]]
                        if source_coins:
                            source_value = sum(self.portfolio["coins"][c] * Decimal(market_data.get(c.lower(), {}).get("price", 1)) for c in source_coins)
                            if source_value > 0:
                                sell_fraction = min(Decimal('1'), remaining_needed / source_value)
                                actual_sold_value = Decimal('0')
                                for source_coin in source_coins[:]:
                                    amount = self.portfolio["coins"][source_coin]
                                    price_c = Decimal(market_data.get(source_coin.lower(), {}).get("price", 1))
                                    sell_amount = (amount * sell_fraction).quantize(Decimal('0.000001'))
                                    sell_value = (sell_amount * price_c).quantize(Decimal('0.01'))
                                    if await self.confirm_trade("SELL", source_coin, search_level, sell_amount, price_c, sell_value):
                                        actual_sold_value += sell_value
                                        self.portfolio["coins"][source_coin] -= sell_amount
                                        if self.portfolio["coins"][source_coin] < Decimal('0.000001'):
                                            del self.portfolio["coins"][source_coin]
                                            self.trade_history = [t for t in self.trade_history if t["coin"] != source_coin or t["action"] != "BUY"]
                                used_value = min(actual_sold_value, remaining_needed)
                                total_funded += used_value
                                if actual_sold_value > used_value:
                                    excess = actual_sold_value - used_value
                                    self.portfolio["USD"] += excess
                                remaining_needed -= used_value
                                if remaining_needed <= 0:
                                    break
                
                if total_funded > 0 or existing_value > 0:
                    additional_amount = (total_funded / price).quantize(Decimal('0.000001'))
                    total_amount = existing_amount + additional_amount
                    self.portfolio["coins"][coin] = total_amount
                    self.trade_history.append({
                        "coin": coin,
                        "action": "BUY",
                        "usd_amount": float((total_amount * price).quantize(Decimal('0.01'))),
                        "confidence": level,
                        "timestamp": datetime.now()
                    })
                    logger.info(f"Updated portfolio: {coin} = {total_amount}")
        
        self.enforce_level_caps(decisions, market_data)
        logger.info(f"Final portfolio: {self.portfolio}")
                
    def rebalance_for_level_5(self, coin: str, market_data: Dict[str, Dict], decisions: Dict[str, Dict]) -> None:
        """Rebalance portfolio for a Level 5 purchase, ensuring it gets 50% of total value."""
        # Check if coin already exists
        existing_amount = self.portfolio["coins"].get(coin, Decimal('0'))
        if existing_amount > 0:
            del self.portfolio["coins"][coin]
            self.trade_history = [t for t in self.trade_history if t["coin"] != coin or t["action"] != "BUY"]

        total_value = self.calculate_portfolio_value(market_data)
        level_5_coins = [t["coin"] for t in self.trade_history if t["confidence"] == 5 and t["coin"] in self.portfolio["coins"]]
        n_level_5 = len(level_5_coins) + 1
        target_funded = (Decimal('0.50') * total_value / Decimal(n_level_5)).quantize(Decimal('0.01'))

        total_funded = min(self.portfolio["USD"], target_funded)
        self.portfolio["USD"] -= total_funded
        remaining_needed = target_funded - total_funded
        logger.info(f"Remaining funding needed: ${remaining_needed}")

        if remaining_needed > 0:
            non_level_5_coins = [c for c in self.portfolio["coins"] if c not in level_5_coins]
            total_coin_value = sum(
                self.portfolio["coins"][c] * Decimal(market_data.get(c.lower(), {}).get("price", 1))
                for c in non_level_5_coins
            )
            current_value = total_coin_value  # Define current_value here
            logger.info(f"Current value of non-level 5 coins: ${current_value}")

            if total_coin_value > 0:
                sell_fraction = remaining_needed / total_coin_value
                logger.info(f"Sell fraction for non-level 5 coins: {sell_fraction}")
                actual_sold_value = Decimal('0.0')

                for other_coin in non_level_5_coins[:]:  # Use a copy to avoid modification issues
                    amount = self.portfolio["coins"][other_coin]
                    price = Decimal(market_data.get(other_coin.lower(), {}).get("price", 1))
                    sell_amount = min(amount, (amount * sell_fraction).quantize(Decimal('0.000001')))
                    sell_value = (sell_amount * price).quantize(Decimal('0.01'))
                    self.portfolio["coins"][other_coin] -= sell_amount
                    actual_sold_value += sell_value
                    logger.info(f"Sold {sell_amount} of {other_coin} for ${sell_value}")
                    if self.portfolio["coins"][other_coin] < Decimal('0.000001'):
                        del self.portfolio["coins"][other_coin]
                        self.trade_history = [t for t in self.trade_history if t["coin"] != other_coin or t["action"] != "BUY"]

                used_value = min(actual_sold_value, remaining_needed)
                total_funded += used_value
                if actual_sold_value > used_value:
                    excess = actual_sold_value - used_value
                    self.portfolio["USD"] += excess
                    logger.info(f"Returned ${excess} excess from level 5 sales to USD")

        # Calculate buy amount including any existing amount
        price = Decimal(market_data.get(coin.lower(), {}).get("price", 1))
        if existing_amount > 0:
            existing_value = (existing_amount * price).quantize(Decimal('0.01'))
            total_funded += existing_value

        buy_amount = (total_funded / price).quantize(Decimal('0.000001'))
        self.portfolio["coins"][coin] = buy_amount
        coin_value = (buy_amount * price).quantize(Decimal('0.01'))
        logger.info(f"Final {coin} amount: {buy_amount} worth ${coin_value}")

        # Record the trade
        self.trade_history.append({
            "coin": coin, "action": "BUY", "usd_amount": float(total_funded), "confidence": 5, "timestamp": datetime.now()
        })

        # Enforce level caps
        self.enforce_level_caps(decisions, market_data)

        # Verify final allocation
        total_value = self.calculate_portfolio_value(market_data)
        coin_value = (self.portfolio["coins"][coin] * price).quantize(Decimal('0.01'))
        level_5_percentage = (coin_value / total_value * 100).quantize(Decimal('0.01'))
        logger.info(f"Level 5 rebalance complete. {coin} now at {level_5_percentage}")

    def redistribute_level_proceeds(self, level: int, coin: str, usd_value: float, market_data: Dict[str, Dict], decisions: Dict[str, Dict]) -> None:
        proceeds = Decimal(str(usd_value)).quantize(Decimal('0.01'))
        level_coins = [t["coin"] for t in self.trade_history if t["confidence"] == level and t["coin"] in self.portfolio["coins"] and t["coin"] != coin]
        if not level_coins:
            return  # Proceeds already added to USD in execute_trade

        total_value = self.calculate_portfolio_value(market_data)
        max_coins = self.max_coins_per_level[level]
        level_cap = self.level_max_p[level]
        n_coins = len(level_coins)

        target_per_coin_percentage = level_cap / Decimal(max_coins if n_coins <= max_coins else n_coins)
        target_per_coin = (target_per_coin_percentage * total_value).quantize(Decimal('0.01'))

        total_needed = Decimal('0')
        coin_needs = {}
        for level_coin in level_coins:
            current_amount = self.portfolio["coins"][level_coin]
            current_price = Decimal(market_data.get(level_coin.lower(), {}).get("price", 1))
            current_value = (current_amount * current_price).quantize(Decimal('0.01'))
            needed_value = max(Decimal('0'), target_per_coin - current_value)
            coin_needs[level_coin] = needed_value
            total_needed += needed_value

        if total_needed > 0:
            distribution_factor = min(Decimal('1'), proceeds / total_needed)
            for level_coin, needed_value in coin_needs.items():
                if needed_value > 0:
                    usd_to_add = (needed_value * distribution_factor).quantize(Decimal('0.01'))
                    if usd_to_add > 0:
                        price = Decimal(market_data.get(level_coin.lower(), {}).get("price", 1))
                        buy_amount = (usd_to_add / price).quantize(Decimal('0.000001'))
                        self.portfolio["coins"][level_coin] += buy_amount
                        self.portfolio["USD"] -= usd_to_add  # Add this line
                        proceeds -= usd_to_add
                        logger.info(f"Distributed ${usd_to_add} to {level_coin}, new amount: {self.portfolio['coins'][level_coin]}")
        if proceeds > 0:
            self.portfolio["USD"] += proceeds

    def redistribute_level_5_proceeds(self, coin: str, usd_value: float, market_data: Dict[str, Dict], decisions: Dict[str, Dict]) -> None:
        proceeds = Decimal(str(usd_value)).quantize(Decimal('0.01'))
        total_value = self.calculate_portfolio_value(market_data)
        
        level_5_coins = [t["coin"] for t in self.trade_history if t["confidence"] == 5 and t["coin"] in self.portfolio["coins"]]
        
        if level_5_coins:
            # Distribute to remaining Level 5 coins (not applicable in this test case)
            target_value = (Decimal('0.50') * total_value).quantize(Decimal('0.01'))
            for level_5_coin in level_5_coins:
                current_amount = self.portfolio["coins"][level_5_coin]
                current_price = Decimal(market_data.get(level_5_coin.lower(), {}).get("price", 1))
                current_value = (current_amount * current_price).quantize(Decimal('0.01'))
                needed_value = max(Decimal('0'), target_value - current_value)
                if needed_value > 0 and proceeds > 0:
                    usd_to_add = min(needed_value, proceeds).quantize(Decimal('0.01'))
                    if usd_to_add > 0:
                        buy_amount = (usd_to_add / current_price).quantize(Decimal('0.000001'))
                        self.portfolio["coins"][level_5_coin] += buy_amount
                        self.portfolio["USD"] -= usd_to_add  # Decrease USD balance
                        proceeds -= usd_to_add
                        logger.info(f"Distributed ${usd_to_add} to level 5 coin {level_5_coin}")
        else:
            # Restore normal caps for Levels 1-4
            for level in range(1, 5):
                level_coins = [t["coin"] for t in self.trade_history if t["confidence"] == level and t["coin"] in self.portfolio["coins"]]
                if level_coins:
                    n_coins = len(level_coins)
                    target_per_coin = (self.level_max_p[level] * total_value / Decimal(n_coins)).quantize(Decimal('0.01'))
                    for level_coin in level_coins:
                        current_amount = self.portfolio["coins"][level_coin]
                        current_price = Decimal(market_data.get(level_coin.lower(), {}).get("price", 1))
                        current_value = (current_amount * current_price).quantize(Decimal('0.01'))
                        needed_value = max(Decimal('0'), target_per_coin - current_value)
                        if needed_value > 0 and proceeds > 0:
                            usd_to_add = min(needed_value, proceeds).quantize(Decimal('0.01'))
                            buy_amount = (usd_to_add / current_price).quantize(Decimal('0.000001'))
                            self.portfolio["coins"][level_coin] += buy_amount
                            self.portfolio["USD"] -= usd_to_add  # Decrease USD balance
                            proceeds -= usd_to_add
                            logger.info(f"Distributed ${usd_to_add} to {level_coin} at level {level}")
        
        if proceeds > 0:
            self.portfolio["USD"] += proceeds
            logger.info(f"Stored remaining ${proceeds} in USD")

    # In TradeService
    def calculate_portfolio_value(self, market_data):
        total_value = Decimal(str(self.portfolio["USD"]))
        for coin, amount in self.portfolio["coins"].items():
            price = Decimal(str(market_data.get(coin.lower(), {}).get("price", 0)))
            value = (Decimal(str(amount)) * price).quantize(Decimal('0.01'))
            logger.debug(f"Coin {coin}: Amount={amount}, Price={price}, Value={value}")  # Downgrade to debug
            total_value += value
        logger.info(f"Total Value: {total_value}")
        return total_value
    
if __name__ == "__main__":
    service = TradeService()
    service.some_method()