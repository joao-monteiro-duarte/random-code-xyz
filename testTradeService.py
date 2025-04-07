import unittest
from unittest.mock import patch, MagicMock
import asyncio
from datetime import datetime, timedelta
import json
from trade_service import TradeService  # Assume this is the file name

class TestTradeService(unittest.TestCase):
    def setUp(self):
        self.service = TradeService(api_key="mock_key")
        self.service.model_ready = True
        self.loop = asyncio.get_event_loop()
        self.market_data = {
            "btc": {"price": 50000, "volume": 1000, "market_cap": 1000000},
            "eth": {"price": 2000, "volume": 2000, "market_cap": 500000},
            "sol": {"price": 100, "volume": 3000, "market_cap": 100000}
        }
        self.video_scores = [
            {"BTC": {"sign": 1, "I": 8, "E": 6}, "ETH": {"sign": 0, "I": 5, "E": 5}, "SOL": {"sign": -1, "I": 7, "E": 4}},
            {"BTC": {"sign": 1, "I": 6, "E": 7}, "ETH": {"sign": 1, "I": 4, "E": 6}, "SOL": {"sign": 0, "I": 5, "E": 5}}
        ]
        self.videos_with_transcripts = {
            "vid1": {"vph": 50, "hours_since": "24"},
            "vid2": {"vph": 30, "hours_since": "12"}
        }

    def run_async(self, coro):
        return self.loop.run_until_complete(coro)

    @patch("trade_service.TradeService.llm_response")
    def test_initial_buy(self, mock_llm):
        mock_llm.side_effect = [
            MagicMock(content=json.dumps({
                "BTC": {"decision": "BUY", "confidence": 3, "reasoning": "Bullish sentiment"},
                "ETH": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"},
                "SOL": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"}
            })),
            MagicMock(content=json.dumps({
                "coin": "BTC",
                "decision": "BUY",
                "confidence": 3,
                "reasoning": "Confirmed bullish trend"
            }))
        ]
        decisions = self.run_async(self.service.make_decisions(self.video_scores, self.videos_with_transcripts, self.market_data))
        self.service.execute_trade(decisions, self.market_data)
        self.assertEqual(self.service.portfolio["coins"]["BTC"], 0.04)  # $2000 / 50000 = 0.04 BTC
        self.assertEqual(self.service.portfolio["USD"], 8000)
        self.assertEqual(decisions["BTC"]["usd_allocation"], 2000)
        self.assertEqual(decisions["BTC"]["percentage"], 0.2)

    @patch("trade_service.TradeService.llm_response")
    def test_sell_confidence_1(self, mock_llm):
        self.service.portfolio = {"USD": 5000, "coins": {"BTC": 0.1}}  # $5000 in BTC
        self.service.trade_history.append({"coin": "BTC", "action": "BUY", "usd_amount": 5000, "confidence": 5, "timestamp": datetime.now() - timedelta(days=2)})
        mock_llm.side_effect = [
            MagicMock(content=json.dumps({
                "BTC": {"decision": "SELL", "confidence": 1, "reasoning": "Bearish signal"},
                "ETH": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"},
                "SOL": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"}
            })),
            MagicMock(content=json.dumps({
                "coin": "BTC",
                "decision": "SELL",
                "confidence": 1,
                "reasoning": "Confirmed bearish"
            }))
        ]
        decisions = self.run_async(self.service.make_decisions(self.video_scores, self.videos_with_transcripts, self.market_data))
        self.service.execute_trade(decisions, self.market_data)
        self.assertNotIn("BTC", self.service.portfolio["coins"])
        self.assertEqual(self.service.portfolio["USD"], 10000)

    @patch("trade_service.TradeService.llm_response")
    def test_sell_confidence_3(self, mock_llm):
        self.service.portfolio = {"USD": 5000, "coins": {"BTC": 0.1}}  # $5000 in BTC
        self.service.trade_history.append({"coin": "BTC", "action": "BUY", "usd_amount": 5000, "confidence": 4, "timestamp": datetime.now() - timedelta(days=2)})
        mock_llm.side_effect = [
            MagicMock(content=json.dumps({
                "BTC": {"decision": "SELL", "confidence": 3, "reasoning": "Moderate bearish"},
                "ETH": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"},
                "SOL": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"}
            })),
            MagicMock(content=json.dumps({
                "coin": "BTC",
                "decision": "SELL",
                "confidence": 3,
                "reasoning": "Confirmed moderate bearish"
            }))
        ]
        decisions = self.run_async(self.service.make_decisions(self.video_scores, self.videos_with_transcripts, self.market_data))
        self.service.execute_trade(decisions, self.market_data)
        self.assertNotIn("BTC", self.service.portfolio["coins"])
        self.assertEqual(self.service.portfolio["USD"], 10000)

    @patch("trade_service.TradeService.llm_response")
    def test_rebalance_underperformer(self, mock_llm):
        self.service.portfolio = {"USD": 0, "coins": {"ETH": 1, "SOL": 10}}  # $2000 ETH, $1000 SOL
        self.service.trade_history = [
            {"coin": "ETH", "action": "BUY", "usd_amount": 2000, "confidence": 3, "timestamp": datetime.now() - timedelta(days=2)},
            {"coin": "SOL", "action": "BUY", "usd_amount": 1000, "confidence": 2, "timestamp": datetime.now() - timedelta(days=3)}
        ]
        self.market_data["eth"]["price"] = 1800  # -10% profit, under 0.1 threshold
        mock_llm.side_effect = [
            MagicMock(content=json.dumps({
                "BTC": {"decision": "BUY", "confidence": 5, "reasoning": "Strong bullish"},
                "ETH": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"},
                "SOL": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"}
            })),
            MagicMock(content=json.dumps({
                "coin": "BTC",
                "decision": "BUY",
                "confidence": 5,
                "reasoning": "Confirmed strong bullish"
            }))
        ]
        decisions = self.run_async(self.service.make_decisions(self.video_scores, self.videos_with_transcripts, self.market_data))
        self.service.execute_trade(decisions, self.market_data)
        self.assertEqual(self.service.portfolio["coins"]["BTC"], 0.036)  # $1800 / 50000 = 0.036 BTC
        self.assertNotIn("ETH", self.service.portfolio["coins"])
        self.assertEqual(self.service.portfolio["coins"]["SOL"], 10)
        self.assertEqual(decisions["BTC"]["usd_allocation"], 1800)

    @patch("trade_service.TradeService.llm_response")
    def test_rebalance_performers(self, mock_llm):
        self.service.portfolio = {"USD": 0, "coins": {"ETH": 1, "SOL": 10}}  # $2000 ETH, $1000 SOL
        self.service.trade_history = [
            {"coin": "ETH", "action": "BUY", "usd_amount": 2000, "confidence": 3, "timestamp": datetime.now() - timedelta(days=2)},
            {"coin": "SOL", "action": "BUY", "usd_amount": 1000, "confidence": 2, "timestamp": datetime.now() - timedelta(days=3)}
        ]
        mock_llm.side_effect = [
            MagicMock(content=json.dumps({
                "BTC": {"decision": "BUY", "confidence": 5, "reasoning": "Strong bullish"},
                "ETH": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"},
                "SOL": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"}
            })),
            MagicMock(content=json.dumps({
                "coin": "BTC",
                "decision": "BUY",
                "confidence": 5,
                "reasoning": "Confirmed strong bullish"
            }))
        ]
        decisions = self.run_async(self.service.make_decisions(self.video_scores, self.videos_with_transcripts, self.market_data))
        self.service.execute_trade(decisions, self.market_data)
        self.assertAlmostEqual(self.service.portfolio["coins"]["BTC"], 0.03, places=5)  # $1500 / 50000 = 0.03 BTC
        self.assertAlmostEqual(self.service.portfolio["coins"]["ETH"], 0.625, places=5)  # $1250 / 2000
        self.assertEqual(self.service.portfolio["coins"]["SOL"], 7.5)  # $750 / 100
        self.assertEqual(decisions["BTC"]["usd_allocation"], 1500)

    @patch("trade_service.TradeService.llm_response")
    def test_confirmation_adjust_to_hold(self, mock_llm):
        mock_llm.side_effect = [
            MagicMock(content=json.dumps({
                "BTC": {"decision": "BUY", "confidence": 4, "reasoning": "Bullish"},
                "ETH": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"},
                "SOL": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"}
            })),
            MagicMock(content=json.dumps({
                "coin": "BTC",
                "decision": "HOLD",
                "confidence": 2,
                "reasoning": "Market data unclear"
            }))
        ]
        decisions = self.run_async(self.service.make_decisions(self.video_scores, self.videos_with_transcripts, self.market_data))
        self.service.execute_trade(decisions, self.market_data)
        self.assertNotIn("BTC", self.service.portfolio["coins"])
        self.assertEqual(self.service.portfolio["USD"], 10000)
        self.assertEqual(decisions["BTC"]["decision"], "HOLD")
        self.assertEqual(decisions["BTC"]["confidence"], 2)

    @patch("trade_service.TradeService.llm_response")
    def test_24_hour_postpone(self, mock_llm):
        self.service.trade_history.append({"coin": "BTC", "action": "BUY", "usd_amount": 2000, "confidence": 3, "timestamp": datetime.now() - timedelta(hours=1)})
        self.service.portfolio["coins"]["BTC"] = 0.04
        self.service.portfolio["USD"] = 8000
        mock_llm.side_effect = [
            MagicMock(content=json.dumps({
                "BTC": {"decision": "BUY", "confidence": 4, "reasoning": "Stronger bullish"},
                "ETH": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"},
                "SOL": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"}
            })),
            MagicMock(content=json.dumps({
                "coin": "BTC",
                "decision": "BUY",
                "confidence": 4,
                "reasoning": "Confirmed"
            }))
        ]
        decisions = self.run_async(self.service.make_decisions(self.video_scores, self.videos_with_transcripts, self.market_data))
        self.service.execute_trade(decisions, self.market_data)
        self.assertEqual(self.service.portfolio["coins"]["BTC"], 0.04)  # No change
        self.assertEqual(self.service.portfolio["USD"], 8000)

    @patch("trade_service.TradeService.llm_response")
    def test_no_funds_no_rebalance(self, mock_llm):
        self.service.portfolio = {"USD": 0, "coins": {"ETH": 1}}  # $2000 ETH
        self.service.trade_history.append({"coin": "ETH", "action": "BUY", "usd_amount": 2000, "confidence": 3, "timestamp": datetime.now() - timedelta(hours=1)})
        mock_llm.side_effect = [
            MagicMock(content=json.dumps({
                "BTC": {"decision": "BUY", "confidence": 3, "reasoning": "Bullish"},
                "ETH": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"},
                "SOL": {"decision": "HOLD", "confidence": 1, "reasoning": "Neutral"}
            })),
            MagicMock(content=json.dumps({
                "coin": "BTC",
                "decision": "BUY",
                "confidence": 3,
                "reasoning": "Confirmed"
            }))
        ]
        decisions = self.run_async(self.service.make_decisions(self.video_scores, self.videos_with_transcripts, self.market_data))
        self.service.execute_trade(decisions, self.market_data)
        self.assertNotIn("BTC", self.service.portfolio["coins"])  # No funds, no underperformers
        self.assertEqual(self.service.portfolio["USD"], 0)
        self.assertEqual(decisions["BTC"]["usd_allocation"], 0)

    def tearDown(self):
        self.loop.close()

if __name__ == "__main__":
    unittest.main()