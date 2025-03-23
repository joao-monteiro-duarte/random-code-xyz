import unittest
import asyncio
import sys
import os
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Mock OpenRouterLLM import for testing
sys.modules['langroid'] = MagicMock()
sys.modules['langroid.language_models'] = MagicMock()
sys.modules['langroid.language_models.openrouter_llm'] = MagicMock()
sys.modules['langroid.language_models.openrouter_llm'].OpenRouterLLM = MagicMock()

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import run_cycle from the implementation file
from run_cycle_impl import run_cycle

class TestRunCycle(unittest.TestCase):
    """Test the run_cycle function with Redis transcript caching."""
    
    @pytest.mark.asyncio
    @patch('run_cycle_impl.get_accumulated_videos')
    @patch('transcripts.get_transcript_from_redis')
    @patch('fetch_videos.process_video_stats')
    @patch('trades.get_portfolio_state')
    @patch('trades.global_scores', {'bitcoin': 0, 'ethereum': 0, 'solana': 0})
    async def test_run_cycle_with_redis_transcripts(self, mock_portfolio_state, mock_process_video_stats, mock_get_transcript, mock_get_accumulated):
        """Test that run_cycle uses Redis for transcripts and process_video_stats."""
        # Set up mock behavior
        mock_get_accumulated.return_value = [
            ("video1", 1000, "2023-01-01T00:00:00Z", 100),
            ("video2", 2000, "2023-01-02T00:00:00Z", 200)
        ]
        
        # Mock process_video_stats
        mock_process_video_stats.side_effect = lambda video, now: (
            video[0], video[1], video[2], video[1] / 10  # Simple VPH calculation for test
        )
        
        # Mock transcript retrieval
        mock_get_transcript.side_effect = AsyncMock(
            side_effect=lambda vid_id: f"Transcript for {vid_id}" if vid_id in ["video1", "video2"] else None
        )
        
        # Mock other functions that would be called
        with patch('run_cycle_impl.update_market_data', new_callable=AsyncMock), \
             patch('run_cycle_impl.prune_score_history', new_callable=AsyncMock), \
             patch('run_cycle_impl.update_vph_for_existing_videos', new_callable=AsyncMock), \
             patch('run_cycle_impl.analyze_sentiment', new_callable=AsyncMock), \
             patch('run_cycle_impl.get_portfolio_state'), \
             patch('run_cycle_impl.set_accumulated_videos'), \
             patch('run_cycle_impl.log_step', new_callable=AsyncMock):
            
            # Mock portfolio state
            mock_portfolio_state.return_value = {
                'bitcoin_allocation': 100,
                'ethereum_allocation': 100,
                'solana_allocation': 100,
                'allocated_ada': 300,
                'unallocated_ada': 700
            }
                
            # Run the test
            await run_cycle(500)
            
            # Verify that process_video_stats was called for each video
            self.assertEqual(mock_process_video_stats.call_count, 2)
            
            # Verify it was called with correct parameters
            mock_process_video_stats.assert_any_call(("video1", 1000, "2023-01-01T00:00:00Z", 100), unittest.mock.ANY)
            mock_process_video_stats.assert_any_call(("video2", 2000, "2023-01-02T00:00:00Z", 200), unittest.mock.ANY)

if __name__ == '__main__':
    unittest.main()
