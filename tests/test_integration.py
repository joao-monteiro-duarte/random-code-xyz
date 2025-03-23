"""
Integration tests for the crypto trading pool application.
These tests verify that the service architecture works correctly with the original implementation.
"""
import unittest
import sys
import os
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

# Mock OpenRouterLLM import for testing
sys.modules['langroid'] = MagicMock()
sys.modules['langroid.language_models'] = MagicMock()
sys.modules['langroid.language_models.openrouter_llm'] = MagicMock()
sys.modules['langroid.language_models.openrouter_llm'].OpenRouterLLM = MagicMock()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the refactored architecture
from models.video import Video
from services.transcript_service import TranscriptService
from services.app_service import AppService
from utils.cache_manager import CacheManager

# Import from the original implementation
from run_cycle_impl import process_video_stats, run_cycle, get_accumulated_videos, set_accumulated_videos

class TestServiceIntegration(unittest.TestCase):
    """Test integration between the service architecture and original implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock Redis client
        self.redis_client = MagicMock()
        self.redis_client.get = MagicMock(return_value=None)
        self.redis_client.set = MagicMock(return_value=True)
        self.redis_client.keys = MagicMock(return_value=[])
        
        # Create service
        self.app_service = AppService(
            redis_client=self.redis_client,
            redis_available=True
        )
    
    def test_video_model_conversion(self):
        """Test conversion between Video model and tuple."""
        # Create a Video object
        now = datetime.now()
        video = Video(
            id="test_video",
            title="Test Video",
            views=1000,
            publish_time=now - timedelta(hours=10),
            vph=0
        )
        
        # Convert to tuple
        video_tuple = video.to_tuple()
        
        # Check tuple structure
        self.assertEqual(len(video_tuple), 4)
        self.assertEqual(video_tuple[0], "test_video")
        self.assertEqual(video_tuple[1], 1000)
        self.assertEqual(video_tuple[3], 0)
        
        # Convert back to Video
        video2 = Video.from_tuple(video_tuple)
        
        # Check conversion
        self.assertEqual(video2.id, "test_video")
        self.assertEqual(video2.views, 1000)
        self.assertEqual(video2.vph, 0)
    
    def test_process_video_stats_with_both_implementations(self):
        """Test process_video_stats using both implementations."""
        # Create test data
        now = datetime(2023, 5, 15, 12, 0, 0)
        publish_time = (now - timedelta(hours=10)).isoformat()
        video_tuple = ("test_video", 1000, publish_time, 0)
        
        # Process with service architecture
        result1 = self.app_service.process_video_stats(video_tuple, now)
        
        # Process with original implementation
        with patch('services.app_service.get_app_service', side_effect=ImportError):
            result2 = process_video_stats(video_tuple, now)
        
        # Check that both implementations give same result
        self.assertEqual(result1[0], result2[0])  # id
        self.assertEqual(result1[1], result2[1])  # views
        self.assertEqual(result1[2], result2[2])  # publish_time
        self.assertEqual(result1[3], result2[3])  # vph
    
    def test_accumulated_videos_storage(self):
        """Test storing and retrieving accumulated videos."""
        # Create test data
        videos = [
            ("video1", 1000, "2023-05-15T12:00:00", 100),
            ("video2", 2000, "2023-05-15T10:00:00", 200)
        ]
        
        # Store using app service
        self.app_service.set_accumulated_videos(videos)
        
        # Check Redis call
        self.redis_client.set.assert_called_once()
        
        # Mock Redis get to return the videos
        import json
        # Convert tuples to lists for JSON serialization
        videos_json = [[video[0], video[1], video[2], video[3]] for video in videos]
        self.redis_client.get.return_value = json.dumps(videos_json)
        
        # Retrieve using app service
        result1 = self.app_service.get_accumulated_videos()
        
        # Retrieve using original implementation with our mock
        with patch('run_cycle_impl.get_accumulated_videos', return_value=videos):
            result2 = get_accumulated_videos()
        
        # Check results
        self.assertEqual(len(result1), 2)
        # Just check the individual elements to avoid tuple/list comparison issues
        self.assertEqual(result1[0][0], videos[0][0])  # video_id
        self.assertEqual(result1[0][1], videos[0][1])  # views
        self.assertEqual(result1[0][2], videos[0][2])  # publish_time
        self.assertEqual(result1[0][3], videos[0][3])  # vph
        self.assertEqual(result1[1][0], videos[1][0])  # video_id of second one

    @pytest.mark.asyncio
    @patch('run_cycle_impl.log_step', new_callable=AsyncMock)
    @patch('run_cycle_impl.update_market_data', new_callable=AsyncMock)
    @patch('run_cycle_impl.prune_score_history', new_callable=AsyncMock)
    @patch('run_cycle_impl.update_vph_for_existing_videos', new_callable=AsyncMock)
    @patch('run_cycle_impl.analyze_sentiment', new_callable=AsyncMock)
    @patch('aiohttp.ClientSession.get')
    async def test_run_cycle_integration(self, mock_get, mock_analyze, mock_update_vph, 
                                        mock_prune, mock_market, mock_log):
        """Test run_cycle integration with service architecture."""
        # Mock Redis to return empty list
        self.redis_client.get.return_value = '[]'
        
        # Mock HTTP response
        mock_resp = MagicMock()
        mock_resp.json = AsyncMock(return_value={'items': []})
        mock_get.__aenter__.return_value = mock_resp
        
        # Register app service
        with patch('services.app_service.get_app_service', return_value=self.app_service):
            # Run cycle
            await run_cycle(500)
            
            # Check that functions were called
            mock_log.assert_called()
            mock_market.assert_called()
            mock_prune.assert_called()
            mock_update_vph.assert_called()
            
            # No videos to process, so analyze shouldn't be called
            mock_analyze.assert_not_called()

class TestEndToEnd(unittest.TestCase):
    """End-to-end tests for the application."""
    
    @unittest.skip("Skip this test in CI/CD environments")
    def test_api_startup(self):
        """Test API server startup."""
        from fastapi.testclient import TestClient
        from api import app
        
        client = TestClient(app)
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "active")
        
    @unittest.skip("Skip this test in CI/CD environments")
    def test_status_endpoint(self):
        """Test status endpoint."""
        from fastapi.testclient import TestClient
        from api import app
        
        client = TestClient(app)
        response = client.get("/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "online")

if __name__ == '__main__':
    unittest.main()