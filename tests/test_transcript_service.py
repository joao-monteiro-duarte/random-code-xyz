"""
Tests for the TranscriptService.
"""
import unittest
import sys
import os
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from services import TranscriptService
from models import Video

class TestTranscriptService(unittest.TestCase):
    """Test TranscriptService functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock Redis client
        self.redis_client = MagicMock()
        self.redis_client.get = MagicMock(return_value=None)
        self.redis_client.set = MagicMock(return_value=True)
        self.redis_client.keys = MagicMock(return_value=[])
        
        # Create service
        self.service = TranscriptService(
            redis_client=self.redis_client,
            redis_available=True
        )
    
    def test_get_transcript(self):
        """Test get_transcript method."""
        # Test with empty cache
        self.assertIsNone(self.service.get_transcript("test_video"))
        
        # Add to cache and test again
        self.service.cache_manager.local_cache["test_video"] = "Test transcript"
        self.assertEqual(self.service.get_transcript("test_video"), "Test transcript")
    
    def test_save_transcript(self):
        """Test save_transcript method."""
        # Save a transcript
        result = self.service.save_transcript("test_video", "Test transcript")
        
        # Check result
        self.assertTrue(result)
        
        # Check local cache
        self.assertEqual(self.service.cache_manager.local_cache["test_video"], "Test transcript")
        
        # Check Redis call
        self.redis_client.set.assert_called_once_with("transcript:test_video", "Test transcript")
    
    def test_process_video_stats_with_tuple(self):
        """Test process_video_stats with a tuple."""
        now = datetime(2023, 5, 15, 12, 0, 0)
        publish_time = (now - timedelta(hours=10)).isoformat()
        video = ("test_video", 1000, publish_time, 0)
        
        # Process stats
        result = self.service.process_video_stats(video, now)
        
        # Check result
        self.assertEqual(result[0], "test_video")
        self.assertEqual(result[1], 1000)
        self.assertEqual(result[2], publish_time)
        self.assertEqual(result[3], 100.0)  # 1000 views / 10 hours = 100 VPH
    
    def test_process_video_stats_with_object(self):
        """Test process_video_stats with a Video object."""
        now = datetime(2023, 5, 15, 12, 0, 0)
        publish_time = now - timedelta(hours=10)
        video = Video(
            id="test_video",
            title="Test Video",
            views=1000,
            publish_time=publish_time,
            vph=0
        )
        
        # Process stats
        result = self.service.process_video_stats(video, now)
        
        # Check result
        self.assertEqual(result.id, "test_video")
        self.assertEqual(result.views, 1000)
        self.assertEqual(result.publish_time, publish_time)
        self.assertEqual(result.vph, 100.0)  # 1000 views / 10 hours = 100 VPH
    
    @pytest.mark.asyncio
    @patch.object(TranscriptService, 'fetch_transcript', new_callable=AsyncMock)
    @patch.object(TranscriptService, 'get_transcript')
    @patch.object(TranscriptService, 'save_transcript')
    async def test_process_video_with_cache_miss(self, mock_save, mock_get, mock_fetch):
        """Test process_video with cache miss."""
        # Set up mocks
        mock_get.return_value = None
        mock_fetch.return_value = "New transcript"
        
        # Process video
        video = ("test_video", 1000, "2023-05-15T12:00:00", 100)
        video_id, transcript = await self.service.process_video(video)
        
        # Check result
        self.assertEqual(video_id, "test_video")
        self.assertEqual(transcript, "New transcript")
        
        # Check mock calls
        mock_get.assert_called_once_with("test_video")
        mock_fetch.assert_called_once_with("test_video")
        mock_save.assert_called_once_with("test_video", "New transcript")
    
    @pytest.mark.asyncio
    @patch.object(TranscriptService, 'fetch_transcript', new_callable=AsyncMock)
    @patch.object(TranscriptService, 'get_transcript')
    @patch.object(TranscriptService, 'save_transcript')
    async def test_process_video_with_cache_hit(self, mock_save, mock_get, mock_fetch):
        """Test process_video with cache hit."""
        # Set up mocks
        mock_get.return_value = "Cached transcript"
        
        # Process video
        video = ("test_video", 1000, "2023-05-15T12:00:00", 100)
        video_id, transcript = await self.service.process_video(video)
        
        # Check result
        self.assertEqual(video_id, "test_video")
        self.assertEqual(transcript, "Cached transcript")
        
        # Check mock calls
        mock_get.assert_called_once_with("test_video")
        mock_fetch.assert_not_called()
        mock_save.assert_not_called()
    
    @pytest.mark.asyncio
    @patch.object(TranscriptService, 'process_video', new_callable=AsyncMock)
    async def test_process_videos_parallel(self, mock_process):
        """Test process_videos with parallel execution."""
        # Set up mock
        mock_process.side_effect = lambda v: ("video_" + v[0], "Transcript for " + v[0])
        
        # Process videos
        videos = [("1", 1000, "2023-05-15T12:00:00", 100), ("2", 2000, "2023-05-15T10:00:00", 200)]
        results = await self.service.process_videos(videos, parallel=True)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], ("video_1", "Transcript for 1"))
        self.assertEqual(results[1], ("video_2", "Transcript for 2"))
        
        # Check mock calls
        self.assertEqual(mock_process.call_count, 2)
    
    @pytest.mark.asyncio
    @patch.object(TranscriptService, 'process_video', new_callable=AsyncMock)
    async def test_process_videos_sequential(self, mock_process):
        """Test process_videos with sequential execution."""
        # Set up mock
        mock_process.side_effect = lambda v: ("video_" + v[0], "Transcript for " + v[0])
        
        # Process videos
        videos = [("1", 1000, "2023-05-15T12:00:00", 100), ("2", 2000, "2023-05-15T10:00:00", 200)]
        results = await self.service.process_videos(videos, parallel=False)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], ("video_1", "Transcript for 1"))
        self.assertEqual(results[1], ("video_2", "Transcript for 2"))
        
        # Check mock calls
        self.assertEqual(mock_process.call_count, 2)
        
    def test_export_transcripts(self):
        """Test export_transcripts method."""
        # Set up mock
        original_local_cache = self.service.cache_manager.local_cache
        self.service.cache_manager.local_cache = {"video1": "transcript1", "video2": "transcript2"}
        
        # Mock set_many method
        self.service.cache_manager.set_many = MagicMock(return_value=True)
        
        # Export transcripts
        result = self.service.export_transcripts()
        
        # Check result
        self.assertEqual(result, 2)
        
        # Check mock calls
        self.service.cache_manager.set_many.assert_called_once_with({"video1": "transcript1", "video2": "transcript2"})
        
        # Reset mock
        self.service.cache_manager.local_cache = original_local_cache
        
    @pytest.mark.asyncio
    async def test_transcript_fetching_with_fallbacks(self):
        """Test transcript fetching with fallbacks to audio transcription and simulation."""
        # Replace the implementation with test implementations
        self.service._try_fetch_captions = AsyncMock()
        self.service._try_transcribe_audio = AsyncMock()
        self.service._generate_simulated_transcript = MagicMock()
        
        # Test case 1: Captions available
        self.service._try_fetch_captions.return_value = "Caption transcript"
        transcript = await self.service.fetch_transcript("video1")
        self.assertEqual(transcript, "Caption transcript")
        self.service._try_fetch_captions.assert_called_once_with("video1")
        self.service._try_transcribe_audio.assert_not_called()
        self.service._generate_simulated_transcript.assert_not_called()
        
        # Reset mocks
        self.service._try_fetch_captions.reset_mock()
        self.service._try_transcribe_audio.reset_mock()
        self.service._generate_simulated_transcript.reset_mock()
        
        # Test case 2: Captions unavailable, audio transcription works
        self.service._try_fetch_captions.return_value = None
        self.service._try_transcribe_audio.return_value = "Audio transcript"
        transcript = await self.service.fetch_transcript("video2")
        self.assertEqual(transcript, "Audio transcript")
        self.service._try_fetch_captions.assert_called_once_with("video2")
        self.service._try_transcribe_audio.assert_called_once_with("video2")
        self.service._generate_simulated_transcript.assert_not_called()
        
        # Reset mocks
        self.service._try_fetch_captions.reset_mock()
        self.service._try_transcribe_audio.reset_mock()
        self.service._generate_simulated_transcript.reset_mock()
        
        # Test case 3: Both captions and audio transcription fail
        self.service._try_fetch_captions.return_value = None
        self.service._try_transcribe_audio.return_value = None
        self.service._generate_simulated_transcript.return_value = "Simulated transcript"
        transcript = await self.service.fetch_transcript("video3")
        self.assertEqual(transcript, "Simulated transcript")
        self.service._try_fetch_captions.assert_called_once_with("video3")
        self.service._try_transcribe_audio.assert_called_once_with("video3")
        self.service._generate_simulated_transcript.assert_called_once_with("video3")

if __name__ == '__main__':
    unittest.main()