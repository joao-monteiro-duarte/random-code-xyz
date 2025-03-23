import unittest
import asyncio
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from transcripts module
from transcripts import (
    get_transcript_from_redis,
    save_transcript_to_redis,
    process_single_video
)
from config import REDIS_AVAILABLE, redis_client

class TestTranscripts(unittest.TestCase):
    """Test transcript handling functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Clear transcripts from Redis before each test
        if REDIS_AVAILABLE and redis_client:
            keys = redis_client.keys('transcript:*')
            if keys:
                redis_client.delete(*keys)
    
    def tearDown(self):
        """Clean up after tests."""
        # Clear transcripts from Redis after each test
        if REDIS_AVAILABLE and redis_client:
            keys = redis_client.keys('transcript:*')
            if keys:
                redis_client.delete(*keys)
    
    def test_save_and_get_transcript(self):
        """Test saving and retrieving transcripts from Redis."""
        video_id = "test_video_123"
        transcript = "This is a test transcript for video test_video_123"
        
        # Save transcript
        save_transcript_to_redis(video_id, transcript)
        
        # Retrieve transcript
        retrieved = get_transcript_from_redis(video_id)
        
        # Verify retrieval worked
        self.assertEqual(retrieved, transcript)
        
        # Test with non-existent video ID
        self.assertIsNone(get_transcript_from_redis("nonexistent_video"))
    
    @unittest.skipIf(not REDIS_AVAILABLE, "Redis not available")
    def test_process_single_video_with_redis(self):
        """Test processing a single video with Redis caching."""
        # Create a mock video entry
        video = ("mock_video_id", 1000, "2023-01-01T00:00:00Z", 100)
        
        # Pre-save a transcript to Redis to test cache retrieval
        mock_transcript = "This is a pre-cached transcript for mock_video_id"
        save_transcript_to_redis("mock_video_id", mock_transcript)
        
        # Run the process function - mock the fetch_transcript
        with unittest.mock.patch('transcripts.fetch_transcript', 
                                 return_value=asyncio.Future()) as mock_fetch:
            mock_fetch.return_value.set_result("New transcript")
            
            # Create an event loop and run process_single_video
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the test
                result = loop.run_until_complete(process_single_video(video))
                
                # Verify results
                self.assertEqual(result[0], "mock_video_id")
                self.assertEqual(result[1], mock_transcript)
                
                # Verify fetch_transcript was not called (used cache)
                mock_fetch.assert_not_called()
            finally:
                loop.close()

if __name__ == '__main__':
    unittest.main()
