import unittest
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the function to test
from fetch_videos import process_video_stats

class TestProcessVideoStats(unittest.TestCase):
    """Test process_video_stats function."""
    
    def test_process_video_stats_calculation(self):
        """Test that process_video_stats correctly calculates VPH."""
        # Create test data
        now = datetime(2023, 5, 15, 12, 0, 0)  # 2023-05-15 12:00:00
        publish_time = (now - timedelta(hours=10)).isoformat()  # 10 hours ago
        views = 1000
        video = ("test_video_id", views, publish_time, 0)  # Initial VPH doesn't matter
        
        # Call the function
        result = process_video_stats(video, now)
        
        # Verify result
        expected_vph = views / 10  # 1000 views / 10 hours = 100 VPH
        self.assertEqual(result[0], "test_video_id")  # Video ID unchanged
        self.assertEqual(result[1], views)  # Views unchanged
        self.assertEqual(result[2], publish_time)  # Publish time unchanged
        self.assertEqual(result[3], expected_vph)  # VPH correctly calculated
    
    def test_process_video_stats_with_string_time(self):
        """Test process_video_stats with string publish time."""
        # Create test data with string publish time
        now = datetime(2023, 5, 15, 12, 0, 0)
        publish_time = "2023-05-15T06:00:00Z"  # ISO format with Z
        views = 1200
        video = ("test_video_id", views, publish_time, 0)
        
        # Call the function
        result = process_video_stats(video, now)
        
        # Calculate expected VPH (views divided by hours since publication)
        # In this case: 1200 views / 6 hours = 200 VPH
        expected_vph = 200
        self.assertAlmostEqual(result[3], expected_vph, delta=1)
    
    def test_process_video_stats_minimum_hours(self):
        """Test that process_video_stats uses a minimum of 1 hour."""
        # Create test data with recent publish time (less than 1 hour ago)
        now = datetime(2023, 5, 15, 12, 0, 0)
        publish_time = (now - timedelta(minutes=30)).isoformat()  # 30 minutes ago
        views = 500
        video = ("test_video_id", views, publish_time, 0)
        
        # Call the function
        result = process_video_stats(video, now)
        
        # Verify minimum 1 hour is used
        expected_vph = views / 1  # 500 views / 1 hour = 500 VPH
        self.assertEqual(result[3], expected_vph)

if __name__ == '__main__':
    unittest.main()