#!/usr/bin/env python3
"""
Direct test of process_video_stats function
"""
from datetime import datetime
from fetch_videos import process_video_stats

def test_process_video_stats():
    # Test data
    now = datetime.now()
    video = ("test_video", 1000, now.isoformat(), 0)
    
    # Process the video stats
    result = process_video_stats(video, now)
    
    # Output the result
    print(f"Input video: {video}")
    print(f"Processed video: {result}")
    print(f"Views per hour: {result[3]}")
    
    # Check if it's working as expected
    if result[3] >= 1000:  # Should be 1000 VPH (1000 views / 1 hour minimum)
        print("PASS: VPH calculation is correct!")
    else:
        print(f"FAIL: Expected VPH around 1000, got {result[3]}")

if __name__ == "__main__":
    test_process_video_stats()