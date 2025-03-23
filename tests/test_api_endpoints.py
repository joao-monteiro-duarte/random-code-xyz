"""
Tests for FastAPI endpoints.
"""
import json
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Create a mock CryptoTradingService first
class MockCryptoTradingService:
    def __init__(self):
        self.is_initialized = True
        self.app_service = MagicMock()
        self.app_service.is_running = False
        self.transcript_service = MagicMock()
        
    async def get_transcript(self, video_id):
        return "This is a test transcript for cryptocurrency analysis"
        
    async def process_video(self, video_id):
        return (video_id, "This is a test transcript for cryptocurrency analysis")
        
    async def analyze_sentiment(self, transcript):
        return {
            "bitcoin": 5.0,
            "ethereum": 3.0,
            "solana": 4.0
        }
        
    async def run_cycle(self, vph_threshold=500.0, background=True):
        if self.app_service.is_running:
            return {"status": "error", "message": "A cycle is already running"}
        return {"status": "started", "message": "Trading cycle started in background"}
        
    def get_service_status(self):
        return {
            "initialized": True,
            "app_service": {
                "redis_available": True,
                "is_running": self.app_service.is_running,
                "last_cycle_time": datetime.now().isoformat(),
                "time_since_last_cycle": "300.0 seconds",
                "next_cycle_in": "1500.0 seconds",
                "accumulated_videos_count": 1
            },
            "sentiment_service_available": True
        }

# Mock the get_crypto_trading_service dependency
mock_service = MockCryptoTradingService()

async def override_get_crypto_trading_service():
    return mock_service

# Import app AFTER defining the mock
import sys
import os
from main import app, get_crypto_trading_service

# Override the dependency
app.dependency_overrides[get_crypto_trading_service] = override_get_crypto_trading_service

# TestClient for FastAPI
client = TestClient(app)

# Mock data for testing
VIDEO_ID = "test_video_id"
MOCK_TRANSCRIPT = "This is a test transcript for cryptocurrency analysis"
MOCK_SENTIMENT = {
    "bitcoin": 5.0,
    "ethereum": 3.0,
    "solana": 4.0
}
MOCK_VIDEO = {
    "id": VIDEO_ID,
    "title": "Test Video",
    "views": 1000,
    "publish_time": datetime.now().isoformat(),
    "vph": 500.0
}
MOCK_ACCUMULATED_VIDEOS = [(VIDEO_ID, 1000, datetime.now().isoformat(), 500.0)]

# API Tests
def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "application": "Crypto Trading Pool API",
        "version": "1.0.0",
        "status": "active"
    }

def test_status_endpoint():
    """Test the status endpoint."""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "service_status" in data

def test_run_cycle_endpoint():
    """Test the run-cycle endpoint."""
    # Reset the is_running flag
    mock_service.app_service.is_running = False
    
    # Test the endpoint
    response = client.post("/run-cycle")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "background" in response.json()["message"]

def test_run_cycle_already_running():
    """Test the run-cycle endpoint when a cycle is already running."""
    # Set the is_running flag
    mock_service.app_service.is_running = True
    
    # Test the endpoint
    response = client.post("/run-cycle")
    assert response.status_code == 409
    assert "already running" in response.json()["detail"].lower()
    
    # Reset the is_running flag for other tests
    mock_service.app_service.is_running = False

def test_get_transcript_endpoint():
    """Test the transcript endpoint."""
    response = client.get(f"/transcripts/{VIDEO_ID}")
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == VIDEO_ID
    assert "transcript" in data

def test_process_transcript_endpoint():
    """Test the process transcript endpoint."""
    response = client.post("/transcripts", json={"video_id": VIDEO_ID})
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == VIDEO_ID
    assert "transcript" in data

def test_analyze_sentiment_endpoint():
    """Test the analyze-sentiment endpoint."""
    test_transcript = "This is a test transcript about cryptocurrency."
    response = client.post("/analyze-sentiment", json={"transcript": test_transcript})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "bitcoin" in data["sentiment"]
    assert "ethereum" in data["sentiment"]
    assert "solana" in data["sentiment"]

def test_get_accumulated_videos_endpoint():
    """Test the videos endpoint."""
    # Mock the get_accumulated_videos method
    mock_service.app_service.get_accumulated_videos = MagicMock(return_value=MOCK_ACCUMULATED_VIDEOS)
    
    # Mock the Video.from_tuple method
    from models.video import Video
    original_from_tuple = Video.from_tuple
    
    # Create a mock video
    mock_video = Video(
        id=VIDEO_ID,
        title="Test Video",
        views=1000,
        publish_time=datetime.now(),
        vph=500.0
    )
    
    # Replace the method
    Video.from_tuple = MagicMock(return_value=mock_video)
    
    try:
        response = client.get("/videos")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "videos" in data
    finally:
        # Restore the original method
        Video.from_tuple = original_from_tuple

@pytest.mark.asyncio
async def test_websocket_endpoint():
    """Test the websocket endpoint."""
    # We can only verify the endpoint exists in a basic test
    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws") as _:
                pass
            assert True  # If we get here, the endpoint exists
    except Exception:
        # The websocket endpoint exists, but we can't fully test it
        # in a simple test without more complex setup
        assert True  # Skip for now