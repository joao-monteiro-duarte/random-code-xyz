#!/usr/bin/env python3
"""
Manual test script for the crypto trading pool API endpoints.
This script sends HTTP requests to the API endpoints and logs the responses.

Usage:
    1. Start the API server in a separate terminal:
       $ ./run.sh --api
    
    2. Run this script to test the API endpoints:
       $ python test_api.py

Note: This script assumes the API server is running on http://localhost:8000.
"""
import os
import sys
import json
import asyncio
import aiohttp
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api_test.log')
    ]
)
logger = logging.getLogger(__name__)

# API base URL (change if needed)
API_BASE_URL = "http://localhost:8000"

async def test_api_endpoint(session, method, endpoint, data=None, params=None):
    """Test an API endpoint and log the response."""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method.lower() == "get":
            async with session.get(url, params=params) as response:
                status = response.status
                response_data = await response.json()
        elif method.lower() == "post":
            async with session.post(url, json=data, params=params) as response:
                status = response.status
                response_data = await response.json()
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            return None
        
        # Log the response
        logger.info(f"{method.upper()} {endpoint} - Status: {status}")
        logger.info(f"Response: {json.dumps(response_data, indent=2)}")
        
        return response_data
    except Exception as e:
        logger.error(f"Error testing {method.upper()} {endpoint}: {e}")
        return None

async def main():
    """Run tests for API endpoints."""
    logger.info("Starting API endpoint tests")
    
    # Test variables
    video_id = "dQw4w9WgXcQ"  # Replace with a real YouTube video ID
    
    async with aiohttp.ClientSession() as session:
        # Test root endpoint
        await test_api_endpoint(session, "get", "/")
        
        # Test status endpoint
        await test_api_endpoint(session, "get", "/status")
        
        # Test run-cycle endpoint
        await test_api_endpoint(session, "post", "/run-cycle", params={"vph_threshold": 500.0})
        
        # Test get transcript endpoint
        await test_api_endpoint(session, "get", f"/transcripts/{video_id}")
        
        # Test process transcript endpoint
        await test_api_endpoint(session, "post", "/transcripts", data={"video_id": video_id})
        
        # Test analyze sentiment endpoint
        test_transcript = (
            "This is a test transcript about cryptocurrency. Bitcoin is looking very promising this week "
            "with strong fundamentals and growing adoption. Ethereum on the other hand is facing challenges "
            "with scaling and high gas fees. Solana has seen remarkable growth recently."
        )
        await test_api_endpoint(session, "post", "/analyze-sentiment", data={"transcript": test_transcript})
        
        # Test get videos endpoint
        await test_api_endpoint(session, "get", "/videos")
        
        logger.info("API endpoint tests completed")

if __name__ == "__main__":
    asyncio.run(main())