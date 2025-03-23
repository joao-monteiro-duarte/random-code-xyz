#!/usr/bin/env python3
"""
Test script for the transcript acquisition and sentiment analysis
without using the full service architecture
"""
import asyncio
import logging
import sys
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Import individual services directly to avoid circular dependencies
from services.transcript_service import TranscriptService
from services.sentiment_service import SentimentAnalysisService

async def test_transcripts_and_sentiment():
    """Test transcript acquisition and sentiment analysis"""
    
    logger.info("=== Testing Enhanced Transcript Service and Sentiment Analysis ===")
    
    # Initialize services without Redis for simplicity
    transcript_service = TranscriptService(redis_client=None, redis_available=False)
    sentiment_service = SentimentAnalysisService()
    
    # Test video IDs - these are example cryptocurrency videos
    # We'll use both a popular channel (likely to have captions)
    # and a smaller channel (may need audio transcription)
    test_videos = [
        {"id": "UNt_K4AH8s8", "title": "Crypto News Today", "expected_captions": False},
        {"id": "z0dD9ROl7t8", "title": "Bitcoin Technical Analysis", "expected_captions": False}
    ]
    
    # Process each video
    for video in test_videos:
        video_id = video["id"]
        title = video["title"]
        
        logger.info(f"\nProcessing video: {title} (ID: {video_id})")
        
        # 1. Try to fetch transcript
        logger.info("Fetching transcript...")
        transcript = await transcript_service.fetch_transcript(video_id)
        
        if transcript:
            transcript_length = len(transcript)
            logger.info(f"Transcript acquired ({transcript_length} chars)")
            logger.info(f"Sample: {transcript[:200]}..." if transcript_length > 200 else transcript)
            
            # If transcript is successful and has enough content, analyze sentiment
            if transcript_length > 100:
                # 2. Analyze sentiment
                logger.info("\nAnalyzing sentiment...")
                sentiment_data = await sentiment_service.analyze_transcript(video_id, transcript)
                
                if sentiment_data:
                    logger.info("Sentiment analysis results:")
                    for crypto, data in sentiment_data.items():
                        if isinstance(data, dict):
                            score = data.get("score", 0)
                            is_small_cap = data.get("is_small_cap", False)
                            urgency = data.get("urgency", "low")
                            reasons = data.get("reason", "No reason provided")
                            
                            logger.info(f"- {crypto.upper()}: {score}/10 (small cap: {is_small_cap}, urgency: {urgency})")
                            logger.info(f"  Reason: {reasons}")
                        else:
                            # Handle legacy format
                            logger.info(f"- {crypto.upper()}: {data}/10")
                else:
                    logger.info("No sentiment data returned from analysis")
            else:
                logger.info("Transcript too short for sentiment analysis")
        else:
            logger.info("Failed to acquire transcript")
    
    # Test simulated transcript as fallback
    logger.info("\n=== Testing Simulated Transcript Fallback ===")
    simulated_transcript = transcript_service._generate_simulated_transcript("test_id")
    logger.info(f"Simulated transcript ({len(simulated_transcript)} chars): {simulated_transcript}")
    
    # Analyze the simulated transcript
    logger.info("\nAnalyzing simulated transcript sentiment:")
    simulated_sentiment = await sentiment_service.analyze_transcript("test_id", simulated_transcript)
    
    if simulated_sentiment:
        logger.info("Sentiment results for simulated transcript:")
        for crypto, data in simulated_sentiment.items():
            if isinstance(data, dict):
                score = data.get("score", 0)
                logger.info(f"- {crypto.upper()}: {score}/10")
            else:
                logger.info(f"- {crypto.upper()}: {data}/10")
    
    logger.info("\nTests completed!")

if __name__ == "__main__":
    asyncio.run(test_transcripts_and_sentiment())