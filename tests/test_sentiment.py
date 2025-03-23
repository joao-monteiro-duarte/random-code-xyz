"""
Simple test script to verify the sentiment service is working properly.
"""

import asyncio
import logging
from services.sentiment_service import SentimentAnalysisService

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_sentiment_service():
    """Test the sentiment service with a sample transcript"""
    service = SentimentAnalysisService()
    
    # Test with a mock video ID and transcript
    video_id = 'testid123'
    transcript = '''
    Bitcoin is looking bullish for 2023 with institutional adoption increasing.
    Ethereum's merge was successful but there are still some scalability concerns.
    Solana has seen amazing developer growth and their ecosystem is thriving.
    
    The small cap gem I'm watching is definitely Pepe coin - it's about to explode 
    with massive gains! I think we could easily see a 10x in the next few weeks.
    '''
    
    print("Analyzing sample transcript...")
    result = await service.analyze_transcript(video_id, transcript)
    print(f"Analysis result: {result}")
    
    # Test global score calculation
    videos = [
        ('testid123', 5000, '2023-01-01T12:00:00Z', 600.0),
        ('testid456', 3000, '2023-01-02T12:00:00Z', 400.0)
    ]
    
    video_sentiments = {
        'testid123': result,
        'testid456': {
            "bitcoin": {
                "score": 4.0,
                "reason": "Some positive news",
                "price_prediction": None,
                "is_small_cap": False,
                "urgency": "low"
            },
            "ethereum": {
                "score": 6.0,
                "reason": "Better than expected",
                "price_prediction": None,
                "is_small_cap": False, 
                "urgency": "medium"
            }
        }
    }
    
    print("Calculating global scores...")
    global_scores = await service.calculate_global_scores(video_sentiments, videos)
    print(f"Global scores: {global_scores}")
    
    # Test the score comparison logic from run_cycle_impl.py
    previous_scores = {
        "bitcoin": {
            "score": 4.0,
            "is_small_cap": False,
            "urgency": "low"
        },
        "ethereum": {
            "score": 2.0,
            "is_small_cap": False,
            "urgency": "low"
        }
    }
    
    print("\nTesting score comparison logic:")
    for coin, new_score_data in global_scores.items():
        old_score_data = previous_scores.get(coin, 0)
        
        # Extract score values based on data type
        new_score_value = new_score_data["score"] if isinstance(new_score_data, dict) else new_score_data
        old_score_value = old_score_data["score"] if isinstance(old_score_data, dict) else old_score_data
        
        print(f"{coin}: old={old_score_value}, new={new_score_value}, diff={abs(new_score_value - old_score_value)}")
        if abs(new_score_value - old_score_value) > 3:
            print(f"  ⚠️ Significant change detected!")

if __name__ == "__main__":
    asyncio.run(test_sentiment_service())