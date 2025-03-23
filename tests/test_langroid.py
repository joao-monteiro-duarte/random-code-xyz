"""
Test script to verify the Langroid integration for sentiment analysis.
"""

import asyncio
import logging
from services.sentiment_service import SentimentAnalysisService

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_sentiment_service():
    """Test the sentiment service with a sample transcript"""
    # Create the service
    service = SentimentAnalysisService()
    
    # Test with a mock video ID and transcript
    video_id = 'test123'
    transcript = '''
    Bitcoin is looking bullish for 2023 with institutional adoption increasing.
    Ethereum's merge was successful but there are still some scalability concerns.
    Solana has seen amazing developer growth and their ecosystem is thriving.
    
    The small cap gem I'm watching is definitely Pepe coin - it's about to explode 
    with massive gains! I think we could easily see a 10x in the next few weeks.
    '''
    
    # Testing analyze_transcript
    print("\n=== Testing analyze_transcript ===")
    result = await service.analyze_transcript(video_id, transcript)
    print(f"Analysis result: {result}")
    
    # Testing batch_analyze
    print("\n=== Testing batch_analyze ===")
    batch_result = await service.batch_analyze([
        (video_id, transcript),
        ('test456', 'Ethereum is showing strong fundamentals, while Dogecoin might see a pump soon!')
    ])
    print(f"Batch analysis result: {batch_result}")
    
    # Testing calculate_global_scores
    print("\n=== Testing calculate_global_scores ===")
    videos = [
        ('test123', 5000, '2023-01-01T12:00:00Z', 600.0),
        ('test456', 3000, '2023-01-02T12:00:00Z', 400.0)
    ]
    global_scores = await service.calculate_global_scores(batch_result, videos)
    print(f"Global scores: {global_scores}")
    
    # Testing update_global_scores_incremental
    print("\n=== Testing update_global_scores_incremental ===")
    # Create new data
    new_video_id = 'test789'
    new_transcript = 'Solana ecosystem is growing rapidly. I think Solana is a good investment.'
    new_result = await service.analyze_transcript(new_video_id, new_transcript)
    
    updated_scores, changes = await service.update_global_scores_incremental(
        {new_video_id: new_result},
        [(new_video_id, 2000, '2023-01-03T12:00:00Z', 300.0)],
        global_scores
    )
    
    print(f"Updated scores: {updated_scores}")
    print(f"Sentiment changes: {changes}")
    
    # Test sentinel detection to ensure the dictionary handling works properly
    print("\n=== Testing sentiment score comparison ===")
    for crypto, new_score_data in updated_scores.items():
        old_score_data = global_scores.get(crypto, {"score": 0})
        
        new_score_value = new_score_data["score"] if isinstance(new_score_data, dict) else new_score_data
        old_score_value = old_score_data["score"] if isinstance(old_score_data, dict) else old_score_data
        
        print(f"{crypto}: old_score={old_score_value:.2f}, new_score={new_score_value:.2f}, diff={abs(new_score_value - old_score_value):.2f}")
        if abs(new_score_value - old_score_value) > 3:
            print(f"  ⚠️ SIGNIFICANT SCORE CHANGE detected!")

if __name__ == "__main__":
    asyncio.run(test_sentiment_service())