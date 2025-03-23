"""
Test script to verify the Langroid OpenRouter integration for sentiment analysis.
"""

import asyncio
import logging
import os
from services.sentiment_service import SentimentAnalysisService

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_openrouter_integration():
    """Test the OpenRouter integration with Langroid"""
    # Get the API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("OPENROUTER_API_KEY environment variable not set. Skipping test.")
        return
    
    # Create the service with the API key
    service = SentimentAnalysisService(api_key=api_key)
    
    # Test with a simple crypto-related prompt
    transcript = """
    I'm really excited about Bitcoin this year. The price could easily hit $100,000 with all the institutional adoption.
    Ethereum is also looking solid after the merge, but there are still scalability concerns.
    
    The altcoin I'm most bullish on is Solana, with its incredible speed and growing ecosystem.
    
    And don't sleep on Pepe coin! This meme coin is going to absolutely moon in the next few weeks.
    I won't be surprised if we see a 10x from here. You need to get in now before it explodes!
    """
    
    try:
        # Test LLM response if available
        if hasattr(service, 'agent') and service.agent and service.model_ready:
            print("\n=== Testing direct LLM response ===")
            response = service.agent.llm_response("Respond with a short 'OK' if you're working properly.")
            print(f"LLM response: {response}")
            
            # Test analyze_transcript
            print("\n=== Testing analyze_transcript with real model ===")
            result = await service.analyze_transcript('test_openrouter', transcript)
            print(f"Analysis result from OpenRouter: {result}")
        else:
            print("Langroid/OpenRouter integration not available.")
            print("Reason:", "No agent available" if not hasattr(service, 'agent') or not service.agent else "Model not ready")
    except Exception as e:
        print(f"Error testing OpenRouter: {e}")
    
    print("\n=== Testing mock implementation fallback ===")
    # Force using mock by temporarily setting model_ready to False
    if hasattr(service, 'model_ready'):
        old_model_ready = service.model_ready
        service.model_ready = False
    
    # Test mock implementation
    mock_result = await service.analyze_transcript('test_mock', transcript)
    print(f"Mock analysis result: {mock_result}")
    
    # Restore original state
    if hasattr(service, 'model_ready'):
        service.model_ready = old_model_ready

if __name__ == "__main__":
    asyncio.run(test_openrouter_integration())