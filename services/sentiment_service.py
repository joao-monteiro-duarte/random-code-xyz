"""
Service for analyzing sentiment in cryptocurrency video transcripts using Mixtral via Langroid.
"""
import logging
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime

# Import models
from models.video import Video

logger = logging.getLogger(__name__)

# Try to import Langroid components - wrap in try/except to allow tests to run without Langroid
try:
    import langroid as lr
    import langroid.language_models as lm
    from langroid.language_models.openrouter_llm import OpenRouterLLM
    LANGROID_AVAILABLE = True
except ImportError:
    logger.warning("Langroid not available, sentiment analysis will be mocked")
    LANGROID_AVAILABLE = False

class SentimentAnalysisService:
    """
    Service for analyzing sentiment in video transcripts using Mixtral model via Langroid.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the sentiment analysis service with Langroid.
        
        Args:
            api_key: OpenRouter API key (default: from environment variable)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        
        if not self.api_key:
            logger.warning("No OpenRouter API key provided. Sentiment analysis will be mocked.")
        
        # Initialize state
        self.last_analysis_time = datetime.now()
        self.sentiment_cache = {}  # Cache of video_id -> sentiment scores
        
        # Set up Langroid LLM if available
        if LANGROID_AVAILABLE and self.api_key:
            try:
                # Configure OpenRouter LLM with Mixtral model
                self.llm_config = lm.OpenRouterLLMConfig(
                    model="mistralai/mixtral-8x7b-instruct",
                    api_key=self.api_key,
                    temperature=0.1,  # Lower temperature for more consistent outputs
                    max_tokens=1000,
                )
                self.llm = OpenRouterLLM(self.llm_config)
                
                # Set up ChatAgent for analysis with focus on small-cap coins
                self.agent_config = lr.ChatAgentConfig(
                    llm=self.llm_config,
                    system_message="""You analyze cryptocurrency video transcripts for sentiment with special focus on identifying early signs of pumps and dumps.
                    Analyze the given transcript and identify sentiment towards specific cryptocurrencies with EXTRA ATTENTION to smaller, lesser-known coins.
                    
                    Rate sentiment for each mentioned crypto on a scale of -10 to +10:
                    - -10: Extremely negative (strong warnings about scams, rug pulls)
                    - -5: Moderately negative (criticism, skepticism)
                    - 0: Neutral or balanced coverage
                    - +5: Moderately positive (optimistic outlook, growth potential)
                    - +10: Extremely positive (price predictions, "moon" talk, strong buying signals)
                    
                    IMPORTANT: While you should include major cryptocurrencies like Bitcoin and Ethereum if mentioned, pay SPECIAL ATTENTION to:
                    1. Low-cap altcoins and meme coins that may be subject to pump and dump schemes
                    2. New coins or tokens being heavily promoted
                    3. Any price predictions or claims about imminent price movements
                    4. Emotional language like "going to moon", "100x", "guaranteed gains", etc.
                    
                    For smaller coins, detect euphoric sentiment that could indicate a pump is starting, or panic selling that could indicate a dump."""
                )
                self.agent = lr.ChatAgent(self.agent_config)
                logger.info("Successfully initialized Langroid agent with Mixtral model")
            except Exception as e:
                logger.error(f"Error initializing Langroid agent: {e}")
                self.agent = None
        else:
            self.agent = None
    
    async def analyze_transcript(self, video_id: str, transcript: str) -> Dict[str, float]:
        """
        Analyze sentiment in a video transcript using Mixtral via Langroid.
        
        Args:
            video_id: ID of the video
            transcript: The video transcript text
            
        Returns:
            Dictionary mapping cryptocurrency names to sentiment scores (-10 to +10)
        """
        if not LANGROID_AVAILABLE or not self.agent:
            # Mock implementation with detailed format
            logger.info(f"Mocking sentiment analysis for video {video_id}")
            return {
                "bitcoin": {
                    "score": 5.0,
                    "reason": "Positive outlook on adoption and institutional interest",
                    "price_prediction": "$100,000 by end of year",
                    "is_small_cap": False,
                    "urgency": "low"
                },
                "ethereum": {
                    "score": 3.0,
                    "reason": "Neutral-positive sentiment with some scalability concerns",
                    "price_prediction": None,
                    "is_small_cap": False,
                    "urgency": "low"
                },
                "solana": {
                    "score": 7.0,
                    "reason": "Strong enthusiasm about performance and developer activity",
                    "price_prediction": "$300 target mentioned",
                    "is_small_cap": False,
                    "urgency": "medium"
                },
                "pepe": {
                    "score": 9.0,
                    "reason": "Extreme hype about potential short-term gains",
                    "price_prediction": "10x in coming weeks",
                    "is_small_cap": True,
                    "urgency": "high"
                }
            }
            
        logger.info(f"Analyzing sentiment for video {video_id}")
        
        # Check if transcript is valid
        if not transcript or len(transcript) < 50:
            logger.warning(f"Transcript for video {video_id} is too short for analysis")
            return {}
            
        # Truncate transcript to avoid token limits
        max_length = 6000  # Mixtral can handle longer contexts
        if len(transcript) > max_length:
            logger.info(f"Truncating transcript for video {video_id} from {len(transcript)} to {max_length} chars")
            transcript = transcript[:max_length]
        
        # Prepare the enhanced prompt for sentiment analysis focused on small-cap opportunities
        prompt = f"""Analyze the sentiment towards cryptocurrencies in this transcript, with particular attention to small cap and meme coins:
        ```
        {transcript}
        ```
        
        For each cryptocurrency mentioned:
        1. Rate the sentiment on a scale of -10 to +10
        2. Provide a brief reason for your rating
        3. Indicate if there are any price predictions or pump signals
        4. For smaller coins, note the urgency level (high, medium, low)
        
        Return your analysis in this JSON format ONLY:
        {{
            "bitcoin": {{
                "score": 5,
                "reason": "Positive outlook on adoption and price increases",
                "price_prediction": null,
                "is_small_cap": false,
                "urgency": "low"
            }},
            "solana": {{
                "score": 8,
                "reason": "Strong enthusiasm about ecosystem growth and developer adoption",
                "price_prediction": "$500 by end of year",
                "is_small_cap": false,
                "urgency": "medium"
            }},
            "pepe": {{
                "score": 9,
                "reason": "Extreme hype about potential for 10x gains in short term",
                "price_prediction": "10x in next month",
                "is_small_cap": true,
                "urgency": "high"
            }}
        }}
        
        IMPORTANT GUIDELINES:
        - Include ALL cryptocurrencies that are meaningfully discussed
        - Pay special attention to altcoins and meme coins
        - For major coins (BTC, ETH), use is_small_cap: false
        - For smaller coins, use is_small_cap: true
        - Set urgency to "high" if the video suggests immediate action is needed
        - Extract specific price predictions when mentioned
        - Accurately represent the sentiment in the video, capturing excitement or fear
        """
        
        try:
            # Use Langroid agent to analyze sentiment
            response = await self.agent.llm_response(prompt)
            
            # Extract JSON from the response
            try:
                # Find JSON content in the response
                import re
                json_match = re.search(r'({.*})', response.strip(), re.DOTALL)
                
                if json_match:
                    json_text = json_match.group(1)
                    sentiment_data = json.loads(json_text)
                    
                    # Extract detailed sentiment data including small cap info
                    detailed_scores = {}
                    for crypto, data in sentiment_data.items():
                        detailed_scores[crypto.lower()] = {
                            "score": data.get("score", 0),
                            "reason": data.get("reason", ""),
                            "price_prediction": data.get("price_prediction"),
                            "is_small_cap": data.get("is_small_cap", False),
                            "urgency": data.get("urgency", "low")
                        }
                    
                    # Cache the results
                    self.sentiment_cache[video_id] = detailed_scores
                    return detailed_scores
                else:
                    logger.error(f"No JSON found in response for video {video_id}")
                    logger.debug(f"Raw response: {response}")
                    return {}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse sentiment analysis for video {video_id}: {e}")
                logger.debug(f"Raw response: {response}")
                return {}
        except Exception as e:
            logger.error(f"Error in sentiment analysis for video {video_id}: {e}")
            return {}
    
    async def batch_analyze(self, video_transcripts: List[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
        """
        Analyze sentiment for multiple video transcripts.
        
        Args:
            video_transcripts: List of (video_id, transcript) tuples
            
        Returns:
            Dictionary mapping video_ids to sentiment scores
        """
        results = {}
        
        for video_id, transcript in video_transcripts:
            try:
                # Check cache first
                if video_id in self.sentiment_cache:
                    results[video_id] = self.sentiment_cache[video_id]
                    continue
                
                # Analyze transcript
                sentiment = await self.analyze_transcript(video_id, transcript)
                results[video_id] = sentiment
            except Exception as e:
                logger.error(f"Error analyzing video {video_id}: {e}")
                continue
        
        return results
    
    async def calculate_global_scores(self, video_sentiments: Dict[str, Dict[str, Dict[str, Any]]], 
                                    videos: List[Union[Video, Tuple]], 
                                    vph_threshold: float = 500.0) -> Dict[str, Dict[str, Any]]:
        """
        Calculate global sentiment scores by aggregating individual video sentiments.
        Weight videos by their views per hour with a minimum impact for low-VPH videos.
        Prioritize small-cap coins with high sentiment scores and urgency.
        
        Args:
            video_sentiments: Dictionary mapping video_ids to sentiment data
            videos: List of Video objects or tuples with VPH information
            vph_threshold: Threshold for applying minimum weight to low-VPH videos
            
        Returns:
            Dictionary mapping cryptocurrencies to aggregated sentiment data
        """
        # Initialize accumulators
        weighted_scores = {}
        total_weights = {}
        crypto_metadata = {}  # Store additional metadata like is_small_cap
        
        # Process each video
        for video in videos:
            # Extract video ID and VPH
            if isinstance(video, Video):
                video_id = video.id
                vph = video.vph
            else:
                video_id = video[0]
                vph = video[3]
            
            # Skip if no sentiment data for this video
            if video_id not in video_sentiments:
                continue
                
            # Get sentiment data for this video
            video_sentiment = video_sentiments[video_id]
            
            # Calculate video weight based on VPH (with minimum impact for low-VPH videos)
            weight = max(vph, vph_threshold * 0.05)  # Minimum 5% of threshold weight
            
            # Add weighted scores
            for crypto, data in video_sentiment.items():
                # Extract score and metadata
                score = data.get("score", 0)
                is_small_cap = data.get("is_small_cap", False)
                urgency = data.get("urgency", "low")
                
                # Boost weight for small caps with high urgency
                boost = 1.0
                if is_small_cap:
                    if urgency == "high":
                        boost = 2.0
                    elif urgency == "medium":
                        boost = 1.5
                
                effective_weight = weight * boost
                
                # Initialize if new coin
                if crypto not in weighted_scores:
                    weighted_scores[crypto] = 0
                    total_weights[crypto] = 0
                    crypto_metadata[crypto] = {
                        "is_small_cap": is_small_cap,
                        "videos_mentioned": 0,
                        "reasons": [],
                        "price_predictions": [],
                        "max_urgency": "low"
                    }
                
                # Update metadata
                crypto_metadata[crypto]["videos_mentioned"] += 1
                if data.get("reason") and data.get("reason") not in crypto_metadata[crypto]["reasons"]:
                    crypto_metadata[crypto]["reasons"].append(data.get("reason"))
                if data.get("price_prediction") and data.get("price_prediction") not in crypto_metadata[crypto]["price_predictions"]:
                    crypto_metadata[crypto]["price_predictions"].append(data.get("price_prediction"))
                
                # Track maximum urgency level
                urgency_levels = {"low": 0, "medium": 1, "high": 2}
                current_urgency = crypto_metadata[crypto]["max_urgency"]
                if urgency_levels.get(urgency, 0) > urgency_levels.get(current_urgency, 0):
                    crypto_metadata[crypto]["max_urgency"] = urgency
                
                # Add weighted score
                weighted_scores[crypto] += score * effective_weight
                total_weights[crypto] += effective_weight
        
        # Calculate final scores (weighted average) with metadata
        global_scores = {}
        for crypto, weighted_score in weighted_scores.items():
            if total_weights.get(crypto, 0) > 0:
                avg_score = weighted_score / total_weights[crypto]
                
                # Assemble final data
                global_scores[crypto] = {
                    "score": avg_score,
                    "is_small_cap": crypto_metadata[crypto]["is_small_cap"],
                    "videos_mentioned": crypto_metadata[crypto]["videos_mentioned"],
                    "reasons": crypto_metadata[crypto]["reasons"],
                    "price_predictions": crypto_metadata[crypto]["price_predictions"],
                    "urgency": crypto_metadata[crypto]["max_urgency"]
                }
        
        # Sort by priority score (combination of sentiment score, small cap status, and urgency)
        sorted_scores = {}
        urgency_values = {"low": 0, "medium": 1, "high": 2}
        
        # Calculate priority scores and sort
        crypto_priority = [(crypto, 
                           data["score"] * (2 if data["is_small_cap"] else 1) * (1 + 0.5 * urgency_values[data["urgency"]]))
                          for crypto, data in global_scores.items()]
        crypto_priority.sort(key=lambda x: x[1], reverse=True)
        
        # Rebuild dictionary in priority order
        for crypto, _ in crypto_priority:
            sorted_scores[crypto] = global_scores[crypto]
        
        return sorted_scores
        
    async def update_global_scores_incremental(self, 
                                             new_video_sentiments: Dict[str, Dict[str, Dict[str, Any]]], 
                                             new_videos: List[Union[Video, Tuple]], 
                                             current_global_scores: Dict[str, Dict[str, Any]],
                                             vph_threshold: float = 500.0) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
        """
        Incrementally update global sentiment scores by adding new video data.
        This maintains the existing scores and enhances them with new data,
        allowing for more frequent decision making without waiting for a full cycle.
        
        Args:
            new_video_sentiments: Dictionary mapping video_ids to new sentiment data
            new_videos: List of new Video objects or tuples with VPH information
            current_global_scores: The current global sentiment scores to update
            vph_threshold: Threshold for applying minimum weight to low-VPH videos
            
        Returns:
            Tuple of (
                updated_global_scores: Dictionary mapping cryptocurrencies to updated sentiment data,
                sentiment_changes: Dictionary mapping cryptocurrencies to sentiment change magnitude
            )
        """
        # Copy current global scores to preserve original data
        updated_global_scores = {k: v.copy() for k, v in current_global_scores.items()} if current_global_scores else {}
        
        # Maps to track existing weights and scores for proper weighted averaging
        existing_weights = {}
        for crypto, data in current_global_scores.items():
            # Estimate the weight - using videos_mentioned as a proxy
            videos_count = data.get("videos_mentioned", 1)
            score = data.get("score", 0)
            # Store estimated weight - will be balanced with new data
            existing_weights[crypto] = videos_count * 100  # Rough approximation
        
        # Initialize trackers for new data
        new_weighted_scores = {}
        new_total_weights = {}
        new_crypto_metadata = {}
        sentiment_changes = {}  # Track magnitude of changes
        
        # First, identify newly discovered coins and record them
        newly_discovered_coins = set()
        
        # Process each new video
        for video in new_videos:
            # Extract video ID and VPH
            if isinstance(video, Video):
                video_id = video.id
                vph = video.vph
            else:
                video_id = video[0]
                vph = video[3]
            
            # Skip if no sentiment data for this video
            if video_id not in new_video_sentiments:
                continue
                
            # Get sentiment data for this video
            video_sentiment = new_video_sentiments[video_id]
            
            # Calculate video weight based on VPH with minimum impact
            weight = max(vph, vph_threshold * 0.05)  # Minimum 5% of threshold
            
            # Give new videos a freshness boost for responsiveness
            freshness_boost = 1.2  # 20% boost for new content
            
            # Process each cryptocurrency mentioned in the video
            for crypto, data in video_sentiment.items():
                # Check if this is a newly discovered coin
                if crypto not in current_global_scores:
                    newly_discovered_coins.add(crypto)
                
                # Extract score and metadata
                score = data.get("score", 0)
                is_small_cap = data.get("is_small_cap", False)
                urgency = data.get("urgency", "low")
                
                # Boost weight for small caps with high urgency
                boost = 1.0
                if is_small_cap:
                    if urgency == "high":
                        boost = 2.0  # Double weight for high urgency small caps
                    elif urgency == "medium":
                        boost = 1.5  # 50% boost for medium urgency
                
                # Apply freshness boost
                effective_weight = weight * boost * freshness_boost
                
                # Initialize if first time seeing this crypto in new data
                if crypto not in new_weighted_scores:
                    new_weighted_scores[crypto] = 0
                    new_total_weights[crypto] = 0
                    new_crypto_metadata[crypto] = {
                        "is_small_cap": is_small_cap,
                        "videos_mentioned": 0,
                        "reasons": [],
                        "price_predictions": [],
                        "max_urgency": "low"
                    }
                
                # Update metadata
                new_crypto_metadata[crypto]["videos_mentioned"] += 1
                if data.get("reason") and data.get("reason") not in new_crypto_metadata[crypto]["reasons"]:
                    new_crypto_metadata[crypto]["reasons"].append(data.get("reason"))
                if data.get("price_prediction") and data.get("price_prediction") not in new_crypto_metadata[crypto]["price_predictions"]:
                    new_crypto_metadata[crypto]["price_predictions"].append(data.get("price_prediction"))
                
                # Track maximum urgency level
                urgency_levels = {"low": 0, "medium": 1, "high": 2}
                current_urgency = new_crypto_metadata[crypto]["max_urgency"]
                if urgency_levels.get(urgency, 0) > urgency_levels.get(current_urgency, 0):
                    new_crypto_metadata[crypto]["max_urgency"] = urgency
                
                # Add weighted score
                new_weighted_scores[crypto] += score * effective_weight
                new_total_weights[crypto] += effective_weight
        
        # Merge new data with existing data
        for crypto, weighted_score in new_weighted_scores.items():
            if new_total_weights[crypto] > 0:
                # Calculate score from just the new data
                new_avg_score = weighted_score / new_total_weights[crypto]
                
                # If the crypto already exists in current global scores
                if crypto in updated_global_scores:
                    # Get existing data
                    existing_data = updated_global_scores[crypto]
                    existing_score = existing_data.get("score", 0)
                    existing_videos = existing_data.get("videos_mentioned", 0)
                    
                    # Calculate combined weighted score
                    combined_weight = existing_weights.get(crypto, 0) + new_total_weights[crypto]
                    combined_score = (
                        (existing_score * existing_weights.get(crypto, 0)) + 
                        (new_avg_score * new_total_weights[crypto])
                    ) / combined_weight
                    
                    # Record the magnitude of sentiment change
                    sentiment_changes[crypto] = abs(combined_score - existing_score)
                    
                    # Update the global score
                    updated_global_scores[crypto]["score"] = combined_score
                    updated_global_scores[crypto]["videos_mentioned"] += new_crypto_metadata[crypto]["videos_mentioned"]
                    
                    # Update metadata
                    updated_global_scores[crypto]["reasons"].extend([
                        r for r in new_crypto_metadata[crypto]["reasons"] 
                        if r not in updated_global_scores[crypto]["reasons"]
                    ])
                    updated_global_scores[crypto]["price_predictions"].extend([
                        p for p in new_crypto_metadata[crypto]["price_predictions"] 
                        if p not in updated_global_scores[crypto]["price_predictions"]
                    ])
                    
                    # Update urgency if the new urgency is higher
                    urgency_levels = {"low": 0, "medium": 1, "high": 2}
                    current_urgency = updated_global_scores[crypto]["urgency"]
                    new_urgency = new_crypto_metadata[crypto]["max_urgency"]
                    if urgency_levels.get(new_urgency, 0) > urgency_levels.get(current_urgency, 0):
                        updated_global_scores[crypto]["urgency"] = new_urgency
                        
                # If this is a newly discovered cryptocurrency
                else:
                    # Add as a new entry with extra freshness boost for discoverability
                    if crypto in newly_discovered_coins:
                        new_coin_boost = 1.1  # 10% boost for newly discovered coins
                        new_avg_score *= new_coin_boost
                        
                    # Record a high sentiment change for new coins
                    sentiment_changes[crypto] = abs(new_avg_score)
                        
                    # Add the new coin to global scores
                    updated_global_scores[crypto] = {
                        "score": new_avg_score,
                        "is_small_cap": new_crypto_metadata[crypto]["is_small_cap"],
                        "videos_mentioned": new_crypto_metadata[crypto]["videos_mentioned"],
                        "reasons": new_crypto_metadata[crypto]["reasons"],
                        "price_predictions": new_crypto_metadata[crypto]["price_predictions"],
                        "urgency": new_crypto_metadata[crypto]["max_urgency"],
                        "is_newly_discovered": True
                    }
        
        # Sort by priority score as in the original method
        sorted_scores = {}
        urgency_values = {"low": 0, "medium": 1, "high": 2}
        
        # Calculate priority scores and sort
        crypto_priority = [(crypto, 
                           data["score"] * (2 if data["is_small_cap"] else 1) * (1 + 0.5 * urgency_values[data["urgency"]]))
                          for crypto, data in updated_global_scores.items()]
        crypto_priority.sort(key=lambda x: x[1], reverse=True)
        
        # Rebuild dictionary in priority order
        for crypto, _ in crypto_priority:
            sorted_scores[crypto] = updated_global_scores[crypto]
        
        return sorted_scores, sentiment_changes