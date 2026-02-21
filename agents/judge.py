"""Judge Agent - AI-Powered Trade Decision

Uses Google Vertex AI (Gemini) to analyze Polymarket odds vs news sentiment
and make a trade decision (BUY/SELL/HOLD) with reasoning and script.

Includes self-improvement system that learns from prediction accuracy.
"""

import os
import json
import logging
from typing import Optional, List, Dict
from dotenv import load_dotenv
from utils.vertex_ai_client import generate_content_with_fallback
from agents.self_improvement import get_improvement_system

# Load environment variables
load_dotenv()

try:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logging.warning("Vertex AI not available. Install google-cloud-aiplatform")

logger = logging.getLogger(__name__)

# Initialize Vertex AI (lazy initialization)
_model = None


def _get_model():
    """Initialize and return Vertex AI model (singleton pattern)."""
    global _model
    
    if not VERTEX_AI_AVAILABLE:
        return None
    
    if _model is None:
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            logger.error("GCP_PROJECT_ID not set in environment")
            return None
        
        try:
            vertexai.init(project=project_id, location="us-central1")
            _model = GenerativeModel("gemini-2.5-flash-lite")
            logger.info("Vertex AI initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            return None
    
    return _model


def _weight_to_trust(weight: float) -> str:
    """Convert weight multiplier to trust level description."""
    if weight >= 1.3:
        return "very high trust"
    elif weight >= 1.1:
        return "high trust"
    elif weight >= 0.9:
        return "normal trust"
    elif weight >= 0.7:
        return "reduced trust"
    else:
        return "low trust"


def decide_trade(
    gambler_data: Dict,
    gossip_data: Dict,
    video_gossip_data: Dict = None,
    ticker: str = None,
    company_name: str = None
) -> Dict[str, str]:
    """
    Analyze inputs from both agents and generate trade decision using Vertex AI.
    
    Args:
        gambler_data: Dictionary from Gambler agent with:
            - explanation: Explanation of Polymarket odds
            - sentiment_score: 1-10 score
            - raw_odds: Probability (0.0-1.0)
            - reasoning: Why this sentiment
        gossip_data: Dictionary from Gossip agent with:
            - summary: 2-paragraph news summary
            - sentiment_score: 1-10 score
            - articles_count: Number of articles
            - reasoning: Why this sentiment
        video_gossip_data: Dictionary from Video Gossip agent with:
            - summary: Summary of video content
            - sentiment_score: 1-10 score
            - videos_analyzed: Number of videos
            - reasoning: Why this sentiment
        ticker: Optional ticker symbol for context
        company_name: Optional company name for context
    
    Returns:
        Dictionary with 'decision', 'explanation', 'confidence', and 'key_factors'
    """
    logger.info("")
    logger.info("[JUDGE] " + "=" * 70)
    logger.info(f"[JUDGE] decide_trade() called")
    logger.info(f"[JUDGE] Parameters: ticker='{ticker}', company_name='{company_name or 'None'}'")
    logger.info("[JUDGE] " + "=" * 70)
    
    logger.info(f"[JUDGE] Step 1: Extracting input data from agents...")
    logger.info(f"[JUDGE] Gambler data keys: {list(gambler_data.keys())}")
    logger.info(f"[JUDGE] Gossip data keys: {list(gossip_data.keys())}")
    logger.info(f"[JUDGE] Video Gossip data: {video_gossip_data is not None}")
    if video_gossip_data:
        logger.info(f"[JUDGE] Video Gossip data keys: {list(video_gossip_data.keys())}")
    
    model = _get_model()
    logger.info(f"[JUDGE] Vertex AI model: {model is not None}")
    
    if model is None:
        logger.warning("[JUDGE] Vertex AI not available, returning default HOLD decision")
        result = {
            "decision": "HOLD",
            "explanation": "AI analysis unavailable. Please check Vertex AI credentials.",
            "confidence": "low",
            "key_factors": ["AI unavailable"]
        }
        logger.info(f"[JUDGE] Returning fallback result: {result}")
        return result
    
    company_display = company_name or ticker or "the company"
    ticker_display = ticker or "N/A"
    
    # Extract data from agents
    logger.info(f"[JUDGE] Extracting Gambler agent data...")
    gambler_explanation = gambler_data.get("explanation", "No analysis available")
    gambler_score = gambler_data.get("sentiment_score", 5)
    gambler_odds = gambler_data.get("raw_odds", None)
    gambler_reasoning = gambler_data.get("reasoning", "")
    
    logger.info(f"[JUDGE]   - Gambler sentiment: {gambler_score}/10")
    logger.info(f"[JUDGE]   - Gambler odds: {gambler_odds}")
    logger.info(f"[JUDGE]   - Gambler explanation: {gambler_explanation[:80]}...")
    
    logger.info(f"[JUDGE] Extracting Gossip agent data...")
    gossip_summary = gossip_data.get("summary", "No news summary available")
    gossip_score = gossip_data.get("sentiment_score", 5)
    gossip_count = gossip_data.get("articles_count", 0)
    gossip_reasoning = gossip_data.get("reasoning", "")
    
    logger.info(f"[JUDGE]   - Gossip sentiment: {gossip_score}/10")
    logger.info(f"[JUDGE]   - Gossip articles: {gossip_count}")
    logger.info(f"[JUDGE]   - Gossip summary: {gossip_summary[:80]}...")
    
    # Extract video gossip data
    if video_gossip_data:
        logger.info(f"[JUDGE] Extracting Video Gossip agent data...")
        video_gossip_summary = video_gossip_data.get("summary", "No video analysis available")
        video_gossip_score = video_gossip_data.get("sentiment_score", 5)
        video_gossip_count = video_gossip_data.get("videos_analyzed", 0)
        video_gossip_reasoning = video_gossip_data.get("reasoning", "")
        
        logger.info(f"[JUDGE]   - Video Gossip sentiment: {video_gossip_score}/10")
        logger.info(f"[JUDGE]   - Video Gossip videos: {video_gossip_count}")
        logger.info(f"[JUDGE]   - Video Gossip summary: {video_gossip_summary[:80]}...")
    else:
        video_gossip_summary = "No video analysis available"
        video_gossip_score = 5
        video_gossip_count = 0
        video_gossip_reasoning = ""
    
    # Get learned agent weights from self-improvement system
    logger.info(f"[JUDGE] Fetching agent weights from self-improvement system...")
    try:
        improvement_system = get_improvement_system()
        agent_weights = improvement_system.get_agent_weights()
        logger.info(f"[JUDGE] Agent weights: {agent_weights}")
        
        # Apply weights to sentiment scores
        weighted_gambler = gambler_score * agent_weights.get("gambler", 1.0)
        weighted_gossip = gossip_score * agent_weights.get("gossip", 1.0)
        weighted_video = video_gossip_score * agent_weights.get("video_gossip", 1.0)
        
        # Calculate weighted average (normalize back to 1-10 scale)
        total_weight = sum(agent_weights.values())
        weighted_avg = (weighted_gambler + weighted_gossip + weighted_video) / total_weight
        
        logger.info(f"[JUDGE] Weighted scores:")
        logger.info(f"[JUDGE]   - Gambler: {gambler_score} × {agent_weights.get('gambler', 1.0):.2f} = {weighted_gambler:.2f}")
        logger.info(f"[JUDGE]   - Gossip: {gossip_score} × {agent_weights.get('gossip', 1.0):.2f} = {weighted_gossip:.2f}")
        logger.info(f"[JUDGE]   - Video: {video_gossip_score} × {agent_weights.get('video_gossip', 1.0):.2f} = {weighted_video:.2f}")
        logger.info(f"[JUDGE]   - Weighted average: {weighted_avg:.2f}/10")
        
    except Exception as e:
        logger.warning(f"[JUDGE] Could not fetch agent weights: {e}. Using equal weights.")
        agent_weights = {"gambler": 1.0, "gossip": 1.0, "video_gossip": 1.0}
        weighted_avg = (gambler_score + gossip_score + video_gossip_score) / 3
    
    # Format odds for display
    odds_display = f"{gambler_odds:.1%}" if gambler_odds is not None else "N/A"
    logger.info(f"[JUDGE] Formatted odds display: {odds_display}")
    
    # Construct prompt with weight information
    prompt = f"""You are a hedge fund manager analyzing {company_display} ({ticker_display}).

AGENT PERFORMANCE WEIGHTS (based on historical accuracy):
- Smart Money Agent: {agent_weights.get('gambler', 1.0):.2f}x weight ({_weight_to_trust(agent_weights.get('gambler', 1.0))})
- News Sentiment Agent: {agent_weights.get('gossip', 1.0):.2f}x weight ({_weight_to_trust(agent_weights.get('gossip', 1.0))})
- Video Sentiment Agent: {agent_weights.get('video_gossip', 1.0):.2f}x weight ({_weight_to_trust(agent_weights.get('video_gossip', 1.0))})

NOTE: Higher weights indicate historically more accurate predictions. Consider this when weighing conflicting signals.

Smart Money (Prediction Markets):
- Analysis: {gambler_explanation}
- Sentiment Score: {gambler_score}/10 (Weighted: {weighted_gambler:.1f}/10)
- Raw Odds: {odds_display}
- Reasoning: {gambler_reasoning}

Public Sentiment (News):
- Summary: {gossip_summary}
- Sentiment Score: {gossip_score}/10 (Weighted: {weighted_gossip:.1f}/10)
- Articles Analyzed: {gossip_count}
- Reasoning: {gossip_reasoning}

Video Sentiment (YouTube Analysis):
- Summary: {video_gossip_summary}
- Sentiment Score: {video_gossip_score}/10 (Weighted: {weighted_video:.1f}/10)
- Videos Analyzed: {video_gossip_count}
- Reasoning: {video_gossip_reasoning}

WEIGHTED AVERAGE SENTIMENT: {weighted_avg:.1f}/10

Task:
1. Compare three sentiment sources:
   - Smart Money (prediction markets): Real money betting on outcomes
   - Public Sentiment (news): Written news and articles
   - Video Sentiment (YouTube): Spoken sentiment from video content
2. Identify patterns and divergence:
   - If all three align → strong confirmation
   - If smart money differs from public/video → potential opportunity
   - If public and video differ → mixed signals
3. Make decision: BUY, SELL, or HOLD
   - BUY: Positive signals, good opportunity
   - SELL: Negative signals, risk concerns
   - HOLD: Mixed/unclear signals, wait for clarity
4. Assess confidence level: high, medium, or low
5. Identify 2-3 key factors driving the decision
6. Provide detailed explanation (3-4 sentences)

Output JSON only (no markdown):
{{
    "decision": "BUY" or "SELL" or "HOLD",
    "explanation": "Detailed explanation (3-4 sentences)",
    "confidence": "high" or "medium" or "low",
    "key_factors": ["factor1", "factor2", "factor3"]
}}"""

    try:
        logger.info(f"[JUDGE] Step 2: Sending data to Vertex AI for decision analysis...")
        logger.info(f"[JUDGE] Prompt length: {len(prompt)} characters")
        logger.info(f"[JUDGE] Prompt preview: {prompt[:200]}...")
        
        # Try SDK first, fallback to REST API
        project_id = os.getenv("GCP_PROJECT_ID")
        logger.info(f"[JUDGE] Using project_id: {project_id}")
        logger.info(f"[JUDGE] Calling generate_content_with_fallback()...")
        
        response_text = generate_content_with_fallback(prompt, model, project_id)
        
        logger.info(f"[JUDGE] AI response received: {response_text is not None}")
        if response_text:
            logger.info(f"[JUDGE] Response length: {len(response_text)} characters")
            logger.info(f"[JUDGE] Response preview: {response_text[:200]}...")
        
        if not response_text:
            logger.error("[JUDGE] Both SDK and REST API failed - no response text")
            raise ValueError("Both SDK and REST API failed")
        
        # Clean response text
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise
        
        # Validate and normalize decision
        decision = result.get("decision", "HOLD").upper()
        if decision not in ["BUY", "SELL", "HOLD"]:
            decision = "HOLD"
        
        # Validate confidence
        confidence = result.get("confidence", "medium").lower()
        if confidence not in ["high", "medium", "low"]:
            confidence = "medium"
        
        result["decision"] = decision
        result["explanation"] = result.get("explanation", "Analysis completed.")
        result["confidence"] = confidence
        result["key_factors"] = result.get("key_factors", [])
        
        logger.info("")
        logger.info("[JUDGE] " + "=" * 70)
        logger.info(f"[JUDGE] ✓✓✓ DECISION ANALYSIS COMPLETE ✓✓✓")
        logger.info(f"[JUDGE] Final Decision: {decision}")
        logger.info(f"[JUDGE] Confidence: {confidence}")
        logger.info(f"[JUDGE] Key factors: {', '.join(result['key_factors'])}")
        logger.info(f"[JUDGE] Explanation: {result['explanation'][:150]}...")
        logger.info("[JUDGE] " + "=" * 70)
        logger.info("")
        
        return result
        
    except Exception as e:
        logger.error("")
        logger.error("[JUDGE] " + "=" * 70)
        logger.error(f"[JUDGE] ✗✗✗ ERROR IN AI ANALYSIS ✗✗✗")
        logger.error(f"[JUDGE] Error type: {type(e).__name__}")
        logger.error(f"[JUDGE] Error message: {str(e)}")
        logger.error(f"[JUDGE] Full traceback:", exc_info=True)
        logger.error("[JUDGE] " + "=" * 70)
        
        result = {
            "decision": "HOLD",
            "explanation": f"Error during AI analysis: {str(e)}. Please try again.",
            "confidence": "low",
            "key_factors": [f"Analysis error: {type(e).__name__}"]
        }
        logger.warning(f"[JUDGE] Returning fallback result: {result}")
        return result