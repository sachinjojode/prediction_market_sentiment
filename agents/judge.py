"""Judge Agent - AI-Powered Trade Decision

Uses Google Vertex AI (Gemini) to analyze Polymarket odds vs news sentiment
and make a trade decision (BUY/SELL/HOLD) with reasoning and script.
"""

import os
import json
import logging
from typing import Optional, List, Dict
from dotenv import load_dotenv
from utils.vertex_ai_client import generate_content_with_fallback

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


def decide_trade(
    gambler_data: Dict,
    gossip_data: Dict,
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
    
    # Format odds for display
    odds_display = f"{gambler_odds:.1%}" if gambler_odds is not None else "N/A"
    logger.info(f"[JUDGE] Formatted odds display: {odds_display}")
    
    # Construct prompt
    prompt = f"""You are a hedge fund manager analyzing {company_display} ({ticker_display}).

Smart Money (Prediction Markets):
- Analysis: {gambler_explanation}
- Sentiment Score: {gambler_score}/10
- Raw Odds: {odds_display}
- Reasoning: {gambler_reasoning}

Public Sentiment (News):
- Summary: {gossip_summary}
- Sentiment Score: {gossip_score}/10
- Articles Analyzed: {gossip_count}
- Reasoning: {gossip_reasoning}

Task:
1. Compare smart money (prediction markets) vs public sentiment (news)
2. Identify if there's divergence (opportunity) or alignment (confirmation)
   - Divergence: Smart money and public disagree → potential opportunity
   - Alignment: Both agree → confirmation of trend
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