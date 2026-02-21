"""Broadcaster Agent - Video Generation

Generates video reports using Flora API with mock fallback
for hackathon/demo reliability.
"""

import os
import logging
from typing import Dict
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Style mappings for different decisions
STYLE_MAPPING = {
    "BUY": "Golden Bull, Upward Arrows, Bright Green",
    "SELL": "Red Bear, Storm Clouds, Downward Chart",
    "HOLD": "Neutral Market, Balanced Scales, Blue Tones"
}

FLORA_API_TIMEOUT = 30  # seconds


def generate_video(script: str, decision: str) -> Dict[str, str]:
    """
    Generate video using Flora API.
    
    Args:
        script: 15-second TV news script text
        decision: Trade decision (BUY/SELL/HOLD)
    
    Returns:
        Dictionary with 'video_url' and 'status' keys
    """
    # Get style for decision
    style = STYLE_MAPPING.get(decision.upper(), STYLE_MAPPING["HOLD"])
    
    # Try Flora API key method first
    flora_api_key = os.getenv("FLORA_API_KEY")
    flora_webhook_url = os.getenv("FLORA_WEBHOOK_URL")
    
    # Determine which method to use
    use_api_key = bool(flora_api_key)
    use_webhook = bool(flora_webhook_url)
    
    if use_api_key:
        return _generate_video_api_key(script, style, flora_api_key)
    elif use_webhook:
        return _generate_video_webhook(script, style, flora_webhook_url)
    else:
        logger.warning("No Flora credentials found, using mock video")
        return _generate_mock_video(script, decision)


def _generate_video_api_key(script: str, style: str, api_key: str) -> Dict[str, str]:
    """Generate video using Flora API key method."""
    try:
        # Flora API endpoint (adjust based on actual API documentation)
        url = "https://api.flora.ai/v1/generate"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": script,
            "style": style,
            "duration": 15  # 15 seconds
        }
        
        logger.info("Requesting video generation from Flora API (API key method)")
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=FLORA_API_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            video_url = data.get("video_url") or data.get("url") or data.get("video")
            
            if video_url:
                logger.info(f"Video generated successfully: {video_url}")
                return {
                    "video_url": video_url,
                    "status": "success"
                }
            else:
                logger.warning("Flora API returned success but no video URL")
                return _generate_mock_video(script, "BUY" if "BUY" in style else "SELL")
        else:
            logger.warning(f"Flora API returned status {response.status_code}")
            return _generate_mock_video(script, "BUY" if "BUY" in style else "SELL")
            
    except requests.exceptions.Timeout:
        logger.warning("Flora API request timed out, using mock video")
        return _generate_mock_video(script, "BUY" if "BUY" in style else "SELL")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Flora API request failed: {e}, using mock video")
        return _generate_mock_video(script, "BUY" if "BUY" in style else "SELL")
    except Exception as e:
        logger.error(f"Unexpected error in Flora API call: {e}")
        return _generate_mock_video(script, "BUY" if "BUY" in style else "SELL")


def _generate_video_webhook(script: str, style: str, webhook_url: str) -> Dict[str, str]:
    """Generate video using Flora webhook method."""
    try:
        payload = {
            "prompt": script,
            "style": style
        }
        
        logger.info("Requesting video generation from Flora API (webhook method)")
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=FLORA_API_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            video_url = data.get("video_url") or data.get("url") or data.get("video")
            
            if video_url:
                logger.info(f"Video generated successfully: {video_url}")
                return {
                    "video_url": video_url,
                    "status": "success"
                }
            else:
                logger.warning("Flora webhook returned success but no video URL")
                return _generate_mock_video(script, "BUY" if "BUY" in style else "SELL")
        else:
            logger.warning(f"Flora webhook returned status {response.status_code}")
            return _generate_mock_video(script, "BUY" if "BUY" in style else "SELL")
            
    except requests.exceptions.Timeout:
        logger.warning("Flora webhook request timed out, using mock video")
        return _generate_mock_video(script, "BUY" if "BUY" in style else "SELL")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Flora webhook request failed: {e}, using mock video")
        return _generate_mock_video(script, "BUY" if "BUY" in style else "SELL")
    except Exception as e:
        logger.error(f"Unexpected error in Flora webhook call: {e}")
        return _generate_mock_video(script, "BUY" if "BUY" in style else "SELL")


def _generate_mock_video(script: str, decision: str) -> Dict[str, str]:
    """
    Generate mock video path for demo purposes.
    CRITICAL: This ensures the demo never breaks.
    """
    print(f">> MOCK VIDEO GENERATED: [Path to local asset]")
    print(f">> Script: {script[:50]}...")
    print(f">> Decision: {decision}")
    
    # Return a placeholder URL that can be displayed in the UI
    mock_url = f"https://via.placeholder.com/800x450/667eea/ffffff?text=MarketMinds+Video+({decision})"
    
    logger.info("Using mock video (Flora API unavailable or failed)")
    return {
        "video_url": mock_url,
        "status": "mock"
    }
