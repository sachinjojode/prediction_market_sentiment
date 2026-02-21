"""Vertex AI Client Utility

Provides both SDK and REST API methods for calling Gemini models.
Falls back to REST API if SDK fails.
"""

import os
import json
import requests
import logging
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Model name
MODEL_NAME = "gemini-2.5-flash-lite"
REST_API_BASE = "https://aiplatform.googleapis.com/v1"


def generate_content_rest_api(prompt: str, project_id: str = None) -> Optional[str]:
    """
    Generate content using REST API with API key.
    
    Args:
        prompt: The prompt text
        project_id: Optional project ID (for constructing model path)
    
    Returns:
        Generated text or None if error
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("[REST API] GOOGLE_API_KEY not found in environment")
        return None
    
    # Construct model path
    if project_id:
        model_path = f"publishers/google/models/{MODEL_NAME}"
    else:
        model_path = f"publishers/google/models/{MODEL_NAME}"
    
    url = f"{REST_API_BASE}/{model_path}:generateContent"
    
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    params = {
        "key": api_key
    }
    
    try:
        logger.info("[REST API] Calling Gemini via REST API")
        response = requests.post(
            url,
            json=payload,
            params=params,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract text from response
        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if parts and len(parts) > 0 and "text" in parts[0]:
                    text = parts[0]["text"]
                    logger.info("[REST API] Successfully generated content")
                    return text
        
        logger.warning("[REST API] Unexpected response structure")
        return None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[REST API] Request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"[REST API] Unexpected error: {e}")
        return None


def generate_content_with_fallback(prompt: str, sdk_model=None, project_id: str = None) -> Optional[str]:
    """
    Try SDK first, fallback to REST API.
    
    Args:
        prompt: The prompt text
        sdk_model: SDK GenerativeModel instance (optional)
        project_id: Project ID for REST API
    
    Returns:
        Generated text or None if both methods fail
    """
    # Try SDK first
    if sdk_model is not None:
        try:
            logger.info("[AI] Attempting SDK call")
            response = sdk_model.generate_content(prompt)
            
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                response_text = response.candidates[0].content.parts[0].text
            else:
                raise ValueError("Unexpected SDK response format")
            
            logger.info("[AI] SDK call successful")
            return response_text
            
        except Exception as e:
            logger.warning(f"[AI] SDK call failed: {e}, trying REST API fallback")
    
    # Fallback to REST API
    return generate_content_rest_api(prompt, project_id)
