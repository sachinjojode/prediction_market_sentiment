"""Gossip Agent - News Sentiment Analysis

Uses Google News RSS to fetch news articles and Vertex AI to summarize
them and calculate sentiment scores.
"""

import os
import json
import requests
import xml.etree.ElementTree as ET
import logging
from typing import List, Dict
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

GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search"

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
            logger.info("Vertex AI initialized successfully for Gossip agent")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            return None
    
    return _model


def _fetch_news_articles(ticker: str, company_name: str = None) -> List[Dict]:
    """
    Fetch news articles using Google News RSS.
    
    Args:
        ticker: Stock ticker symbol
        company_name: Optional company name for better search
    
    Returns:
        List of article dictionaries with title, description, link
    """
    try:
        # Construct RSS feed URL
        query = f"{company_name or ticker} stock news"
        url = f"{GOOGLE_NEWS_RSS_BASE}?q={query}&hl=en&gl=US&ceid=US:en"
        
        logger.info(f"[GOSSIP] Fetching news articles from RSS: '{query}'")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        # Find all item elements (news articles)
        items = root.findall(".//item")
        logger.info(f"[GOSSIP] Found {len(items)} articles in RSS feed")
        
        articles = []
        for item in items[:10]:  # Get top 10 articles
            title_elem = item.find("title")
            description_elem = item.find("description")
            link_elem = item.find("link")
            
            if title_elem is not None and title_elem.text:
                title = title_elem.text.strip()
                # Clean up common RSS formatting
                title = title.replace("&apos;", "'")
                title = title.replace("&quot;", '"')
                title = title.replace("&amp;", "&")
                title = title.replace("&lt;", "<")
                title = title.replace("&gt;", ">")
                
                description = ""
                if description_elem is not None and description_elem.text:
                    description = description_elem.text.strip()
                    # Clean HTML tags from description
                    import re
                    description = re.sub(r'<[^>]+>', '', description)
                    description = description.replace("&apos;", "'")
                    description = description.replace("&quot;", '"')
                    description = description.replace("&amp;", "&")
                
                link = ""
                if link_elem is not None and link_elem.text:
                    link = link_elem.text.strip()
                
                articles.append({
                    "title": title,
                    "snippet": description,
                    "link": link
                })
                logger.info(f"[GOSSIP] Article {len(articles)}: {title[:60]}...")
        
        logger.info(f"[GOSSIP] Retrieved {len(articles)} articles")
        return articles
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[GOSSIP] Error fetching RSS feed: {e}")
        return []
    except ET.ParseError as e:
        logger.error(f"[GOSSIP] Error parsing RSS XML: {e}")
        return []
    except Exception as e:
        logger.error(f"[GOSSIP] Unexpected error fetching articles: {e}")
        return []


def get_news_sentiment(ticker: str, company_name: str = None) -> Dict:
    """
    Fetch news articles and use Vertex AI to summarize and calculate sentiment.
    
    Args:
        ticker: Stock ticker symbol (e.g., "NVDA", "AAPL")
        company_name: Optional company name for context
    
    Returns:
        Dictionary with:
        - summary: 2-paragraph summary of articles
        - sentiment_score: 1-10 score (10 = most positive)
        - articles_count: Number of articles analyzed
        - reasoning: Why this sentiment score
    """
    logger.info("")
    logger.info("[GOSSIP] " + "=" * 70)
    logger.info(f"[GOSSIP] get_news_sentiment() called")
    logger.info(f"[GOSSIP] Parameters: ticker='{ticker}', company_name='{company_name or 'None'}'")
    logger.info("[GOSSIP] " + "=" * 70)
    
    # Fetch news articles
    logger.info(f"[GOSSIP] Step 1: Fetching news articles from Google News RSS...")
    articles = _fetch_news_articles(ticker, company_name)
    logger.info(f"[GOSSIP] _fetch_news_articles() returned {len(articles)} articles")
    
    if not articles:
        logger.warning(f"[GOSSIP] No news articles found for {ticker}")
        logger.warning(f"[GOSSIP] Returning fallback result with neutral sentiment")
        result = {
            "summary": f"No recent news articles found for {ticker}.",
            "sentiment_score": 5,
            "articles_count": 0,
            "reasoning": "No articles available to analyze.",
            "sources": []
        }
        logger.info(f"[GOSSIP] Returning result: {result}")
        return result
    
    logger.info(f"[GOSSIP] Step 2: Processing {len(articles)} articles for AI analysis...")
    logger.info(f"[GOSSIP] Article titles:")
    for i, art in enumerate(articles[:5], 1):
        logger.info(f"[GOSSIP]   {i}. {art.get('title', 'N/A')[:70]}...")
    
    # Use Vertex AI to summarize and analyze
    logger.info(f"[GOSSIP] Step 3: Initializing Vertex AI model...")
    model = _get_model()
    logger.info(f"[GOSSIP] Vertex AI model: {model is not None}")
    
    if model is None:
        logger.warning("[GOSSIP] Vertex AI not available, returning basic summary")
        # Basic fallback: just list titles
        titles = "\n".join([f"- {art['title']}" for art in articles[:5]])
        sources = []
        for article in articles[:10]:
            sources.append({
                "title": article.get("title", "Unknown"),
                "url": article.get("link", "")
            })
        
        result = {
            "summary": f"Found {len(articles)} recent articles about {company_name or ticker}:\n{titles}",
            "sentiment_score": 5,
            "articles_count": len(articles),
            "reasoning": "Basic summary - AI analysis unavailable.",
            "sources": sources
        }
        logger.info(f"[GOSSIP] Returning basic result (no AI)")
        return result
    
    # Prepare articles list for AI
    company_display = company_name or ticker
    articles_text = ""
    for i, article in enumerate(articles, 1):
        articles_text += f"""
Article {i}:
Title: {article['title']}
Summary: {article['snippet']}
URL: {article['link']}
---
"""
    
    # Construct AI prompt
    prompt = f"""You are analyzing news articles about {company_display} ({ticker}).

Recent News Articles ({len(articles)} articles):
{articles_text}

Task:
1. Summarize the key points from all articles in 2 coherent paragraphs
   - First paragraph: Main news and developments
   - Second paragraph: Market implications and trends
2. Analyze overall consumer sentiment from the news
3. Rate sentiment from 1-10 where:
   - 1-3 = Very negative (bad news, concerns, problems)
   - 4-6 = Neutral/mixed (balanced coverage)
   - 7-10 = Very positive (good news, growth, opportunities)
4. Provide reasoning for the sentiment score

Output JSON only (no markdown):
{{
    "summary": "2 paragraph summary (paragraph 1. paragraph 2.)",
    "sentiment_score": 6,
    "reasoning": "Why this sentiment score"
}}"""

    try:
        logger.info(f"[GOSSIP] Step 4: Sending articles to Vertex AI for summarization...")
        logger.info(f"[GOSSIP] Prompt length: {len(prompt)} characters")
        logger.info(f"[GOSSIP] Prompt preview: {prompt[:200]}...")
        
        # Try SDK first, fallback to REST API
        project_id = os.getenv("GCP_PROJECT_ID")
        logger.info(f"[GOSSIP] Using project_id: {project_id}")
        logger.info(f"[GOSSIP] Calling generate_content_with_fallback()...")
        
        response_text = generate_content_with_fallback(prompt, model, project_id)
        
        logger.info(f"[GOSSIP] AI response received: {response_text is not None}")
        if response_text:
            logger.info(f"[GOSSIP] Response length: {len(response_text)} characters")
            logger.info(f"[GOSSIP] Response preview: {response_text[:200]}...")
        
        if not response_text:
            logger.error("[GOSSIP] Both SDK and REST API failed - no response text")
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
            ai_result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[^{}]*"summary"[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                ai_result = json.loads(json_match.group())
            else:
                raise
        
        # Validate sentiment score
        sentiment_score = ai_result.get("sentiment_score", 5)
        try:
            sentiment_score = int(float(sentiment_score))
            sentiment_score = max(1, min(10, sentiment_score))  # Clamp to 1-10
        except (ValueError, TypeError):
            sentiment_score = 5
        
        # Prepare sources list
        sources = []
        for article in articles[:10]:  # Top 10 articles as sources
            source_entry = {
                "title": article.get("title", "Unknown"),
                "url": article.get("link", "")
            }
            sources.append(source_entry)
        
        result = {
            "summary": ai_result.get("summary", f"Found {len(articles)} articles about {company_display}."),
            "sentiment_score": sentiment_score,
            "articles_count": len(articles),
            "reasoning": ai_result.get("reasoning", "Analysis completed."),
            "sources": sources
        }
        
        logger.info("")
        logger.info("[GOSSIP] " + "=" * 70)
        logger.info(f"[GOSSIP] ✓✓✓ AI ANALYSIS COMPLETE ✓✓✓")
        logger.info(f"[GOSSIP] Final sentiment score: {sentiment_score}/10")
        logger.info(f"[GOSSIP] Articles analyzed: {len(articles)}")
        logger.info(f"[GOSSIP] Summary preview: {result['summary'][:150]}...")
        logger.info(f"[GOSSIP] Reasoning: {result['reasoning'][:150]}...")
        logger.info("[GOSSIP] " + "=" * 70)
        logger.info("")
        
        return result
        
    except Exception as e:
        logger.error("")
        logger.error("[GOSSIP] " + "=" * 70)
        logger.error(f"[GOSSIP] ✗✗✗ ERROR IN AI ANALYSIS ✗✗✗")
        logger.error(f"[GOSSIP] Error type: {type(e).__name__}")
        logger.error(f"[GOSSIP] Error message: {str(e)}")
        logger.error(f"[GOSSIP] Full traceback:", exc_info=True)
        logger.error("[GOSSIP] " + "=" * 70)
        
        # Fallback: basic summary
        titles = "\n".join([f"- {art['title']}" for art in articles[:5]])
        sources = []
        for article in articles[:10]:
            sources.append({
                "title": article.get("title", "Unknown"),
                "url": article.get("link", "")
            })
        
        result = {
            "summary": f"Found {len(articles)} articles about {company_display}:\n{titles}\n\nAI summarization encountered an error.",
            "sentiment_score": 5,
            "articles_count": len(articles),
            "reasoning": f"Fallback summary due to AI error: {str(e)}",
            "sources": sources
        }
        logger.warning(f"[GOSSIP] Returning fallback result: sentiment=5/10")
        return result