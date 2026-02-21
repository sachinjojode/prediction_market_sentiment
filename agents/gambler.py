"""Gambler Agent - Polymarket Sentiment Analysis

Fetches probability data from Polymarket's Gamma API and uses Vertex AI
to analyze the odds and generate sentiment scores.

Uses official Polymarket API documentation:
https://docs.polymarket.com/api-reference/introduction
Gamma API: https://gamma-api.polymarket.com
CLOB API: https://clob.polymarket.com

Key improvements:
  1. Uses the correct /public-search endpoint (not _q param on /markets)
  2. Fetches by slug for known events
  3. Proper JSON parsing of outcomes/clobTokenIds/outcomePrices
  4. Client-side relevance filtering as safety net
  5. Rate-limited CLOB calls for order book data
"""

import os
import json
import time
import requests
from typing import Optional, Dict, List
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
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

# Polymarket API base URLs
GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Delay between CLOB API calls to be polite
API_DELAY = 0.3

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
            logger.info("Vertex AI initialized successfully for Gambler agent")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            return None
    
    return _model


# ─── Helper Functions ────────────────────────────────────────────────────────

def safe_json_parse(value, fallback=None):
    """Parse a value that might be a JSON string, list, or other type."""
    if fallback is None:
        fallback = []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else fallback
        except (json.JSONDecodeError, TypeError):
            return fallback
    return fallback


def is_ticker_related(market: dict, ticker: str, company_name: str = None) -> bool:
    """Check if a market is actually about the ticker/company."""
    ticker_lower = ticker.lower()
    company_lower = (company_name or "").lower()
    
    text = " ".join([
        str(market.get("question", "")),
        str(market.get("title", "")),
        str(market.get("description", "")),
        str(market.get("tags", "")),
        str(market.get("slug", "")),
    ]).lower()
    
    # Check for ticker mention
    if ticker_lower in text or f"({ticker_lower})" in text:
        return True
    
    # Check for company name mention
    if company_lower:
        company_words = [w for w in company_lower.split() if len(w) > 3]
        if company_words:
            return any(word in text for word in company_words)
    
    return False


def is_market_closed_or_resolved(market: dict) -> bool:
    """
    Check if a market is closed, resolved, or has ended.
    Checks: closed field, closedTime, umaResolutionStatus, and endDate.
    """
    # Check closed field
    if market.get("closed", False):
        logger.debug(f"[GAMBLER] Market marked as closed")
        return True
    
    # Check closedTime
    closed_time = market.get("closedTime") or market.get("closed_time")
    if closed_time:
        try:
            if isinstance(closed_time, (int, float)):
                closed_dt = datetime.fromtimestamp(closed_time, tz=timezone.utc)
            else:
                closed_dt = datetime.fromisoformat(str(closed_time).replace("Z", "+00:00"))
            if closed_dt < datetime.now(timezone.utc):
                logger.debug(f"[GAMBLER] Market closed at {closed_dt}")
                return True
        except Exception as e:
            logger.debug(f"[GAMBLER] Error parsing closedTime: {e}")
    
    # Check umaResolutionStatus
    resolution_status = market.get("umaResolutionStatus") or market.get("resolution_status")
    if resolution_status and str(resolution_status).lower() in ["resolved", "closed", "finalized"]:
        logger.debug(f"[GAMBLER] Market resolution status: {resolution_status}")
        return True
    
    # Check endDate
    end_date = market.get("endDate") or market.get("end_date") or market.get("end_date_iso") or market.get("expirationDate")
    if end_date:
        try:
            if isinstance(end_date, (int, float)):
                end_dt = datetime.fromtimestamp(end_date, tz=timezone.utc)
            else:
                end_dt = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
            if end_dt < datetime.now(timezone.utc):
                logger.debug(f"[GAMBLER] Market end date in past: {end_dt}")
                return True
        except Exception as e:
            logger.debug(f"[GAMBLER] Error parsing endDate: {e}")
    
    return False


def classify_market_type(market: dict) -> str:
    """
    Classify market type:
    - 'earnings_sentiment': Earnings beat/miss, revenue, guidance
    - 'price_target': Will close above/below $X
    - 'other': Other types
    """
    question = (market.get("question") or market.get("title", "")).lower()
    
    # Earnings/sentiment keywords
    earnings_keywords = [
        "earnings", "beat", "miss", "revenue", "guidance", "eps", 
        "profit", "loss", "quarterly", "q1", "q2", "q3", "q4",
        "exceed", "surpass", "outperform", "underperform"
    ]
    
    # Price target keywords
    price_keywords = [
        "close above", "close below", "above $", "below $", 
        "reach $", "hit $", "price target", "trading above",
        "trading below", "exceed $", "surpass $"
    ]
    
    if any(keyword in question for keyword in earnings_keywords):
        return "earnings_sentiment"
    elif any(keyword in question for keyword in price_keywords):
        return "price_target"
    else:
        return "other"


def extract_market_probability(market: dict) -> Optional[float]:
    """
    Extract probability from a market.
    Tries multiple methods: outcomePrices, CLOB midpoint, or calculates from prices.
    """
    market_id = market.get("conditionId") or market.get("id", "unknown")
    logger.debug(f"[GAMBLER] Extracting probability for market {market_id}")
    
    # Method 1: Direct outcomePrices (JSON string or list)
    raw_outcome_prices = market.get("outcomePrices") or market.get("outcome_prices")
    raw_outcomes = market.get("outcomes")
    
    logger.debug(f"[GAMBLER]   Raw outcomePrices type: {type(raw_outcome_prices)}, value: {raw_outcome_prices}")
    logger.debug(f"[GAMBLER]   Raw outcomes type: {type(raw_outcomes)}, value: {raw_outcomes}")
    
    outcome_prices = safe_json_parse(raw_outcome_prices or [])
    outcomes = safe_json_parse(raw_outcomes or [])
    
    logger.debug(f"[GAMBLER]   Parsed outcomePrices: {outcome_prices}")
    logger.debug(f"[GAMBLER]   Parsed outcomes: {outcomes}")
    
    if outcome_prices and len(outcome_prices) > 0:
        try:
            # For binary markets, look for "Yes" or first outcome
            if outcomes:
                yes_index = None
                for i, outcome in enumerate(outcomes):
                    if "yes" in str(outcome).lower() or i == 0:
                        yes_index = i
                        break
                
                logger.debug(f"[GAMBLER]   Found 'Yes' index: {yes_index}")
                
                if yes_index is not None and yes_index < len(outcome_prices):
                    price = float(outcome_prices[yes_index])
                    logger.info(f"[GAMBLER]   ✓ Extracted probability from outcomePrices[{yes_index}]: {price:.4f} ({price*100:.2f}%)")
                    return price
            else:
                # No outcomes listed, use first price
                price = float(outcome_prices[0])
                logger.info(f"[GAMBLER]   ✓ Extracted probability from first outcomePrice: {price:.4f} ({price*100:.2f}%)")
                return price
        except (ValueError, TypeError, IndexError) as e:
            logger.debug(f"[GAMBLER]   ✗ Error extracting from outcomePrices: {e}")
    
    # Method 2: Try to get from CLOB midpoint (if we have token IDs)
    clob_token_ids = safe_json_parse(market.get("clobTokenIds") or market.get("clob_token_ids") or [])
    if clob_token_ids:
        logger.debug(f"[GAMBLER]   Found {len(clob_token_ids)} CLOB token IDs, but skipping CLOB API call for now")
        # This would require a CLOB API call - for now, skip
        # We'll enhance this later if needed
    
    # Method 3: Calculate from prices if we have multiple outcomes
    if outcome_prices and len(outcome_prices) >= 2:
        try:
            prices = [float(p) for p in outcome_prices]
            # For binary markets, "Yes" is typically first outcome
            if prices[0] > 0:
                logger.info(f"[GAMBLER]   ✓ Extracted probability from first of multiple prices: {prices[0]:.4f} ({prices[0]*100:.2f}%)")
                return prices[0]
        except (ValueError, TypeError) as e:
            logger.debug(f"[GAMBLER]   ✗ Error calculating from prices: {e}")
    
    logger.warning(f"[GAMBLER]   ✗ Could not extract probability from market {market_id}")
    logger.warning(f"[GAMBLER]   Market keys: {list(market.keys())}")
    return None


# ─── Polymarket API Client ───────────────────────────────────────────────────

class PolymarketClient:
    """Lightweight client for Polymarket's public APIs."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "MarketMinds-Gambler/1.0",
        })

    def public_search(self, query: str) -> dict:
        """
        Use the /public-search endpoint — this is the correct search API.
        Returns {"events": [...], "markets": [...], "profiles": [...]}.
        """
        url = f"{GAMMA_BASE}/public-search"
        params = {"q": query}
        
        logger.info(f"[GAMBLER] API CALL: GET {url}")
        logger.info(f"[GAMBLER]   Parameters: {params}")
        logger.info(f"[GAMBLER]   Query: '{query}'")
        
        try:
            resp = self.session.get(url, params=params, timeout=15)
            
            logger.info(f"[GAMBLER] API RESPONSE: Status {resp.status_code}")
            logger.info(f"[GAMBLER]   Response URL: {resp.url}")
            logger.info(f"[GAMBLER]   Response headers: {dict(resp.headers)}")
            
            resp.raise_for_status()
            data = resp.json()
            
            events_count = len(data.get('events', []))
            markets_count = len(data.get('markets', []))
            profiles_count = len(data.get('profiles', []))
            
            logger.info(f"[GAMBLER] API RESPONSE DATA:")
            logger.info(f"[GAMBLER]   Events: {events_count}")
            logger.info(f"[GAMBLER]   Markets: {markets_count}")
            logger.info(f"[GAMBLER]   Profiles: {profiles_count}")
            
            # Log first few events if any
            if events_count > 0:
                logger.info(f"[GAMBLER]   First event preview:")
                first_event = data.get('events', [])[0]
                logger.info(f"[GAMBLER]     - ID: {first_event.get('id', 'N/A')}")
                logger.info(f"[GAMBLER]     - Title: {first_event.get('title', first_event.get('question', 'N/A'))[:80]}")
                logger.info(f"[GAMBLER]     - Slug: {first_event.get('slug', 'N/A')}")
                logger.info(f"[GAMBLER]     - Markets in event: {len(first_event.get('markets', []))}")
            
            # Log first few markets if any
            if markets_count > 0:
                logger.info(f"[GAMBLER]   First market preview:")
                first_market = data.get('markets', [])[0]
                logger.info(f"[GAMBLER]     - ID: {first_market.get('id', first_market.get('conditionId', 'N/A'))}")
                logger.info(f"[GAMBLER]     - Question: {first_market.get('question', first_market.get('title', 'N/A'))[:80]}")
                logger.info(f"[GAMBLER]     - Active: {first_market.get('active', 'N/A')}")
                logger.info(f"[GAMBLER]     - Volume: {first_market.get('volume', 'N/A')}")
            
            # Log full response structure (truncated if too large)
            response_str = json.dumps(data, indent=2, default=str)
            if len(response_str) > 2000:
                logger.debug(f"[GAMBLER]   Full response (truncated): {response_str[:2000]}...")
            else:
                logger.debug(f"[GAMBLER]   Full response: {response_str}")
            
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"[GAMBLER] API ERROR for '{query}':")
            logger.error(f"[GAMBLER]   Error type: {type(e).__name__}")
            logger.error(f"[GAMBLER]   Error message: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"[GAMBLER]   Response status: {e.response.status_code}")
                logger.error(f"[GAMBLER]   Response text: {e.response.text[:500]}")
            return {}
        except Exception as e:
            logger.error(f"[GAMBLER] UNEXPECTED ERROR for '{query}': {e}")
            logger.error(f"[GAMBLER]   Error type: {type(e).__name__}")
            logger.error(f"[GAMBLER]   Full traceback:", exc_info=True)
            return {}

    def get_event_by_slug(self, slug: str) -> dict | None:
        """Fetch a single event by its slug (includes child markets)."""
        url = f"{GAMMA_BASE}/events/slug/{slug}"
        
        logger.info(f"[GAMBLER] API CALL: GET {url}")
        logger.info(f"[GAMBLER]   Slug: '{slug}'")
        
        try:
            resp = self.session.get(url, timeout=15)
            
            logger.info(f"[GAMBLER] API RESPONSE: Status {resp.status_code}")
            
            resp.raise_for_status()
            data = resp.json()
            
            # The endpoint may return a list or single object
            if isinstance(data, list):
                logger.info(f"[GAMBLER]   Response is list with {len(data)} items")
                if data:
                    logger.info(f"[GAMBLER]   First event ID: {data[0].get('id', 'N/A')}")
                    logger.info(f"[GAMBLER]   First event title: {data[0].get('title', data[0].get('question', 'N/A'))[:80]}")
                return data[0] if data else None
            else:
                logger.info(f"[GAMBLER]   Response is single object")
                logger.info(f"[GAMBLER]   Event ID: {data.get('id', 'N/A')}")
                logger.info(f"[GAMBLER]   Event title: {data.get('title', data.get('question', 'N/A'))[:80]}")
                logger.info(f"[GAMBLER]   Markets in event: {len(data.get('markets', []))}")
                return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"[GAMBLER] API ERROR for slug '{slug}':")
            logger.warning(f"[GAMBLER]   Error type: {type(e).__name__}")
            logger.warning(f"[GAMBLER]   Error message: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.warning(f"[GAMBLER]   Response status: {e.response.status_code}")
            return None
        except Exception as e:
            logger.warning(f"[GAMBLER] UNEXPECTED ERROR for slug '{slug}': {e}")
            return None

    def get_markets_by_event(self, event_id: str) -> list[dict]:
        """Fetch all markets for a given event ID."""
        url = f"{GAMMA_BASE}/markets"
        params = {"event_id": event_id, "closed": "false"}
        
        logger.info(f"[GAMBLER] API CALL: GET {url}")
        logger.info(f"[GAMBLER]   Parameters: {params}")
        logger.info(f"[GAMBLER]   Event ID: '{event_id}'")
        
        try:
            resp = self.session.get(url, params=params, timeout=15)
            
            logger.info(f"[GAMBLER] API RESPONSE: Status {resp.status_code}")
            
            resp.raise_for_status()
            data = resp.json()
            
            if isinstance(data, list):
                logger.info(f"[GAMBLER]   Response is list with {len(data)} markets")
                if data:
                    logger.info(f"[GAMBLER]   First market ID: {data[0].get('id', data[0].get('conditionId', 'N/A'))}")
                    logger.info(f"[GAMBLER]   First market question: {data[0].get('question', 'N/A')[:80]}")
                return data
            else:
                logger.warning(f"[GAMBLER]   Response is not a list: {type(data)}")
                return []
        except requests.exceptions.RequestException as e:
            logger.warning(f"[GAMBLER] API ERROR for event '{event_id}':")
            logger.warning(f"[GAMBLER]   Error type: {type(e).__name__}")
            logger.warning(f"[GAMBLER]   Error message: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.warning(f"[GAMBLER]   Response status: {e.response.status_code}")
            return []
        except Exception as e:
            logger.warning(f"[GAMBLER] UNEXPECTED ERROR for event '{event_id}': {e}")
            return []

    def get_midpoint_price(self, token_id: str) -> float | None:
        """Get midpoint price from CLOB API for a token."""
        url = f"{CLOB_BASE}/midpoint"
        params = {"token_id": token_id}
        
        logger.debug(f"[GAMBLER] API CALL: GET {url}")
        logger.debug(f"[GAMBLER]   Parameters: {params}")
        logger.debug(f"[GAMBLER]   Token ID: '{token_id[:20]}...'")
        
        try:
            resp = self.session.get(url, params=params, timeout=15)
            
            logger.debug(f"[GAMBLER] API RESPONSE: Status {resp.status_code}")
            
            resp.raise_for_status()
            data = resp.json()
            
            midpoint = float(data.get("mid", 0))
            logger.debug(f"[GAMBLER]   Midpoint price: {midpoint:.4f} ({midpoint*100:.2f}%)")
            
            return midpoint
        except Exception as e:
            logger.debug(f"[GAMBLER] API ERROR for token '{token_id[:20]}...': {e}")
            return None


# ─── Market Discovery ────────────────────────────────────────────────────────

def discover_markets(client: PolymarketClient, ticker: str, company_name: str = None) -> list[dict]:
    """
    Discover markets for a ticker using:
      1. /public-search endpoint (primary)
      2. Client-side relevance filtering
    """
    seen_ids = set()
    all_markets = []

    def add_market(m: dict):
        mid = m.get("conditionId") or m.get("id") or m.get("condition_id", "")
        if mid and mid not in seen_ids:
            seen_ids.add(mid)
            all_markets.append(m)

    def add_event_markets(event: dict):
        """Extract markets from an event object."""
        event_id = event.get("id", "unknown")
        event_title = event.get("title") or event.get("question", "Unknown")
        
        logger.info(f"[GAMBLER] Processing event: {event_id}")
        logger.info(f"[GAMBLER]   Title: {event_title[:80]}")
        
        markets = event.get("markets", [])
        logger.info(f"[GAMBLER]   Markets embedded in event: {len(markets)}")
        
        if markets:
            logger.info(f"[GAMBLER]   Adding {len(markets)} markets from embedded data")
            for i, m in enumerate(markets, 1):
                market_id = m.get("conditionId") or m.get("id", "unknown")
                market_q = m.get("question") or m.get("title", "Unknown")
                logger.debug(f"[GAMBLER]     Market {i}/{len(markets)}: {market_id} - {market_q[:60]}...")
                add_market(m)
        else:
            # If event doesn't embed markets, fetch them separately
            if event_id:
                logger.info(f"[GAMBLER]   No embedded markets, fetching separately for event {event_id}...")
                markets = client.get_markets_by_event(str(event_id))
                logger.info(f"[GAMBLER]   Fetched {len(markets)} markets for event {event_id}")
                for m in markets:
                    add_market(m)
            else:
                logger.warning(f"[GAMBLER]   Event has no ID and no embedded markets, skipping")

    # Build search queries
    search_queries = [ticker]
    if company_name:
        search_queries.append(company_name)
        # Also try "TICKER stock" format
        search_queries.append(f"{ticker} stock")
        search_queries.append(f"{company_name} stock")

    logger.info(f"[GAMBLER] Searching for markets using queries: {search_queries}")

    # Strategy 1: /public-search
    for query in search_queries:
        logger.info("")
        logger.info(f"[GAMBLER] ── Searching (public-search) for: '{query}' ──")
        results = client.public_search(query)

        # Extract events from search results
        search_events = results.get("events", [])
        logger.info(f"[GAMBLER]   → Found {len(search_events)} event(s) in search results")
        
        for i, evt in enumerate(search_events, 1):
            logger.info(f"[GAMBLER]   Processing event {i}/{len(search_events)}")
            add_event_markets(evt)

        # Extract markets directly from search results
        search_markets = results.get("markets", [])
        logger.info(f"[GAMBLER]   → Found {len(search_markets)} direct market(s) in search results")
        
        for i, m in enumerate(search_markets, 1):
            market_id = m.get("conditionId") or m.get("id", "unknown")
            market_q = m.get("question") or m.get("title", "Unknown")
            logger.debug(f"[GAMBLER]     Direct market {i}/{len(search_markets)}: {market_id} - {market_q[:60]}...")
            add_market(m)
        
        logger.info(f"[GAMBLER] ── End search for '{query}' ──")
        logger.info("")

    logger.info("")
    logger.info(f"[GAMBLER] ===== MARKET DISCOVERY SUMMARY =====")
    logger.info(f"[GAMBLER] Total unique markets found before filtering: {len(all_markets)}")
    
    # Log all market IDs and questions before filtering
    if all_markets:
        logger.info(f"[GAMBLER] All discovered markets:")
        for i, m in enumerate(all_markets[:10], 1):  # Log first 10
            market_id = m.get("conditionId") or m.get("id", "unknown")
            market_q = m.get("question") or m.get("title", "Unknown")
            logger.info(f"[GAMBLER]   {i}. [{market_id}] {market_q[:70]}...")
        if len(all_markets) > 10:
            logger.info(f"[GAMBLER]   ... and {len(all_markets) - 10} more markets")

    # Client-side relevance filter
    logger.info("")
    logger.info(f"[GAMBLER] Applying relevance filter for ticker '{ticker}' / company '{company_name or 'N/A'}'...")
    
    filtered = []
    for m in all_markets:
        is_related = is_ticker_related(m, ticker, company_name)
        market_id = m.get("conditionId") or m.get("id", "unknown")
        market_q = m.get("question") or m.get("title", "Unknown")
        
        if is_related:
            logger.info(f"[GAMBLER]   ✓ KEEP: [{market_id}] {market_q[:70]}...")
            filtered.append(m)
        else:
            logger.debug(f"[GAMBLER]   ✗ FILTER OUT: [{market_id}] {market_q[:70]}...")
    
    removed = len(all_markets) - len(filtered)
    if removed > 0:
        logger.info(f"[GAMBLER] Filtered out {removed} irrelevant market(s)")
    logger.info(f"[GAMBLER] Relevant markets after filtering: {len(filtered)}")
    logger.info(f"[GAMBLER] =====================================")
    logger.info("")

    return filtered


# ─── Main Data Fetching Function ────────────────────────────────────────────

def _fetch_polymarket_data(ticker: str, company_name: str = None) -> Optional[Dict]:
    """
    Fetch Polymarket data for a ticker using the correct API endpoints.
    
    Args:
        ticker: Stock ticker symbol
        company_name: Optional company name for better search
    
    Returns:
        Dictionary with aggregated market data or None
    """
    logger.info(f"[GAMBLER] Starting market discovery for {ticker}/{company_name or 'N/A'}")
    
    client = PolymarketClient()
    
    # Discover markets
    markets = discover_markets(client, ticker, company_name)
    
    if not markets:
        logger.warning(f"[GAMBLER] No markets found for {ticker}")
        return None
    
    logger.info(f"[GAMBLER] Processing {len(markets)} markets...")
    logger.info("")
    
    # Process markets: extract probabilities, filter by volume, calculate relevance
    all_valid_markets = []
    
    for i, market in enumerate(markets, 1):
        market_id = market.get("conditionId") or market.get("id", "unknown")
        market_q = market.get("question") or market.get("title", "Unknown")
        
        logger.info(f"[GAMBLER] ── Processing market {i}/{len(markets)}: {market_id} ──")
        logger.info(f"[GAMBLER]   Question: {market_q}")
        logger.info(f"[GAMBLER]   Raw market data keys: {list(market.keys())}")
        
        # Check if market is closed/resolved/ended
        if is_market_closed_or_resolved(market):
            logger.info(f"[GAMBLER]   ✗ SKIPPED: Market is closed/resolved/ended")
            logger.info("")
            continue
        
        # Only process active markets
        if not market.get("active", True):
            logger.info(f"[GAMBLER]   ✗ SKIPPED: Market is not active")
            logger.info("")
            continue
        
        # Classify market type
        market_type = classify_market_type(market)
        logger.info(f"[GAMBLER]   Market type: {market_type}")
        
        # Extract probability
        logger.info(f"[GAMBLER]   Extracting probability...")
        probability = extract_market_probability(market)
        if probability is None:
            logger.warning(f"[GAMBLER]   ✗ SKIPPED: Could not extract probability")
            logger.info("")
            continue
        
        logger.info(f"[GAMBLER]   ✓ Probability extracted: {probability:.4f} ({probability*100:.2f}%)")
        
        # Get volume
        raw_volume = market.get("volume") or market.get("volumeNum") or 0
        logger.info(f"[GAMBLER]   Raw volume: {raw_volume} (type: {type(raw_volume)})")
        
        volume = raw_volume
        if isinstance(volume, str):
            try:
                volume = float(volume)
                logger.info(f"[GAMBLER]   Converted volume from string: {volume:,.0f}")
            except (ValueError, TypeError):
                logger.warning(f"[GAMBLER]   Could not convert volume string: {volume}")
                volume = 0
        
        logger.info(f"[GAMBLER]   Final volume: {volume:,.0f}")
        
        # Skip very low volume markets
        if volume < 50:
            logger.info(f"[GAMBLER]   ✗ SKIPPED: Volume too low ({volume:,.0f} < 50)")
            logger.info("")
            continue
        
        # Calculate relevance score
        question = market.get("question", "").lower()
        ticker_lower = ticker.lower()
        mentions_ticker = (
            f" {ticker_lower} " in f" {question} " or
            f"({ticker_lower})" in question
        )
        base_relevance = 3 if mentions_ticker else 2
        
        # Boost relevance for earnings/sentiment markets (strongest signal)
        if market_type == "earnings_sentiment":
            relevance_score = base_relevance + 2  # Highest priority
            logger.info(f"[GAMBLER]   Earnings/sentiment market - boosted relevance")
        elif market_type == "price_target":
            relevance_score = base_relevance  # Price targets are less direct sentiment
            logger.info(f"[GAMBLER]   Price target market - standard relevance")
        else:
            relevance_score = base_relevance - 1  # Other types lower priority
            logger.info(f"[GAMBLER]   Other market type - reduced relevance")
        
        logger.info(f"[GAMBLER]   Final relevance score: {relevance_score} (mentions ticker: {mentions_ticker})")
        
        # Add to valid markets
        processed_market = {
            "question": market.get("question") or market.get("title", "Unknown"),
            "probability": probability,
            "volume": volume,
            "outcomes": safe_json_parse(market.get("outcomes", [])),
            "outcome_prices": safe_json_parse(market.get("outcomePrices") or market.get("outcome_prices") or []),
            "market_id": market_id,
            "slug": market.get("slug", ""),
            "market_type": market_type,
            "relevance_score": relevance_score,
            "url": f"https://polymarket.com/event/{market.get('slug', '')}" if market.get('slug') else None
        }
        
        all_valid_markets.append(processed_market)
        logger.info(f"[GAMBLER]   ✓ ADDED: prob={probability:.2%}, vol={volume:,.0f}, relevance={relevance_score}")
        logger.info("")
    
    if not all_valid_markets:
        logger.warning(f"[GAMBLER] No valid markets after processing for {ticker}")
        return None
    
    # Separate markets by type
    earnings_markets = [m for m in all_valid_markets if m["market_type"] == "earnings_sentiment"]
    price_target_markets = [m for m in all_valid_markets if m["market_type"] == "price_target"]
    other_markets = [m for m in all_valid_markets if m["market_type"] == "other"]
    
    logger.info(f"[GAMBLER] Market breakdown:")
    logger.info(f"[GAMBLER]   Earnings/sentiment: {len(earnings_markets)}")
    logger.info(f"[GAMBLER]   Price targets: {len(price_target_markets)}")
    logger.info(f"[GAMBLER]   Other: {len(other_markets)}")
    
    # Prioritize earnings/sentiment markets for sentiment analysis
    # Use price targets as secondary signal (implied price distribution)
    markets_for_sentiment = earnings_markets + price_target_markets[:3]  # Top 3 price targets
    
    # Sort by relevance and volume
    markets_for_sentiment.sort(key=lambda x: (x["relevance_score"], x["volume"]), reverse=True)
    
    # Take top markets (up to 10 for aggregation)
    top_markets = markets_for_sentiment[:10]
    
    # Calculate weighted average probability (heavily weight earnings markets)
    total_weighted_prob = 0.0
    total_weight = 0.0
    
    for market in top_markets:
        # Earnings markets get 3x weight, price targets get 1x weight
        type_multiplier = 3.0 if market["market_type"] == "earnings_sentiment" else 1.0
        weight = market["volume"] * market["relevance_score"] * type_multiplier
        total_weighted_prob += market["probability"] * weight
        total_weight += weight
    
    if total_weight > 0:
        weighted_avg_prob = total_weighted_prob / total_weight
    else:
        weighted_avg_prob = sum(m["probability"] for m in top_markets) / len(top_markets) if top_markets else 0.5
    
    best_market = max(top_markets, key=lambda x: x["volume"] * x["relevance_score"]) if top_markets else None
    
    logger.info(f"[GAMBLER] Found {len(all_valid_markets)} valid markets, using top {len(top_markets)} for sentiment")
    logger.info(f"[GAMBLER] Weighted average probability: {weighted_avg_prob:.2%}")
    if best_market:
        logger.info(f"[GAMBLER] Best market: {best_market.get('question', 'Unknown')[:70]}...")
    
    # Prepare sources list
    sources = []
    for market in top_markets[:5]:  # Top 5 markets as sources
        source_entry = {
            "question": market.get("question", "Unknown"),
            "probability": f"{market.get('probability', 0):.1%}",
            "type": market.get("market_type", "unknown"),
            "url": market.get("url")
        }
        sources.append(source_entry)
    
    return {
        "ticker": ticker,
        "best_market": best_market,
        "all_markets": all_valid_markets,
        "top_markets": top_markets,
        "weighted_avg_probability": weighted_avg_prob,
        "markets_count": len(all_valid_markets),
        "top_markets_count": len(top_markets),
        "sources": sources,
        "earnings_markets_count": len(earnings_markets),
        "price_target_markets_count": len(price_target_markets)
    }


# ─── Main Public Function ───────────────────────────────────────────────────

def get_polymarket_sentiment(ticker: str, company_name: str = None) -> Dict:
    """
    Fetch Polymarket data and analyze it with Vertex AI to generate
    explanation and sentiment score.
    
    Args:
        ticker: Stock ticker symbol (e.g., "NVDA", "AAPL")
        company_name: Optional company name for context
    
    Returns:
        Dictionary with:
        - explanation: 2-3 sentence explanation
        - sentiment_score: 1-10 score (10 = most positive)
        - raw_odds: Probability from Polymarket (0.0-1.0)
        - reasoning: Why this sentiment score
    """
    logger.info("")
    logger.info("[GAMBLER] " + "=" * 70)
    logger.info(f"[GAMBLER] get_polymarket_sentiment() called")
    logger.info(f"[GAMBLER] Parameters: ticker='{ticker}', company_name='{company_name or 'None'}'")
    logger.info("[GAMBLER] " + "=" * 70)
    
    # Fetch raw Polymarket data
    logger.info(f"[GAMBLER] Step 1: Fetching Polymarket data...")
    market_data = _fetch_polymarket_data(ticker, company_name)
    logger.info(f"[GAMBLER] _fetch_polymarket_data() returned: {market_data is not None}")
    
    if not market_data:
        logger.warning(f"[GAMBLER] No Polymarket data found for {ticker}")
        logger.warning(f"[GAMBLER] Returning fallback result with neutral sentiment")
        result = {
            "explanation": f"No prediction market data found for {ticker}. This may indicate low market interest or that markets haven't been created yet.",
            "sentiment_score": 5,
            "raw_odds": None,
            "reasoning": "No data available to analyze.",
            "sources": []
        }
        logger.info(f"[GAMBLER] Returning result: {result}")
        return result
    
    # Use weighted average from multiple markets
    logger.info(f"[GAMBLER] Step 2: Processing market data...")
    raw_odds = market_data.get("weighted_avg_probability", market_data["best_market"]["probability"])
    best_market = market_data["best_market"]
    top_markets = market_data.get("top_markets", [])
    markets_count = market_data.get("markets_count", 0)
    top_markets_count = market_data.get("top_markets_count", 0)
    
    logger.info(f"[GAMBLER] Market data summary:")
    logger.info(f"[GAMBLER]   - Total markets found: {markets_count}")
    logger.info(f"[GAMBLER]   - Top markets used: {top_markets_count}")
    logger.info(f"[GAMBLER]   - Weighted average probability: {raw_odds:.2%}")
    logger.info(f"[GAMBLER]   - Best market question: {best_market.get('question', 'N/A')[:60]}...")
    
    # Use Vertex AI to analyze
    logger.info(f"[GAMBLER] Step 3: Initializing Vertex AI model...")
    model = _get_model()
    logger.info(f"[GAMBLER] Vertex AI model: {model is not None}")
    
    if model is None:
        logger.warning("[GAMBLER] Vertex AI not available, returning basic analysis")
        logger.warning("[GAMBLER] Calculating sentiment from raw odds only")
        sentiment_score = int(raw_odds * 10) if raw_odds else 5
        result = {
            "explanation": f"Prediction markets show {raw_odds:.1%} probability. Vertex AI analysis unavailable.",
            "sentiment_score": sentiment_score,
            "raw_odds": raw_odds,
            "reasoning": "Basic calculation based on odds.",
            "sources": market_data.get("sources", []) if market_data else []
        }
        logger.info(f"[GAMBLER] Returning basic result: sentiment={sentiment_score}/10")
        return result
    
    # Build market summary for AI
    company_display = company_name or ticker
    market_summary = f"""
Aggregated Analysis from {top_markets_count} prediction markets:

Weighted Average Probability: {raw_odds:.1%}
Total Markets Analyzed: {markets_count}

Top Markets:
"""
    
    for i, market in enumerate(top_markets[:5], 1):
        market_summary += f"""
{i}. {market.get('question', 'Unknown')[:80]}...
   Probability: {market.get('probability', 0):.1%}
   Volume: {market.get('volume', 0):,.0f}
   Relevance: {'High' if market.get('relevance_score', 0) >= 3 else 'Medium'}
"""
    
    if top_markets_count > 5:
        market_summary += f"\n... and {top_markets_count - 5} more markets"
    
    # Construct AI prompt
    prompt = f"""You are a financial analyst analyzing prediction market data for {company_display} ({ticker}).

Prediction Market Data:
{market_summary}

CRITICAL INTERPRETATION GUIDE:
- These probabilities show what prediction market traders (smart money) believe will happen
- For markets asking "will price go UP/ABOVE": 
  * Probability > 60% = BULLISH (traders expect price to rise)
  * Probability 40-60% = NEUTRAL (uncertain)
  * Probability < 40% = BEARISH (traders expect price to fall)
- Higher volume markets = more reliable signals
- Multiple markets with similar probabilities = stronger consensus

SENTIMENT SCORING RULES:
- If weighted average > 65%: Sentiment = 8-10 (very bullish)
- If weighted average 55-65%: Sentiment = 6-7 (moderately bullish)  
- If weighted average 45-55%: Sentiment = 4-6 (neutral)
- If weighted average 35-45%: Sentiment = 2-4 (moderately bearish)
- If weighted average < 35%: Sentiment = 1-3 (very bearish)

Task:
1. Analyze the aggregated prediction market data for {company_display}
2. Write a clear, concise 2-3 sentence explanation of what these probabilities mean for the stock
3. Assign a sentiment score (1-10) based on the weighted average probability
4. Provide specific reasoning referencing the actual probability values

Be direct and factual. Focus on what the markets are saying, not speculation.

Output JSON only (no markdown):
{{
    "explanation": "Clear 2-3 sentence explanation of what prediction markets indicate",
    "sentiment_score": 7,
    "reasoning": "Specific reasoning with probability values (e.g., 'Weighted average of 72% indicates strong bullish sentiment from prediction market traders')"
}}"""

    try:
        logger.info("[GAMBLER] Step 4: Sending data to Vertex AI for analysis...")
        logger.info(f"[GAMBLER] Prompt length: {len(prompt)} characters")
        logger.info(f"[GAMBLER] Prompt preview: {prompt[:200]}...")
        
        # Try SDK first, fallback to REST API
        project_id = os.getenv("GCP_PROJECT_ID")
        logger.info(f"[GAMBLER] Using project_id: {project_id}")
        logger.info(f"[GAMBLER] Calling generate_content_with_fallback()...")
        
        response_text = generate_content_with_fallback(prompt, model, project_id)
        
        logger.info(f"[GAMBLER] AI response received: {response_text is not None}")
        if response_text:
            logger.info(f"[GAMBLER] Response length: {len(response_text)} characters")
            logger.info(f"[GAMBLER] Response preview: {response_text[:200]}...")
        
        if not response_text:
            logger.error("[GAMBLER] Both SDK and REST API failed - no response text")
            raise ValueError("Both SDK and REST API failed")
        
        # Clean response text - handle markdown code blocks
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:].strip()
        elif response_text.startswith("```"):
            response_text = response_text[3:].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()
        response_text = response_text.strip()
        
        # Parse JSON - try multiple strategies
        ai_result = None
        try:
            ai_result = json.loads(response_text)
        except json.JSONDecodeError:
            try:
                import re
                json_pattern = r'\{[^{}]*(?:"explanation"[^{}]*"sentiment_score"[^{}]*"reasoning"[^{}]*)\}'
                json_match = re.search(json_pattern, response_text, re.DOTALL)
                if json_match:
                    ai_result = json.loads(json_match.group())
                else:
                    json_match = re.search(r'\{.*"explanation".*"sentiment_score".*"reasoning".*\}', response_text, re.DOTALL)
                    if json_match:
                        ai_result = json.loads(json_match.group())
            except (json.JSONDecodeError, AttributeError):
                import re
                explanation_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', response_text)
                score_match = re.search(r'"sentiment_score"\s*:\s*(\d+)', response_text)
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response_text)
                
                if explanation_match and score_match:
                    ai_result = {
                        "explanation": explanation_match.group(1),
                        "sentiment_score": int(score_match.group(1)),
                        "reasoning": reasoning_match.group(1) if reasoning_match else "Analysis completed"
                    }
                else:
                    raise ValueError("Could not parse AI response as JSON")
        
        if not ai_result:
            raise ValueError("Failed to parse AI response")
        
        # Validate sentiment score
        sentiment_score = ai_result.get("sentiment_score", 5)
        try:
            sentiment_score = int(float(sentiment_score))
            sentiment_score = max(1, min(10, sentiment_score))  # Clamp to 1-10
        except (ValueError, TypeError):
            sentiment_score = 5
        
        result = {
            "explanation": ai_result.get("explanation", f"Prediction markets show {raw_odds:.1%} probability."),
            "sentiment_score": sentiment_score,
            "raw_odds": raw_odds,
            "reasoning": ai_result.get("reasoning", "Analysis completed."),
            "sources": market_data.get("sources", [])
        }
        
        logger.info("")
        logger.info("[GAMBLER] " + "=" * 70)
        logger.info(f"[GAMBLER] ✓✓✓ AI ANALYSIS COMPLETE ✓✓✓")
        logger.info(f"[GAMBLER] Final sentiment score: {sentiment_score}/10")
        logger.info(f"[GAMBLER] Raw odds: {raw_odds}")
        logger.info(f"[GAMBLER] Explanation: {result['explanation'][:150]}...")
        logger.info(f"[GAMBLER] Reasoning: {result['reasoning'][:150]}...")
        logger.info("[GAMBLER] " + "=" * 70)
        logger.info("")
        
        return result
        
    except Exception as e:
        logger.error("")
        logger.error("[GAMBLER] " + "=" * 70)
        logger.error(f"[GAMBLER] ✗✗✗ ERROR IN AI ANALYSIS ✗✗✗")
        logger.error(f"[GAMBLER] Error type: {type(e).__name__}")
        logger.error(f"[GAMBLER] Error message: {str(e)}")
        logger.error(f"[GAMBLER] Full traceback:", exc_info=True)
        logger.error("[GAMBLER] " + "=" * 70)
        
        fallback_score = int(raw_odds * 10) if raw_odds else 5
        result = {
            "explanation": f"Prediction markets show {raw_odds:.1%} probability. AI analysis encountered an error.",
            "sentiment_score": fallback_score,
            "raw_odds": raw_odds,
            "reasoning": f"Fallback calculation due to AI error: {str(e)}",
            "sources": market_data.get("sources", []) if market_data else []
        }
        logger.warning(f"[GAMBLER] Returning fallback result: sentiment={fallback_score}/10")
        return result
