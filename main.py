"""MarketMinds - Multi-Agent Sentiment Trader

FastAPI web server that orchestrates the multi-agent system
for market sentiment analysis and trade decision making.
"""

import os
import uuid
import logging
import asyncio
from typing import Dict, Optional
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from agents.gambler import get_polymarket_sentiment
from agents.gossip import get_news_sentiment
from agents.video_gossip import get_video_sentiment
from agents.judge import decide_trade
from media.broadcaster import generate_video
from utils.ticker_converter import company_name_to_ticker
import requests
from dateutil import parser as date_parser

# Load environment variables
load_dotenv()

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set detailed logging for all agents
logging.getLogger('agents.gambler').setLevel(logging.INFO)
logging.getLogger('agents.gossip').setLevel(logging.INFO)
logging.getLogger('agents.judge').setLevel(logging.INFO)
logging.getLogger('utils.vertex_ai_client').setLevel(logging.INFO)

# Initialize FastAPI app
app = FastAPI(title="MarketMinds - Multi-Agent Sentiment Trader")

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory job store for progress tracking
jobs: Dict[str, Dict] = {}

# Hardcoded company descriptions (replacing Airia API)
COMPANY_DESCRIPTIONS = {
    "PLTR": "Palantir Technologies Inc. is a software company that specializes in big data analytics. Founded in 2003, Palantir provides platforms for integrating, managing, and securing data. The company's software is used by government agencies, financial institutions, and large enterprises for data analysis, intelligence gathering, and operational decision-making. Palantir's flagship products include Palantir Gotham for government clients and Palantir Foundry for commercial enterprises.",
    "AAPL": "Apple Inc. is a multinational technology company that designs, develops, and sells consumer electronics, computer software, and online services. Founded in 1976, Apple is one of the world's largest technology companies by revenue. The company's hardware products include the iPhone smartphone, iPad tablet, Mac personal computers, Apple Watch smartwatch, AirPods earbuds, and Apple TV. Apple's software includes the iOS, iPadOS, macOS, watchOS, and tvOS operating systems, and the iTunes media player.",
    "NVDA": "NVIDIA Corporation is a technology company that designs graphics processing units (GPUs) for gaming, professional visualization, data centers, and automotive markets. Founded in 1993, NVIDIA is a leader in artificial intelligence computing and has expanded beyond gaming to become a key player in AI, machine learning, autonomous vehicles, and data center solutions. The company's GPUs are widely used for AI training and inference, cryptocurrency mining, and high-performance computing applications.",
    "TSM": "Taiwan Semiconductor Manufacturing Company (TSMC) is the world's largest dedicated independent semiconductor foundry. Founded in 1987, TSMC manufactures chips for other companies rather than designing its own. The company produces advanced semiconductor chips used in smartphones, computers, servers, automotive electronics, and IoT devices. TSMC is a critical supplier to major technology companies including Apple, AMD, NVIDIA, and Qualcomm, and is known for its leading-edge manufacturing processes and production capacity."
}


async def get_company_description(ticker: str, company_name: str = None) -> str:
    """
    Get company description from hardcoded dictionary.
    
    Args:
        ticker: Stock ticker symbol
        company_name: Optional company name (not used, kept for compatibility)
    
    Returns:
        Company description string, or generic description if ticker not found
    """
    logger.info(f"[COMPANY_DESC] ===== get_company_description() called =====")
    logger.info(f"[COMPANY_DESC] Parameters: ticker='{ticker}', company_name='{company_name or None}'")
    
    ticker_upper = ticker.upper()
    
    if ticker_upper in COMPANY_DESCRIPTIONS:
        description = COMPANY_DESCRIPTIONS[ticker_upper]
        logger.info(f"[COMPANY_DESC] ✓ Found description for {ticker_upper} ({len(description)} chars)")
        logger.info(f"[COMPANY_DESC] Description preview: {description[:150]}...")
        return description
    else:
        # Fallback for unknown tickers
        fallback = f"{company_name or ticker} is a publicly traded company."
        logger.info(f"[COMPANY_DESC] ⚠ No description found for {ticker_upper}, using fallback")
        return fallback


async def run_analysis(input_text: str, job_id: str) -> Dict:
    """
    Orchestrate the multi-agent analysis workflow.
    
    Args:
        input_text: Company name or ticker symbol
        job_id: Unique job identifier
    
    Returns:
        Complete analysis results dictionary
    """
    logger.info("=" * 80)
    logger.info(f"[{job_id}] ===== STARTING ANALYSIS WORKFLOW =====")
    logger.info(f"[{job_id}] Input text: '{input_text}'")
    logger.info("=" * 80)
    
    try:
        # Convert company name to ticker if needed
        logger.info(f"[{job_id}] Step 0: Converting company name to ticker...")
        logger.info(f"[{job_id}] Calling company_name_to_ticker('{input_text}')")
        
        ticker = await asyncio.to_thread(company_name_to_ticker, input_text)
        logger.info(f"[{job_id}] Ticker conversion result: '{ticker}'")
        
        company_name = input_text if input_text.upper() != ticker else None
        logger.info(f"[{job_id}] Final values - Ticker: '{ticker}', Company: '{company_name or 'N/A'}'")
        
        # Step 0.5: Get company description
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] ===== STEP 0.5: COMPANY DESCRIPTION =====")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] Getting company description for ticker: '{ticker}'")
        logger.info(f"[{job_id}] About to call get_company_description(ticker='{ticker}', company_name='{company_name or None}')")
        
        try:
            company_description = await get_company_description(ticker, company_name)
            logger.info(f"[{job_id}] get_company_description() returned: {type(company_description).__name__}")
            logger.info(f"[{job_id}] Returned value length: {len(company_description) if company_description else 0}")
            
            if company_description:
                logger.info(f"[{job_id}] ✓ Company description retrieved ({len(company_description)} chars)")
                logger.info(f"[{job_id}] Description preview: {company_description[:150]}...")
            else:
                logger.warning(f"[{job_id}] ⚠ No company description available (empty string)")
                company_description = f"{company_name or ticker} is a publicly traded company."
        except Exception as e:
            logger.error(f"[{job_id}] ✗✗✗ ERROR in Step 0.5 (Company Description) ✗✗✗")
            logger.error(f"[{job_id}] Error type: {type(e).__name__}")
            logger.error(f"[{job_id}] Error message: {str(e)}")
            logger.error(f"[{job_id}] Full traceback:", exc_info=True)
            company_description = f"{company_name or ticker} is a publicly traded company."
        
        logger.info(f"[{job_id}] Step 0.5 complete. Final company_description length: {len(company_description)}")
        
        # Initialize job status
        logger.info(f"[{job_id}] Initializing job status in jobs store...")
        jobs[job_id] = {
            "status": "running",
            "progress": {
                "gambler": {"status": "pending", "data": None},
                "gossip": {"status": "pending", "data": None},
                "video_gossip": {"status": "pending", "data": None},
                "judge": {"status": "pending", "data": None},
                "broadcaster": {"status": "pending", "data": None}
            },
            "result": None,
            "error": None
        }
        logger.info(f"[{job_id}] Job status initialized successfully")
        
        # Step 1: Gambler Agent
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] ===== STEP 1: GAMBLER AGENT =====")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] Starting Gambler agent...")
        logger.info(f"[{job_id}] Parameters: ticker='{ticker}', company_name='{company_name or 'None'}'")
        
        jobs[job_id]["progress"]["gambler"]["status"] = "running"
        logger.info(f"[{job_id}] Updated job status: gambler.status = 'running'")
        
        try:
            logger.info(f"[{job_id}] Calling get_polymarket_sentiment() in thread...")
            start_time = asyncio.get_event_loop().time()
            
            gambler_result = await asyncio.to_thread(
                get_polymarket_sentiment, 
                ticker, 
                company_name
            )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(f"[{job_id}] Gambler agent returned after {elapsed:.2f} seconds")
            logger.info(f"[{job_id}] Gambler result keys: {list(gambler_result.keys())}")
            
            jobs[job_id]["progress"]["gambler"]["status"] = "completed"
            jobs[job_id]["progress"]["gambler"]["data"] = gambler_result
            logger.info(f"[{job_id}] Updated job status: gambler.status = 'completed'")
            
            logger.info("")
            logger.info(f"[{job_id}] ✓✓✓ GAMBLER AGENT COMPLETED ✓✓✓")
            logger.info(f"[{job_id}]   - Sentiment Score: {gambler_result.get('sentiment_score', 'N/A')}/10")
            logger.info(f"[{job_id}]   - Raw Odds: {gambler_result.get('raw_odds', 'N/A')}")
            logger.info(f"[{job_id}]   - Explanation: {gambler_result.get('explanation', '')[:100]}...")
            logger.info(f"[{job_id}]   - Reasoning: {gambler_result.get('reasoning', 'N/A')[:100]}...")
            
        except Exception as e:
            logger.error("")
            logger.error(f"[{job_id}] ✗✗✗ GAMBLER AGENT ERROR ✗✗✗")
            logger.error(f"[{job_id}] Error type: {type(e).__name__}")
            logger.error(f"[{job_id}] Error message: {str(e)}")
            logger.error(f"[{job_id}] Full traceback:", exc_info=True)
            
            jobs[job_id]["progress"]["gambler"]["status"] = "error"
            gambler_result = {
                "explanation": f"Error: {str(e)}",
                "sentiment_score": 5,
                "raw_odds": None,
                "reasoning": f"Error occurred: {type(e).__name__}"
            }
            jobs[job_id]["progress"]["gambler"]["data"] = gambler_result
            logger.info(f"[{job_id}] Set fallback gambler_result due to error")
        
        # Step 2: Gossip Agent
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] ===== STEP 2: GOSSIP AGENT =====")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] Starting Gossip agent...")
        logger.info(f"[{job_id}] Parameters: ticker='{ticker}', company_name='{company_name or 'None'}'")
        
        jobs[job_id]["progress"]["gossip"]["status"] = "running"
        logger.info(f"[{job_id}] Updated job status: gossip.status = 'running'")
        
        try:
            logger.info(f"[{job_id}] Calling get_news_sentiment() in thread...")
            start_time = asyncio.get_event_loop().time()
            
            gossip_result = await asyncio.to_thread(
                get_news_sentiment, 
                ticker, 
                company_name
            )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(f"[{job_id}] Gossip agent returned after {elapsed:.2f} seconds")
            logger.info(f"[{job_id}] Gossip result keys: {list(gossip_result.keys())}")
            
            jobs[job_id]["progress"]["gossip"]["status"] = "completed"
            jobs[job_id]["progress"]["gossip"]["data"] = gossip_result
            logger.info(f"[{job_id}] Updated job status: gossip.status = 'completed'")
            
            logger.info("")
            logger.info(f"[{job_id}] ✓✓✓ GOSSIP AGENT COMPLETED ✓✓✓")
            logger.info(f"[{job_id}]   - Sentiment Score: {gossip_result.get('sentiment_score', 'N/A')}/10")
            logger.info(f"[{job_id}]   - Articles Found: {gossip_result.get('articles_count', 0)}")
            logger.info(f"[{job_id}]   - Summary preview: {gossip_result.get('summary', '')[:100]}...")
            logger.info(f"[{job_id}]   - Reasoning: {gossip_result.get('reasoning', 'N/A')[:100]}...")
            
        except Exception as e:
            logger.error("")
            logger.error(f"[{job_id}] ✗✗✗ GOSSIP AGENT ERROR ✗✗✗")
            logger.error(f"[{job_id}] Error type: {type(e).__name__}")
            logger.error(f"[{job_id}] Error message: {str(e)}")
            logger.error(f"[{job_id}] Full traceback:", exc_info=True)
            
            jobs[job_id]["progress"]["gossip"]["status"] = "error"
            gossip_result = {
                "summary": f"Error: {str(e)}",
                "sentiment_score": 5,
                "articles_count": 0,
                "reasoning": f"Error occurred: {type(e).__name__}"
            }
            jobs[job_id]["progress"]["gossip"]["data"] = gossip_result
            logger.info(f"[{job_id}] Set fallback gossip_result due to error")
        
        # Step 2.5: Video Gossip Agent
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] ===== STEP 2.5: VIDEO GOSSIP AGENT =====")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] Starting Video Gossip agent...")
        logger.info(f"[{job_id}] Parameters: ticker='{ticker}', company_name='{company_name or 'None'}'")
        
        jobs[job_id]["progress"]["video_gossip"]["status"] = "running"
        logger.info(f"[{job_id}] Updated job status: video_gossip.status = 'running'")
        
        try:
            logger.info(f"[{job_id}] Calling get_video_sentiment() in thread...")
            start_time = asyncio.get_event_loop().time()
            
            video_gossip_result = await asyncio.to_thread(
                get_video_sentiment,
                ticker,
                company_name
            )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(f"[{job_id}] Video Gossip agent returned after {elapsed:.2f} seconds")
            logger.info(f"[{job_id}] Video Gossip result keys: {list(video_gossip_result.keys())}")
            
            jobs[job_id]["progress"]["video_gossip"]["status"] = "completed"
            jobs[job_id]["progress"]["video_gossip"]["data"] = video_gossip_result
            logger.info(f"[{job_id}] Updated job status: video_gossip.status = 'completed'")
            
            logger.info("")
            logger.info(f"[{job_id}] ✓✓✓ VIDEO GOSSIP AGENT COMPLETED ✓✓✓")
            logger.info(f"[{job_id}]   - Sentiment Score: {video_gossip_result.get('sentiment_score', 'N/A')}/10")
            logger.info(f"[{job_id}]   - Videos Analyzed: {video_gossip_result.get('videos_analyzed', 0)}")
            logger.info(f"[{job_id}]   - Summary preview: {video_gossip_result.get('summary', '')[:100]}...")
            logger.info(f"[{job_id}]   - Reasoning: {video_gossip_result.get('reasoning', 'N/A')[:100]}...")
            
        except Exception as e:
            logger.error("")
            logger.error(f"[{job_id}] ✗✗✗ VIDEO GOSSIP AGENT ERROR ✗✗✗")
            logger.error(f"[{job_id}] Error type: {type(e).__name__}")
            logger.error(f"[{job_id}] Error message: {str(e)}")
            logger.error(f"[{job_id}] Full traceback:", exc_info=True)
            
            jobs[job_id]["progress"]["video_gossip"]["status"] = "error"
            video_gossip_result = {
                "summary": f"Error: {str(e)}",
                "sentiment_score": 5,
                "videos_analyzed": 0,
                "reasoning": f"Error occurred: {type(e).__name__}",
                "sources": []
            }
            jobs[job_id]["progress"]["video_gossip"]["data"] = video_gossip_result
            logger.info(f"[{job_id}] Set fallback video_gossip_result due to error")
        
        # Step 3: Judge Agent
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] ===== STEP 3: JUDGE AGENT =====")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] Starting Judge agent...")
        logger.info(f"[{job_id}] Preparing inputs for Judge:")
        logger.info(f"[{job_id}]   - Gambler sentiment: {gambler_result.get('sentiment_score', 'N/A')}/10")
        logger.info(f"[{job_id}]   - Gossip sentiment: {gossip_result.get('sentiment_score', 'N/A')}/10")
        logger.info(f"[{job_id}]   - Video Gossip sentiment: {video_gossip_result.get('sentiment_score', 'N/A')}/10")
        logger.info(f"[{job_id}]   - Ticker: '{ticker}', Company: '{company_name or 'None'}'")
        
        jobs[job_id]["progress"]["judge"]["status"] = "running"
        logger.info(f"[{job_id}] Updated job status: judge.status = 'running'")
        
        try:
            logger.info(f"[{job_id}] Calling decide_trade() in thread...")
            start_time = asyncio.get_event_loop().time()
            
            verdict = await asyncio.to_thread(
                decide_trade,
                gambler_result,
                gossip_result,
                video_gossip_result,
                ticker,
                company_name
            )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(f"[{job_id}] Judge agent returned after {elapsed:.2f} seconds")
            logger.info(f"[{job_id}] Verdict keys: {list(verdict.keys())}")
            
            jobs[job_id]["progress"]["judge"]["status"] = "completed"
            jobs[job_id]["progress"]["judge"]["data"] = verdict
            logger.info(f"[{job_id}] Updated job status: judge.status = 'completed'")
            
            logger.info("")
            logger.info(f"[{job_id}] ✓✓✓ JUDGE AGENT COMPLETED ✓✓✓")
            logger.info(f"[{job_id}]   - Decision: {verdict.get('decision', 'N/A')}")
            logger.info(f"[{job_id}]   - Confidence: {verdict.get('confidence', 'N/A')}")
            logger.info(f"[{job_id}]   - Key Factors: {', '.join(verdict.get('key_factors', []))}")
            logger.info(f"[{job_id}]   - Explanation preview: {verdict.get('explanation', '')[:100]}...")
            
        except Exception as e:
            logger.error("")
            logger.error(f"[{job_id}] ✗✗✗ JUDGE AGENT ERROR ✗✗✗")
            logger.error(f"[{job_id}] Error type: {type(e).__name__}")
            logger.error(f"[{job_id}] Error message: {str(e)}")
            logger.error(f"[{job_id}] Full traceback:", exc_info=True)
            
            jobs[job_id]["progress"]["judge"]["status"] = "error"
            verdict = {
                "decision": "HOLD",
                "explanation": f"Error during analysis: {str(e)}",
                "confidence": "low",
                "key_factors": [f"Error occurred: {type(e).__name__}"]
            }
            jobs[job_id]["progress"]["judge"]["data"] = verdict
            logger.info(f"[{job_id}] Set fallback verdict due to error")
        
        # Step 4: Broadcaster Agent (optional - can skip for now)
        logger.info(f"[{job_id}] ===== STEP 4: BROADCASTER AGENT (SKIPPED) =====")
        logger.info(f"[{job_id}] Broadcaster agent skipped per requirements")
        jobs[job_id]["progress"]["broadcaster"]["status"] = "skipped"
        video_result = {"video_url": "", "status": "skipped"}
        jobs[job_id]["progress"]["broadcaster"]["data"] = video_result
        
        # Compile final result
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] ===== COMPILING FINAL RESULTS =====")
        logger.info("=" * 80)
        
        result = {
            "ticker": ticker,
            "company_name": company_name,
            "company_description": company_description,
            "timestamp": datetime.now().isoformat(),
            "gambler": gambler_result,
            "gossip": gossip_result,
            "video_gossip": video_gossip_result,
            "judge": verdict,
            "video": video_result
        }
        
        logger.info(f"[{job_id}] Final result structure:")
        logger.info(f"[{job_id}]   - Ticker: {result['ticker']}")
        logger.info(f"[{job_id}]   - Company: {result.get('company_name', 'N/A')}")
        logger.info(f"[{job_id}]   - Timestamp: {result['timestamp']}")
        logger.info(f"[{job_id}]   - Has gambler data: {bool(result['gambler'])}")
        logger.info(f"[{job_id}]   - Has gossip data: {bool(result['gossip'])}")
        logger.info(f"[{job_id}]   - Has video_gossip data: {bool(result['video_gossip'])}")
        logger.info(f"[{job_id}]   - Has judge data: {bool(result['judge'])}")
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] ===== ANALYSIS WORKFLOW COMPLETE =====")
        logger.info("=" * 80)
        logger.info(f"[{job_id}] Final Decision: {verdict.get('decision', 'N/A')}")
        logger.info(f"[{job_id}] Confidence: {verdict.get('confidence', 'N/A')}")
        logger.info(f"[{job_id}] Job status set to 'completed'")
        logger.info(f"[{job_id}] Result stored in jobs[{job_id}]['result']")
        logger.info("=" * 80)
        logger.info("")
        
        return result
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error(f"[{job_id}] ===== ANALYSIS WORKFLOW ERROR =====")
        logger.error("=" * 80)
        logger.error(f"[{job_id}] Error type: {type(e).__name__}")
        logger.error(f"[{job_id}] Error message: {str(e)}")
        logger.error(f"[{job_id}] Full traceback:", exc_info=True)
        
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        logger.error(f"[{job_id}] Job status set to 'error'")
        logger.error("=" * 80)
        raise


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze")
async def analyze_ticker(request: Request):
    """Start analysis workflow for a ticker or company name."""
    logger.info("=" * 80)
    logger.info("NEW ANALYSIS REQUEST RECEIVED")
    logger.info("=" * 80)
    try:
        logger.info("[API] Parsing request body...")
        data = await request.json()
        logger.info(f"[API] Request data received: {data}")
        
        input_text = data.get("ticker", "").strip()  # Can be ticker or company name
        logger.info(f"[API] Extracted input_text: '{input_text}'")
        
        if not input_text:
            logger.error("[API] Empty input_text - returning 400 error")
            raise HTTPException(status_code=400, detail="Ticker symbol or company name is required")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        logger.info(f"[API] Generated job_id: {job_id}")
        logger.info(f"[API] Starting analysis job {job_id} for input: '{input_text}'")
        
        # Start analysis in background
        logger.info(f"[API] Creating background task for job {job_id}")
        asyncio.create_task(run_analysis(input_text, job_id))
        logger.info(f"[API] Background task created successfully")
        
        response = {
            "job_id": job_id,
            "input": input_text,
            "status": "started"
        }
        logger.info(f"[API] Returning response: {response}")
        return JSONResponse(response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[API] ERROR starting analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get analysis progress status."""
    logger.debug(f"[API] Status request for job_id: {job_id}")
    
    if job_id not in jobs:
        logger.warning(f"[API] Job {job_id} not found in jobs store")
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    logger.debug(f"[API] Job {job_id} status: {job['status']}")
    
    return JSONResponse({
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "result": job.get("result"),
        "error": job.get("error")
    })


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Get final analysis result."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed yet. Status: {job['status']}"
        )
    
    return JSONResponse(job["result"])


@app.get("/chart/{ticker}")
async def get_chart_data(ticker: str):
    """Get historical stock price data using SerpApi Google Finance API."""
    logger.info(f"[API] Fetching chart data for ticker: {ticker}")
    
    try:
        # Get SerpApi API key from environment
        serpapi_key = os.getenv("SERPAPI_KEY")
        if not serpapi_key:
            logger.warning("[API] SERPAPI_KEY not set, using direct Google Finance scraping as fallback")
            # Fallback to direct scraping (will implement if needed)
            raise HTTPException(status_code=500, detail="SERPAPI_KEY not configured. Please add SERPAPI_KEY to your .env file")
        
        # Determine exchange (default to NASDAQ for most tech stocks, NYSE for others)
        # Common exchanges mapping
        exchange_map = {
            "NVDA": "NASDAQ",
            "AAPL": "NASDAQ",
            "MSFT": "NASDAQ",
            "GOOGL": "NASDAQ",
            "GOOG": "NASDAQ",
            "META": "NASDAQ",
            "TSLA": "NASDAQ",
            "AMZN": "NASDAQ",
            "NFLX": "NASDAQ",
            "AMD": "NASDAQ",
            "INTC": "NASDAQ",
        }
        
        exchange = exchange_map.get(ticker.upper(), "NASDAQ")  # Default to NASDAQ
        query = f"{ticker.upper()}:{exchange}"
        
        logger.info(f"[API] Using SerpApi Google Finance API for {query}")
        
        # SerpApi endpoint
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_finance",
            "q": query,
            "hl": "en",
            "window": "6M",  # 6 months of data
            "api_key": serpapi_key
        }
        
        logger.info(f"[API] Calling SerpApi: {url}")
        logger.info(f"[API] Query: {query}")
        
        response = requests.get(url, params=params, timeout=20)
        
        logger.info(f"[API] Response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"[API] SerpApi returned status {response.status_code}: {response.text[:200]}")
            raise HTTPException(status_code=response.status_code, detail=f"SerpApi returned {response.status_code}")
        
        data = response.json()
        
        logger.info(f"[API] Response keys: {list(data.keys())}")
        
        # Parse SerpApi response
        if "graph" not in data:
            logger.error(f"[API] No 'graph' field in response. Available keys: {list(data.keys())}")
            if "error" in data:
                logger.error(f"[API] SerpApi error: {data.get('error')}")
            raise HTTPException(status_code=404, detail=f"No graph data found for ticker {ticker}")
        
        graph_data = data["graph"]
        
        if not graph_data or len(graph_data) == 0:
            logger.error(f"[API] Graph data is empty")
            raise HTTPException(status_code=404, detail=f"No price data found for ticker {ticker}")
        
        logger.info(f"[API] Found {len(graph_data)} data points in graph")
        
        # Parse graph data
        dates = []
        prices = []
        volumes = []
        highs = []
        lows = []
        opens = []
        
        for point in graph_data:
            # Parse date
            date_str = point.get("date", "")
            if date_str:
                try:
                    # Parse date string like "Oct 17 2023, 04:00 PM UTC-04:00"
                    date_obj = date_parser.parse(date_str)
                    dates.append(date_obj.isoformat())
                except Exception as e:
                    logger.debug(f"[API] Error parsing date '{date_str}': {e}")
                    continue
            
            # Extract price
            price = point.get("price")
            if price is None:
                logger.debug(f"[API] Skipping point with no price: {point}")
                continue
            
            prices.append(float(price))
            
            # Extract other fields (may not always be present)
            volumes.append(point.get("volume", 0))
            # Graph data may not have high/low/open, use price as fallback
            highs.append(point.get("high", price))
            lows.append(point.get("low", price))
            opens.append(point.get("open", price))
        
        if not dates or not prices:
            logger.error(f"[API] No valid data points after parsing")
            raise HTTPException(status_code=404, detail=f"No valid price data for ticker {ticker}")
        
        # Ensure all arrays have same length
        min_len = min(len(dates), len(prices), len(volumes), len(highs), len(lows), len(opens))
        dates = dates[:min_len]
        prices = prices[:min_len]
        volumes = volumes[:min_len]
        highs = highs[:min_len]
        lows = lows[:min_len]
        opens = opens[:min_len]
        
        chart_data = {
            "dates": dates,
            "prices": prices,
            "volumes": volumes,
            "high": highs,
            "low": lows,
            "open": opens
        }
        
        logger.info(f"[API] Successfully fetched {len(dates)} data points from SerpApi")
        logger.info(f"[API] Price range: ${min(prices):.2f} - ${max(prices):.2f}")
        logger.info(f"[API] Latest price: ${prices[-1]:.2f}")
        logger.info(f"[API] Date range: {dates[0]} to {dates[-1]}")
        
        return JSONResponse(chart_data)
        
    except HTTPException:
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"[API] Network error fetching chart data: {e}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"[API] Error fetching chart data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching chart data: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
