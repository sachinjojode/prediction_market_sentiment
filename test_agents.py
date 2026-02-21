"""Test script for individual agents

Tests each agent step by step to verify they're working correctly.
"""

import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("MarketMinds Agent Testing")
print("=" * 60)
print()

# Test ticker/company
TEST_INPUT = "NVDA"  # Can be ticker or company name
print(f"Testing with: {TEST_INPUT}\n")

# Test 1: Company Name Converter
print("=" * 60)
print("TEST 1: Company Name to Ticker Converter")
print("=" * 60)
try:
    from utils.ticker_converter import company_name_to_ticker
    
    test_cases = [
        "NVDA",
        "Nvidia",
        "Apple",
        "AAPL",
        "Microsoft"
    ]
    
    for test in test_cases:
        result = company_name_to_ticker(test)
        print(f"  '{test}' → '{result}'")
    
    ticker = company_name_to_ticker(TEST_INPUT)
    print(f"\n✓ Converter working. '{TEST_INPUT}' → '{ticker}'")
except Exception as e:
    print(f"✗ Converter failed: {e}")
    sys.exit(1)

print()

# Test 2: Gambler Agent
print("=" * 60)
print("TEST 2: Gambler Agent (Polymarket + AI Analysis)")
print("=" * 60)
try:
    from agents.gambler import get_polymarket_probability
    
    print(f"Fetching Polymarket data for {ticker}...")
    gambler_result = get_polymarket_probability(ticker, "Nvidia")
    
    print(f"\n✓ Gambler Agent Results:")
    print(f"  - Sentiment Score: {gambler_result.get('sentiment_score', 'N/A')}/10")
    print(f"  - Raw Odds: {gambler_result.get('raw_odds', 'N/A')}")
    print(f"  - Explanation: {gambler_result.get('explanation', 'N/A')[:100]}...")
    print(f"  - Reasoning: {gambler_result.get('reasoning', 'N/A')[:80]}...")
    
    if gambler_result.get('sentiment_score') is None:
        print("\n⚠ Warning: Sentiment score is None")
    
except Exception as e:
    print(f"✗ Gambler Agent failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 3: Gossip Agent
print("=" * 60)
print("TEST 3: Gossip Agent (RSS + AI Summarization)")
print("=" * 60)
try:
    from agents.gossip import get_news_sentiment
    
    print(f"Fetching news articles for {ticker}...")
    gossip_result = get_news_sentiment(ticker, "Nvidia")
    
    print(f"\n✓ Gossip Agent Results:")
    print(f"  - Sentiment Score: {gossip_result.get('sentiment_score', 'N/A')}/10")
    print(f"  - Articles Found: {gossip_result.get('articles_count', 0)}")
    summary = gossip_result.get('summary', 'N/A')
    print(f"  - Summary (first 200 chars): {summary[:200]}...")
    print(f"  - Reasoning: {gossip_result.get('reasoning', 'N/A')[:80]}...")
    
    if gossip_result.get('articles_count', 0) == 0:
        print("\n⚠ Warning: No articles found")
    
except Exception as e:
    print(f"✗ Gossip Agent failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Judge Agent
print("=" * 60)
print("TEST 4: Judge Agent (Decision Making)")
print("=" * 60)
try:
    from agents.judge import decide_trade
    
    print("Analyzing with Judge agent using results from Gambler and Gossip...")
    judge_result = decide_trade(gambler_result, gossip_result, ticker, "Nvidia")
    
    print(f"\n✓ Judge Agent Results:")
    print(f"  - Decision: {judge_result.get('decision', 'N/A')}")
    print(f"  - Confidence: {judge_result.get('confidence', 'N/A')}")
    print(f"  - Key Factors: {', '.join(judge_result.get('key_factors', []))}")
    explanation = judge_result.get('explanation', 'N/A')
    print(f"  - Explanation (first 200 chars): {explanation[:200]}...")
    
except Exception as e:
    print(f"✗ Judge Agent failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Summary
print("=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("✓ All agents tested successfully!")
print()
print("Results Summary:")
print(f"  Gambler Sentiment: {gambler_result.get('sentiment_score', 'N/A')}/10")
print(f"  Gossip Sentiment: {gossip_result.get('sentiment_score', 'N/A')}/10")
print(f"  Final Decision: {judge_result.get('decision', 'N/A')} ({judge_result.get('confidence', 'N/A')} confidence)")
print()
print("=" * 60)
