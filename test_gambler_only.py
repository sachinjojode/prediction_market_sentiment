"""Test script for Gambler agent only - step by step testing"""

import sys
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 60)
print("GAMBLER AGENT TESTING - Step by Step")
print("=" * 60)
print()

# Test ticker
TEST_TICKER = "NVDA"
TEST_COMPANY = "Nvidia"

print(f"Testing with: {TEST_TICKER} ({TEST_COMPANY})\n")

# Test 1: Fetch Polymarket data
print("=" * 60)
print("STEP 1: Fetching Polymarket Data")
print("=" * 60)
try:
    from agents.gambler import _fetch_polymarket_data
    
    market_data = _fetch_polymarket_data(TEST_TICKER)
    
    if market_data:
        print(f"✓ Found {len(market_data.get('all_markets', []))} valid markets")
        print(f"\nAll Markets Found:")
        for i, market in enumerate(market_data.get('all_markets', [])[:10], 1):
            print(f"\n  Market {i}:")
            print(f"    Question: {market.get('question', 'N/A')[:80]}...")
            print(f"    Probability: {market.get('probability', 0):.2%}")
            print(f"    Volume: {market.get('volume', 0):,.0f}")
        
        print(f"\n  Best Market (highest volume):")
        best = market_data.get('best_market', {})
        print(f"    Question: {best.get('question', 'N/A')}")
        print(f"    Probability: {best.get('probability', 0):.2%}")
        print(f"    Volume: {best.get('volume', 0):,.0f}")
    else:
        print("✗ No markets found")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Error fetching data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 2: Full analysis
print("=" * 60)
print("STEP 2: Full Analysis with AI")
print("=" * 60)
try:
    from agents.gambler import get_polymarket_probability
    
    result = get_polymarket_probability(TEST_TICKER, TEST_COMPANY)
    
    print(f"\n✓ Analysis Complete:")
    print(f"  - Sentiment Score: {result.get('sentiment_score', 'N/A')}/10")
    print(f"  - Raw Odds: {result.get('raw_odds', 'N/A')}")
    if result.get('raw_odds'):
        print(f"    (as percentage: {result.get('raw_odds') * 100:.2f}%)")
    print(f"  - Explanation: {result.get('explanation', 'N/A')}")
    print(f"  - Reasoning: {result.get('reasoning', 'N/A')}")
    
except Exception as e:
    print(f"✗ Error in analysis: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
