#!/usr/bin/env python3
"""
Quick test script for video_gossip YouTube search functionality.
Run this to verify YouTube search is working before running the full analysis.
"""

import logging
from agents.video_gossip import _yt_search

# Set up logging to see detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_youtube_search():
    """Test YouTube search with a simple query."""
    print("\n" + "="*80)
    print("Testing YouTube Search for 'NVDA latest news'")
    print("="*80 + "\n")
    
    results = _yt_search("NVDA latest news", max_results=5)
    
    print(f"\n{'='*80}")
    print(f"RESULTS: Found {len(results)} videos")
    print("="*80 + "\n")
    
    if results:
        print("✓ SUCCESS: YouTube search is working!\n")
        print("Sample videos found:")
        for i, video in enumerate(results[:3], 1):
            title = video.get('title', 'Unknown')
            channel = video.get('channel', video.get('uploader', 'Unknown'))
            duration = video.get('duration', 0)
            print(f"\n{i}. {title}")
            print(f"   Channel: {channel}")
            print(f"   Duration: {duration/60:.1f} minutes")
    else:
        print("✗ FAILED: No videos found")
        print("\nPossible reasons:")
        print("1. YouTube is blocking your IP (try waiting a few minutes)")
        print("2. Browser cookies not accessible (check Chrome is installed)")
        print("3. Network/firewall issues")
        print("\nCheck the detailed logs above for specific error messages.")
    
    print("\n" + "="*80 + "\n")
    return len(results) > 0

if __name__ == "__main__":
    success = test_youtube_search()
    exit(0 if success else 1)
