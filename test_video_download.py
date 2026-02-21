#!/usr/bin/env python3
"""
Test script to verify YouTube video download is working.
"""

import logging
import tempfile
import os
from agents.video_gossip import _download_video_audio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_video_download():
    """Test downloading a short NVDA-related YouTube video."""
    # Use the same video that was found in the search
    test_video_url = "https://www.youtube.com/watch?v=gjQyaFHUD3Q"
    test_video_title = "Massive News for Nvidia Stock Investors | NVDA Stock Analysis"
    
    print("\n" + "="*80)
    print(f"Testing YouTube Video Download")
    print("="*80 + "\n")
    print(f"Video: {test_video_title}")
    print(f"URL: {test_video_url}")
    print("\nAttempting download...")
    print("="*80 + "\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_file = _download_video_audio(test_video_url, temp_dir)
        
        print("\n" + "="*80)
        if audio_file and os.path.exists(audio_file):
            file_size = os.path.getsize(audio_file)
            print(f"✓ SUCCESS: Video download and audio extraction worked!")
            print(f"\nAudio file: {audio_file}")
            print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            print("\nThe video_gossip agent should now work properly!")
            return True
        else:
            print(f"✗ FAILED: Could not download video")
            print("\nPossible reasons:")
            print("1. YouTube is still blocking downloads (may need to wait)")
            print("2. Your IP might be temporarily rate-limited by YouTube")
            print("3. Network/firewall issues")
            print("\nRecommendations:")
            print("- Try again in a few minutes")
            print("- Make sure you're logged into YouTube in Chrome browser")
            print("- Check if you can watch the video normally in your browser")
            return False
    
    print("="*80 + "\n")

if __name__ == "__main__":
    success = test_video_download()
    exit(0 if success else 1)
