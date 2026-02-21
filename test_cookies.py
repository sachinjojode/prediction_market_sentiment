#!/usr/bin/env python3
"""
Helper script to test cookie extraction from Chrome.
This will help diagnose cookie issues.
"""

import logging
import yt_dlp

logging.basicConfig(level=logging.INFO)

def test_cookie_extraction():
    """Test if we can extract cookies from Chrome."""
    print("\n" + "="*80)
    print("Testing Cookie Extraction from Chrome")
    print("="*80 + "\n")
    
    ydl_opts = {
        'quiet': False,
        'verbose': True,
        'cookiesfrombrowser': ('chrome',),
        'extract_flat': 'in_playlist',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("✓ Successfully initialized yt-dlp with Chrome cookies")
            
            # Try to extract info from a simple search
            result = ydl.extract_info("ytsearch1:NVDA", download=False)
            entries = result.get('entries', [])
            
            if entries:
                print(f"✓ Successfully fetched video info with cookies!")
                print(f"  Found {len(entries)} video(s)")
                print(f"  First video: {entries[0].get('title', 'Unknown')}")
                return True
            else:
                print("✗ No results found")
                return False
                
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {str(e)[:200]}")
        return False
    
    print("="*80 + "\n")

if __name__ == "__main__":
    success = test_cookie_extraction()
    exit(0 if success else 1)
