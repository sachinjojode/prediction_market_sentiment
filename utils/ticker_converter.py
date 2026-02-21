"""Company Name to Ticker Symbol Converter

Converts company names to ticker symbols using a mapping dictionary.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Common company name to ticker mapping
COMPANY_TO_TICKER = {
    # Tech Companies
    "nvidia": "NVDA",
    "nvidia corporation": "NVDA",
    "apple": "AAPL",
    "apple inc": "AAPL",
    "microsoft": "MSFT",
    "microsoft corporation": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "alphabet inc": "GOOGL",
    "amazon": "AMZN",
    "amazon.com": "AMZN",
    "meta": "META",
    "meta platforms": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "tesla inc": "TSLA",
    "netflix": "NFLX",
    "netflix inc": "NFLX",
    "intel": "INTC",
    "intel corporation": "INTC",
    "amd": "AMD",
    "advanced micro devices": "AMD",
    "nvidia": "NVDA",
    
    # Financial
    "jpmorgan": "JPM",
    "jpmorgan chase": "JPM",
    "bank of america": "BAC",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "wells fargo": "WFC",
    
    # Other
    "disney": "DIS",
    "walt disney": "DIS",
    "coca cola": "KO",
    "coca-cola": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "walmart": "WMT",
    "walmart inc": "WMT",
    "target": "TGT",
    "target corporation": "TGT",
    "nike": "NKE",
    "nike inc": "NKE",
    "starbucks": "SBUX",
    "starbucks corporation": "SBUX",
    "boeing": "BA",
    "boeing company": "BA",
    "general electric": "GE",
    "ge": "GE",
    "ford": "F",
    "ford motor": "F",
    "general motors": "GM",
    "gm": "GM",
}


def company_name_to_ticker(company_name: str) -> Optional[str]:
    """
    Convert company name to ticker symbol.
    
    Args:
        company_name: Company name (e.g., "Nvidia", "Apple Inc")
    
    Returns:
        Ticker symbol if found, None otherwise. If input is already a ticker, returns it.
    """
    if not company_name:
        return None
    
    # Clean input
    cleaned = company_name.strip().lower()
    
    # If it's already a ticker (all caps, short), return as-is
    if cleaned.isupper() and len(cleaned) <= 5:
        logger.info(f"Input '{company_name}' appears to be a ticker, using as-is")
        return cleaned.upper()
    
    # Check mapping
    ticker = COMPANY_TO_TICKER.get(cleaned)
    
    if ticker:
        logger.info(f"Converted '{company_name}' → '{ticker}'")
        return ticker
    
    # Try partial matches
    for company, tick in COMPANY_TO_TICKER.items():
        if cleaned in company or company in cleaned:
            logger.info(f"Partial match: '{company_name}' → '{tick}'")
            return tick
    
    # If not found, assume it might be a ticker and return uppercase
    logger.warning(f"Company name '{company_name}' not found in mapping, using as ticker")
    return cleaned.upper()
