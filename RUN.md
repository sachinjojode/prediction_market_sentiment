# How to Run MarketMinds

## Quick Start

### 1. Activate Virtual Environment
```bash
cd /Users/sachinjojode/Desktop/Master/01_Personal/Projects/prediction_market_sentiment
source venv/bin/activate
```

### 2. Start the Server
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open in Browser
Navigate to: **http://localhost:8000**

### 4. Use the Web Interface
1. Enter a ticker symbol (e.g., `NVDA`, `AAPL`, `TSLA`) or company name (e.g., `Nvidia`, `Apple`)
2. Click "Analyze"
3. Watch the progress as each agent runs:
   - **Gambler Agent**: Fetches Polymarket prediction data
   - **Gossip Agent**: Fetches news articles from Google News RSS
   - **Judge Agent**: Analyzes both and makes a decision (BUY/SELL/HOLD)
   - **Broadcaster Agent**: Generates video report (mock for now)

## API Endpoints

### Start Analysis
```bash
POST http://localhost:8000/api/analyze
Content-Type: application/json

{
  "ticker": "NVDA"
}
```

### Check Job Status
```bash
GET http://localhost:8000/api/status/{job_id}
```

### Get Results
```bash
GET http://localhost:8000/api/result/{job_id}
```

## Testing Individual Agents

### Test Gambler Agent Only
```bash
python test_gambler_only.py
```

### Test All Agents
```bash
python test_agents.py
```

## Troubleshooting

### Port Already in Use
If port 8000 is busy, change it in `.env`:
```
PORT=8001
```

### Vertex AI Errors
Make sure:
1. `GCP_PROJECT_ID` is set in `.env`
2. `credentials.json` exists in project root
3. Vertex AI API is enabled in Google Cloud Console

### No Markets Found
The Polymarket API may not return markets for all tickers. This is an API limitation, not a code issue.
