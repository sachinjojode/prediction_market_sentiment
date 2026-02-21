# MarketMinds - Multi-Agent Sentiment Trader

An autonomous multi-agent system that analyzes market sentiment for assets using prediction markets, news feeds, AI-powered analysis, and automated video report generation.

## 🎯 Overview

MarketMinds combines multiple data sources and AI agents to make intelligent trading decisions:

1. **Gambler Agent**: Analyzes "real money" sentiment from Polymarket prediction markets
2. **Gossip Agent**: Scrapes current news headlines from Google News RSS
3. **Judge Agent**: Uses Google Vertex AI (Gemini) to compare money vs. hype and decide (BUY/SELL/HOLD)
4. **Broadcaster Agent**: Generates video reports using Flora API
5. **Web Interface**: Interactive FastAPI web app for ticker analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Google Cloud Platform account (for Vertex AI)
- (Optional) Flora API account (for video generation - mock fallback available)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd prediction_market_sentiment
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up credentials** (see [Credential Setup](#credential-setup) below)

5. **Run the application:**
   ```bash
   python main.py
   ```

6. **Open your browser:**
   Navigate to `http://localhost:8000`

## 📋 Credential Setup

### Step 1: Google Cloud Platform (Vertex AI) Setup

**Required for:** Judge Agent (Gemini AI)

1. **Create a Google Cloud Project:**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Click "Select a project" → "New Project"
   - Enter project name (e.g., "marketminds-ai")
   - Note your **Project ID** (different from project name)

2. **Enable Vertex AI API:**
   - In Cloud Console, go to "APIs & Services" → "Library"
   - Search for "Vertex AI API"
   - Click "Enable"

3. **Create Service Account:**
   - Go to "IAM & Admin" → "Service Accounts"
   - Click "Create Service Account"
   - Name: `marketminds-service`
   - Description: "Service account for MarketMinds AI agent"
   - Click "Create and Continue"

4. **Grant Permissions:**
   - Role: Select "Vertex AI User" (or "AI Platform Developer")
   - Click "Continue" → "Done"

5. **Generate Service Account Key:**
   - Click on the created service account
   - Go to "Keys" tab → "Add Key" → "Create new key"
   - Choose "JSON" format
   - Download the JSON file
   - **Rename it to `credentials.json`** and place it in the project root directory

6. **Set Environment Variable:**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add:
     ```
     GCP_PROJECT_ID="your-project-id-here"
     GOOGLE_APPLICATION_CREDENTIALS="./credentials.json"
     ```
   - Replace `your-project-id-here` with your actual Project ID from step 1

**Verification:**
```bash
# Test credentials (optional)
python -c "from google.cloud import aiplatform; print('Credentials OK')"
```

### Step 2: Flora API Setup

**Required for:** Broadcaster Agent (Video Generation)

**Option A: Using API Key**

1. **Sign up for Flora:**
   - Visit [Flora.ai](https://flora.ai) or their API documentation
   - Create an account or sign in

2. **Generate API Key:**
   - Navigate to API settings/dashboard
   - Generate a new API key
   - Copy the API key

3. **Add to `.env`:**
   ```
   FLORA_API_KEY="your-flora-api-key-here"
   ```

**Option B: Using Webhook URL**

1. **Get Webhook URL:**
   - If Flora provides webhook-based integration
   - Copy the webhook URL from your Flora dashboard

2. **Add to `.env`:**
   ```
   FLORA_WEBHOOK_URL="https://api.flora.ai/v1/webhook/run"
   ```

**Note:** You only need ONE of the above (API key OR webhook URL). The system will use whichever is provided. If both are missing, the mock fallback will activate automatically (demo will still work).

### Step 3: Polymarket Gamma API

**Required for:** Gambler Agent

**No credentials needed!** The Gamma API is public and read-only. No API key required.

**Verification:**
```bash
# Test endpoint (optional)
curl "https://gamma-api.polymarket.com/events?q=NVDA"
```

### Step 4: Google News RSS

**Required for:** Gossip Agent

**No credentials needed!** Google News RSS is publicly accessible. No API key required.

**Verification:**
```bash
# Test endpoint (optional)
curl "https://news.google.com/rss/search?q=NVDA+stock+news"
```

### Step 5: SerpApi Setup (for Stock Charts)

**Required for:** Interactive Stock Price Charts

1. **Sign up for SerpApi:**
   - Visit [SerpApi](https://serpapi.com/)
   - Create a free account (100 searches/month free tier available)
   - Go to your [API Dashboard](https://serpapi.com/dashboard)

2. **Get your API Key:**
   - Copy your API key from the dashboard
   - It will look like: `abc123def456...`

3. **Add to `.env`:**
   ```
   SERPAPI_KEY="your-serpapi-api-key-here"
   ```

**Note:** SerpApi provides a free tier with 100 searches per month. For production use, you may need a paid plan. The chart will not work without this API key.

**Verification:**
```bash
# Test API key (optional)
curl "https://serpapi.com/search.json?engine=google_finance&q=NVDA:NASDAQ&api_key=YOUR_KEY"
```

### Step 6: Complete `.env` File

Your final `.env` file should look like:

```bash
# Google Cloud Platform (Required for AI analysis)
GCP_PROJECT_ID="your-gcp-project-id"
GOOGLE_APPLICATION_CREDENTIALS="./credentials.json"

# SerpApi (Required for stock charts)
SERPAPI_KEY="your-serpapi-api-key"

# Flora API (Optional - for video generation)
FLORA_API_KEY="your-flora-api-key"
# OR
FLORA_WEBHOOK_URL="https://api.flora.ai/v1/webhook/run"

# Optional Configuration
DEFAULT_TICKER="NVDA"
PORT=8000
```

## 🔒 Security Best Practices

1. **Never commit credentials:**
   - The `.gitignore` file already excludes `.env` and `credentials.json`
   - Never push these files to version control

2. **File permissions (Linux/Mac):**
   ```bash
   chmod 600 credentials.json
   chmod 600 .env
   ```

3. **Verify `.gitignore`:**
   Make sure these are in your `.gitignore`:
   ```
   .env
   credentials.json
   ```

## 🧪 Testing Individual Agents

You can test each agent independently:

```python
# Test Gambler (no credentials needed)
from agents.gambler import get_polymarket_probability
print(get_polymarket_probability("NVDA"))

# Test Gossip (no credentials needed)
from agents.gossip import get_news_sentiment
print(get_news_sentiment("NVDA"))

# Test Judge (requires GCP credentials)
from agents.judge import decide_trade
result = decide_trade(0.65, ["Headline 1", "Headline 2"])
print(result)

# Test Broadcaster (requires Flora or will use mock)
from media.broadcaster import generate_video
result = generate_video("Test script", "BUY")
print(result)
```

## 🐛 Troubleshooting

### Google Cloud Issues

- **Error: "Could not automatically determine credentials"**
  - Solution: Verify `GOOGLE_APPLICATION_CREDENTIALS` path is correct and file exists
  - Check that the path in `.env` is relative to the project root or absolute

- **Error: "Permission denied" or "API not enabled"**
  - Solution: Enable Vertex AI API in Cloud Console
  - Verify service account has "Vertex AI User" role

- **Error: "Invalid project ID"**
  - Solution: Double-check `GCP_PROJECT_ID` matches your actual project ID (not project name)
  - Project ID is different from project name in Google Cloud

### Flora API Issues

- **Error: "Invalid API key" or "401 Unauthorized"**
  - Solution: Regenerate API key in Flora dashboard
  - Verify the key is correctly set in `.env` (no extra spaces or quotes)

- **Error: "Webhook URL not found"**
  - Solution: Verify webhook URL is correct
  - Or switch to API key method

- **Note:** If Flora fails, the system automatically uses mock video (demo will still work)

### General Issues

- **Module not found errors:**
  - Make sure virtual environment is activated
  - Run `pip install -r requirements.txt` again

- **Port already in use:**
  - Change `PORT` in `.env` to a different port (e.g., 8001)
  - Or stop the process using port 8000

## 📁 Project Structure

```
/market-minds
  ├── main.py                    # FastAPI web server + orchestration
  ├── agents/
  │   ├── __init__.py
  │   ├── gambler.py             # Polymarket Gamma API integration
  │   ├── gossip.py              # Google News RSS parser
  │   └── judge.py               # Vertex AI Gemini integration
  ├── media/
  │   ├── __init__.py
  │   └── broadcaster.py        # Flora API integration with mock fallback
  ├── static/                    # Web assets (CSS)
  │   └── style.css
  ├── templates/                 # HTML templates
  │   └── index.html
  ├── .env                       # Environment variables (create from .env.example)
  ├── credentials.json           # Google Cloud service account (user-provided)
  ├── requirements.txt           # Python dependencies
  └── README.md                  # This file
```

## 🎨 Usage

1. **Start the server:**
   ```bash
   python main.py
   ```

2. **Open the web interface:**
   - Navigate to `http://localhost:8000`
   - Enter a ticker symbol (e.g., NVDA, AAPL, TSLA)
   - Click "Analyze"

3. **Watch the progress:**
   - Real-time progress indicators show each agent's status
   - Results appear automatically when analysis completes

4. **View results:**
   - Trade decision (BUY/SELL/HOLD) with reasoning
   - Polymarket probability (if available)
   - News headlines
   - Generated video report

## 🔄 API Endpoints

- `GET /` - Web interface
- `POST /analyze` - Start analysis for a ticker
- `GET /status/{job_id}` - Get analysis progress
- `GET /result/{job_id}` - Get final analysis result

## 🚀 Deployment

For production deployment:

1. Set environment variables on your hosting platform
2. Use a production ASGI server (e.g., Gunicorn with Uvicorn workers)
3. Set up proper SSL/TLS certificates
4. Configure firewall rules
5. Use environment-specific `.env` files (never commit them)

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📝 License

This project is for educational and hackathon purposes.

## 🤝 Contributing

This is a hackathon project. Feel free to fork and modify for your needs!

## ⚠️ Disclaimer

This tool is for educational purposes only. It does not constitute financial advice. Always do your own research before making trading decisions.
