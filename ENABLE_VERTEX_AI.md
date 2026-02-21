# How to Enable Vertex AI API for MarketMinds

## Current Status
✅ Vertex AI API is initialized successfully
✅ Code updated to use `gemini-2.5-flash-lite` model
✅ REST API fallback implemented

## Two Ways to Use Gemini Models

### Option 1: Service Account (Current Setup)
- Uses `GOOGLE_APPLICATION_CREDENTIALS` pointing to `credentials.json`
- Tries Python SDK first, falls back to REST API if SDK fails

### Option 2: API Key (Alternative)
- Uses `GOOGLE_API_KEY` in `.env` file
- Uses REST API directly (more reliable for some setups)

## Step-by-Step Instructions

### Step 1: Enable Vertex AI API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select project: **marketminds-ai-488116**
3. Go to **"APIs & Services"** → **"Library"**
4. Search for: **"Vertex AI API"**
5. Click **"Enable"** (if not already enabled)

### Step 2: Get API Key (For REST API Fallback)
1. Go to **"APIs & Services"** → **"Credentials"**
2. Click **"Create Credentials"** → **"API Key"**
3. Copy the API key
4. Add it to your `.env` file:
   ```
   GOOGLE_API_KEY="your-api-key-here"
   ```

### Step 3: Verify Service Account Permissions (For SDK)
1. Go to **"IAM & Admin"** → **"Service Accounts"**
2. Find your service account (used in credentials.json)
3. Ensure it has: **"Vertex AI User"** role

### Step 4: Test
Run this command:
```bash
source venv/bin/activate
python test_agents.py
```

## How It Works

The system now:
1. **Tries Python SDK first** - Uses `GenerativeModel("gemini-2.5-flash-lite")`
2. **Falls back to REST API** - If SDK fails, uses REST API with API key
3. **Both methods use the same model** - `gemini-2.5-flash-lite`

## Model Name
- **Current**: `gemini-2.5-flash-lite`
- Updated in all three agents (Gambler, Gossip, Judge)

## Troubleshooting

### If SDK fails but REST API works:
- The system will automatically use REST API
- Check logs for `[REST API]` messages

### If both fail:
1. **Check API Key**: Make sure `GOOGLE_API_KEY` is set in `.env`
2. **Check Billing**: Ensure billing is enabled on your project
3. **Check Region**: Model should be available in `us-central1`

### To test REST API directly:
```bash
curl "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash-lite:generateContent?key=${GOOGLE_API_KEY}" \
-X POST \
-H "Content-Type: application/json" \
-d '{
  "contents": [
    {
      "role": "user",
      "parts": [{"text": "Say hello"}]
    }
  ]
}'
```

## Files Updated
- ✅ `agents/gambler.py` - Updated model name + REST API fallback
- ✅ `agents/gossip.py` - Updated model name + REST API fallback
- ✅ `agents/judge.py` - Updated model name + REST API fallback
- ✅ `utils/vertex_ai_client.py` - New REST API utility
