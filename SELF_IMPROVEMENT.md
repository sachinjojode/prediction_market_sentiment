# Self-Improvement System for MarketMinds

## Overview

The MarketMinds AI agents now feature a **self-improvement system** that learns from prediction accuracy over time and automatically adjusts agent weights to improve future predictions.

## How It Works

### 1. **Prediction Recording**
Every time you analyze a stock, the system records:
- Sentiment scores from each agent (Gambler, Gossip, Video)
- The Judge's final decision (BUY/SELL/HOLD)
- Confidence level
- Current stock price at time of prediction

### 2. **Outcome Tracking**
The system tracks actual outcomes by:
- Monitoring price changes over 7 days
- Comparing predictions to actual results:
  - **BUY** is correct if price rises >2%
  - **SELL** is correct if price falls <-2%
  - **HOLD** is correct if price stays within ±2%

### 3. **Performance Calculation**
For each agent, the system calculates:
- **Overall Accuracy**: % of correct predictions
- **Recent Accuracy**: % of correct predictions in last 10
- **Confidence Calibration**: Confidence levels when right vs. wrong
- **Weight Adjustment**: Multiplier based on accuracy (0.5x to 1.45x)

### 4. **Adaptive Weighting**
The Judge agent automatically applies learned weights:
- **High performers** (>70% accuracy) → Get up to 1.45x weight
- **Average performers** (50-70%) → Normal 1.0x weight  
- **Low performers** (<50%) → Reduced down to 0.5x weight

### 5. **Insight Generation**
The system generates actionable insights like:
- ⚠️ Low accuracy warnings
- 📉 Recent performance decline alerts
- ⚡ Overconfidence detection
- ✅ High performer recognition

## User Interface

### Performance Dashboard
Located at the top of the analysis page (expandable section):
- **Agent Cards**: Show accuracy, recent performance, and current weight
- **Insights Panel**: Displays AI-generated recommendations
- **Total Predictions**: Tracks how many analyses have been made

### In-Analysis Integration
When the Judge makes decisions, it receives:
- Current agent weights
- Weighted sentiment scores
- Historical performance context

This information is visible in the logs and used to make more informed decisions.

## Data Storage

All data is stored locally in JSON files:
- `data/agent_history.json` - All historical predictions
- `data/agent_performance.json` - Current performance metrics

These files are automatically created and updated. They're excluded from git via `.gitignore`.

## Key Features

### 🎯 **Accuracy-Based Weighting**
Agents that make better predictions get more influence in the Judge's decision.

### 📊 **Real-Time Updates**
Performance metrics update automatically as outcomes are determined.

### 🔄 **Continuous Learning**
The system improves over time as more predictions are made and validated.

### 🛡️ **Graceful Degradation**
If the system can't load, agents default to equal weights (1.0x).

### 📈 **Transparency**
Full visibility into agent performance, weights, and reasoning.

## Example Workflow

1. **Day 0**: You analyze NVDA
   - Gambler: 9/10 sentiment
   - Gossip: 8/10 sentiment
   - Video: 7/10 sentiment
   - Judge decides: BUY (high confidence)
   - System records prediction at current price: $189.82

2. **Day 7**: System checks outcome
   - New price: $195.50 (+3%)
   - Prediction was correct!
   - All agents get credit for the accurate call
   - Accuracy metrics update

3. **Next Analysis**: 
   - If Gambler has been consistently accurate → Gets 1.2x weight
   - If Video has been less accurate → Gets 0.8x weight
   - Judge now weighs Gambler's opinion more heavily

## Performance Metrics Explained

### **Accuracy**
Overall percentage of correct predictions across all time.

### **Recent Accuracy**  
Percentage of correct predictions in the last 10 analyses. Helps detect if an agent's performance is declining.

### **Weight Adjustment**
Current multiplier applied to this agent's sentiment score:
- `1.45x` = Very High Trust (>85% accuracy)
- `1.20x` = High Trust (70-85% accuracy)
- `1.00x` = Normal Trust (50-70% accuracy)
- `0.80x` = Reduced Trust (40-50% accuracy)
- `0.50x` = Low Trust (<40% accuracy)

### **Confidence Calibration**
Measures if an agent is overconfident (high confidence when wrong) or well-calibrated (high confidence when right).

## Technical Implementation

### Core Components

1. **`agents/self_improvement.py`**
   - Main system logic
   - Prediction recording
   - Performance calculation
   - Weight adjustment algorithms

2. **`agents/judge.py`** (Enhanced)
   - Fetches agent weights
   - Applies weighted scoring
   - Includes weight context in AI prompt

3. **`main.py`** (Enhanced)
   - Records predictions after analysis
   - Provides `/api/performance` endpoint
   - Integrates with chart data for price tracking

4. **`templates/index.html`** (Enhanced)
   - Performance dashboard UI
   - Real-time metric display
   - Insight visualization

### API Endpoints

**GET `/api/performance`**
Returns current performance metrics and insights:
```json
{
  "performance": {
    "gambler": {
      "accuracy": "75.0%",
      "recent_accuracy": "80.0%",
      "total_predictions": 20,
      "weight_adjustment": "1.15x"
    },
    ...
  },
  "insights": [
    "✅ GAMBLER: Consistently high accuracy (75.0%). Good performance!"
  ],
  "total_predictions": 60
}
```

## Best Practices

### For Optimal Learning
1. **Make regular predictions** - System needs data to learn
2. **Analyze diverse stocks** - Tests agents across conditions
3. **Check back after 7 days** - Outcomes are evaluated then
4. **Review insights** - Follow AI recommendations to improve prompts

### Interpreting Results
- **Early predictions (<10 total)**: Weights may be unstable
- **Established history (>50 predictions)**: Weights are reliable
- **Recent decline**: May indicate changing market conditions
- **High variance**: Agent may be inconsistent

## Limitations

- Requires 7 days to evaluate outcomes
- Needs minimum 5 predictions per agent for stable weights
- Price-based evaluation may not capture all nuances
- Historical performance doesn't guarantee future accuracy

## Future Enhancements

Potential improvements:
- Multiple time horizons (1 day, 3 days, 30 days)
- Sector-specific weighting
- Market condition awareness (bull/bear markets)
- Individual ticker performance tracking
- A/B testing different prompts
- Reinforcement learning integration

## Troubleshooting

### "No performance data yet"
- System is new, make some predictions first
- Check that `data/` directory exists
- Verify write permissions

### Weights seem wrong
- May need more historical data
- Check that outcomes are being updated
- Review logs for errors in performance calculation

### Dashboard not loading
- Check browser console for errors
- Verify `/api/performance` endpoint is accessible
- Ensure FastAPI server is running with latest code

## Summary

The self-improvement system makes MarketMinds **truly intelligent** by:
- 📝 Learning from every prediction
- 🎯 Adapting to agent performance
- 📊 Providing transparency and insights
- 🚀 Continuously improving accuracy

This creates a **feedback loop** where the system gets better the more you use it!
