# Troubleshooting Guide

## Frontend Stuck on Loading

### Issue: Clicked "Analyze" but everything is just loading

**Solution:**

1. **Check Browser Console:**
   - Open Developer Tools (F12 or Cmd+Option+I)
   - Go to Console tab
   - Look for JavaScript errors (red messages)

2. **Check Server Logs:**
   - Look at the terminal where you ran `python main.py`
   - You should see logs like:
     ```
     INFO: Starting analysis job {job_id} for input: 'NVDA'
     INFO: [job_id] Input received: 'NVDA'
     INFO: [job_id] ===== STEP 1: GAMBLER AGENT =====
     ```

3. **Restart the Server:**
   - Press `Ctrl+C` in the terminal to stop the server
   - Run `python main.py` again
   - Refresh the browser (Cmd+Shift+R or Ctrl+Shift+R to hard refresh)

4. **Check Network Tab:**
   - Open Developer Tools → Network tab
   - Click "Analyze" again
   - Check if `/analyze` request is being sent
   - Check the response status (should be 200)

## Common Issues

### Issue: "Failed to start analysis"
- **Cause:** JavaScript error or server not running
- **Fix:** Check browser console and restart server

### Issue: Polling keeps running but no progress
- **Cause:** Agent is stuck or taking very long
- **Fix:** Check server logs to see which agent is running

### Issue: "No markets found" for Gambler
- **Cause:** Polymarket API limitation (markets may not be indexed)
- **Fix:** This is expected - the system will still work with news analysis

### Issue: Vertex AI errors
- **Cause:** API not enabled or credentials missing
- **Fix:** Check `ENABLE_VERTEX_AI.md` for setup instructions

## Debug Mode

To see more detailed logs, the server already logs at INFO level. Check the terminal output for:
- `[job_id]` - Each analysis job has a unique ID
- `[GAMBLER]` - Gambler agent logs
- `[GOSSIP]` - Gossip agent logs  
- `[JUDGE]` - Judge agent logs

## Test Endpoints Directly

```bash
# Test analyze endpoint
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA"}'

# Check status (replace {job_id} with actual ID from above)
curl http://localhost:8000/status/{job_id}
```
