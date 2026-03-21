#!/bin/bash
# Run a single trading cycle via Claude Code.
#
# This is what the cron job calls. It invokes Claude Code in headless mode
# with the trading prompt, allowing it to use Bash to call the CLI tools.
#
# Usage:
#   ./hybrid/run_cycle.sh           # Normal cycle
#   ./hybrid/run_cycle.sh --force   # Run even outside market hours (testing)

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/hybrid/trade_logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
CYCLE_LOG="$LOG_DIR/cycle_${TIMESTAMP}.log"

# Check if market hours (skip on weekends/off-hours unless --force)
if [ "$1" != "--force" ]; then
    MARKET_CHECK=$(python3 -m hybrid.cli daily-state 2>/dev/null || echo '{"is_market_hours": false}')
    IS_OPEN=$(echo "$MARKET_CHECK" | python3 -c "import sys,json; print(json.load(sys.stdin).get('is_market_hours', False))" 2>/dev/null || echo "False")

    if [ "$IS_OPEN" = "False" ]; then
        echo "[$(date)] Market closed — skipping cycle" >> "$LOG_DIR/cron.log"
        exit 0
    fi
fi

echo "[$(date)] Starting trading cycle" >> "$LOG_DIR/cron.log"

# Read the trading prompt
PROMPT=$(cat "$PROJECT_DIR/hybrid/trading_prompt.md")

# Run Claude Code in headless mode
# --allowedTools: Only allow Bash commands that run our CLI + alerts
# --max-turns: Limit how many tool calls Claude can make per cycle
claude -p "$PROMPT" \
    --allowedTools "Bash(python3 -m hybrid.cli *),Bash(python3 -c *from hybrid*),Read" \
    --model sonnet \
    --max-turns 25 \
    --output-format json \
    2>>"$CYCLE_LOG" | tee -a "$CYCLE_LOG" | \
    python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    result = data.get('result', '')
    cost = data.get('cost_usd', 0)
    turns = data.get('num_turns', 0)
    print(f'  Result: {result[:200]}...' if len(result) > 200 else f'  Result: {result}')
    print(f'  Turns: {turns} | Cost: \${cost:.4f}')
except:
    pass
" >> "$LOG_DIR/cron.log" 2>&1

echo "[$(date)] Cycle complete" >> "$LOG_DIR/cron.log"
