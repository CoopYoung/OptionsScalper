#!/bin/bash
# Setup cron job for the hybrid trader (Claude Code edition).
#
# Runs every 10 minutes during market hours (Mon-Fri, 9:30 AM - 4:00 PM ET).
# Also sends a daily summary at 4:15 PM ET.
#
# Usage:
#   chmod +x hybrid/setup_cron.sh
#   ./hybrid/setup_cron.sh          # Install cron
#   ./hybrid/setup_cron.sh remove   # Remove cron
#   ./hybrid/setup_cron.sh status   # Show current cron entries

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(which python3)"
CYCLE_CMD="$PROJECT_DIR/hybrid/run_cycle.sh"
SUMMARY_CMD="cd $PROJECT_DIR && $PYTHON -c \"from hybrid.alerts.telegram import notify_daily_summary; from hybrid.broker.alpaca import get_account, get_positions; from hybrid.risk.validator import get_daily_state; notify_daily_summary(get_daily_state(), get_positions(), get_account())\""
LOG_FILE="$PROJECT_DIR/hybrid/trade_logs/cron.log"

if [ "$1" = "remove" ]; then
    crontab -l 2>/dev/null | grep -v "hybrid" | crontab -
    echo "Hybrid trader cron removed."
    exit 0
fi

if [ "$1" = "status" ]; then
    echo "Current hybrid trader cron entries:"
    crontab -l 2>/dev/null | grep "hybrid" || echo "  (none)"
    exit 0
fi

# Verify claude CLI is available
if ! command -v claude &> /dev/null; then
    echo "ERROR: 'claude' CLI not found in PATH."
    echo "Install Claude Code first: https://code.claude.com"
    exit 1
fi

# Create the cron entries
# Trading cycles: every 10 min, Mon-Fri, 9:30 AM - 4:00 PM ET
CRON_ENTRY="*/10 9-15 * * 1-5 $CYCLE_CMD >> $LOG_FILE 2>&1"

# Daily summary: Mon-Fri at 4:15 PM ET
SUMMARY_ENTRY="15 16 * * 1-5 $SUMMARY_CMD >> $LOG_FILE 2>&1"

# Add to crontab (preserving existing non-hybrid entries)
(crontab -l 2>/dev/null | grep -v "hybrid"; echo "$CRON_ENTRY"; echo "$SUMMARY_ENTRY") | crontab -

echo "Hybrid trader cron installed:"
echo ""
echo "  Trading cycles:  Every 10 min, Mon-Fri 9:30 AM - 4:00 PM ET"
echo "  Daily summary:   Mon-Fri at 4:15 PM ET"
echo "  Cycle script:    $CYCLE_CMD"
echo "  Log file:        $LOG_FILE"
echo ""
echo "Commands:"
echo "  View cron:     crontab -l"
echo "  View logs:     tail -f $LOG_FILE"
echo "  Manual cycle:  ./hybrid/run_cycle.sh --force"
echo "  Remove cron:   $0 remove"
echo ""
echo "Cost: \$0 — uses your Claude Max subscription"
