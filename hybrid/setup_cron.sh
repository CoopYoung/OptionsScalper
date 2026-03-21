#!/bin/bash
# Setup cron job for the hybrid trader.
#
# Runs every 10 minutes during market hours (Mon-Fri, 9:30 AM - 4:15 PM ET).
# Also sends a daily summary at 4:15 PM ET.
#
# Usage:
#   chmod +x hybrid/setup_cron.sh
#   ./hybrid/setup_cron.sh          # Install cron
#   ./hybrid/setup_cron.sh remove   # Remove cron

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$(which python3)"
HYBRID_CMD="cd $PROJECT_DIR && $PYTHON -m hybrid"
LOG_FILE="$PROJECT_DIR/hybrid/trade_logs/cron.log"

if [ "$1" = "remove" ]; then
    crontab -l 2>/dev/null | grep -v "hybrid" | crontab -
    echo "Hybrid trader cron removed."
    exit 0
fi

# Create the cron entries
CRON_ENTRY="*/10 9-15 * * 1-5 $HYBRID_CMD >> $LOG_FILE 2>&1"
SUMMARY_ENTRY="15 16 * * 1-5 $HYBRID_CMD --summary >> $LOG_FILE 2>&1"

# Add to crontab (preserving existing entries)
(crontab -l 2>/dev/null | grep -v "hybrid"; echo "$CRON_ENTRY"; echo "$SUMMARY_ENTRY") | crontab -

echo "Hybrid trader cron installed:"
echo "  Trading:  Every 10 min, Mon-Fri 9:30 AM - 3:59 PM ET"
echo "  Summary:  Mon-Fri at 4:15 PM ET"
echo ""
echo "View cron: crontab -l"
echo "View logs: tail -f $LOG_FILE"
echo "Remove:    $0 remove"
