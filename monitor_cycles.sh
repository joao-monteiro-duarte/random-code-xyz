#!/bin/bash
# Monitor cycle timing from the logs in real-time

# Get log file path from command line or use default
LOG_FILE=$1

if [ -z "$LOG_FILE" ]; then
  # Find the most recent log file
  LOG_FILE=$(find . -name "app_*.log" -type f -print0 | xargs -0 ls -t | head -1)
  if [ -z "$LOG_FILE" ]; then
    echo "No log file found. Please provide a log file path."
    exit 1
  fi
  echo "Using most recent log file: $LOG_FILE"
fi

echo "Monitoring cycle timing from $LOG_FILE"
echo "Press Ctrl+C to exit"
echo "======================================="

# Monitor cycle timings
tail -f "$LOG_FILE" | grep -E --color=auto "Checking cycle timing|Quick decision cycle triggered|Auto-triggering trading cycle|Making quick decisions|Trading cycle completed|fetch_videos_loop|cycle_loop"