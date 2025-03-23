#!/bin/bash
# Script to check the contents of the latest test log file

LATEST_LOG=$(ls -1t test_quick_dual_cycle_*.log | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No log file found."
    exit 1
fi

echo "Checking log file: $LATEST_LOG"
echo "File size: $(ls -lh $LATEST_LOG | awk '{print $5}')"

# Get the error message for the comprehensive cycle
echo "=== Comprehensive Cycle Error ==="
grep -A 10 "Error in comprehensive cycle" $LATEST_LOG || echo "No comprehensive cycle error found."

# Get specific error messages from the transcript
echo "=== Error Traceback for Video Update ==="
grep -A 10 "list.update_vph" $LATEST_LOG || echo "No update_vph error found."

# Check if accumulated_videos was set properly
echo "=== Accumulated Videos Check ==="
grep -A 5 "accumulated_videos" $LATEST_LOG

# Show overall test results
echo "=== Test Results ==="
grep -A 5 "Test Summary" $LATEST_LOG || echo "No test summary found."