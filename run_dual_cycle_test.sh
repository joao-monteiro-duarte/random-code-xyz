#!/bin/bash
# Run the dual-cycle test with proper environment setup

# Ensure the virtual environment is activated if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
  echo "Virtual environment activated"
fi

# Set test mode environment variables
export REDIS_TEST_MODE=true
export LOG_PATH="dual_cycle_test_$(date +%Y%m%d_%H%M%S).log"

# Create test directory for output
mkdir -p test_results

# Run the test with timestamp
echo "Starting dual-cycle test at $(date)"
echo "Results will be logged to $LOG_PATH"

# Run the test script
python test_dual_cycle.py

# Print completion
echo "Test completed at $(date)"
echo "Test logs available at $LOG_PATH"