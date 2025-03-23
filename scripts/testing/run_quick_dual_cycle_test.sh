#!/bin/bash
# Run the quick dual-cycle test with proper environment setup

# Ensure the virtual environment is activated if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
  echo "Virtual environment activated"
fi

# Set test mode environment variables
export REDIS_TEST_MODE=true
export LOG_PATH="quick_dual_cycle_test_$(date +%Y%m%d_%H%M%S).log"
export USE_MOCK_DATA=true  # Enable mock data for market data service
export CYCLE_DEBUG=true  # Enable detailed cycle debugging

# Run the test with timestamp
echo "Starting quick dual-cycle test at $(date)"
echo "Results will be logged to $LOG_PATH"
echo "Using mock market data to avoid API rate limits"

# Run the test script
python quick_dual_cycle_test.py

# Print completion
echo "Test completed at $(date)"
echo "Test logs available at $LOG_PATH"