#!/bin/bash
# Run the quick dual cycle test script with proper environment setup

# Set environment variables for testing
export REDIS_TEST_MODE=true
export OPENROUTER_API_KEY="test_key_123"  # Mock key for testing
export ANTHROPIC_API_KEY="test_key_456"   # Mock key for testing
export EXCHANGE_API_KEY="test_key_789"    # Mock key for testing
export EXCHANGE_API_SECRET="test_secret"  # Mock secret for testing

# Set cycle intervals for faster testing
export CYCLE_INTERVAL=600   # 10 minutes for testing (instead of 30)
export FETCH_INTERVAL=120   # 2 minutes for testing (instead of 5)

# Set log level
export LOG_LEVEL=INFO

echo "Running test_quick_dual_cycle.py with test environment..."
python3 test_quick_dual_cycle.py

echo "Test completed. Check the log file for results."
# Show most recent log file
ls -t test_quick_dual_cycle_*.log | head -1