#!/bin/bash
# Script to run the dual-cycle integration test

echo "Starting dual-cycle integration test..."

# Check if Python venv exists and activate if it does
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set environment variables for testing
export CYCLE_INTERVAL=900  # 15 minutes for quicker testing (default 1800s/30min)
export FETCH_INTERVAL=300  # 5 minutes (default is usually higher)
export VPH_THRESHOLD=100   # Lower threshold for testing
export REDIS_TEST_MODE=true
export TEST_MODE=true
export THROTTLE_MINUTES=15  # Reduce throttling for testing (default 60min)

echo "Environment variables set:"
echo "CYCLE_INTERVAL=$CYCLE_INTERVAL"
echo "FETCH_INTERVAL=$FETCH_INTERVAL"
echo "VPH_THRESHOLD=$VPH_THRESHOLD"
echo "THROTTLE_MINUTES=$THROTTLE_MINUTES"

# Run the integration test
echo "Running tests/test_dual_cycle_integration.py..."
python tests/test_dual_cycle_integration.py

# Capture exit code
EXIT_CODE=$?

# Check if Python venv was activated
if [ -d "venv" ]; then
    echo "Deactivating virtual environment..."
    deactivate
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "Test completed successfully!"
else
    echo "Test failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE