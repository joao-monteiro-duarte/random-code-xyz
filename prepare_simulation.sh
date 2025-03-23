#!/bin/bash
# Script to prepare for the 2-week simulation starting March 22, 2025

echo "======================================="
echo "Preparing for 2-week simulation"
echo "======================================="

# Kill any running processes
echo "Stopping any running processes..."
pkill -f uvicorn
pkill -f python

# Clear Redis cache (if needed)
echo "Clearing Redis cache..."
redis-cli FLUSHALL

# Install any missing dependencies
echo "Installing dependencies..."
pip install --upgrade -r requirements.txt

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found. Installing..."
    sudo apt-get update && sudo apt-get install -y ffmpeg
else
    echo "ffmpeg is already installed: $(ffmpeg -version | head -n 1)"
fi

# Create data directory if it doesn't exist
echo "Creating data directory..."
mkdir -p data

# Initialize portfolio with $10,000 in cash
echo "Initializing portfolio with $10,000 in cash..."
cat > data/portfolio.json << EOF
{
  "timestamp": "$(date -Iseconds)",
  "total_value": 10000.0,
  "unallocated_value": 10000.0,
  "coins": {}
}
EOF

# Initialize trade history
echo "Initializing trade history..."
cat > data/trade_history.json << EOF
[]
EOF

# Create timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Test the transcript service
echo "Testing transcript service..."
TRANSCRIPT_LOG="transcript_test_${TIMESTAMP}.log"
python test_transcripts_and_sentiment.py > "$TRANSCRIPT_LOG" 2>&1
if [ $? -eq 0 ]; then
    echo "Transcript service test successful!"
else
    echo "Transcript service test failed. See $TRANSCRIPT_LOG for details."
fi

# Test service initialization (circular dependency fix)
echo "Testing service initialization..."
INIT_LOG="initialization_test_${TIMESTAMP}.log"
python test_initialization.py > "$INIT_LOG" 2>&1
if [ $? -eq 0 ]; then
    echo "Service initialization test successful!"
else
    echo "Service initialization test failed. See $INIT_LOG for details."
fi

# Check the configuration file
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "IMPORTANT: Please edit .env to add your API keys before starting the simulation."
fi

# Print status
echo -e "\nSystem prepared for simulation!"
echo "To start the simulation, run: ./run.sh --api"
echo "The simulation will run from March 22 to April 5, 2025"
echo "======================================="