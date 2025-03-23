#!/bin/bash
# Start the FastAPI server for the crypto trading pool application

# Check if Python virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
else
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    echo "Virtual environment created and dependencies installed"
fi

# Check for Redis
if command -v redis-cli >/dev/null 2>&1; then
    # Check if Redis is running
    if redis-cli ping >/dev/null 2>&1; then
        echo "Redis is running"
    else
        echo "Starting Redis server..."
        redis-server --daemonize yes
        sleep 1
        if redis-cli ping >/dev/null 2>&1; then
            echo "Redis server started"
        else
            echo "Warning: Failed to start Redis server, using local fallback"
        fi
    fi
else
    echo "Warning: Redis not found, using local fallback"
fi

# Start the FastAPI server
echo "Starting the FastAPI server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Note: Press CTRL+C to stop the server