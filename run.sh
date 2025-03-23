#!/bin/bash
# Run script for the crypto trading pool application

# Default options
TEST_MODE=true
DEBUG=false
VPH_THRESHOLD=500.0

# Load environment variables if .env file exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    # Only export valid key=value pairs, skip comments and empty lines
    while IFS='=' read -r key value; do
        # Skip lines starting with # or empty lines
        [[ "$key" =~ ^#.*$ ]] || [[ -z "$key" ]] && continue
        # Trim whitespace and export
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        export "$key=$value"
    done < .env
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --api|-a)
            RUN_MODE="api"
            shift
            ;;
        --cycle|-c)
            RUN_MODE="cycle"
            shift
            ;;
        --test|-t)
            if [[ "$2" == "api" ]]; then
                RUN_MODE="test_api"
                shift
            elif [[ "$2" == "trade" ]]; then
                RUN_MODE="test_trade"
                shift
            else
                RUN_MODE="test"
            fi
            shift
            ;;
        --docker|-d)
            RUN_MODE="docker"
            shift
            ;;
        --live)
            TEST_MODE=false
            shift
            ;;
        --debug)
            DEBUG=true
            export LOG_LEVEL=DEBUG
            shift
            ;;
        --vph)
            VPH_THRESHOLD="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run.sh [OPTION]"
            echo "Options:"
            echo "  --api, -a       Start the API server"
            echo "  --cycle, -c     Run a single processing cycle"
            echo "  --test, -t      Run tests"
            echo "  --test api      Run API tests"
            echo "  --test trade    Test trade execution directly"
            echo "  --docker, -d    Run using Docker Compose"
            echo "  --live          Run in live trading mode (default: test mode)"
            echo "  --debug         Enable debug logging"
            echo "  --vph VALUE     Set VPH threshold (default: 500.0)"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run './run.sh --help' for usage information"
            exit 1
            ;;
    esac
done

# Set test mode environment variable
if [ "$TEST_MODE" = true ]; then
    export TEST_MODE=true
    echo "Running in TEST MODE (simulated trading)"
else
    export TEST_MODE=false
    echo "Running in LIVE MODE (real trading)"
    
    # Check if API keys are set for live trading
    if [ -z "$EXCHANGE_API_KEY" ] || [ -z "$EXCHANGE_API_SECRET" ]; then
        echo "ERROR: EXCHANGE_API_KEY and EXCHANGE_API_SECRET must be set for live trading"
        echo "Please add them to your .env file or export them as environment variables"
        exit 1
    fi
    
    # Confirm live mode
    read -p "WARNING: You are about to execute REAL trades. Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting."
        exit 1
    fi
fi

# Create data directory if it doesn't exist
mkdir -p data

# Execute the selected run mode
case ${RUN_MODE:-api} in
    api)
        echo "Starting API server..."
        if [ "$DEBUG" = true ]; then
            uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level debug
        else
            uvicorn main:app --reload --host 0.0.0.0 --port 8000
        fi
        ;;
    cycle)
        echo "Running single processing cycle with VPH threshold: $VPH_THRESHOLD..."
        python -c "import asyncio; from run_cycle_impl import run_cycle; asyncio.run(run_cycle($VPH_THRESHOLD))"
        ;;
    test)
        echo "Running tests..."
        python run_tests.py
        ;;
    test_api)
        echo "Running API tests..."
        python test_api.py
        ;;
    test_trade)
        echo "Testing trade execution directly..."
        python test_trade_execution.py
        ;;
    docker)
        echo "Starting with Docker Compose..."
        docker-compose up --build
        ;;
    *)
        echo "No run mode specified, defaulting to API server..."
        uvicorn main:app --reload --host 0.0.0.0 --port 8000
        ;;
esac