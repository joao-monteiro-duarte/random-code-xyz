#!/bin/bash
# Script to start the 2-week simulation with proper logging and dual-cycle debug options

# Parse command line options
DEBUG_MODE=false
SHORT_MODE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --debug)
      DEBUG_MODE=true
      shift
      ;;
    --short)
      SHORT_MODE=true
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Create timestamp for this simulation run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SIMULATION_ID="sim_${TIMESTAMP}"

if [ "$SHORT_MODE" = true ]; then
  echo "Running in short mode (24 hours)"
  SIMULATION_PERIOD="24-hour test simulation"
  # Set shorter cycle intervals for testing
  export CYCLE_INTERVAL=1800  # 30 minutes in seconds
  export FETCH_INTERVAL=300   # 5 minutes in seconds
else
  SIMULATION_PERIOD="March 22 - April 5, 2025 (2 weeks)"
fi

# Create a directory for this simulation's logs
mkdir -p "simulations/${SIMULATION_ID}"
mkdir -p "simulations/${SIMULATION_ID}/logs"

# Metadata and configuration
echo "Simulation started at: $(date)" > "simulations/${SIMULATION_ID}/metadata.txt"
echo "Simulation ID: ${SIMULATION_ID}" >> "simulations/${SIMULATION_ID}/metadata.txt"
echo "Starting portfolio value: \$10,000" >> "simulations/${SIMULATION_ID}/metadata.txt"
echo "Cycle intervals: comprehensive=${CYCLE_INTERVAL}s, quick=${FETCH_INTERVAL}s" >> "simulations/${SIMULATION_ID}/metadata.txt"
echo "Debug mode: ${DEBUG_MODE}" >> "simulations/${SIMULATION_ID}/metadata.txt"
echo "Short mode: ${SHORT_MODE}" >> "simulations/${SIMULATION_ID}/metadata.txt"
echo "Verbose: ${VERBOSE}" >> "simulations/${SIMULATION_ID}/metadata.txt"

# Copy the .env file for reference
if [ -f .env ]; then
    cp .env "simulations/${SIMULATION_ID}/.env.snapshot"
fi

# Set enhanced logging options for dual-cycle debugging
if [ "$DEBUG_MODE" = true ]; then
    echo "Running in DEBUG mode with enhanced logging"
    export LOG_LEVEL="DEBUG"
    
    # Create debug directory for cycle logs
    mkdir -p "simulations/${SIMULATION_ID}/logs/cycles"
    
    # Set log path environment variable with detailed cycle logging
    export LOG_PATH="simulations/${SIMULATION_ID}/logs/app_${TIMESTAMP}.log"
    export CYCLE_LOG_PATH="simulations/${SIMULATION_ID}/logs/cycles"
    
    # Enable detailed cycle logging
    export CYCLE_DEBUG=true
    
    if [ "$VERBOSE" = true ]; then
        echo "Verbose mode enabled - logs will be shown in terminal"
    fi
else
    # Standard logging if not in debug mode
    export LOG_PATH="simulations/${SIMULATION_ID}/app_${TIMESTAMP}.log"
    export LOG_LEVEL="INFO"
fi

# Create trade directories
mkdir -p "simulations/${SIMULATION_ID}/trades"
mkdir -p "simulations/${SIMULATION_ID}/portfolio"

# Start the API server with logging
echo "Starting simulation ${SIMULATION_ID}..."
echo "Logs will be saved to ${LOG_PATH}"
echo "======================================="
echo "Simulation start: $(date)"
echo "Simulation ID: ${SIMULATION_ID}"
echo "Simulation period: ${SIMULATION_PERIOD}"
echo "Cycle intervals: comprehensive=${CYCLE_INTERVAL}s, quick=${FETCH_INTERVAL}s"
echo "======================================="

# Run the API server with appropriate options
echo "Starting API server..."

if [ "$VERBOSE" = true ] && [ "$DEBUG_MODE" = true ]; then
    # In verbose debug mode, show logs in terminal
    ./run.sh --api --debug 2>&1 | tee "simulations/${SIMULATION_ID}/server_${TIMESTAMP}.log"
else
    # Otherwise log to file only
    ./run.sh --api --debug > "simulations/${SIMULATION_ID}/server_${TIMESTAMP}.log" 2>&1
fi

# Save a tail of the last 100 log lines for quick reference
tail -n 100 "${LOG_PATH}" > "simulations/${SIMULATION_ID}/last_logs.txt"