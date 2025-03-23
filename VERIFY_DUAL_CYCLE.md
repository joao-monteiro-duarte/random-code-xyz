# Verifying the Dual-Cycle Framework

This document provides instructions on how to verify that the dual-cycle framework is working correctly, with the 5-minute quick decision cycle and 30-minute comprehensive cycle functioning as expected.

## Overview of the Dual-Cycle Framework

The crypto trading system uses a dual-cycle approach for trading decisions:

1. **Quick Decision Cycle (5 minutes)**:
   - Triggered when new videos are fetched and significant sentiment changes are detected
   - Focuses on rapid response to emerging trends
   - Uses incremental updates to sentiment scores
   - Adds a freshness boost (20%) for new content
   - Adds a new coin boost (10%) for newly discovered cryptocurrencies
   - Only executes trades when sentiment changes by 1.0 or more

2. **Comprehensive Cycle (30 minutes)**:
   - Processes all accumulated videos thoroughly
   - Performs complete sentiment analysis on all transcripts
   - Makes more thorough trading decisions based on full data
   - Implements portfolio rebalancing and position adjustments

## Enhanced Logging for Cycle Verification

We've added enhanced logging to make it easier to verify that both cycles are running properly:

1. **In `main.py`**: 
   - Added detailed logging in the `cycle_loop` function to track comprehensive cycle timing
   - Added logging for why cycles are or aren't triggered

2. **In `app_service.py`**:
   - Added detailed logging in the `fetch_videos_loop` function for quick decision cycles
   - Added timestamp logging when quick decisions are triggered
   - Added logging of sentiment changes that trigger or don't trigger quick decisions

3. **Added New Scripts**:
   - `test_dual_cycle.py`: Dedicated test script for the dual-cycle framework
   - `run_dual_cycle_test.sh`: Shell script to run the test with proper environment setup
   - `monitor_cycles.sh`: Tool to monitor cycle execution in real-time
   - `analyze_cycles.py`: Tool to verify cycle timing from logs
   - Enhanced `start_simulation.sh` with debug options for cycle verification

## Verification Steps

### 1. Run the Dedicated Test Script (30 minutes)

```bash
./run_dual_cycle_test.sh
```

This will:
- Run the system for 30 minutes in a controlled environment
- Add test videos with mock transcripts to trigger cycles
- Log detailed timing information for both cycle types
- Create a comprehensive log file for analysis

### 2. Monitor Cycles in Real-Time

While running a test or simulation, you can monitor cycle execution in real-time:

```bash
./monitor_cycles.sh [path/to/logfile]
```

This will show only the cycle-related log lines, making it easier to see if cycles are running correctly.

### 3. Run a Short Simulation with Enhanced Debugging

```bash
./start_simulation.sh --short --debug --verbose
```

This will:
- Run a 24-hour simulation instead of the full 2 weeks
- Enable enhanced debugging for cycles
- Show logs in the terminal for real-time monitoring
- Create detailed logs in the simulations directory

### 4. Analyze Cycle Timing from Logs

After running a test or simulation, you can analyze the logs to verify cycle timing:

```bash
./analyze_cycles.py path/to/logfile [--output chart.png]
```

This will:
- Extract all cycle execution times from the logs
- Calculate the average interval between cycles
- Verify that cycles are running at the expected intervals
- Generate a chart showing cycle execution timing

### What to Look For

1. **Comprehensive Cycles**:
   - Should run approximately every 30 minutes (1800 seconds)
   - Log lines will contain "Auto-triggering trading cycle"
   - Should see full sentiment analysis and trading decisions

2. **Quick Decision Cycles**:
   - Should trigger when new videos cause significant sentiment changes
   - Log lines will contain "Quick decision cycle triggered"
   - Should only execute trades when sentiment changes by 1.0 or more
   - Throttling should prevent trading the same coin too frequently

3. **Expected Patterns**:
   - Quick decisions should be more frequent than comprehensive cycles
   - Quick decisions should respond rapidly to new information
   - Comprehensive cycles should be more thorough and rebalance the portfolio

## Expected Output

When functioning correctly, you should see log output similar to this:

```
2023-03-23 10:00:00,123 - main - INFO - Checking cycle timing - Last cycle: 2023-03-23T09:30:00.123456, Now: 2023-03-23T10:00:00.123456, Time since: 1800.0s
2023-03-23 10:00:00,456 - main - INFO - Auto-triggering trading cycle after 1800.0s

... (comprehensive cycle execution) ...

2023-03-23 10:05:00,789 - app_service - INFO - Quick decision cycle triggered at 2023-03-23T10:05:00.789123
2023-03-23 10:05:00,790 - app_service - INFO - Significant changes: True, Running: False
2023-03-23 10:05:00,791 - app_service - INFO - bitcoin: change of 1.50 from previous
2023-03-23 10:05:00,792 - app_service - INFO - Significant sentiment changes detected, triggering quick decisions

... (quick decision cycle execution) ...
```

## Troubleshooting

1. **Cycles Not Running at Expected Intervals**:
   - Check the `CYCLE_INTERVAL` and `FETCH_INTERVAL` settings in `config/settings.py`
   - Verify that `cycle_loop` and `fetch_videos_loop` are properly started in `main.py`
   - Check for errors or exceptions in the logs

2. **Quick Decisions Not Triggering**:
   - Ensure you have accumulated videos with transcripts
   - Verify that sentiment scores are changing significantly (by ≥ 1.0)
   - Check `sentiment_changes` values in the logs

3. **No Trades Being Executed**:
   - Check sentiment scores to ensure they meet thresholds
   - Verify that the throttling mechanism isn't preventing trades
   - Check that the market data service is providing valid data

## Additional Notes

- The dual-cycle framework is designed for different time scales in production:
  - Quick decision cycle: Every 5 minutes
  - Comprehensive cycle: Every 30 minutes

- For testing purposes, you can adjust these intervals in `config/settings.py` or using environment variables:
  ```
  export CYCLE_INTERVAL=600  # 10 minutes for testing
  export FETCH_INTERVAL=120  # 2 minutes for testing
  ```

- When running short tests, consider adding more logging to understand the system's behavior.