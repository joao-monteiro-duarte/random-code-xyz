#!/usr/bin/env python3
"""
Script to analyze cycle timing from logs
This helps verify that both the 5-minute quick decision cycle and 30-minute
comprehensive cycle are executing at the expected intervals.
"""
import argparse
import re
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any

def parse_log_file(log_file: str) -> Tuple[List[datetime], List[datetime]]:
    """
    Parse a log file for cycle execution times.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Tuple of (comprehensive_cycle_times, quick_decision_times)
    """
    comprehensive_cycle_times = []
    quick_decision_times = []
    
    # Regular expressions for cycle timing
    comprehensive_pattern = r"Auto-triggering trading cycle after (\d+\.\d+)s"
    quick_pattern = r"Quick decision cycle triggered at (\S+)"
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract timestamp from log line
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
            if not timestamp_match:
                continue
                
            timestamp_str = timestamp_match.group(1)
            
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
            except ValueError:
                continue
            
            # Check for comprehensive cycle
            if "Auto-triggering trading cycle" in line:
                comprehensive_cycle_times.append(timestamp)
                print(f"Comprehensive cycle at {timestamp}")
            
            # Check for quick decision cycle
            if "Quick decision cycle triggered" in line:
                quick_match = re.search(quick_pattern, line)
                if quick_match:
                    quick_decision_times.append(timestamp)
                    print(f"Quick decision at {timestamp}")
    
    return comprehensive_cycle_times, quick_decision_times

def analyze_cycle_timing(
    comprehensive_times: List[datetime], 
    quick_times: List[datetime]
) -> Dict[str, Any]:
    """
    Analyze the cycle timing to check if cycles are running at expected intervals.
    
    Args:
        comprehensive_times: List of comprehensive cycle times
        quick_times: List of quick decision times
        
    Returns:
        Dictionary of analysis results
    """
    results = {
        "comprehensive_count": len(comprehensive_times),
        "quick_count": len(quick_times),
        "comprehensive_intervals": [],
        "quick_intervals": [],
        "comprehensive_avg_interval": 0,
        "quick_avg_interval": 0,
        "is_comprehensive_ok": False,
        "is_quick_ok": False
    }
    
    # Calculate intervals between comprehensive cycles
    if len(comprehensive_times) > 1:
        intervals = []
        for i in range(1, len(comprehensive_times)):
            interval = (comprehensive_times[i] - comprehensive_times[i-1]).total_seconds()
            intervals.append(interval)
        
        results["comprehensive_intervals"] = intervals
        results["comprehensive_avg_interval"] = sum(intervals) / len(intervals)
        
        # Check if comprehensive cycles are running at ~30min intervals
        results["is_comprehensive_ok"] = 1700 <= results["comprehensive_avg_interval"] <= 1900
    
    # Calculate intervals between quick decision cycles
    if len(quick_times) > 1:
        intervals = []
        for i in range(1, len(quick_times)):
            interval = (quick_times[i] - quick_times[i-1]).total_seconds()
            intervals.append(interval)
        
        results["quick_intervals"] = intervals
        results["quick_avg_interval"] = sum(intervals) / len(intervals)
        
        # Check if quick decision cycles are running at ~5min intervals
        results["is_quick_ok"] = 250 <= results["quick_avg_interval"] <= 350
    
    return results

def create_timing_chart(
    comprehensive_times: List[datetime], 
    quick_times: List[datetime],
    results: Dict[str, Any],
    output_file: str = None
):
    """
    Create a timing chart showing when cycles were executed.
    
    Args:
        comprehensive_times: List of comprehensive cycle times
        quick_times: List of quick decision times
        results: Analysis results
        output_file: Path to save the chart image
    """
    plt.figure(figsize=(12, 6))
    
    # Normalize times to start from 0
    if comprehensive_times and quick_times:
        start_time = min(comprehensive_times[0] if comprehensive_times else datetime.max,
                        quick_times[0] if quick_times else datetime.max)
    elif comprehensive_times:
        start_time = comprehensive_times[0]
    elif quick_times:
        start_time = quick_times[0]
    else:
        return  # No data to plot
    
    # Convert to minutes from start
    comp_minutes = [(t - start_time).total_seconds() / 60 for t in comprehensive_times]
    quick_minutes = [(t - start_time).total_seconds() / 60 for t in quick_times]
    
    # Plot timelines
    if comprehensive_times:
        plt.scatter(comp_minutes, [1] * len(comp_minutes), s=100, marker='o', 
                  color='blue', label=f'Comprehensive Cycles ({len(comprehensive_times)})')
        
        for x in comp_minutes:
            plt.axvline(x=x, color='lightblue', linestyle='--', alpha=0.5)
    
    if quick_times:
        plt.scatter(quick_minutes, [0.5] * len(quick_minutes), s=80, marker='^', 
                  color='green', label=f'Quick Decisions ({len(quick_times)})')
    
    # Add guidelines for expected intervals
    max_time = max(comp_minutes + quick_minutes) if comp_minutes or quick_minutes else 60
    
    # 30-minute comprehensive cycle lines
    for t in range(0, int(max_time) + 30, 30):
        plt.axvline(x=t, color='blue', linestyle=':', alpha=0.3)
    
    # 5-minute quick cycle lines
    for t in range(0, int(max_time) + 5, 5):
        plt.axvline(x=t, color='green', linestyle=':', alpha=0.2)
    
    # Add text annotations summarizing results
    plt.figtext(0.5, 0.01, 
               f"Comprehensive Cycles: {results['comprehensive_count']} (Avg interval: {results['comprehensive_avg_interval']:.1f}s, Expected: 1800s)\n"
               f"Quick Decisions: {results['quick_count']} (Avg interval: {results['quick_avg_interval']:.1f}s, Expected: 300s)", 
               ha="center", fontsize=12, 
               bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Configure plot
    plt.yticks([0.5, 1], ['Quick Decisions', 'Comprehensive Cycles'])
    plt.xlabel('Minutes since start')
    plt.title('Cycle Timing Analysis')
    plt.grid(True, axis='x')
    plt.legend()
    
    # Add status indicators
    comp_status = "✓" if results["is_comprehensive_ok"] else "✗"
    quick_status = "✓" if results["is_quick_ok"] else "✗"
    
    plt.annotate(f"{comp_status} Comprehensive", xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=12, color='green' if results["is_comprehensive_ok"] else 'red')
    plt.annotate(f"{quick_status} Quick", xy=(0.02, 0.90), xycoords='axes fraction', 
                fontsize=12, color='green' if results["is_quick_ok"] else 'red')
    
    if output_file:
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Chart saved to {output_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze cycle timing from logs")
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument("--output", "-o", help="Output chart file path", default=None)
    
    args = parser.parse_args()
    
    print(f"Analyzing cycle timing from {args.log_file}")
    
    try:
        # Parse the log file
        comprehensive_times, quick_times = parse_log_file(args.log_file)
        
        if not comprehensive_times and not quick_times:
            print("No cycle data found in the log file")
            return 1
        
        # Analyze timing
        results = analyze_cycle_timing(comprehensive_times, quick_times)
        
        # Print summary
        print("\n===== CYCLE TIMING ANALYSIS =====")
        print(f"Comprehensive cycles: {results['comprehensive_count']}")
        print(f"Quick decisions: {results['quick_count']}")
        
        if results['comprehensive_intervals']:
            print(f"Comprehensive cycle interval: {results['comprehensive_avg_interval']:.1f}s average (expected ~1800s)")
            print(f"Interval status: {'OK' if results['is_comprehensive_ok'] else 'Not within expected range'}")
        
        if results['quick_intervals']:
            print(f"Quick decision interval: {results['quick_avg_interval']:.1f}s average (expected ~300s)")
            print(f"Interval status: {'OK' if results['is_quick_ok'] else 'Not within expected range'}")
        
        # Create chart
        create_timing_chart(comprehensive_times, quick_times, results, args.output)
        
        return 0
        
    except Exception as e:
        print(f"Error analyzing log file: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())