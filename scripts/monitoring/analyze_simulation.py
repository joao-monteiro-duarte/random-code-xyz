#!/usr/bin/env python3
"""
Script to analyze the results of a simulation run
"""
import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any

def load_simulation_data(simulation_id: str) -> Dict[str, Any]:
    """Load data from a simulation run"""
    
    simulation_dir = f"simulations/{simulation_id}"
    
    if not os.path.exists(simulation_dir):
        raise ValueError(f"Simulation directory {simulation_dir} not found")
    
    # Load metadata
    metadata = {}
    metadata_path = f"{simulation_dir}/metadata.txt"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
    
    # Load portfolio data
    portfolio_files = sorted([f for f in os.listdir(f"{simulation_dir}/portfolio") if f.endswith(".json")])
    portfolio_history = []
    
    for pf in portfolio_files:
        with open(f"{simulation_dir}/portfolio/{pf}", "r") as f:
            portfolio_history.append(json.load(f))
    
    # Load trade data
    trade_files = sorted([f for f in os.listdir(f"{simulation_dir}/trades") if f.endswith(".json")])
    trades = []
    
    for tf in trade_files:
        with open(f"{simulation_dir}/trades/{tf}", "r") as f:
            trades.extend(json.load(f))
    
    return {
        "metadata": metadata,
        "portfolio_history": portfolio_history,
        "trades": trades,
        "simulation_id": simulation_id
    }

def create_performance_report(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a performance report from simulation data"""
    
    metadata = data["metadata"]
    portfolio_history = data["portfolio_history"]
    trades = data["trades"]
    
    # Calculate basic metrics
    if portfolio_history:
        initial_value = portfolio_history[0]["total_value"] if portfolio_history else 10000.0
        final_value = portfolio_history[-1]["total_value"] if portfolio_history else initial_value
        profit_loss = final_value - initial_value
        profit_loss_pct = (profit_loss / initial_value) * 100 if initial_value > 0 else 0
    else:
        initial_value = 10000.0
        final_value = 10000.0
        profit_loss = 0.0
        profit_loss_pct = 0.0
    
    # Analyze trades
    total_trades = len(trades)
    buy_trades = sum(1 for t in trades if t.get("action") == "buy")
    sell_trades = sum(1 for t in trades if t.get("action") == "sell")
    
    # Calculate trade performance
    profitable_trades = sum(1 for t in trades if t.get("action") == "sell" and t.get("profit", 0) > 0)
    profit_ratio = profitable_trades / sell_trades if sell_trades > 0 else 0
    
    # Analyze by cycle type
    quick_decision_trades = sum(1 for t in trades if t.get("triggered_by") == "quick_decision")
    comprehensive_trades = total_trades - quick_decision_trades
    
    # Coins traded
    traded_coins = set(t.get("symbol", "").upper() for t in trades)
    
    # Create performance report
    report = {
        "simulation_id": data["simulation_id"],
        "initial_value": initial_value,
        "final_value": final_value,
        "profit_loss": profit_loss,
        "profit_loss_pct": profit_loss_pct,
        "total_trades": total_trades,
        "buy_trades": buy_trades,
        "sell_trades": sell_trades,
        "profitable_trades": profitable_trades,
        "profit_ratio": profit_ratio,
        "quick_decision_trades": quick_decision_trades,
        "comprehensive_trades": comprehensive_trades,
        "traded_coins": list(traded_coins),
        "unique_coins_count": len(traded_coins)
    }
    
    return report

def generate_charts(data: Dict[str, Any], output_dir: str):
    """Generate charts from simulation data"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    portfolio_history = data["portfolio_history"]
    trades = data["trades"]
    
    # Create timestamp for report filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create portfolio value chart
    if portfolio_history:
        df = pd.DataFrame(portfolio_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["total_value"])
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plt.savefig(f"{output_dir}/portfolio_value_{timestamp}.png")
        plt.close()
    
    # Create trade distribution chart
    if trades:
        trade_types = {}
        
        # Count trades by coin
        for trade in trades:
            symbol = trade.get("symbol", "unknown").upper()
            if symbol not in trade_types:
                trade_types[symbol] = 0
            trade_types[symbol] += 1
        
        # Sort by count
        sorted_trades = sorted(trade_types.items(), key=lambda x: x[1], reverse=True)
        
        # Plot top 10 coins
        top_coins = sorted_trades[:10]
        
        plt.figure(figsize=(12, 6))
        plt.bar([coin[0] for coin in top_coins], [coin[1] for coin in top_coins])
        plt.title("Top 10 Traded Coins")
        plt.xlabel("Coin")
        plt.ylabel("Number of Trades")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/trade_distribution_{timestamp}.png")
        plt.close()
    
    # Create cycle type comparison
    cycle_types = {"quick_decision": 0, "comprehensive": 0}
    
    for trade in trades:
        cycle_type = trade.get("triggered_by", "comprehensive")
        cycle_types[cycle_type] += 1
    
    plt.figure(figsize=(8, 6))
    plt.pie(
        cycle_types.values(), 
        labels=cycle_types.keys(), 
        autopct='%1.1f%%',
        startangle=90
    )
    plt.axis('equal')
    plt.title("Trades by Cycle Type")
    plt.savefig(f"{output_dir}/cycle_types_{timestamp}.png")
    plt.close()
    
    # Write report to file
    report = create_performance_report(data)
    
    with open(f"{output_dir}/report_{timestamp}.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate HTML report
    html_report = f"""
    <html>
    <head>
        <title>Simulation Report - {data['simulation_id']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .metric {{ margin-bottom: 5px; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .chart {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Simulation Report</h1>
        <p>Simulation ID: {data['simulation_id']}</p>
        <p>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Performance Summary</h2>
        <div class="metric">Initial Portfolio Value: ${report['initial_value']:.2f}</div>
        <div class="metric">Final Portfolio Value: ${report['final_value']:.2f}</div>
        <div class="metric">Profit/Loss: <span class="{'positive' if report['profit_loss'] >= 0 else 'negative'}">${report['profit_loss']:.2f} ({report['profit_loss_pct']:.2f}%)</span></div>
        
        <h2>Trading Activity</h2>
        <div class="metric">Total Trades: {report['total_trades']}</div>
        <div class="metric">Buy Trades: {report['buy_trades']}</div>
        <div class="metric">Sell Trades: {report['sell_trades']}</div>
        <div class="metric">Profitable Trades: {report['profitable_trades']} ({report['profit_ratio']*100:.2f}% of sells)</div>
        <div class="metric">Quick Decision Trades: {report['quick_decision_trades']} ({report['quick_decision_trades']/report['total_trades']*100:.2f}% of total)</div>
        <div class="metric">Comprehensive Cycle Trades: {report['comprehensive_trades']} ({report['comprehensive_trades']/report['total_trades']*100:.2f}% of total)</div>
        <div class="metric">Unique Coins Traded: {report['unique_coins_count']}</div>
        
        <h2>Charts</h2>
        <div class="chart">
            <h3>Portfolio Value Over Time</h3>
            <img src="portfolio_value_{timestamp}.png" alt="Portfolio Value Chart" width="800">
        </div>
        
        <div class="chart">
            <h3>Top 10 Traded Coins</h3>
            <img src="trade_distribution_{timestamp}.png" alt="Trade Distribution Chart" width="800">
        </div>
        
        <div class="chart">
            <h3>Trades by Cycle Type</h3>
            <img src="cycle_types_{timestamp}.png" alt="Cycle Types Chart" width="600">
        </div>
        
        <h2>Traded Coins</h2>
        <ul>
            {''.join([f'<li>{coin}</li>' for coin in sorted(report['traded_coins'])])}
        </ul>
    </body>
    </html>
    """
    
    with open(f"{output_dir}/report_{timestamp}.html", "w") as f:
        f.write(html_report)
    
    print(f"Report generated in {output_dir}/report_{timestamp}.html")

def main():
    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument("simulation_id", help="ID of the simulation to analyze")
    parser.add_argument("--output", "-o", default=None, help="Output directory for reports and charts")
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    output_dir = args.output or f"simulations/{args.simulation_id}/analysis"
    
    # Load simulation data
    try:
        data = load_simulation_data(args.simulation_id)
        
        # Generate report and charts
        generate_charts(data, output_dir)
        
        # Print summary to console
        report = create_performance_report(data)
        
        print("=" * 40)
        print(f"SIMULATION REPORT: {args.simulation_id}")
        print("=" * 40)
        print(f"Portfolio: ${report['initial_value']:.2f} → ${report['final_value']:.2f}")
        print(f"Profit/Loss: ${report['profit_loss']:.2f} ({report['profit_loss_pct']:.2f}%)")
        print(f"Total Trades: {report['total_trades']} ({report['quick_decision_trades']} quick, {report['comprehensive_trades']} comprehensive)")
        print(f"Profit Ratio: {report['profit_ratio']*100:.2f}% of sell trades were profitable")
        print(f"Unique Coins: {report['unique_coins_count']}")
        print("=" * 40)
        print(f"Full report available at: {output_dir}/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
    except Exception as e:
        print(f"Error analyzing simulation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())