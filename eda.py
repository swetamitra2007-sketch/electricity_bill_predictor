"""
=============================================================================
STAGE 3: Exploratory Data Analysis (EDA) & Consumption Pattern Analysis
=============================================================================
This script analyzes historical electricity consumption data to uncover:
1. Daily and weekly consumption patterns (e.g., Weekday vs Weekend differences).
2. Seasonal/monthly consumption trends across 4 years (Winter vs Summer peaks).
3. Long-term trends using rolling averages.

Generates and saves visual charts into the 'outputs/' directory.
=============================================================================
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set clean visual styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "daily_power_consumption.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")


def run_eda():
    print("=" * 70)
    print("STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Processed data not found at: {DATA_FILE}")
        print("Please run 'python data_preprocessing.py' first.")
        sys.exit(1)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1. Load Clean Daily Data
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 2. Extract calendar attributes for analysis
    df["Day_Name"] = df["Date"].dt.day_name()
    df["Day_of_Week"] = df["Date"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["Is_Weekend"] = df["Day_of_Week"].isin([5, 6])
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["Year"] = df["Date"].dt.year

    print(f"Dataset covers: {df['Date'].min().date()} to {df['Date'].max().date()} ({len(df)} days)")

    # 3. Analyze Day of Week Consumption
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_stats = df.groupby("Day_Name")["daily_consumption_kwh"].agg(["mean", "std", "count"]).reindex(dow_order)

    print("\n--- Day-of-Week Consumption Averages (kWh) ---")
    for day, row in dow_stats.iterrows():
        print(f"  {day:<10}: {row['mean']:.2f} kWh (± {row['std']:.2f})")

    weekday_avg = df[~df["Is_Weekend"]]["daily_consumption_kwh"].mean()
    weekend_avg = df[df["Is_Weekend"]]["daily_consumption_kwh"].mean()
    pct_diff = ((weekend_avg - weekday_avg) / weekday_avg) * 100

    print(f"\nWeekday Average: {weekday_avg:.2f} kWh")
    print(f"Weekend Average: {weekend_avg:.2f} kWh")
    print(f"Insight: Household consumes {abs(pct_diff):.1f}% {'more' if pct_diff > 0 else 'less'} electricity on weekends.")

    # 4. Monthly/Seasonal Consumption
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_stats = df.groupby("Month_Name")["daily_consumption_kwh"].mean().reindex(month_order)
    highest_month = monthly_stats.idxmax()
    lowest_month = monthly_stats.idxmin()

    print(f"\nHighest Consumption Month: {highest_month} ({monthly_stats.max():.2f} kWh/day)")
    print(f"Lowest Consumption Month : {lowest_month} ({monthly_stats.min():.2f} kWh/day)")
    print("Insight: Electricity demand peaks in winter months (December/January/February) due to electric heating.")

    # =========================================================================
    # PLOT 1: Full Historical Time Series Trend (with 30-day Rolling Average)
    # =========================================================================
    print("\nGenerating Plot 1: Historical Daily Trend...")
    df["Rolling_30_Mean"] = df["daily_consumption_kwh"].rolling(window=30, center=True).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["Date"], df["daily_consumption_kwh"], color="#4A90E2", alpha=0.45, linewidth=0.9, label="Daily Consumption (kWh)")
    ax.plot(df["Date"], df["Rolling_30_Mean"], color="#D9534F", linewidth=2.4, label="30-Day Moving Average (Trend)")
    ax.axhline(df["daily_consumption_kwh"].mean(), color="#333333", linestyle="--", linewidth=1.2, label=f"Overall Mean ({df['daily_consumption_kwh'].mean():.1f} kWh)")
    
    ax.set_title("Historical Household Electricity Consumption (Dec 2006 - Nov 2010)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Daily Energy Consumption (kWh)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    trend_plot_path = os.path.join(OUTPUTS_DIR, "daily_consumption_trend.png")
    fig.savefig(trend_plot_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {trend_plot_path}")

    # =========================================================================
    # PLOT 2: Weekly & Seasonal Pattern Analysis
    # =========================================================================
    print("Generating Plot 2: Weekly & Monthly Patterns...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot A: Day of Week
    colors_dow = ["#5B9BD5"] * 5 + ["#ED7D31", "#ED7D31"]  # Highlight weekend days
    ax1.bar(dow_stats.index, dow_stats["mean"], yerr=dow_stats["std"], capsize=4, color=colors_dow, edgecolor="#333333", alpha=0.85)
    ax1.set_title("Average Electricity Consumption by Day of Week", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Average Consumption (kWh)", fontsize=10)
    ax1.set_xticks(range(len(dow_stats.index)))
    ax1.set_xticklabels(dow_stats.index, rotation=35, ha="right")
    ax1.grid(axis="y", linestyle=":", alpha=0.7)
    for i, v in enumerate(dow_stats["mean"]):
        ax1.text(i, v + 1.0, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")

    # Subplot B: Monthly Seasonality
    colors_month = plt.cm.viridis(np.linspace(0.2, 0.85, 12))
    ax2.bar(month_order, monthly_stats.values, color=colors_month, edgecolor="#333333", alpha=0.85)
    ax2.set_title("Seasonal Electricity Consumption by Month", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Average Consumption (kWh)", fontsize=10)
    ax2.set_xticks(range(len(month_order)))
    ax2.set_xticklabels(month_order)
    ax2.grid(axis="y", linestyle=":", alpha=0.7)
    for i, v in enumerate(monthly_stats.values):
        ax2.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    patterns_plot_path = os.path.join(OUTPUTS_DIR, "weekly_and_seasonal_patterns.png")
    fig.savefig(patterns_plot_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {patterns_plot_path}")

    print("\n[SUCCESS] EDA completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_eda()
