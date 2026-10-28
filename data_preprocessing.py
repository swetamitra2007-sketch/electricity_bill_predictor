"""
=============================================================================
STAGE 2: Data Preprocessing & Energy Unit Conversion (kW -> daily kWh)
=============================================================================
This script loads the raw minute-level household electricity dataset,
cleans missing values, and correctly converts minute-level power readings (kW)
into total daily energy consumption (kWh).

Mathematical Principle:
- Power is measured in Kilowatts (kW).
- Energy = Power x Time.
- Since readings are recorded every 1 minute:
      Time = 1 minute = (1 / 60) hour.
      Energy (kWh) for 1 minute = Global_active_power (kW) * (1 / 60)
- Total Daily Energy Consumption (kWh) = Sum of all 1-minute kWh readings in that day.
  Mathematically: Daily_kWh = (Sum of 1440 minute kW readings) / 60.0
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_FILE = os.path.join(DATA_DIR, "household_power_consumption.txt")
PROCESSED_FILE = os.path.join(DATA_DIR, "daily_power_consumption.csv")


def load_and_preprocess_data():
    """
    Loads raw power data, handles missing values, converts to daily kWh,
    and saves the cleaned daily time series.
    """
    print("=" * 70)
    print("STAGE 2: PREPROCESSING AND CONVERTING POWER DATA TO DAILY kWh")
    print("=" * 70)

    # 1. Check if raw file exists
    if not os.path.exists(RAW_FILE):
        print(f"[ERROR] Raw dataset not found at: {RAW_FILE}")
        print("Please run 'python download_data.py' first to download the dataset.")
        sys.exit(1)

    print(f"Loading raw dataset from '{RAW_FILE}'...")
    print("This file contains over 2 million rows; loading will take 10-20 seconds...")

    # 2. Read only the columns we need to optimize memory and speed
    # Missing values in this dataset are recorded as '?'
    use_cols = ["Date", "Time", "Global_active_power"]
    
    try:
        df = pd.read_csv(
            RAW_FILE,
            sep=';',
            usecols=use_cols,
            na_values=['?'],
            dtype={'Global_active_power': float},
            low_memory=False
        )
    except Exception as e:
        print(f"[ERROR] Failed to read dataset: {e}")
        sys.exit(1)

    total_rows = len(df)
    print(f"[OK] Successfully loaded {total_rows:,} raw minute-level records.")

    # 3. Handle missing values
    missing_count = df["Global_active_power"].isna().sum()
    missing_pct = (missing_count / total_rows) * 100.0
    print(f"Missing readings detected: {missing_count:,} ({missing_pct:.2f}% of total data).")

    # Forward-fill short missing gaps (standard for continuous time series power readings)
    # If the power meter missed 1-2 minutes, the last known power is the best estimate.
    print("Handling missing values using forward-fill (carrying forward last known reading)...")
    df["Global_active_power"] = df["Global_active_power"].ffill().bfill()

    # 4. Parse Dates
    print("Parsing date formats (DD/MM/YYYY)...")
    # Date in raw data is in format '16/12/2006'
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

    # 5. Physics conversion: minute kW -> minute kWh
    # Energy (kWh) = Power (kW) * Time (hours)
    # 1 minute = 1 / 60 hours
    print("Converting minute-level power (kW) to energy (kWh)...")
    df["Energy_kWh_minute"] = df["Global_active_power"] / 60.0

    # 6. Group by Date to obtain total daily consumption in kWh
    print("Aggregating minute readings to daily totals...")
    daily_df = df.groupby("Date").agg(
        daily_consumption_kwh=("Energy_kWh_minute", "sum"),
        valid_minutes=("Energy_kWh_minute", "count")
    ).reset_index()

    # A complete day must have 24 hours * 60 minutes = 1,440 readings.
    # The first day (16/12/2006) and last day (26/11/2010) are partial days.
    # Filter out days with fewer than 1,300 minutes to ensure training integrity.
    initial_days = len(daily_df)
    daily_df = daily_df[daily_df["valid_minutes"] >= 1300].copy()
    dropped_days = initial_days - len(daily_df)
    if dropped_days > 0:
        print(f"Dropped {dropped_days} incomplete boundary day(s) (e.g. partial start/end days).")

    # Drop valid_minutes column as it is no longer needed
    daily_df = daily_df[["Date", "daily_consumption_kwh"]].sort_values("Date").reset_index(drop=True)

    # 7. Summary Statistics
    print("\n--- Daily Consumption Summary Statistics ---")
    print(f"Date Range: {daily_df['Date'].min().strftime('%Y-%m-%d')} to {daily_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Total Valid Days: {len(daily_df):,}")
    print(f"Average Daily Consumption : {daily_df['daily_consumption_kwh'].mean():.2f} kWh")
    print(f"Minimum Daily Consumption : {daily_df['daily_consumption_kwh'].min():.2f} kWh")
    print(f"Maximum Daily Consumption : {daily_df['daily_consumption_kwh'].max():.2f} kWh")
    print(f"Standard Deviation        : {daily_df['daily_consumption_kwh'].std():.2f} kWh")

    # 8. Save processed dataset
    daily_df.to_csv(PROCESSED_FILE, index=False)
    print(f"\n[SUCCESS] Clean daily dataset saved to: {PROCESSED_FILE}")
    print("=" * 70)
    return daily_df


if __name__ == "__main__":
    load_and_preprocess_data()
