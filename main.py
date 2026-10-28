"""
=============================================================================
ELECTRICITY BILL PREDICTOR: MASTER PIPELINE SCRIPT
=============================================================================
This master script runs the entire project pipeline end-to-end:
  [Step 1] Download & Extract Official UCI Dataset
  [Step 2] Clean Raw Minute Data & Convert kW to Daily kWh
  [Step 3] Run Exploratory Data Analysis (EDA) & Save Pattern Plots
  [Step 4] Train Machine Learning Models (Linear Regression & Decision Tree),
           Evaluate with Mean Absolute Error (MAE), Plot Actual vs Predicted,
           Forecast Tomorrow's Consumption & Calculate the Electricity Bill.

Usage:
  python main.py
=============================================================================
"""

import os
import sys

# Ensure UTF-8 output encoding for cross-platform terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def print_banner():
    banner = """
=============================================================================
           SINGULARITY RESEARCH LAB - AI DIVISION ADMISSION TASK
                     PROJECT: ELECTRICITY BILL PREDICTOR
=============================================================================
    """
    print(banner)


def check_dependencies():
    """
    Verifies that all required third-party libraries are installed.
    """
    missing_packages = []
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn"
    }
    for mod, pkg in packages.items():
        try:
            __import__(mod)
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        print("\n" + "!" * 70)
        print("ERROR: Missing required Python packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install all requirements by running:")
        print("  pip install -r requirements.txt")
        print("!" * 70 + "\n")
        sys.exit(1)


def main():
    print_banner()
    check_dependencies()

    # Step 1: Download dataset if needed
    print("\n>>> STEP 1: VERIFYING / DOWNLOADING OFFICIAL UCI DATASET...")
    from download_data import download_and_extract_dataset
    download_and_extract_dataset()

    # Step 2: Preprocess raw data to daily kWh
    print("\n>>> STEP 2: PREPROCESSING DATA & CONVERTING kW TO DAILY kWh...")
    from data_preprocessing import load_and_preprocess_data
    load_and_preprocess_data()

    # Step 3: Run EDA & Visualizations
    print("\n>>> STEP 3: PERFORMING EXPLORATORY DATA ANALYSIS (EDA)...")
    from eda import run_eda
    run_eda()

    # Step 4: Machine Learning & Bill Forecasting
    print("\n>>> STEP 4: TRAINING MODELS, EVALUATING MAE & ESTIMATING BILL...")
    from train_predict import train_and_evaluate
    train_and_evaluate()

    print("\n" + "=" * 70)
    print("ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("All visual charts and metrics report are saved in the 'outputs/' folder:")
    print("  1. outputs/daily_consumption_trend.png")
    print("  2. outputs/weekly_and_seasonal_patterns.png")
    print("  3. outputs/actual_vs_predicted_linear_regression.png")
    print("  4. outputs/model_comparison.png")
    print("  5. outputs/evaluation_results.txt")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
