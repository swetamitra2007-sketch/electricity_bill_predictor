"""
=============================================================================
STAGE 4: Feature Engineering, Model Training, Evaluation & Bill Estimation
=============================================================================
This script executes the core Machine Learning workflow:
1. Feature Engineering (Lags and Calendar features without data leakage).
2. Chronological Data Split (Time-series train/test split: 80% train, 20% test).
3. Main Model: Linear Regression.
4. Optional Comparison: Decision Tree Regressor.
5. Evaluation Metric: Mean Absolute Error (MAE) in kWh (plus RMSE & R2).
6. Generates Actual vs. Predicted comparison plots.
7. Forecasts next-day consumption and computes an illustrative electricity bill.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Clean styling for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "daily_power_consumption.csv")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_FILE = os.path.join(OUTPUTS_DIR, "evaluation_results.txt")


def create_features(df):
    """
    Creates lag features and calendar features with strict prevention of data leakage.
    
    Data Leakage Prevention:
    - Lag features (lag_1, lag_2, lag_3, lag_7) look strictly backwards into previous days.
    - Rolling 7-day average uses shift(1), meaning day t only averages days t-1 through t-7.
      Day t's own target value is NEVER used to compute its features.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # 1. Target variable: Daily consumption in kWh
    # 2. Lag features (Past consumption values)
    df["lag_1"] = df["daily_consumption_kwh"].shift(1)  # Yesterday's consumption
    df["lag_2"] = df["daily_consumption_kwh"].shift(2)  # 2 days ago
    df["lag_3"] = df["daily_consumption_kwh"].shift(3)  # 3 days ago
    df["lag_7"] = df["daily_consumption_kwh"].shift(7)  # Same day last week (weekly cycle)

    # 3. Rolling Window Feature (7-day moving average of previous days)
    # Notice .shift(1): this ensures no leakage of the current day's value!
    df["rolling_mean_7"] = df["daily_consumption_kwh"].shift(1).rolling(window=7).mean()

    # 4. Calendar Features (Derived only from the date)
    df["day_of_week"] = df["Date"].dt.dayofweek          # 0=Monday, 6=Sunday
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)  # 1 for Saturday/Sunday
    df["month"] = df["Date"].dt.month                    # 1 to 12 (captures seasonality)
    df["day_of_month"] = df["Date"].dt.day               # 1 to 31

    # Drop the first 7 rows that contain NaN due to shifting by up to 7 days
    cleaned_df = df.dropna().reset_index(drop=True)
    return cleaned_df


def train_and_evaluate():
    print("=" * 70)
    print("STAGE 4: MODEL TRAINING, EVALUATION & BILL PREDICTION")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Processed daily data not found at: {DATA_FILE}")
        print("Please run 'python data_preprocessing.py' first.")
        sys.exit(1)

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # 1. Load Data
    raw_daily = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(raw_daily)} daily records.")

    # 2. Engineer Features
    df_features = create_features(raw_daily)
    print(f"Features created successfully. Rows available after dropping initial lag NaNs: {len(df_features)}")

    # Define Feature columns and Target
    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_7",
        "rolling_mean_7",
        "day_of_week", "is_weekend", "month", "day_of_month"
    ]
    target_col = "daily_consumption_kwh"

    X = df_features[feature_cols]
    y = df_features[target_col]
    dates = df_features["Date"]

    # 3. Chronological Train-Test Split (80% Train, 20% Test)
    # We NEVER use random train_test_split on time series data!
    # Shuffling would cause future leakage into the past.
    split_index = int(len(df_features) * 0.80)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    dates_train = dates.iloc[:split_index]

    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    dates_test = dates.iloc[split_index:]

    print(f"\n--- Chronological Split Details ---")
    print(f"Training Set : {len(X_train)} days ({dates_train.min().date()} to {dates_train.max().date()})")
    print(f"Test Set     : {len(X_test)} days ({dates_test.min().date()} to {dates_test.max().date()})")

    # =========================================================================
    # MODEL 1: MAIN MODEL - Linear Regression
    # =========================================================================
    print("\n--- Training Main Model: Linear Regression ---")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Predict on Test Set
    lr_pred_test = lr_model.predict(X_test)

    # Evaluation Metrics
    lr_mae = mean_absolute_error(y_test, lr_pred_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred_test))
    lr_r2 = r2_score(y_test, lr_pred_test)

    print(f"Linear Regression Results:")
    print(f"  -> Mean Absolute Error (MAE) : {lr_mae:.3f} kWh/day")
    print(f"  -> Root Mean Squared Error   : {lr_rmse:.3f} kWh/day")
    print(f"  -> R-squared (R2 Score)      : {lr_r2:.3f}")

    # Inspect Linear Regression Coefficients to see which features matter most
    print("\nFeature Importances (Linear Regression Coefficients):")
    for feat, coef in zip(feature_cols, lr_model.coef_):
        print(f"  {feat:<15}: {coef:>8.4f}")
    print(f"  {'Intercept':<15}: {lr_model.intercept_:>8.4f}")

    # =========================================================================
    # MODEL 2: COMPARISON MODEL - Decision Tree Regressor
    # =========================================================================
    print("\n--- Training Comparison Model: Decision Tree Regressor ---")
    # Using max_depth=5 to avoid overfitting on time series
    dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)

    dt_pred_test = dt_model.predict(X_test)
    dt_mae = mean_absolute_error(y_test, dt_pred_test)
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred_test))
    dt_r2 = r2_score(y_test, dt_pred_test)

    print(f"Decision Tree Regressor Results:")
    print(f"  -> Mean Absolute Error (MAE) : {dt_mae:.3f} kWh/day")
    print(f"  -> Root Mean Squared Error   : {dt_rmse:.3f} kWh/day")
    print(f"  -> R-squared (R2 Score)      : {dt_r2:.3f}")

    # Model Comparison Summary
    best_model_name = "Linear Regression" if lr_mae <= dt_mae else "Decision Tree Regressor"
    print(f"\nModel Comparison Winner: {best_model_name} (Lowest Test MAE)")

    # =========================================================================
    # PLOT 1: Actual vs. Predicted (Main Model - Linear Regression)
    # =========================================================================
    print("\nGenerating Actual vs. Predicted plot for Linear Regression...")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates_test, y_test.values, label="Actual Daily Consumption (kWh)", color="#2C3E50", linewidth=1.8, alpha=0.9)
    ax.plot(dates_test, lr_pred_test, label=f"Linear Regression Prediction (MAE: {lr_mae:.2f} kWh)", color="#E74C3C", linewidth=1.5, linestyle="--")
    
    ax.set_title("Test Set: Actual vs. Predicted Electricity Consumption (Linear Regression)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Daily Consumption (kWh)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    lr_plot_path = os.path.join(OUTPUTS_DIR, "actual_vs_predicted_linear_regression.png")
    fig.savefig(lr_plot_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {lr_plot_path}")

    # =========================================================================
    # PLOT 2: Direct Model Comparison Plot
    # =========================================================================
    print("Generating Model Comparison plot (Linear Regression vs Decision Tree)...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Subplot 1: Linear Regression
    ax1.plot(dates_test, y_test.values, label="Actual (kWh)", color="#2C3E50", alpha=0.8, linewidth=1.4)
    ax1.plot(dates_test, lr_pred_test, label=f"Linear Regression (MAE: {lr_mae:.2f} kWh)", color="#2980B9", linewidth=1.4, linestyle="-")
    ax1.set_title("Main Model: Linear Regression Predictions vs Actual", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Daily kWh", fontsize=10)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: Decision Tree Regressor
    ax2.plot(dates_test, y_test.values, label="Actual (kWh)", color="#2C3E50", alpha=0.8, linewidth=1.4)
    ax2.plot(dates_test, dt_pred_test, label=f"Decision Tree (MAE: {dt_mae:.2f} kWh)", color="#27AE60", linewidth=1.4, linestyle="-")
    ax2.set_title("Comparison Model: Decision Tree Regressor Predictions vs Actual", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_ylabel("Daily kWh", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    comparison_plot_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")
    fig.savefig(comparison_plot_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {comparison_plot_path}")

    # =========================================================================
    # FORECAST NEXT DAY CONSUMPTION & ELECTRICITY BILL ESTIMATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("NEXT-DAY FORECAST & ELECTRICITY BILL ESTIMATION")
    print("=" * 70)

    # The last known day in the dataset
    last_row = df_features.iloc[-1]
    last_date = last_row["Date"]
    next_date = last_date + pd.Timedelta(days=1)

    # Build the feature vector for tomorrow (next_date):
    # - lag_1 for tomorrow is today's actual consumption
    # - lag_2 for tomorrow is yesterday's actual consumption
    # - lag_3 for tomorrow is 2 days ago consumption
    # - lag_7 for tomorrow is 6 days ago consumption
    # - rolling_mean_7 is the average of the last 7 known days
    last_7_days_values = df_features["daily_consumption_kwh"].iloc[-7:].values
    
    next_day_features = pd.DataFrame([{
        "lag_1": df_features["daily_consumption_kwh"].iloc[-1],
        "lag_2": df_features["daily_consumption_kwh"].iloc[-2],
        "lag_3": df_features["daily_consumption_kwh"].iloc[-3],
        "lag_7": df_features["daily_consumption_kwh"].iloc[-7],
        "rolling_mean_7": np.mean(last_7_days_values),
        "day_of_week": next_date.dayofweek,
        "is_weekend": 1 if next_date.dayofweek in [5, 6] else 0,
        "month": next_date.month,
        "day_of_month": next_date.day
    }])

    # Predict next day's consumption using Main Model (Linear Regression)
    forecast_kwh_lr = lr_model.predict(next_day_features)[0]
    forecast_kwh_dt = dt_model.predict(next_day_features)[0]

    # Electricity Bill Estimation
    # Tariff Assumptions:
    # Stated illustrative flat domestic tariff rate:
    # INR: Rs. 7.50 per kWh (typical Indian urban domestic benchmark)
    TARIFF_INR_PER_KWH = 7.50

    # Model 1: Linear Regression Bill Calculations
    daily_cost_lr = forecast_kwh_lr * TARIFF_INR_PER_KWH
    monthly_kwh_lr = forecast_kwh_lr * 30.0
    monthly_cost_lr = monthly_kwh_lr * TARIFF_INR_PER_KWH

    # Model 2: Decision Tree Bill Calculations
    daily_cost_dt = forecast_kwh_dt * TARIFF_INR_PER_KWH
    monthly_kwh_dt = forecast_kwh_dt * 30.0
    monthly_cost_dt = monthly_kwh_dt * TARIFF_INR_PER_KWH

    # Set stdout to utf-8 if supported to prevent Windows charmap issues
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"Last Observed Date in Dataset : {last_date.date()} ({last_row['daily_consumption_kwh']:.2f} kWh/day)")
    print(f"Forecast Target Date          : {next_date.date()} ({next_date.day_name()})")
    print(f"Linear Regression Forecast    : {forecast_kwh_lr:.2f} kWh/day")
    print(f"Decision Tree Forecast        : {forecast_kwh_dt:.2f} kWh/day")
    
    print("\n--- Illustrative Electricity Bill Calculation (Assumed Rate: Rs. 7.50/kWh) ---")
    print("[DISCLAIMER] These are illustrative estimates based on the assumed tariff, not actual utility bills.")
    print(f"\n1. Main Model (Linear Regression - Forecast: {forecast_kwh_lr:.2f} kWh/day):")
    print(f"   -> Estimated Daily Bill         : Rs. {daily_cost_lr:.2f}")
    print(f"   -> Estimated 30-Day Consumption : {monthly_kwh_lr:.1f} kWh/month")
    print(f"   -> Estimated 30-Day Bill        : Rs. {monthly_cost_lr:.2f}")

    print(f"\n2. Comparison Model (Decision Tree - Forecast: {forecast_kwh_dt:.2f} kWh/day):")
    print(f"   -> Estimated Daily Bill         : Rs. {daily_cost_dt:.2f}")
    print(f"   -> Estimated 30-Day Consumption : {monthly_kwh_dt:.1f} kWh/month")
    print(f"   -> Estimated 30-Day Bill        : Rs. {monthly_cost_dt:.2f}")

    # =========================================================================
    # SAVE EVALUATION RESULTS TO A TEXT REPORT
    # =========================================================================
    report_content = f"""======================================================================
ELECTRICITY BILL PREDICTOR: ACTUAL EVALUATION & MODEL METRICS REPORT
Singularity Research Lab - AI Division Task
======================================================================

1. DATASET & CHRONOLOGICAL SPLIT
----------------------------------------------------------------------
- Dataset Name          : Individual Household Electric Power Consumption (UCI)
- Total Clean Days      : {len(df_features)} days
- Train Set Range       : {dates_train.min().date()} to {dates_train.max().date()} ({len(X_train)} days, 80%)
- Test Set Range        : {dates_test.min().date()} to {dates_test.max().date()} ({len(X_test)} days, 20%)
- Splitting Strategy    : Chronological Split (Strictly preserves time order, prevents data leakage)

2. MAIN MODEL: LINEAR REGRESSION EVALUATION
----------------------------------------------------------------------
- Test Mean Absolute Error (MAE) : {lr_mae:.3f} kWh/day
- Test Root Mean Squared Error   : {lr_rmse:.3f} kWh/day
- Test R-squared (R2 Score)      : {lr_r2:.3f}

Feature Weights (Coefficients):
"""
    for feat, coef in zip(feature_cols, lr_model.coef_):
        report_content += f"  - {feat:<15}: {coef:>8.4f}\n"
    report_content += f"  - Intercept      : {lr_model.intercept_:>8.4f}\n"

    report_content += f"""
3. COMPARISON MODEL: DECISION TREE REGRESSOR (max_depth=5)
----------------------------------------------------------------------
- Test Mean Absolute Error (MAE) : {dt_mae:.3f} kWh/day
- Test Root Mean Squared Error   : {dt_rmse:.3f} kWh/day
- Test R-squared (R2 Score)      : {dt_r2:.3f}

Model Comparison:
- Best Model on Test Set: {best_model_name} (Lowest Test MAE: {min(lr_mae, dt_mae):.3f} kWh/day)

4. NEXT-DAY CONSUMPTION FORECASTS & ILLUSTRATIVE BILL ESTIMATES
----------------------------------------------------------------------
- Forecast Target Date         : {next_date.date()} ({next_date.day_name()})
- Assumed Electricity Tariff   : Rs. {TARIFF_INR_PER_KWH:.2f} per kWh (Illustrative flat rate)
[NOTE: These are illustrative estimates for demonstration, not actual bills.]

A. Main Model (Linear Regression):
- Predicted Daily Consumption  : {forecast_kwh_lr:.2f} kWh/day
- Estimated Daily Bill         : Rs. {daily_cost_lr:.2f}
- Projected 30-Day Consumption : {monthly_kwh_lr:.1f} kWh/month
- Projected 30-Day Bill        : Rs. {monthly_cost_lr:.2f}

B. Comparison Model (Decision Tree Regressor):
- Predicted Daily Consumption  : {forecast_kwh_dt:.2f} kWh/day
- Estimated Daily Bill         : Rs. {daily_cost_dt:.2f}
- Projected 30-Day Consumption : {monthly_kwh_dt:.1f} kWh/month
- Projected 30-Day Bill        : Rs. {monthly_cost_dt:.2f}

Formula:
  Daily Bill Estimate = Predicted Daily kWh * Tariff Rate
  30-Day Bill Estimate = (Predicted Daily kWh * 30 days) * Tariff Rate
======================================================================
"""

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[Saved] Detailed evaluation results saved to: {METRICS_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
