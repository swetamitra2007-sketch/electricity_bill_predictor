# Engineering & Research Report: Electricity Bill Predictor

**Task:** Singularity Research Lab — AI Division Admission Task  
**Project Title:** Household Electricity Consumption Modeling & Bill Forecasting  
**Author:** First-Year Computer Science and Engineering Student  
**Primary Frameworks:** Python, pandas, NumPy, scikit-learn, matplotlib  

---

## Executive Summary

This report documents the design, mathematical foundations, implementation, and experimental evaluation of an **Electricity Bill Predictor** built using the benchmark *Individual Household Electric Power Consumption* dataset from the UCI Machine Learning Repository.

The system processes **2,075,259 minute-level power measurements** across 4 years (December 2006 – November 2010), converts raw power ($\text{kW}$) into daily energy ($\text{kWh}$), engineers autoregressive lag and calendar features with strict prevention of data leakage, and evaluates two regression architectures using a **chronological time-series split (80% train, 20% test)**.

### Primary Verified Results
- **Main Model (Linear Regression):** Mean Absolute Error (MAE) of **4.557 kWh/day** ($R^2 = 0.328$, RMSE = $6.214\text{ kWh/day}$).
- **Comparison Model (Decision Tree Regressor, max depth 5):** MAE of **4.795 kWh/day** ($R^2 = 0.253$, RMSE = $6.551\text{ kWh/day}$).
- **Historical Next-Day Forecast (Target Date: November 26, 2010):**
  - Linear Regression: **25.11 kWh/day**
  - Decision Tree Regressor: **25.32 kWh/day**
- **Illustrative Utility Bill Estimates (Assumed Flat Rate: ₹7.50/kWh):**
  - Linear Regression: **₹188.32/day** | **₹5,649.67/month** (30 days)
  - Decision Tree Regressor: **₹189.89/day** | **₹5,696.65/month** (30 days)
  *(Important: These estimates are illustrative calculations based on the assumed flat tariff, not real utility invoices.)*

---

## 1. Project Objective

The core objective is to design an automated end-to-end pipeline capable of:
1. Converting high-frequency, minute-level instantaneous power readings into physically sound daily energy consumption totals.
2. Uncovering behavioral and seasonal consumption patterns (weekday vs. weekend habits, winter vs. summer peaks).
3. Formulating a feature-engineering strategy that eliminates future data leakage in temporal forecasting.
4. Training a **Linear Regression** model as the primary predictive engine, and benchmarking it against an optional **Decision Tree Regressor**.
5. Measuring predictive error using **Mean Absolute Error (MAE)** in physically interpretable units ($\text{kWh/day}$).
6. Generating an out-of-sample historical forecast for the day immediately following the dataset's conclusion, and computing illustrative electricity costs.

---

## 2. Dataset Description & Official Source

- **Name:** Individual Household Electric Power Consumption
- **Originating Institution:** EDF (Électricité de France) R&D / UCI Machine Learning Repository
- **Official URL:** `https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption`
- **Measurement Span:** December 16, 2006 (17:24:00) to November 26, 2010 (21:02:00) — nearly 4 full calendar years.
- **Total Records:** 2,075,259 minute-level observations across 9 attributes.
- **Attributes Utilized:**
  - `Date`: String in format `DD/MM/YYYY`.
  - `Time`: String in format `HH:MM:SS`.
  - `Global_active_power`: Household global minute-averaged active power in kilowatts ($\text{kW}$).

---

## 3. Data Preprocessing & Physical Unit Conversion

### 3.1 Handling Missing Readings
The official dataset represents missing sensor transmissions using a semicolon-delimited question mark (`?`). 
- **Missing Observations:** 25,979 records (**1.25%** of the entire 2.07M dataset).
- **Imputation Strategy:** Continuous electrical appliances exhibit physical inertia; power demand does not randomly drop to zero during transient meter communication interruptions. We applied **forward-fill (`ffill`) followed by backward-fill (`bfill`)** at the minute level. This carries forward the last confirmed load reading, preserving time-series continuity without introducing synthetic distribution distortions.

### 3.2 Mathematical Derivation: Converting Power (kW) to Energy (kWh)
In physical sciences, **Power ($P$)** is an instantaneous rate of energy transfer, whereas **Energy ($E$)** is the accumulation of power over an interval of time ($t$):
$$\text{Energy (kWh)} = \text{Power (kW)} \times \text{Time (hours)}$$

Because each observation in this dataset represents an average over $\Delta t = 1\text{ minute} = \frac{1}{60}\text{ hour}$, the energy consumed in minute $m$ is:
$$E_m = \text{Global\_active\_power}_m \times \frac{1}{60} \text{ kWh}$$

For a complete 24-hour day containing $24 \times 60 = 1,440\text{ minutes}$, the aggregate daily energy consumption is:
$$E_{\text{day}} = \sum_{m=1}^{1440} \frac{\text{Global\_active\_power}_m}{60} = \frac{1}{60} \sum_{m=1}^{1440} \text{Global\_active\_power}_m \quad (\text{kWh/day})$$

### 3.3 Boundary Filtering & Continuity Verification
- The raw dataset begins late in the afternoon on December 16, 2006 (only 396 minutes recorded) and ends in the evening on November 26, 2010.
- Days with fewer than 1,300 valid minutes were filtered out, safely removing only the 2 partial boundary days.
- **Final Clean Time Series:** Exactly **1,440 continuous daily records** spanning from **December 17, 2006 to November 25, 2010**, with **zero missing calendar dates** in between.
- **Daily Consumption Baseline Statistics:**
  - Mean: $26.06\text{ kWh/day}$
  - Standard Deviation: $10.07\text{ kWh/day}$
  - Minimum: $3.50\text{ kWh/day}$
  - Maximum: $79.56\text{ kWh/day}$

---

## 4. Exploratory Data Analysis & Consumption Patterns

Exploratory analysis was conducted in [`eda.py`](file:///c:/Users/SWETA%20MITRA/.antigravity-ide/eda.py) and confirmed the following key behaviors:

### 4.1 Day-of-Week Patterns (Human Habit Cycle)
| Day | Mean Consumption ($\pm$ Std Dev) |
| :--- | :---: |
| Monday | $23.83 \pm 8.51\text{ kWh}$ |
| Tuesday | $25.61 \pm 9.39\text{ kWh}$ |
| Wednesday | $26.03 \pm 9.12\text{ kWh}$ |
| Thursday | $23.56 \pm 8.44\text{ kWh}$ |
| Friday | $25.01 \pm 8.24\text{ kWh}$ |
| **Saturday** | **$29.48 \pm 12.47\text{ kWh}$** |
| **Sunday** | **$28.88 \pm 11.92\text{ kWh}$** |

- **Weekday Average:** $24.81\text{ kWh/day}$
- **Weekend Average:** $29.18\text{ kWh/day}$
- **Key Insight:** The household consumes **17.6% more electricity on weekends** than on weekdays, as occupants spend more hours at home utilizing laundry appliances, entertainment systems, and electric cooking.

### 4.2 Monthly / Seasonal Trends
- **Peak Month:** **December ($35.66\text{ kWh/day}$)**, followed closely by January and February.
- **Lowest Month:** **August ($14.03\text{ kWh/day}$)**.
- **Key Insight:** Electricity demand is strongly driven by winter space heating and lighting during shorter daylight hours. In August, European vacation schedules and reduced heating needs cause power usage to drop to less than half of winter peaks.
- **Saved Visualizations:**
  - `outputs/daily_consumption_trend.png`: 4-year time series overlaid with a 30-day moving average.
  - `outputs/weekly_and_seasonal_patterns.png`: Side-by-side bar charts of day-of-week and monthly seasonalities.

---

## 5. Feature Engineering Without Data Leakage

To forecast electricity consumption on day $t$ ($y_t$), the feature matrix $X_t$ must be constructed exclusively from information known prior to day $t$.

### 5.1 Engineered Feature Definitions
1. **Autoregressive Lags:**
   - `lag_1`: Consumption from yesterday ($y_{t-1}$).
   - `lag_2`: Consumption from 2 days prior ($y_{t-2}$).
   - `lag_3`: Consumption from 3 days prior ($y_{t-3}$).
   - `lag_7`: Consumption from the same day last week ($y_{t-7}$), capturing weekly periodicity.
2. **Rolling Window Memory:**
   - `rolling_mean_7`: 7-day trailing moving average. Crucially implemented as:
     ```python
     df['rolling_mean_7'] = df['daily_consumption_kwh'].shift(1).rolling(window=7).mean()
     ```
     The `.shift(1)` operation ensures that day $t$'s own observation is strictly excluded from the rolling window.
3. **Deterministic Calendar Features:**
   - `day_of_week` (0 to 6)
   - `is_weekend` (Binary indicator: 1 if Saturday/Sunday, else 0)
   - `month` (1 to 12)
   - `day_of_month` (1 to 31)

### 5.2 Drop of Initial Shift Rows
Because computing `lag_7` requires 7 days of prior history, the first 7 days (December 17–23, 2006) become `NaN` and are dropped. This leaves **1,433 complete feature rows** spanning **December 24, 2006 to November 25, 2010**.

---

## 6. Chronological Evaluation Strategy

Standard cross-validation (random shuffling) is invalid for time-series forecasting because training on future dates to predict past dates introduces severe artificial look-ahead bias (data leakage).

We applied a **strict Chronological Split**:
- **Chronological Split Ratio:** 80% Train, 20% Test.
- **Training Set:** First **1,146 consecutive days** (December 24, 2006 to February 11, 2010).
- **Test Set:** Subsequent **287 consecutive unseen future days** (February 12, 2010 to November 25, 2010).
- **Principle:** The models were fitted exclusively on historical data and tested on future data, simulating real production forecasting.

---

## 7. Model Architectures & Experimental Comparison

### 7.1 Main Model: Linear Regression
Linear Regression assumes the target consumption is a linear combination of input features:
$$\hat{y} = \beta_0 + \sum_{j=1}^9 \beta_j X_j$$
Trained using Ordinary Least Squares (`sklearn.linear_model.LinearRegression`).

**Fitted Feature Coefficients ($\beta_j$):**
- `rolling_mean_7` ($\beta = +0.3891$): The single strongest predictor; overall recent baseline consumption heavily anchors upcoming usage.
- `lag_1` ($\beta = +0.2693$): Yesterday's consumption provides immediate inertia.
- `is_weekend` ($\beta = +3.4931$): Explicitly adds $\approx 3.5\text{ kWh}$ to weekend predictions.
- `lag_3` ($\beta = +0.1909$), `lag_7` ($\beta = +0.0680$), `day_of_week` ($\beta = +0.2565$).
- `Intercept` ($\beta_0 = +1.6551$).

### 7.2 Comparison Model: Decision Tree Regressor
A non-linear tree partitioner (`sklearn.tree.DecisionTreeRegressor(max_depth=5, random_state=42)`). The `max_depth` parameter was capped at 5 to prevent overfitting to high-frequency training fluctuations.

---

## 8. Experimental Evaluation Results

Both models were evaluated on the **exact same 287-day test period** (February 12, 2010 to November 25, 2010).

### Comparison Table

| Metric | Main Model: Linear Regression | Comparison Model: Decision Tree Regressor (`max_depth=5`) | Winner |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **`4.557 kWh/day`** | `4.795 kWh/day` | **Linear Regression** |
| **Root Mean Squared Error (RMSE)** | **`6.214 kWh/day`** | `6.551 kWh/day` | **Linear Regression** |
| **Coefficient of Determination ($R^2$)** | **`0.328`** | `0.253` | **Linear Regression** |

### Analysis of Results
1. **Why MAE was chosen as the primary metric:**  
   MAE expresses prediction error directly in the original physical units ($\text{kWh/day}$). An MAE of **4.557 kWh/day** against an average daily consumption of $26.06\text{ kWh/day}$ represents an average percentage error of $\approx 17.5\%$, which is very solid for single-household consumption characterized by volatile occupant habits.
2. **Why Linear Regression outperformed Decision Tree:**  
   Decision trees create orthogonal step functions that cannot extrapolate beyond local training partitions. Linear Regression captures the continuous autoregressive decay of energy usage and rolling trends more effectively.

---

## 9. Historical Next-Day Forecast & Illustrative Bill Estimation

### 9.1 Context of the Next-Day Forecast
The dataset concludes on **Thursday, November 25, 2010** (actual consumption: $23.85\text{ kWh/day}$). Therefore, the "next-day forecast" produced by the models is an **out-of-sample historical forecast for Friday, November 26, 2010**, rather than a live contemporary prediction.

Constructing the feature vector for November 26, 2010:
- `lag_1` = $23.85\text{ kWh}$ (Nov 25)
- `lag_2` = $24.79\text{ kWh}$ (Nov 24)
- `lag_3` = $24.88\text{ kWh}$ (Nov 23)
- `lag_7` = $24.50\text{ kWh}$ (Nov 19)
- `rolling_mean_7` = $24.88\text{ kWh}$ (average of Nov 19–25)
- `day_of_week` = 4 (Friday), `is_weekend` = 0, `month` = 11, `day_of_month` = 26.

### 9.2 Forecasted Energy Output
- **Linear Regression Forecast:** **`25.11 kWh/day`** *(raw float: `25.109658`)*
- **Decision Tree Forecast:** **`25.32 kWh/day`** *(raw float: `25.318455`)*

### 9.3 Illustrative Utility Bill Calculations (Assumed Flat Rate: ₹7.50 / kWh)
*Disclaimer: The assumed rate of ₹7.50/kWh represents a realistic residential benchmark. Real electricity bills use tiered slab rates, fixed meter charges, and taxes.*

#### A. Main Model (Linear Regression)
- **Daily Bill Estimate:**
  $$\text{Daily Bill} = 25.109658 \times 7.50 = \text{\textbf{₹188.32}}$$
- **Projected 30-Day Monthly Consumption:**
  $$\text{30-Day Energy} = 25.109658 \times 30 = \text{\textbf{753.3 kWh/month}}$$
- **Projected 30-Day Monthly Bill:**
  $$\text{Monthly Bill} = 753.28974 \times 7.50 = \text{\textbf{₹5,649.67}}$$
  *(Note on rounding: Calculating from the 2-decimal rounded forecast of $25.11 \times 30 \times 7.50$ yields ₹5,649.75. The pipeline uses the full floating-point precision of 25.109658, resulting in ₹5,649.67.)*

#### B. Comparison Model (Decision Tree Regressor)
- **Daily Bill Estimate:**
  $$\text{Daily Bill} = 25.318455 \times 7.50 = \text{\textbf{₹189.89}}$$
- **Projected 30-Day Monthly Consumption:**
  $$\text{30-Day Energy} = 25.318455 \times 30 = \text{\textbf{759.6 kWh/month}}$$
- **Projected 30-Day Monthly Bill:**
  $$\text{Monthly Bill} = 759.55365 \times 7.50 = \text{\textbf{₹5,696.65}}$$
  *(Note on rounding: Calculating from $25.32 \times 30 \times 7.50$ yields ₹5,697.00. The floating-point calculation yields ₹5,696.65.)*

---

## 10. Limitations & Future Improvements

1. **Absence of Ambient Weather Data:** Heating and cooling (HVAC) represent the largest portion of residential energy demand. Integrating local outdoor temperature, solar irradiance, and humidity would significantly reduce the test MAE below 4 kWh.
2. **Slab-Based Tiered Tariffs:** Real electricity utility providers (such as state electricity boards) apply graduated billing slabs (e.g., ₹4/kWh for the first 100 units, ₹7/kWh for 101–300 units, and ₹10/kWh above 300 units). Future iterations should implement a piecewise tariff function.
3. **Advanced Time-Series Architectures:** Exploring models like SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors), Prophet, or LightGBM could capture complex multi-seasonal cycles (e.g., annual + weekly cycles) more flexibly.

---

## 11. Instructions to Reproduce the Experiment

To reproduce the verified experimental results from scratch:

### Step 1: Clone or Navigate to the Repository
```bash
cd "c:\Users\SWETA MITRA\.antigravity-ide"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Complete Master Pipeline
```bash
python main.py
```

### Step 4: Inspect Generated Outputs
All charts and evaluation summaries will update in the `outputs/` folder:
- [`outputs/actual_vs_predicted_linear_regression.png`](file:///c:/Users/SWETA%20MITRA/.antigravity-ide/outputs/actual_vs_predicted_linear_regression.png)
- [`outputs/model_comparison.png`](file:///c:/Users/SWETA%20MITRA/.antigravity-ide/outputs/model_comparison.png)
- [`outputs/daily_consumption_trend.png`](file:///c:/Users/SWETA%20MITRA/.antigravity-ide/outputs/daily_consumption_trend.png)
- [`outputs/weekly_and_seasonal_patterns.png`](file:///c:/Users/SWETA%20MITRA/.antigravity-ide/outputs/weekly_and_seasonal_patterns.png)
- [`outputs/evaluation_results.txt`](file:///c:/Users/SWETA%20MITRA/.antigravity-ide/outputs/evaluation_results.txt)

---
*Report submitted for the Singularity Research Lab — AI Division Task.*
