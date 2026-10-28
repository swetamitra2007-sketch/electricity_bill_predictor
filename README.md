# Electricity Bill Predictor

**Singularity Research Lab — AI Division Admission Task**  
**Author:** First-Year CSE Student  
**Technology Stack:** Python, pandas, NumPy, matplotlib, scikit-learn, VS Code, GitHub  

---

## 1. Project Overview

The **Electricity Bill Predictor** is a machine learning system designed to analyze historical household electricity consumption, identify daily and seasonal patterns, and accurately forecast future electricity usage and monthly utility bills.

### Key Highlights:
- **Official Dataset:** Uses the UCI Machine Learning Repository's *Individual Household Electric Power Consumption* dataset (~2.07 million minute-level measurements).
- **Physical Accuracy:** Correctly converts raw 1-minute power readings (kilowatts, kW) into daily energy consumption (kilowatt-hours, kWh).
- **Leak-Free Feature Engineering:** Creates lag features and calendar features using strictly past information to prevent future data leakage.
- **Chronological Time-Series Split:** Evaluates models on unseen future time intervals (80% chronological train, 20% test).
- **Primary Model:** Linear Regression (Main Model required by task).
- **Comparison Model:** Decision Tree Regressor (Demonstrating non-linear comparative analysis).
- **Evaluation:** Evaluated using Mean Absolute Error (MAE) in real kWh units, along with RMSE and $R^2$.
- **Financial Forecasting:** Predicts the next day's electricity consumption and estimates the monthly electricity bill using an illustrative tariff rate.

---

## 2. Project Directory Structure

```text
├── data/
│   ├── household_power_consumption.txt   # Official raw UCI dataset (created by download_data.py)
│   └── daily_power_consumption.csv       # Cleaned, daily-aggregated energy dataset (kWh)
├── outputs/
│   ├── daily_consumption_trend.png       # 4-year daily time-series and 30-day moving average
│   ├── weekly_and_seasonal_patterns.png  # Day-of-week and monthly consumption patterns
│   ├── actual_vs_predicted_linear_regression.png # Main model test predictions vs actuals
│   ├── model_comparison.png              # Linear Regression vs. Decision Tree Regressor
│   └── evaluation_results.txt            # Stored test metrics, forecast, and bill calculation
├── download_data.py                      # Automated downloader for the official UCI dataset
├── data_preprocessing.py                 # Data cleaning and kW -> daily kWh conversion
├── eda.py                                # Exploratory Data Analysis & visual pattern extraction
├── train_predict.py                      # Feature engineering, model training, evaluation & bill estimate
├── main.py                               # Master script to run the entire pipeline end-to-end
├── requirements.txt                      # List of Python dependencies
└── README.md                             # Complete project documentation and interview preparation
```

---

## 3. Explaining Each File in Simple Language

| File | Purpose |
| :--- | :--- |
| `download_data.py` | Automatically connects to the UCI repository, downloads the 20MB compressed zip, extracts the 126.8MB `household_power_consumption.txt` file into `data/`, and cleans up the temporary zip file. If offline, it provides clear manual download instructions. |
| `data_preprocessing.py` | Reads the 2+ million minute-level records, identifies missing values (`?`) and imputes them via forward-fill, converts minute-level active power (kW) to energy (kWh), sums them into full 24-hour days (1,440 minutes), and saves `daily_power_consumption.csv`. |
| `eda.py` | Analyzes historical electricity usage trends. It calculates day-of-week consumption (weekday vs. weekend) and seasonal changes (winter heating vs. summer dips), generating high-resolution charts in `outputs/`. |
| `train_predict.py` | Creates lag features (`lag_1`, `lag_2`, `lag_3`, `lag_7`), rolling 7-day averages (shifted to avoid data leakage), and calendar variables. Performs an 80/20 chronological train-test split, trains Linear Regression and Decision Tree models, evaluates Test MAE, plots predictions, and computes the next-day forecast and electricity bill. |
| `main.py` | The master entry-point script. It executes all four pipeline stages in logical sequence and prints clear terminal feedback. |
| `requirements.txt` | Lists the exact Python packages required (`numpy`, `pandas`, `matplotlib`, `scikit-learn`). |

---

## 4. How to Set Up and Run the Project

### Prerequisites
Make sure you have **Python 3.10+** installed on your system.

### Step 1: Open Terminal in VS Code
Open VS Code, press ``Ctrl + ` `` (backtick) or select **Terminal > New Terminal**, and ensure you are in the project folder:
```bash
cd "C:\Users\SWETA MITRA\Downloads\electricity_bill_predictor"
```

### Step 2: Install Required Dependencies
Install the required scientific packages:
```bash
pip install -r requirements.txt
```
*(On Windows systems using the Python launcher, you can also run: `py -m pip install -r requirements.txt`)*

### Step 3: Run the Entire Pipeline
Run the master script:
```bash
python main.py
```
*(Or `py main.py`)*

### Step 4 (Optional): Running Individual Stages
You can also run any stage independently:
```bash
# 1. Download dataset
python download_data.py

# 2. Preprocess & convert power readings
python data_preprocessing.py

# 3. Generate exploratory plots
python eda.py

# 4. Train models, evaluate MAE & forecast bill
python train_predict.py
```

All generated graphs and reports will be saved directly into the `outputs/` folder.

---

## 5. Mathematical Principle: Converting Power (kW) to Daily Energy (kWh)

A fundamental concept in electrical physics and engineering is the distinction between **Power** and **Energy**:

- **Power ($P$):** The instantaneous rate at which electricity is consumed, measured in **Kilowatts (kW)**.
- **Energy ($E$):** The total electricity consumed over a duration of time ($t$), measured in **Kilowatt-hours (kWh)**:
  $$\text{Energy (kWh)} = \text{Power (kW)} \times \text{Time (hours)}$$

### Derivation for Minute-Level Data:
1. In the UCI dataset, `Global_active_power` is recorded every **1 minute**.
2. One minute expressed in hours is:
   $$\Delta t = \frac{1 \text{ minute}}{60 \text{ minutes/hour}} = \frac{1}{60} \text{ hours}$$
3. Therefore, the energy consumed during any single 1-minute interval is:
   $$E_{\text{minute}} = \text{Global\_active\_power (kW)} \times \frac{1}{60} \text{ kWh}$$
4. A complete day contains $24 \text{ hours} \times 60 \text{ minutes} = 1,440 \text{ minutes}$.
5. The total daily energy consumption in kWh is the sum of all 1,440 minute readings:
   $$E_{\text{day}} = \sum_{m=1}^{1440} \frac{\text{Global\_active\_power}_m}{60} = \frac{1}{60} \sum_{m=1}^{1440} \text{Global\_active\_power}_m \text{ (kWh)}$$

This exact formula is implemented in `data_preprocessing.py`.

---

## 6. Feature Engineering & Preventing Data Leakage

### A. The Features Created
To predict tomorrow's electricity consumption ($y_t$), we supply the model with information available up to today ($t-1$ and prior):

1. **Lag Features (Autoregressive Memory):**
   - `lag_1`: Electricity consumed yesterday ($t-1$).
   - `lag_2`: Electricity consumed 2 days ago ($t-2$).
   - `lag_3`: Electricity consumed 3 days ago ($t-3$).
   - `lag_7`: Electricity consumed on the exact same weekday last week ($t-7$).
2. **Rolling Window Feature:**
   - `rolling_mean_7`: The 7-day moving average of consumption.
3. **Calendar Features:**
   - `day_of_week`: Day index (0 for Monday, 6 for Sunday).
   - `is_weekend`: Binary flag (1 for Saturday/Sunday, 0 for weekdays).
   - `month`: Numerical month (1 to 12, reflecting winter/summer seasonality).
   - `day_of_month`: Day of the month (1 to 31).

### B. Why Random Train-Test Split Causes Data Leakage
In standard tabular machine learning (e.g., predicting house prices or flower species), randomly shuffling rows (`train_test_split(shuffle=True)`) is standard practice.

However, **in time series forecasting, random shuffling is a critical error**:
- If we shuffle the data, the model might be trained on electricity consumption from December 2009 to predict consumption in January 2008.
- The model would learn from future information that would **never be available in real life**, causing artificial "data leakage." The model would appear to have 99% accuracy during testing, but would fail completely in real-world deployment.

### C. The Solution: Strict Chronological Splitting
We split the 1,433 days strictly by date order:
- **Training Set (80%):** The first 1,146 consecutive days (December 24, 2006 to February 11, 2010).
- **Test Set (20%):** The subsequent 287 consecutive days (February 12, 2010 to November 25, 2010).

The model is trained *only* on the past, and tested *strictly* on unseen future days.

---

## 7. Actual Model Evaluation Results

The models were executed on the official dataset. Below are the actual evaluation results:

| Metric | Main Model: Linear Regression | Comparison Model: Decision Tree Regressor (`max_depth=5`) |
| :--- | :---: | :---: |
| **Mean Absolute Error (MAE)** | **4.557 kWh** *(Winner)* | **4.795 kWh** |
| **Root Mean Squared Error (RMSE)** | **6.214 kWh** | **6.551 kWh** |
| **$R^2$ Score (Coefficient of Determination)** | **0.328** | **0.253** |

### Key Observations:
1. **Linear Regression Outperforms Decision Tree:**
   Linear Regression achieved a lower test MAE (**4.557 kWh** vs. **4.795 kWh**). In time series with continuous trends and autoregressive lags, regularized linear models often generalize better than axis-aligned tree partitions, which struggle to extrapolate outside their observed bounding boxes.
2. **Feature Weights (Coefficients in Linear Regression):**
   - `rolling_mean_7` (+0.3891) and `lag_1` (+0.2693) are the strongest predictors, proving that recent consumption habits heavily govern tomorrow's power draw.
   - `is_weekend` (+3.4931) shows that weekend days add roughly 3.5 kWh of extra consumption as residents stay home and use more appliances.

---

## 8. Next-Day Consumption Forecast & Electricity Bill Estimation

### A. Forecasting Tomorrow's Energy
Using the latest recorded date (**November 25, 2010**, actual consumption: 23.85 kWh), the trained Linear Regression model generated the next-day forecast:
- **Forecast Target Date:** November 26, 2010 (Friday)
- **Predicted Daily Consumption:** **25.11 kWh**

### B. Illustrative Electricity Bill Calculation
To calculate an estimated utility bill, we apply a transparent, stated residential tariff:
- **Assumed Tariff Rate:** **Rs. 7.50 per kWh** (or **$0.15 per kWh**)

**Estimated Daily Cost:** 25.11 × ₹7.50 = **₹188.33**

**Estimated 30-Day Monthly Consumption:** 25.11 × 30 = **753.3 kWh**

**Estimated 30-Day Monthly Bill:** 753.3 × ₹7.50 = **₹5,649.75**

---

## 9. Visual Output Gallery

All charts are generated at 300 DPI and saved in the `outputs/` folder:

1. **`outputs/daily_consumption_trend.png`**  
   Shows the full 4-year daily time-series from 2006 to 2010 alongside a 30-day moving average, revealing clear annual cycles and holiday spikes.
2. **`outputs/weekly_and_seasonal_patterns.png`**  
   Illustrates average consumption by day of the week (Saturday/Sunday showing the highest consumption at ~29.5 kWh/day) and seasonal changes (December peak at 35.7 kWh/day vs. August low at 14.0 kWh/day).
3. **`outputs/actual_vs_predicted_linear_regression.png`**  
   Overlays actual test set consumption against Linear Regression predictions across all 287 test days.
4. **`outputs/model_comparison.png`**  
   Two-panel comparative chart contrasting Linear Regression predictions against Decision Tree Regressor predictions.

---

## 10. Technical Interview Preparation (Q&A for CSE Freshers)

Here are the most likely technical interview questions and concise, impressive answers tailored for a first-year student:

### Q1: Why did you choose Mean Absolute Error (MAE) instead of Mean Squared Error (MSE) as your primary evaluation metric?
> **Answer:**  
> "MAE measures the average magnitude of errors directly in the original physical units—kilowatt-hours (kWh). An MAE of 4.56 kWh means that, on average, our forecast is off by approximately 4.56 units of electricity. In contrast, MSE squares the errors ($(\text{kWh})^2$), which distorts interpretability and heavily penalizes occasional unusual spikes (such as running power tools for one afternoon)."

### Q2: What is 'data leakage', and how did your pipeline prevent it?
> **Answer:**  
> "Data leakage occurs when information from the future or test set inadvertently influences model training. We prevented data leakage in two key ways:
> 1. **Chronological Splitting:** We used the first 80% of days for training and the remaining 20% future days for testing, avoiding random row shuffling.
> 2. **Shifted Features:** All lag and rolling window features were calculated strictly on prior observations (`shift(1)`), ensuring the target day's value was never part of its own feature calculation."

### Q3: Why does Linear Regression perform better than a Decision Tree on this task?
> **Answer:**  
> "Decision trees create step-wise, axis-parallel splits. They cannot extrapolate trends beyond the minimum or maximum values seen in the training data. Linear Regression models the smooth linear relationship between yesterday's power draw, weekly averages, and upcoming consumption very effectively, leading to a lower test MAE."

### Q4: How did you convert power in kW to energy in kWh?
> **Answer:**  
> "Power is the instantaneous rate of energy use (kW), whereas energy is power multiplied by time (kWh). Since the sensor recorded power every 1 minute ($\frac{1}{60}\text{ hour}$), each minute reading equals $\frac{\text{kW}}{60}\text{ kWh}$. Summing all 1,440 minute readings across a day yields the total daily energy consumption in kWh."

### Q5: How could this project be improved in a future version?
> **Answer:**  
> "1. Incorporating external weather data (outdoor temperature and humidity), because heating and air conditioning drive major power consumption changes.  
> 2. Testing specialized time series models such as ARIMA, SARIMAX, or Prophet.  
> 3. Introducing tiered/slab-based tariff structures rather than a flat per-unit rate to reflect real-world billing tiers."

---
*Created for Singularity Research Lab — AI Division Task.*
