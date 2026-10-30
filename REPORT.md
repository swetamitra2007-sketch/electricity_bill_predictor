# Electricity Bill Predictor

## 1. Project Overview

The **Electricity Bill Predictor** is a machine learning project that analyzes historical household electricity consumption and predicts daily energy usage. It also estimates electricity costs using an assumed tariff of ₹7.50 per kWh.

The project uses the Individual Household Electric Power Consumption dataset from the UCI Machine Learning Repository. Historical minute-level readings are processed into daily consumption data, explored through visualizations, and used to train and evaluate two regression models.

### Objectives

- Process raw household electricity consumption data.
- Analyze daily, weekly, and seasonal consumption patterns.
- Build machine learning models to predict daily electricity consumption.
- Evaluate model performance using standard regression metrics.
- Estimate daily and monthly electricity costs from predicted consumption.

### Technologies Used

- **Programming language:** Python
- **Development environment:** Visual Studio Code
- **Libraries:** Pandas, NumPy, Matplotlib, and Scikit-learn
- **Models:** Linear Regression and Decision Tree Regression
- **Dataset:** UCI Individual Household Electric Power Consumption

---

## 2. Dataset Description

The project uses the Individual Household Electric Power Consumption dataset, which contains measurements of a household's electricity usage at one-minute intervals.

The original dataset contains **2,075,259 observations**, covering December 2006 to November 2010.

The primary variable used in this project is `Global_active_power`, which represents the household's global active power in kilowatts (kW).

### Dataset Characteristics

| Property | Value |
| :--- | :--- |
| Original observations | 2,075,259 |
| Original time interval | One minute |
| Original data period | December 2006 – November 2010 |
| Missing values in the primary variable | 25,979 |
| Missing-value percentage | Approximately 1.25% |
| Processed observations | 1,440 daily records |
| Processed date range | December 17, 2006 – November 25, 2010 |

The processed dataset contains daily electricity consumption values derived from the original minute-level measurements.

---

## 3. Data Preprocessing

Data preprocessing converts the raw dataset into a format suitable for exploratory data analysis and machine learning.

### 3.1 Handling Missing Values

Missing values in `Global_active_power` are handled using forward filling followed by backward filling:

```python
df["Global_active_power"] = (
    df["Global_active_power"].ffill().bfill()
)
```

Forward filling replaces a missing value with the previous available value. Backward filling handles any remaining missing values, particularly at the beginning of the dataset.

**Limitation:** Backward filling can use a future observation to fill an earlier missing value. Furthermore, imputation is performed globally across the full series prior to the chronological split. This is a preprocessing limitation and should be considered when interpreting the results.

### 3.2 Converting Minute-Level Readings into Daily Consumption

The `Global_active_power` readings are measured in kilowatts (kW). Since each reading represents approximately one minute, the energy contribution of each reading is calculated by dividing the power value by 60.

Daily electricity consumption is calculated as:

$$
E_{\text{daily}} = \sum_{i=1}^{1440} \frac{P_i}{60}
$$

Where:

- $E_{\text{daily}}$ is daily energy consumption in kWh.
- $P_i$ is the active power reading in kW.
- 1,440 represents the number of minutes in a complete day.

Days with incomplete boundary data are excluded from the processed daily dataset.

### 3.3 Processed Dataset

After preprocessing and removal of incomplete boundary days, the dataset contains **1,440 daily observations**, covering December 17, 2006, to November 25, 2010.

The processed dataset is saved as:

`data/daily_power_consumption.csv`

---

## 4. Exploratory Data Analysis

Exploratory Data Analysis (EDA) is used to understand consumption levels and identify daily, weekly, and seasonal patterns.

### 4.1 Daily Consumption Statistics

| Statistic | Consumption |
| :--- | :---: |
| Mean daily consumption | 26.06 kWh |
| Standard deviation | 10.07 kWh |
| Minimum daily consumption | 3.50 kWh |
| Maximum daily consumption | 79.56 kWh |

The average daily consumption is approximately 26.06 kWh. The standard deviation indicates that daily electricity usage varies considerably across the dataset.

### 4.2 Weekday and Weekend Patterns

| Day type | Average daily consumption |
| :--- | :---: |
| Weekday | 24.81 kWh |
| Weekend | 29.18 kWh |

Weekend consumption is approximately 17.6% higher than weekday consumption.

The percentage difference is calculated as:

$$
\text{Increase} = \frac{29.18 - 24.81}{24.81} \times 100 \approx 17.6\%
$$

This indicates that household electricity usage in the dataset tends to be higher on weekends.

### 4.3 Monthly Consumption Patterns

The highest reported monthly average is in December, at approximately 35.66 kWh per day. The lowest is in August, at approximately 14.03 kWh per day.

These differences indicate seasonal variation in household electricity usage within the historical dataset. They do not establish that the same pattern will occur in every household or location.

### 4.4 Visualizations

The project generates the following visualizations:

- `outputs/daily_consumption_trend.png` — displays the trend in daily electricity consumption.
- `outputs/weekly_and_seasonal_patterns.png` — compares weekly and seasonal consumption patterns.
- `outputs/actual_vs_predicted_linear_regression.png` — compares actual and predicted consumption for Linear Regression.
- `outputs/model_comparison.png` — presents the comparison between the trained models.

These visualizations help interpret the data and evaluate how closely model predictions follow actual consumption.

---

## 5. Feature Engineering

Feature engineering creates input variables that help the machine learning models learn patterns from historical consumption.

The project uses historical consumption values and calendar-based features.

### 5.1 Lag Features

Lag features represent electricity consumption on previous days.

- `lag_1`: consumption one day earlier.
- `lag_2`: consumption two days earlier.
- `lag_3`: consumption three days earlier.
- `lag_7`: consumption seven days earlier.

For example:

$$
\text{lag}_1(t) = E_{t-1}
$$

Where $E_{t-1}$ is the previous day's electricity consumption.

### 5.2 Rolling Mean

The feature `rolling_mean_7` represents the average consumption over the preceding seven days.

$$
\text{RollingMean}_7(t) = \frac{1}{7} \sum_{i=1}^{7} E_{t-i}
$$

The implementation uses a shift before calculating the rolling mean:

```python
df["rolling_mean_7"] = (
    df["daily_consumption_kwh"]
    .shift(1)
    .rolling(window=7)
    .mean()
)
```

Shifting the series by one day prevents the current day's target value from being included in its own rolling-mean feature.

### 5.3 Calendar Features

The project also uses these calendar-based features:

- `day_of_week`: the day of the week (0 = Monday, 6 = Sunday).
- `is_weekend`: indicates whether the day falls on a weekend (1 for Saturday/Sunday, 0 for weekdays).
- `month`: the month of the year (1 to 12).
- `day_of_month`: the day of the month (1 to 31).

Together, the lag, rolling-mean, and calendar features provide historical and time-related information for predicting electricity consumption.

After creating the features and removing rows without sufficient historical data (the first 7 days), **1,433 feature rows** remain.

---

## 6. Model Development

Two regression models are trained to predict daily electricity consumption.

### 6.1 Linear Regression

Ordinary Least-Squares Linear Regression estimates the relationship between the input features and daily electricity consumption.

Its general equation is:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n
$$

Where:

- $\hat{y}$ is the predicted daily consumption.
- $\beta_0$ is the intercept ($+1.6551$).
- $\beta_1, \ldots, \beta_n$ are the learned feature coefficients.
- $x_1, \ldots, x_n$ represent the input features.

Linear Regression provides a baseline model for the prediction task.

### 6.2 Decision Tree Regression

Decision Tree Regression predicts consumption by dividing the feature space into regions based on learned decision rules.

The model is configured with a maximum depth of five:

```python
DecisionTreeRegressor(max_depth=5, random_state=42)
```

The depth limit helps control model complexity. Decision Trees can capture nonlinear relationships, although their performance depends on the data and model configuration.

### 6.3 Training and Testing Split

The feature dataset is divided chronologically rather than randomly.

| Dataset | Number of observations | Date range |
| :--- | :---: | :--- |
| Training set | 1,146 | December 24, 2006 – February 11, 2010 |
| Testing set | 287 | February 12, 2010 – November 25, 2010 |
| Total | 1,433 | December 24, 2006 – November 25, 2010 |

This approach trains the models on earlier observations and evaluates them on later observations, which better reflects a time-ordered prediction task.

---

## 7. Model Evaluation

The models are evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination ($R^2$).

### 7.1 Evaluation Metrics

**Mean Absolute Error (MAE)** measures the average absolute difference between actual and predicted values.

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

A lower MAE indicates smaller average prediction errors.

**Root Mean Squared Error (RMSE)** measures the square root of the average squared prediction error.

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

RMSE penalizes larger errors more heavily than MAE.

**Coefficient of Determination ($R^2$)** measures how well the model explains the variation in the target values relative to a mean-based baseline.

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

An $R^2$ value closer to 1 generally indicates a better fit. A value of 0 indicates performance equivalent to predicting the test-set mean under the standard definition, while a negative value indicates worse performance than that baseline.

### 7.2 Results

| Model | MAE (kWh/day) | RMSE (kWh/day) | $R^2$ |
| :--- | :---: | :---: | :---: |
| Linear Regression | 4.557 | 6.214 | 0.328 |
| Decision Tree Regression | 4.795 | 6.551 | 0.253 |

### 7.3 Comparison

Linear Regression performs better than Decision Tree Regression on all three reported evaluation metrics on this test set:

- It has a lower MAE (4.557 vs. 4.795 kWh/day).
- It has a lower RMSE (6.214 vs. 6.551 kWh/day).
- It has a higher $R^2$ (0.328 vs. 0.253).

Therefore, **Linear Regression is the better-performing model among the two models evaluated in this experiment**.

Its MAE of approximately 4.56 kWh/day means that its predictions differ from the actual test values by about 4.56 kWh per day on average (relative to the test-set mean of 23.74 kWh/day and overall historical mean of 26.06 kWh/day).

The $R^2$ value of 0.328 indicates that the model explains a portion of the variation in the test data. Its predictions are useful as a baseline, but the results also indicate room for improvement.

---

## 8. Electricity Consumption Prediction

The project uses the trained models to produce a historical forecast for November 26, 2010, immediately following the last date in the processed dataset.

### 8.1 Predicted Daily Consumption

| Model | Predicted consumption |
| :--- | :---: |
| Linear Regression | 25.11 kWh/day |
| Decision Tree Regression | 25.32 kWh/day |

The Linear Regression prediction is approximately 25.11 kWh/day (raw float: `25.109658`), while the Decision Tree prediction is approximately 25.32 kWh/day (raw float: `25.318455`).

These are historical forecasts generated from the dataset ending in November 2010. They are **not predictions for the present day or for a current household**.

---

## 9. Electricity Bill Estimation

The project converts predicted electricity consumption into an estimated cost using an assumed tariff of **₹7.50 per kWh**.

Actual electricity bills may include fixed charges, taxes, subsidies, slab rates, and other adjustments. The estimates below consider only the assumed per-unit tariff.

### 9.1 Formula

$$
\text{Estimated Bill} = \text{Energy Consumption} \times \text{Tariff}
$$

Where:

- Energy consumption is measured in kWh.
- The assumed tariff is ₹7.50 per kWh.

### 9.2 Daily Cost Estimate

- **Linear Regression:**  
  $$25.109658 \times 7.50 = 188.322435 \approx 188.32$$  
  Estimated daily bill: **₹188.32**

- **Decision Tree Regression:**  
  $$25.318455 \times 7.50 = 189.8884125 \approx 189.89$$  
  Estimated daily bill: **₹189.89**

The calculations in the pipeline use the models' unrounded predictions. When calculating from displayed two-decimal figures (25.11 × ₹7.50 = ₹188.325 ≈ **₹188.33**), slight rounding differences of one paisa appear.

### 9.3 Thirty-Day Estimate

The monthly estimate assumes that the predicted daily consumption remains constant for 30 days.

| Model | Estimated 30-day consumption | Estimated bill |
| :--- | :---: | :---: |
| Linear Regression | 753.29 kWh | ₹5,649.67 |
| Decision Tree Regression | 759.55 kWh | ₹5,696.65 |

**For Linear Regression:**

$$
\text{30-day Consumption} = 25.109658 \times 30 = 753.28974 \approx 753.29 \text{ kWh}
$$

$$
\text{Estimated Bill} = 753.28974 \times 7.50 = 5649.67305 \approx 5649.67
$$

Estimated 30-day bill: **₹5,649.67**  
*(Manual calculation from rounded text values: 753.3 × ₹7.50 = **₹5,649.75**.)*

**For Decision Tree Regression:**

$$
\text{30-day Consumption} = 25.318455 \times 30 = 759.55365 \approx 759.55 \text{ kWh}
$$

$$
\text{Estimated Bill} = 759.55365 \times 7.50 = 5696.652375 \approx 5696.65
$$

Estimated 30-day bill: **₹5,696.65**  
*(Manual calculation from rounded text values: 759.6 × ₹7.50 = **₹5,697.00**.)*

These figures are illustrative estimates based on a constant predicted daily consumption and the assumed tariff. They are not actual utility bills.

---

## 10. Limitations and Future Improvements

### 10.1 Limitations

- **Limited predictive performance:** The Linear Regression model achieves an $R^2$ of 0.328, indicating that much of the variation in daily consumption remains unexplained.
- **Historical dataset:** The data covers a single household from 2006 to 2010. The consumption patterns may not represent other households, regions, or present-day electricity usage.
- **Missing-value imputation:** Forward filling and backward filling provide a straightforward way to handle missing data, but they may introduce bias. In particular, backward filling can use information from a later observation.
- **Simplified tariff:** The bill calculation assumes a constant rate of ₹7.50 per kWh. Real electricity tariffs vary by location, consumer category, and consumption slab.
- **Constant monthly projection:** The 30-day estimate multiplies one predicted daily value by 30. It does not predict the individual consumption of each day in the month.
- **Limited model comparison:** Only Linear Regression and a Decision Tree with a maximum depth of five are evaluated. Other models and hyperparameter configurations may perform differently.

### 10.2 Future Improvements

The project could be improved by:

1. Evaluating additional models such as Random Forest Regression and Gradient Boosting.
2. Tuning model hyperparameters using time-series-aware validation.
3. Comparing the models against simple baseline forecasts.
4. Investigating missing-data handling methods that avoid using future observations to fill past values.
5. Adding more relevant features, where available, such as temperature, holidays, and occupancy patterns.
6. Using location-specific electricity tariffs and applicable billing rules.
7. Predicting daily consumption across an entire future month instead of multiplying a single daily prediction.
8. Testing the model on more recent data and additional households.
9. Providing prediction intervals or uncertainty estimates alongside point predictions.

---

## 11. Project Structure and Reproducibility

### 11.1 Project Structure

```text
electricity_bill_predictor/
│
├── data/
│   ├── household_power_consumption.txt
│   └── daily_power_consumption.csv
│
├── outputs/
│   ├── daily_consumption_trend.png
│   ├── weekly_and_seasonal_patterns.png
│   ├── actual_vs_predicted_linear_regression.png
│   ├── model_comparison.png
│   └── evaluation_results.txt
│
├── data_preprocessing.py
├── download_data.py
├── main.py
├── eda.py
├── train_predict.py
├── requirements.txt
├── README.md
├── REPORT.md
├── .gitignore
└── .vscode/
    └── launch.json
```

The raw dataset is large and should not be committed to GitHub. The `.gitignore` file is used to exclude the raw and processed data files from version control where configured.

### 11.2 Installation

Install Python and the required libraries. Open a terminal in the project directory:

```cmd
cd /d "C:\Users\SWETA MITRA\Downloads\electricity_bill_predictor"
py -m pip install -r requirements.txt
```

The project uses the following Python libraries:

```text
pandas
numpy
matplotlib
scikit-learn
```

### 11.3 Running the Project

From the project directory, run:

```cmd
py main.py
```

The main script coordinates the project pipeline. Depending on the implementation, this includes preprocessing the dataset, exploratory analysis, model training, evaluation, prediction, and output generation.

If the raw dataset is not available locally, run the data-download script first:

```cmd
py download_data.py
```

The raw dataset must be placed at the expected path:

`data/household_power_consumption.txt`

### 11.4 Output Files

After a successful run, the project generates daily consumption data, visualizations, and evaluation results. The main output files are stored in the `outputs/` directory.

The evaluation results are saved in:

`outputs/evaluation_results.txt`

### 11.5 Conclusion

The Electricity Bill Predictor demonstrates a complete introductory machine learning workflow, from preprocessing real-world electricity data to engineering features, training regression models, evaluating predictions, and estimating electricity costs.

Among the two evaluated models, Linear Regression achieves better test-set performance than the Decision Tree Regression model. However, its predictive performance remains limited, and the estimated electricity bill is based on a simplified tariff assumption.

The project provides a useful foundation for further work in time-series forecasting, energy analytics, and machine learning-based electricity consumption estimation.\n