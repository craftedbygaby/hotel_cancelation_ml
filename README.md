# 🏨 Hotel Booking Cancellation Predictor — ML Analysis

**Hotel Cancellation ML** is a machine learning project focused on predicting whether a hotel booking will be cancelled, based on booking characteristics and customer behaviour.

This project was developed as part of the **IronHack Data Analytics Bootcamp** and covers the full ML pipeline: data cleaning, exploratory analysis, feature engineering, model building, tuning, and evaluation.

---

## Author

- Gabriela Cascione

---

## Data Sources

### Hotel Booking Demand

| Detail | Info |
|---|---|
| **Source** | [Kaggle — Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) |
| **File** | `hotel_bookings.csv` |
| **Records** | 119,390 rows |
| **Columns** | 32 |
| **Coverage** | 2015 — 2017 |
| **Note** | Real hotel booking data with personal identifiers removed. Columns `name`, `email`, `phone number`, and `credit_card` were artificially created and added. |
| **Source Paper** | *Hotel Booking Demand Datasets* — Nuno Antonio, Ana Almeida, Luis Nunes. Data in Brief, Volume 22, February 2019. |

---

## Project Objective

Predict whether a hotel booking will be cancelled (`is_canceled`: 0 or 1) using classification models, based on features such as lead time, deposit type, customer type, market segment, and seasonality.

**Key facts about the target variable:**
- ~63% of bookings were **not cancelled**
- ~37% of bookings were **cancelled**

This slight class imbalance will be accounted for during model training.

---

## ML Pipeline

### 1. Data Cleaning

The raw dataset (`hotel_bookings_raw_data.csv`) was cleaned using the following steps:

1. **Dropped columns** with excessive null values: `agent` and `company`
2. **Dropped rows** with missing values in `children` and `country` columns
3. **Removed 31,984 duplicate** entries
4. Final clean dataset contains **86,914 rows and 30 columns**
5. Exported as `hotel_booking_clean_data.csv`

### 2. Feature Engineering & Encoding

- Dropped data leakage columns: `reservation_status`, `reservation_status_date`, `assigned_room_type`
- Converted `country` codes to **continents** using `pycountry_convert`
- Converted `arrival_date_month` to numeric values
- Applied **label encoding** to all categorical columns: `hotel`, `meal`, `continent`, `market_segment`, `distribution_channel`, `reserved_room_type`, `deposit_type`, `customer_type`
- Applied **StandardScaler** normalisation to all numerical features

### 3. Model Building

Trained and evaluated 9 classification models:

| Model | Accuracy |
|---|---|
| Gradient Boosting | ~81.4% |
| Random Forest | ~81.3% |
| Extra Trees | ~80.8% |
| XGBoost | ~80.5% |
| AdaBoost | ~80.3% |
| Bagging | ~79.7% |
| Logistic Regression | ~78.6% |
| KNN | ~78.3% |
| Decision Tree | ~78.0% |

Models were evaluated using Accuracy, Precision, Recall and F1 Score.

### 4. Feature Selection

Used **Gradient Boosting feature importances** to identify and drop low-importance features (importance < 0.01): `children`, `distribution_channel`, `is_repeated_guest`, `days_in_waiting_list`, `babies`, `previous_bookings_not_canceled`.

Retraining on the reduced feature set improved model performance.

### 5. Hyperparameter Tuning

Applied **RandomizedSearchCV** (cv=5, n_iter=10) on `RandomForestClassifier` with the following parameter grid:
- `n_estimators`: [100, 250, 500]
- `max_leaf_nodes`: [100, 250, 500]
- `max_depth`: [5, 10, 20]

### 6. Known Limitations & Next Steps

- Class imbalance (~63% not cancelled / ~37% cancelled) was identified but not addressed due to time constraints. Applying techniques such as SMOTE or `class_weight='balanced'` would be a natural next step.

---

## Repository Structure

```
hotel_cancelation_ml/
│
├── data/
│   ├── hotel_bookings_raw_data.csv       # raw dataset
│   └── hotel_booking_clean_data.csv      # cleaned dataset
│
├── notebooks/
│   ├── 01_cleaning.ipynb
│   └── 02_modeling.ipynb
│
├── .gitignore
└── README.md
```

---

## Tools Used

- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, pycountry_convert) — data analysis, visualisation, and ML
- **Jupyter Notebook** — analysis notebooks
- **Git / GitHub** — version control

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/craftedbygaby/hotel_cancelation_ml.git
   cd hotel_cancelation_ml
   ```

2. Install Python dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Run notebooks in order inside the `notebooks/` folder.

---

## Limitations

- Personal identifiers have been removed from the dataset — columns `name`, `email`, `phone number`, and `credit_card` are artificially generated and should not be used in modelling
- Covers only two hotel types: Resort Hotel and City Hotel
- Data spans 2015–2017 — patterns may not reflect current booking trends

---

*All analysis is for educational and portfolio purposes only.*
