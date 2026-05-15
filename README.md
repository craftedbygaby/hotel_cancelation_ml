# 🏨 Hotel Booking Cancellation Predictor — Machine Learning Project

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
| **File** | `hotel_bookings_raw_data.csv` |
| **Records** | 119,390 rows |
| **Columns** | 32 |
| **Coverage** | 2015 — 2017 |
| **Source Paper** | *Hotel Booking Demand Datasets* — Nuno Antonio, Ana Almeida, Luis Nunes. Data in Brief, Volume 22, February 2019. |

---

## Project Objective

Predict whether a hotel booking will be cancelled (`is_canceled`: 0 or 1) using classification models, based on features such as lead time, deposit type, customer type, market segment, and seasonality.

**Key facts about the target variable (after cleaning):**
- ~72% of bookings were **not cancelled**
- ~28% of bookings were **cancelled**

---

## ML Pipeline

### 1. Data Cleaning

The raw dataset (`hotel_bookings_raw_data.csv`) was cleaned using the following steps:

1. **Dropped columns** with excessive null values: `agent` and `company`
2. **Dropped rows** with missing values in `children` and `country` columns
3. **Removed 31,984 duplicate** entries
4. **Dropped data leakage columns:** `reservation_status`, `reservation_status_date`, `assigned_room_type`
5. Final clean dataset contains **86,914 rows and 30 columns**
6. Exported as `hotel_booking_clean_data.csv`

### 2. Feature Engineering & Encoding

- Converted `country` codes to **continents** using `pycountry_convert` (170+ codes → 7 regions)
- Converted `arrival_date_month` from string to numeric values (1–12)
- Applied **label encoding** to all categorical columns: `hotel`, `meal`, `continent`, `market_segment`, `distribution_channel`, `reserved_room_type`, `deposit_type`, `customer_type`
- Applied **StandardScaler** normalisation to all numerical features

### 3. Model Building

Trained and evaluated 9 classification models. Evaluated using Accuracy, Precision, Recall and F1 Score, ranked by F1 Score:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Gradient Boosting** | 0.8105 | 0.7039 | 0.5455 | **0.6146** |
| Extra Trees | 0.8119 | 0.7129 | 0.5378 | 0.6131 |
| Random Forest | 0.8155 | 0.7442 | 0.5091 | 0.6046 |
| AdaBoost | 0.7996 | 0.6754 | 0.5326 | 0.5955 |
| XGBoost | 0.8053 | 0.7254 | 0.4784 | 0.5766 |
| Decision Tree | 0.7718 | 0.6035 | 0.5139 | 0.5551 |
| Bagging | 0.7995 | 0.7221 | 0.4489 | 0.5536 |
| KNN | 0.7899 | 0.6946 | 0.4313 | 0.5322 |
| Logistic Regression | 0.7743 | 0.6963 | 0.3289 | 0.4468 |

**Gradient Boosting** was selected as the best model based on F1 Score.

### 4. Feature Selection

Used **Gradient Boosting feature importances** to identify and drop low-importance features (importance < 0.01): `children`, `distribution_channel`, `is_repeated_guest`, `days_in_waiting_list`, `babies`, `previous_bookings_not_canceled`.

After retraining on the reduced feature set, model performance did not improve — the full feature set was kept.

| Model | Accuracy | F1 Score |
|---|---|---|
| Gradient Boosting (original) | 0.8113 | 0.616 |
| Gradient Boosting (reduced) | 0.8102 | 0.614 |

### 5. Hyperparameter Tuning

Applied **RandomizedSearchCV** (cv=5, n_iter=10) on `GradientBoostingClassifier`. RandomizedSearch was chosen over GridSearch due to training time constraints (~109 minutes for 10 combinations alone).

Parameter grid:
- `n_estimators`: [100, 250, 500]
- `max_leaf_nodes`: [100, 250, 500]
- `max_depth`: [5, 10, 20]

**Best parameters:** `n_estimators: 250`, `max_depth: 10`, `max_leaf_nodes: 100`

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Gradient Boosting (tuned) | 0.8195 | 0.7378 | 0.5409 | 0.6242 |

### 6. Known Limitations & Next Steps

- Class imbalance (~72% not cancelled / ~28% cancelled) was identified but not addressed due to time constraints — a natural next step for future improvement.
- Training times for ensemble models were significant, limiting the number of experiments possible within the week.
- More powerful boosting models (LightGBM, CatBoost) were not explored and could push performance further.

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
├── models/
│   └── best_model_gradient_boosting.pkl
│
├── project_presentation.pdf
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
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost pycountry_convert jupyter
   ```

3. Run notebooks in order inside the `notebooks/` folder.

---

## Limitations

- Covers only two hotel types: Resort Hotel and City Hotel
- Data spans 2015–2017 — patterns may not reflect current booking trends
- Class imbalance (~72/28) was not addressed during modelling

---

*All analysis is for educational and portfolio purposes only.*
