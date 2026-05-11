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
| **Note** | This is a fictional dataset. All hotel names, locations, and personal data are simulated and do not represent real bookings or individuals. |

---

## Project Objective

Predict whether a hotel booking will be cancelled (`is_canceled`: 0 or 1) using classification models, based on features such as lead time, deposit type, customer type, market segment, and seasonality.

**Key facts about the target variable:**
- ~63% of bookings were **not cancelled**
- ~37% of bookings were **cancelled**

This slight class imbalance will be accounted for during model training.

---

## ML Pipeline

*To be updated as the project progresses.*

---

## Repository Structure

```
hotel_cancelation_ml/
│
├── data/
│   └── hotel_bookings.csv
│
├── notebooks/
│   ├── 01_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
│
├── scripts/
│   ├── cleaning.py
│   ├── features.py
│   └── model.py
│
├── .gitignore
└── README.md
```

---

## Tools Used

- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn) — data analysis, visualisation, and ML
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

- Dataset is **fictional** — findings cannot be generalised to real-world hotel booking behaviour
- Covers only two hotel types: Resort Hotel and City Hotel
- Data spans 2015–2017 — patterns may not reflect current booking trends

---

*All analysis is for educational and portfolio purposes only.*
