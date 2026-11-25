# House Price Prediction

Machine Learning project that predicts house prices from property
features (area, quality, rooms, location, year built, and more).
Built as a B.Tech AI & ML internship project — genuinely functional,
with real data, real metrics, and a working Streamlit app.

## Problem Statement

Given a house's features (living area, quality, garage, basement,
neighborhood, year built, etc.), predict its sale price in dollars.
A buyer, seller, or agent can use the estimate as a starting point —
it is an estimate, not a valuation.

## Dataset

**Ames Housing** — Kaggle, "House Prices: Advanced Regression Techniques"
(https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

- 1460 houses, 81 columns, target variable `SalePrice`
- The data was fetched from the OpenML mirror (id 42165) because the
  Kaggle download needs an API token, and verified against Kaggle's
  `train.csv` (Id 1/2/3 SalePrice = 208500 / 181500 / 223500 — exact match)
- Target: mean $180,921, median $163,000, range $34,900–$755,000,
  skew 1.88 (log1p ke baad 0.12)
- Strongest features: OverallQual (corr 0.79), GrLivArea (0.71),
  GarageCars (0.64), GarageArea (0.62), TotalBsmtSF (0.61)

## Technologies

Python 3.12, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn,
Streamlit, Joblib, Git and GitHub. Nothing else, nothing paid.

## ML Workflow

1. **EDA** (`notebooks/01_eda_feature_engineering.ipynb`) — target
   distribution, missing values (7829 cells), duplicates (0),
   correlations, outliers (Id 1299 and 524 dropped), neighborhood analysis.
2. **Feature engineering** — TotalSF, TotalBath, TotalPorchSF, HouseAge,
   RemodAge, HasPool/HasGarage/HasFireplace/HasBsmt; missing values in
   3 groups ("None" for absence, LotFrontage median by Neighborhood,
   median/mode for the rest); ordinal encoding for 16 ordered features;
   log1p for 19 skewed features + target; one-hot for 27 nominal features.
   Final data: 1458 × 216, 0 missing.
3. **Preprocessing pipeline** (`src/preprocessing.py`) — the same 7 steps
   as an sklearn Pipeline, fit on train data only (no leakage).
4. **Training** (`src/train.py`) — 80/20 split (random_state 42),
   3 models, 5-fold cross-validation on train.
5. **Evaluation** — MAE, RMSE, R² on the test set (292 houses), in both
   log scale and dollars.

## Models and Results

Real results on the test set (run with `.venv/bin/python -m src.train`):

| Model | Test R² | Test RMSE | Test MAE | CV R² (5-fold) |
|---|---|---|---|---|
| Linear Regression | 0.9093 | $21,208 | $14,656 | 0.8843 ± 0.0323 |
| Random Forest (200 trees) | 0.8745 | $23,340 | $16,250 | 0.8788 ± 0.0139 |
| **Gradient Boosting** | **0.9127** | **$19,708** | **$13,970** | **0.9014 ± 0.0066** |

**Best model: Gradient Boosting** — highest test R² and lowest RMSE,
and the most stable CV score (±0.0066). Random Forest stayed behind
even with 200 trees. Full numbers: `reports/model_comparison.json`.

## Project Structure

```
house-price-regression/
├── README.md, requirements.txt, .gitignore, DEPLOYMENT.md
├── .venv/                     # virtual environment (git me nahi jata)
├── data/
│   ├── train.csv              # raw dataset
│   └── train_engineered.csv   # EDA + FE ke baad ka data
├── notebooks/
│   └── 01_eda_feature_engineering.ipynb
├── src/
│   ├── preprocessing.py       # Phase 4: sklearn pipeline
│   ├── train.py               # Phase 5+6: training + evaluation
│   ├── predict.py             # Phase 7: load_model / predict
│   └── save_model.py          # Phase 7: best model ko .pkl me save
├── app/
│   └── app.py                 # Phase 8: Streamlit app
├── models/
│   └── house_price_model.pkl  # saved Gradient Boosting pipeline (178 KB)
├── figures/                   # EDA ke plots
├── tests/
│   └── test_predict.py        # Phase 9: 8 tests, sab PASS
└── reports/
    └── model_comparison.json  # asli metrics
```

## Installation

```bash
# virtual environment banao (ek baar)
python3 -m venv .venv

# activate karo
source .venv/bin/activate        # Linux / Mac
# .venv\Scripts\activate        # Windows

# libraries install karo
pip install -r requirements.txt
```

## How to Run

```bash
# EDA notebook
.venv/bin/jupyter notebook notebooks/01_eda_feature_engineering.ipynb

# Models train karo aur metrics dekho
.venv/bin/python -m src.train

# Best model save karo (pehle se save hai, dobara karne ki zaroorat nahi)
.venv/bin/python -m src.save_model

# Tests chalao
.venv/bin/python tests/test_predict.py

# Streamlit app chalao
.venv/bin/streamlit run app/app.py
```

App me 15 fields bhare jate hain (Overall Quality, Living Area,
Garage, Bathrooms, Neighborhood, Year Built, ...). Baaki 64 features
ke liye train data ki typical value (median/mode) use hoti hai — ye
baat app me saaf likhi hai.

## Screenshots

Deploy ke baad yahan live app ke screenshots add karenge. Abhi ke
liye app ko local chala ke dekho: `.venv/bin/streamlit run app/app.py`

## Limitations

- Data Ames, Iowa (USA) ka hai, 2006–2010 ke sales — aaj ke Indian
  market ya aaj ke prices ke liye seedha use nahi ho sakta.
- RMSE lagbhag $19,708 hai — prediction ek estimate hai, exact price nahi.
- App me sirf 15 fields poochhe jate hain; baaki 64 features median/mode
  pe chalte hain, isliye do alag gharon ka estimate same aa sakta hai
  agar unke 15 fields same hon.
- 2 categories sirf test data me thi (train me nahi) — pipeline unhe
  all-zeros encode karta hai (koi leakage nahi, lekin thodi jaankari
  chhoot ti hai).
- Model ko 2010 ke baad ke ghar (YearBuilt > 2010) ka koi idea nahi hai.

## Deployment

Streamlit Community Cloud (free) — poori steps `DEPLOYMENT.md` me hain.
Deployment link: https://cdmluxb2kty44gxw8cvesf.streamlit.app/
