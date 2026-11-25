# Internship Report — House Price Prediction

> Ye file report ka content hai (Markdown me), asli implementation
> aur asli results ke basis pe likha gaya hai. Apne college ke format
> me convert karte time headings aur personal details bhar lena.
> Personal details (naam, roll no.) placeholders me hain — apne
> hisaab se bhar do.

---

## 1. Title Page (bharne ke liye)

- **Project Title:** House Price Prediction using Machine Learning
- **Name:** [Your Name]
- **Roll No.:** [Your Roll No.]
- **Course:** B.Tech, Computer Science and Engineering
- **Institute:** [Your Institute Name]
- **Internship Domain:** Artificial Intelligence & Machine Learning

---

## 2. Abstract

This project builds a machine learning system that predicts the sale
price of a house from its features such as living area, overall
quality, garage capacity, basement area, neighborhood, and year built.
The Ames Housing dataset (1460 houses, Kaggle) was used. After
exploratory data analysis and feature engineering, three regression
models were trained and compared using 5-fold cross-validation and a
held-out test set of 292 houses: Linear Regression, Random Forest
Regressor, and Gradient Boosting Regressor. Gradient Boosting
performed best with a test R² of 0.9127 and RMSE of $19,708
(MAE $13,970). The complete preprocessing + model pipeline was saved
with Joblib and deployed as a clean Streamlit web application where a
user enters a house's key details and gets an estimated price.
All preprocessing (imputation, encoding, scaling) is fitted on
training data only, so there is no data leakage.

## 3. Introduction

House prices depend on many factors at once — area, quality, location,
age, number of rooms — and the relationship is not a simple straight
line. Manual estimation is slow and inconsistent. Machine learning
can learn these patterns from past sales and give a fast, consistent
estimate. This project covers the full journey: from raw data to a
deployed application, using only free and open-source tools.

## 4. Problem Statement

Given the features of a house, predict its sale price (SalePrice) in
US dollars as accurately as possible, and provide the prediction
through a simple web interface that a non-technical user can operate.

## 5. Objectives

1. Find and understand a real, publicly available house price dataset.
2. Perform exploratory data analysis and explain what the data shows.
3. Handle missing values, outliers, and categorical variables correctly.
4. Build a reusable Scikit-learn preprocessing pipeline with no leakage.
5. Train and compare Linear Regression, Random Forest, and Gradient
   Boosting using cross-validation.
6. Evaluate models with MAE, RMSE, and R², and select the best model
   on real results.
7. Save the final pipeline and build a Streamlit application for it.
8. Test the application and prepare it for free deployment.

## 6. Dataset Description

- **Source:** Kaggle — "House Prices: Advanced Regression Techniques"
  (https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Access note:** the file was fetched from the OpenML mirror
  (id 42165) because Kaggle download needs an API token, and verified
  against Kaggle's `train.csv` (Id 1/2/3 SalePrice = 208500 / 181500 /
  223500 — exact match).
- **Size:** 1460 rows × 81 columns (38 numeric, 43 categorical),
  0 duplicate rows, 7829 missing cells.
- **Target variable:** `SalePrice` — mean $180,921, median $163,000,
  minimum $34,900, maximum $755,000, skew 1.88 (0.12 after log1p).
- **Important features (correlation with SalePrice):** OverallQual 0.79,
  GrLivArea 0.71, GarageCars 0.64, GarageArea 0.62, TotalBsmtSF 0.61,
  1stFlrSF 0.61, FullBath 0.56, TotRmsAbvGrd 0.53, YearBuilt 0.52,
  YearRemodAdd 0.51. GarageCars and GarageArea are highly correlated
  with each other (r = 0.882).
- **Neighborhoods:** 25. Highest median prices: NridgHt $315,000,
  NoRidge $301,500, StoneBr $278,000.

## 7. Exploratory Data Analysis (EDA)

Key observations from the data (see `notebooks/01_eda_feature_engineering.ipynb`
and `figures/`):

1. **Target is right-skewed** (skew 1.88) — a few very expensive houses
   stretch the distribution. log1p brings the skew to 0.12, which
   suits Linear Regression's assumptions.
2. **Missing values are not random.** PoolQC is missing in 1453 of 1460
   rows because 99.5% houses have no pool; Alley (1369), Fence (1179),
   MiscFeature (1406) are similar. Filling these with a median would
   be wrong — "None" itself is information.
3. **Two outliers** (Id 1299 and 524): living area above 4600 sq ft
   with OverallQual 10, but price only $160,000–$184,750. Such a big,
   high-quality house cannot be that cheap — likely a partial sale or
   data issue. Both rows were dropped (1458 rows remain).
4. **Quality drives price:** the median price rises steeply with
   OverallQual, which is the single strongest feature (0.79).
5. **Location matters:** median prices vary from about $128,000 to
   $315,000 across the 25 neighborhoods.

## 8. Methodology

Workflow: Dataset → EDA → Feature Engineering → Train/Test Split (80/20)
→ Preprocessing Pipeline → Model Training → Evaluation → Save → App.

**Feature engineering (7 steps, each with a reason):**
1. Drop outliers Id 1299, 524.
2. Missing values in 3 groups — "None" for absence columns (15),
   LotFrontage = median by Neighborhood, small gaps = median/mode.
3. New features — TotalSF, TotalBath, TotalPorchSF, HouseAge, RemodAge,
   HasPool, HasGarage, HasFireplace, HasBsmt.
4. Ordinal encoding for 16 ordered features (e.g. Ex < Gd < TA < Fa < Po).
5. log1p for 19 skewed features and the target.
6. One-hot encoding for 27 nominal features (drop_first=True).
7. Drop Id. Final data: 1458 × 216, 0 missing values.

**Leakage prevention:** all preprocessing (medians, modes, one-hot
categories, scaler) is fitted on the training split only, inside a
single sklearn Pipeline. The pipeline has 212 features; the notebook's
manual version had 214, because 2 categories exist only in test rows
and the pipeline correctly creates no column for them.

**Tools:** Python 3.12, Pandas, NumPy, Scikit-learn, Matplotlib,
Seaborn, Streamlit, Joblib.

## 9. Model Implementation

Three models, each as `Pipeline(preprocessing + model)`, trained on
1166 houses and tested on 292 (random_state 42):

| Model | Settings |
|---|---|
| Linear Regression | Default (baseline) |
| Random Forest Regressor | 200 trees, random_state 42, n_jobs=-1 |
| Gradient Boosting Regressor | Default (100 estimators), random_state 42 |

Each model was also checked with 5-fold cross-validation (R²) on the
training data, so the comparison does not depend on one lucky split.

## 10. Evaluation

Metrics on the test set (292 houses). The target was trained in log
scale, so metrics are reported in log scale and in dollars (expm1):

| Model | Test R² | Test RMSE | Test MAE | CV R² (mean ± std) |
|---|---|---|---|---|
| Linear Regression | 0.9093 | $21,208 | $14,656 | 0.8843 ± 0.0323 |
| Random Forest | 0.8745 | $23,340 | $16,250 | 0.8788 ± 0.0139 |
| **Gradient Boosting** | **0.9127** | **$19,708** | **$13,970** | **0.9014 ± 0.0066** |

Full numbers: `reports/model_comparison.json`.

## 11. Results and Analysis

- **Gradient Boosting is the best model** — highest test R² (0.9127),
  lowest RMSE ($19,708), and the most stable cross-validation score
  (±0.0066), so the result is not a lucky split.
- Linear Regression is a strong baseline (R² 0.9093) because the
  log-transformed target and engineered features make the relationship
  nearly linear.
- Random Forest performed worst (R² 0.8745) despite 200 trees — on
  this small, structured dataset, boosting's sequential error
  correction worked better than bagging.
- An RMSE of $19,708 against a median price of $163,000 means a
  typical error of about 12% — good for an estimate, not an exact
  valuation.
- The saved model was checked on 3 test houses: predictions
  $208,892 vs actual $190,000; $101,953 vs $100,000; $114,514 vs
  $115,000 — close to the real prices.

## 12. Screenshots

Add here after deployment: (1) Streamlit app home page, (2) a filled
form with the predicted price, (3) the Model Information section.
Take them from the live deployed app, not from local mockups.

## 13. Limitations

1. The data is from Ames, Iowa (USA), sales of 2006–2010 — it cannot
   directly predict today's prices or prices in another country.
2. RMSE ≈ $19,708: predictions are estimates, not valuations.
3. The app asks only 15 fields; the other 64 features use train
   median/mode values, so two houses with the same 15 fields get the
   same estimate.
4. Two categories exist only in test data; the pipeline encodes them
   as all-zeros (correct, but some information is lost).
5. The model has no knowledge of houses built after 2010 or of
   market conditions after 2010.

## 14. Conclusion

The project successfully built an end-to-end house price prediction
system: real data was analyzed, a leakage-free preprocessing pipeline
was created, three models were compared honestly, and the best model
(Gradient Boosting, test R² 0.9127) was deployed as a working
Streamlit application. The system is understandable, testable, and
reproducible on a normal laptop with free tools.

## 15. Future Scope

1. Hyperparameter tuning (GridSearchCV) for Gradient Boosting.
2. Try XGBoost / LightGBM and compare.
3. Add more input fields to the app (full 79 features, grouped in sections).
4. SHAP-based explanations — "price badhne me sabse zyada kis feature
   ka haath hai".
5. Retrain on newer, local (Indian) housing data for real-world use.
6. Add a price-trend chart (price vs year built / living area).
