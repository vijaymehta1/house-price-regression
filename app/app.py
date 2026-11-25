"""House Price — Phase 8: Streamlit app.

User ghar ki khaas baatein bharta hai, app price batata hai.
Design ke faisle (seedha aur imaandaar):
  - Model ko 79 raw features chahiye, lekin user se 79 cheezein
    poochhna bekaar hai. Isliye sirf 15 sabse important fields
    poochhte hain (jo price se sabse zyada judte hain — notebook 01
    ki correlation list se). Baaki sab ke liye train data ka typical
    value (numeric ka median, categorical ka mode) use hota hai —
    aur ye baat UI me saaf likhi hai, chhupayi nahi.
  - Price hamesha saved model (models/house_price_model.pkl) se
    aata hai — koi number haath se nahi likha.
  - Model info (R2, RMSE) reports/model_comparison.json se padha
    jata hai, taaki dikhaya gaya number aur asli result ek hi ho.
"""

from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import load_model, predict_single  # noqa: E402

MODEL_PATH = ROOT / "models" / "house_price_model.pkl"
DATA_PATH = ROOT / "data" / "train.csv"
METRICS_PATH = ROOT / "reports" / "model_comparison.json"


def load_defaults():
    """Har feature ki typical value: numeric -> median, categorical -> mode."""
    df = pd.read_csv(DATA_PATH)
    defaults = {}
    for col in df.columns:
        if col in ("Id", "SalePrice"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            defaults[col] = float(df[col].median())
        else:
            defaults[col] = df[col].mode()[0]
    return defaults, df


def build_feature_row(user_inputs, defaults):
    """User ke 15 inputs + baaki ke typical values = poori 79-column row."""
    row = dict(defaults)
    row.update(user_inputs)
    return row


def main():
    st.title("House Price Prediction")
    st.write("Ghar ki details bharo — model estimated price batayega. "
             "Jo field tum nahi bharte, uske liye typical value "
             "(median/mode) use hoti hai.")

    if not MODEL_PATH.exists():
        st.error("Model file nahi mili. Pehle project root se chalao: "
                 ".venv/bin/python -m src.save_model")
        return

    model = load_model(str(MODEL_PATH))
    defaults, df = load_defaults()

    col1, col2 = st.columns(2)
    with col1:
        overall_qual = st.slider("Overall Quality (1-10)",
                                 1, 10, int(defaults["OverallQual"]))
        gr_liv_area = st.number_input("Living Area — GrLivArea (sq ft)",
                                      min_value=300, max_value=6000,
                                      value=int(defaults["GrLivArea"]))
        garage_cars = st.slider("Garage Cars (0-4)",
                                0, 4, int(defaults["GarageCars"]))
        total_bsmt_sf = st.number_input("Basement Area — TotalBsmtSF (sq ft)",
                                        min_value=0, max_value=6000,
                                        value=int(defaults["TotalBsmtSF"]))
        first_flr_sf = st.number_input("First Floor Area — 1stFlrSF (sq ft)",
                                       min_value=0, max_value=5000,
                                       value=int(defaults["1stFlrSF"]))
        second_flr_sf = st.number_input("Second Floor Area — 2ndFlrSF (sq ft)",
                                        min_value=0, max_value=2000,
                                        value=int(defaults["2ndFlrSF"]))
        full_bath = st.slider("Full Bathrooms (0-3)",
                              0, 3, int(defaults["FullBath"]))
        tot_rms = st.slider("Total Rooms Above Ground (2-14)",
                            2, 14, int(defaults["TotRmsAbvGrd"]))
    with col2:
        neighborhood = st.selectbox(
            "Neighborhood",
            sorted(df["Neighborhood"].dropna().unique().tolist()),
            index=sorted(df["Neighborhood"].dropna().unique().tolist()).index(
                defaults["Neighborhood"]))
        lot_area = st.number_input("Lot Area (sq ft)",
                                   min_value=1000, max_value=200000,
                                   value=int(defaults["LotArea"]))
        garage_area = st.number_input("Garage Area (sq ft)",
                                      min_value=0, max_value=1500,
                                      value=int(defaults["GarageArea"]))
        year_built = st.number_input("Year Built",
                                     min_value=1872, max_value=2010,
                                     value=int(defaults["YearBuilt"]))
        year_remod = st.number_input("Year Remodeled",
                                     min_value=1950, max_value=2010,
                                     value=int(defaults["YearRemodAdd"]))
        ms_zoning = st.selectbox(
            "Zoning (MSZoning)",
            sorted(df["MSZoning"].dropna().unique().tolist()),
            index=sorted(df["MSZoning"].dropna().unique().tolist()).index(
                defaults["MSZoning"]))
        house_style = st.selectbox(
            "House Style",
            sorted(df["HouseStyle"].dropna().unique().tolist()),
            index=sorted(df["HouseStyle"].dropna().unique().tolist()).index(
                defaults["HouseStyle"]))

    if st.button("Predict Price"):
        user_inputs = {
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "GarageCars": garage_cars,
            "TotalBsmtSF": total_bsmt_sf,
            "1stFlrSF": first_flr_sf,
            "2ndFlrSF": second_flr_sf,
            "FullBath": full_bath,
            "TotRmsAbvGrd": tot_rms,
            "Neighborhood": neighborhood,
            "LotArea": lot_area,
            "GarageArea": garage_area,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "MSZoning": ms_zoning,
            "HouseStyle": house_style,
        }
        price = predict_single(model, build_feature_row(user_inputs, defaults))
        st.success(f"Estimated House Price: **${price:,.0f}**")

    st.divider()
    st.subheader("Model Information")
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        best = metrics["best_model"]
        row = metrics["results"][best]
        st.write(f"Model: **Gradient Boosting** (best of 3 models)")
        st.write(f"Test R²: **{row['r2_log']}** | "
                 f"Test RMSE: **${row['rmse_dollar']:,.0f}** | "
                 f"Test MAE: **${row['mae_dollar']:,.0f}**")
        st.write("Dataset: Ames Housing, 1460 houses (Kaggle). "
                 "Target: SalePrice.")
    except Exception:
        st.write("Model: Gradient Boosting. "
                 "Metrics file (reports/model_comparison.json) nahi mili.")


if __name__ == "__main__":
    # Project root se chalao: .venv/bin/streamlit run app/app.py
    main()
