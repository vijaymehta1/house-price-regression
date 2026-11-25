"""House Price — Phase 7: prediction function (model load + predict).

Model khud yahan train nahi hota — wo src/save_model.py karta hai.
Ye file sirf ye batati hai: save hua model kaise load karein, aur
us se price kaise nikalein (Streamlit app Phase 8 me yahi use karega).
"""

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = "models/house_price_model.pkl"


def load_model(path=DEFAULT_MODEL_PATH):
    """Saved pipeline (preprocessing + model) ko joblib se load karo."""
    return joblib.load(path)


def predict_prices(model, df):
    """DataFrame (raw features, train.csv jaisi) -> dollar prices.

    Model log-price predict karta hai, isliye expm1 se wapas
    dollars me badalte hain.
    """
    log_prices = model.predict(df)
    return np.expm1(log_prices)


def predict_single(model, features):
    """Ek ghar ke features (dict) -> ek price (dollars, float).

    features me wahi 79 raw columns hone chahiye jo train.csv me
    hain (Id aur SalePrice ko chhod ke). App me jo field user nahi
    bharta, uski default value Phase 8 me bhari jayegi.
    """
    row = pd.DataFrame([features])
    return float(predict_prices(model, row)[0])
