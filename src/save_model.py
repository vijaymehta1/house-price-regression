"""House Price — Phase 7: best model ko train karke save karo.

Best model (Phase 5+6 se): Gradient Boosting — Test R2 0.9127,
RMSE $19,708. Poori pipeline (Phase 4 preprocessing + model) ko
joblib se models/house_price_model.pkl me save karte hain, taaki
Streamlit app ko training dobara na karni pade.

Ek faisla: model ko TRAIN split (1166 rows) pe fit karte hain, poore
data pe nahi — kyunki Phase 6 me report kiye gaye test metrics
(R2, RMSE) isi model ke hain. Agar poore data pe fit karte, to wo
metrics is model ko describe nahi karte.
"""

import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.predict import load_model, predict_prices
from src.preprocessing import build_preprocessor, drop_outliers, load_data, split_data

MODEL_PATH = "models/house_price_model.pkl"


def main():
    df = drop_outliers(load_data())
    X_train, X_test, y_train_log, y_test_log, y_train, y_test = split_data(df)

    model = Pipeline([
        ("pre", build_preprocessor()),
        ("model", GradientBoostingRegressor(random_state=42)),
    ])
    model.fit(X_train, y_train_log)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")

    # Verify: fresh load karke predict karo, aur consistency check karo
    loaded = load_model(MODEL_PATH)
    sample = X_test.iloc[:3]
    actual = y_test.iloc[:3].values
    pred1 = predict_prices(loaded, sample)
    pred2 = predict_prices(loaded, sample)
    print("Consistency check (do baar same prediction):",
          bool((pred1 == pred2).all()))
    for i in range(3):
        print(f"  Ghar {i + 1}: predicted ${pred1[i]:,.0f} | "
              f"actual ${actual[i]:,.0f}")


if __name__ == "__main__":
    # Project root se module ki tarah chalao:
    # .venv/bin/python -m src.save_model
    main()
