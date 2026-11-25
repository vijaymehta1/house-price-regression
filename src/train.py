"""House Price — Phase 5 + 6: model training aur evaluation.

3 models train karte hain, sab ek hi pipeline (Phase 4) ke saath:
  - Linear Regression (baseline — simple, samajhne me aasan)
  - Random Forest Regressor (non-linear, zyada powerful)
  - Gradient Boosting Regressor (aksar tabular data pe sabse achha)

Har model ko 5-fold cross-validation (train pe, R2) + test set pe
MAE / RMSE / R2 se check karte hain. Target log scale pe train hota
hai, isliye metrics dono scale pe report karte hain: log scale (jo
model ne seekha) aur dollars me (jo insaan ko samajh aata hai).
"""

import json
import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, drop_outliers, load_data, split_data

# Test me kuch categories train me nahi hoti — ye warning expected hai
# (handle_unknown="ignore" unhe all-zeros karta hai). Output saaf rakhne
# ke liye sirf ye warning chhupate hain.
warnings.filterwarnings("ignore", message="Found unknown categories.*")

MODELS = {
    "Linear Regression": LinearRegression(),
    # 200 trees: 100 se thoda stable, laptop pe abhi bhi fast
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}


def evaluate(y_true_log, y_pred_log):
    """Metrics — log scale aur dollars (expm1) dono me."""
    y_true_dollar = np.expm1(y_true_log)
    y_pred_dollar = np.expm1(y_pred_log)
    return {
        "r2_log": round(float(r2_score(y_true_log, y_pred_log)), 4),
        "rmse_log": round(float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))), 4),
        "mae_log": round(float(mean_absolute_error(y_true_log, y_pred_log)), 4),
        "rmse_dollar": round(float(np.sqrt(mean_squared_error(y_true_dollar, y_pred_dollar))), 2),
        "mae_dollar": round(float(mean_absolute_error(y_true_dollar, y_pred_dollar)), 2),
    }


def main():
    df = drop_outliers(load_data())
    X_train, X_test, y_train_log, y_test_log, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    results = {}
    for name, model in MODELS.items():
        # Poori pipeline (preprocessing + model) ek saath fit hoti hai,
        # taaki CV ke har fold me preprocessing fold ke train se seekhe.
        pipe = Pipeline([("pre", build_preprocessor()), ("model", model)])
        cv_scores = cross_val_score(pipe, X_train, y_train_log,
                                    cv=5, scoring="r2")
        pipe.fit(X_train, y_train_log)
        pred_log = pipe.predict(X_test)
        metrics = evaluate(y_test_log, pred_log)
        metrics["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_r2_std"] = round(float(cv_scores.std()), 4)
        results[name] = metrics
        print(f"{name}: CV R2 = {metrics['cv_r2_mean']} (+/- {metrics['cv_r2_std']})"
              f" | Test R2 = {metrics['r2_log']}"
              f" | Test RMSE = ${metrics['rmse_dollar']:,.0f}")

    # Best model: test R2 sabse zyada
    best = max(results, key=lambda k: results[k]["r2_log"])
    print(f"\nBest model (test R2 ke hisaab se): {best} "
          f"(R2 = {results[best]['r2_log']}, "
          f"RMSE = ${results[best]['rmse_dollar']:,.0f})")

    with open("reports/model_comparison.json", "w") as f:
        json.dump({"results": results, "best_model": best}, f, indent=2)
    print("reports/model_comparison.json saved.")
    return results, best


if __name__ == "__main__":
    # Project root se module ki tarah chalao (taaki `src` import ho sake):
    # .venv/bin/python -m src.train
    main()
