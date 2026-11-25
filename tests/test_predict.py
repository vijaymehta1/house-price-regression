"""House Price — Phase 9: tests (prediction pipeline + app).

Plain asserts, pytest nahi chahiye.
Run (project root se): .venv/bin/python tests/test_predict.py
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import load_model, predict_prices, predict_single
from src.preprocessing import build_preprocessor, drop_outliers, load_data, split_data

MODEL_PATH = ROOT / "models" / "house_price_model.pkl"

TYPICAL = {"OverallQual": 7, "GrLivArea": 2000, "GarageCars": 2,
           "TotalBsmtSF": 1000, "1stFlrSF": 1200, "2ndFlrSF": 800,
           "FullBath": 2, "TotRmsAbvGrd": 7, "Neighborhood": "NridgHt",
           "LotArea": 10000, "GarageArea": 500, "YearBuilt": 2005,
           "YearRemodAdd": 2005, "MSZoning": "RL", "HouseStyle": "2Story"}


def load_app_module():
    spec = importlib.util.spec_from_file_location("hp_app", ROOT / "app" / "app.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_model_loads():
    assert MODEL_PATH.exists(), "model file missing — pehle python -m src.save_model chalao"
    assert MODEL_PATH.stat().st_size > 0, "model file khaali hai"
    model = load_model(str(MODEL_PATH))
    assert [name for name, _ in model.steps] == ["pre", "model"]
    print("Case 1 (model loads): PASS")


def test_valid_prediction():
    model = load_model(str(MODEL_PATH))
    app = load_app_module()
    defaults, _ = app.load_defaults()
    price = predict_single(model, app.build_feature_row(TYPICAL, defaults))
    assert isinstance(price, float), "price float honi chahiye"
    assert 30000 < price < 800000, f"price range se bahar: {price}"
    print(f"Case 2 (valid prediction ${price:,.0f}): PASS")


def test_consistency():
    model = load_model(str(MODEL_PATH))
    app = load_app_module()
    defaults, _ = app.load_defaults()
    row = app.build_feature_row(TYPICAL, defaults)
    p1 = predict_single(model, row)
    p2 = predict_single(model, row)
    assert p1 == p2, "ek hi input pe do alag predictions"
    print("Case 3 (consistency, do baar same): PASS")


def test_missing_column_raises():
    model = load_model(str(MODEL_PATH))
    app = load_app_module()
    defaults, _ = app.load_defaults()
    row = app.build_feature_row(TYPICAL, defaults)
    del row["GrLivArea"]  # ek column gayab
    try:
        predict_single(model, row)
    except Exception:
        print("Case 4 (missing column raises error): PASS")
        return
    raise AssertionError("missing column pe error aana chahiye tha")


def test_invalid_dtype_raises():
    model = load_model(str(MODEL_PATH))
    app = load_app_module()
    defaults, _ = app.load_defaults()
    row = app.build_feature_row(TYPICAL, defaults)
    row["GrLivArea"] = "two thousand"  # number ki jagah text
    try:
        predict_single(model, row)
    except Exception:
        print("Case 5 (invalid dtype raises error): PASS")
        return
    raise AssertionError("galat dtype pe error aana chahiye tha")


def test_defaults_fill_missing():
    app = load_app_module()
    defaults, _ = app.load_defaults()
    row = app.build_feature_row({}, defaults)  # user ne kuch nahi bhara
    assert len(row) == 79, f"poori row 79 columns ki honi chahiye, mili {len(row)}"
    print("Case 6 (defaults fill all 79 cols): PASS")


def test_preprocessing_no_nan():
    df = drop_outliers(load_data())
    X_train, _, _, _, _, _ = split_data(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X_train)
    assert int(Xt.isna().sum().sum()) == 0, "transform ke baad NaN nahi hona chahiye"
    print("Case 7 (preprocessing 0 NaN): PASS")


def test_batch_predict():
    model = load_model(str(MODEL_PATH))
    df = drop_outliers(load_data())
    _, X_test, _, _, _, _ = split_data(df)
    preds = predict_prices(model, X_test.iloc[:5])
    assert len(preds) == 5 and all(p > 0 for p in preds)
    print("Case 8 (batch predict, 5 rows): PASS")


if __name__ == "__main__":
    test_model_loads()
    test_valid_prediction()
    test_consistency()
    test_missing_column_raises()
    test_invalid_dtype_raises()
    test_defaults_fill_missing()
    test_preprocessing_no_nan()
    test_batch_predict()
    print("\nSab 8/8 cases PASS")
