"""House Price — Phase 4: sklearn preprocessing pipeline.

Notebook 01 (EDA + FE) me jo 7 steps pandas se kiye the, wahi kaam ab
sklearn Pipeline me hai. Kyun? Do wajah:
  1. Leakage nahi — median, mode, one-hot categories, scaler sab sirf
     TRAIN data se seekhe jate hain (fit sirf train pe hota hai).
  2. Dobara use ho sakega — Streamlit app me ek naye ghar ka raw input
     bhi isi pipeline se guzrega (Phase 7 me model ke saath save hoga).

Target (SalePrice) ka log1p yahan nahi hai — wo training script me
y = log1p(SalePrice) se hota hai, aur prediction pe expm1 se wapas
aata hai (Phase 5 aur Phase 8 me).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

RANDOM_STATE = 42
OUTLIER_IDS = [1299, 524]  # GrLivArea 4600+ lekin price 1.6-1.8 lakh (notebook 01)

# --- Group A: "missing matlab cheez hai hi nahi" -> "None" bharo ---
NONE_COLS = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
             "GarageType", "GarageFinish", "GarageQual", "GarageCond",
             "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
             "BsmtFinType2", "MasVnrType"]

# --- Ordinal columns: order wali categories, list ka order hi number hai ---
ORDINAL_MAPS = {
    "ExterQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "HeatingQC": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
}
ORD_COLS = list(ORDINAL_MAPS.keys())

# --- Nominal columns: bina order wali categories -> one-hot ---
NOMINAL_COLS = ["MSZoning", "Street", "Alley", "LandContour", "Utilities",
                "LotConfig", "LandSlope", "Neighborhood", "Condition1",
                "Condition2", "BldgType", "HouseStyle", "RoofStyle",
                "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType",
                "Foundation", "Heating", "CentralAir", "Electrical",
                "GarageType", "GarageFinish", "PavedDrive", "MiscFeature",
                "SaleType", "SaleCondition"]

# --- Right-skewed numeric columns -> log1p ---
SKEWED = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
          "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea",
          "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
          "ScreenPorch", "MiscVal", "PoolArea", "LowQualFinSF", "TotalSF"]


def add_features(df):
    """Naye features — domain knowledge se (notebook 01, step 5.3).

    Kharidne wala 'kul kitna area, kul kitne bathroom, ghar kitna
    purana' sochta hai, isliye tukdon ko ek meaningful number me jodo.
    """
    df = df.copy()
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"]
                          + df["3SsnPorch"] + df["ScreenPorch"])
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    return df


class HandleMissing(BaseEstimator, TransformerMixin):
    """Missing values — pehle KYUN poochho, phir bharo (notebook 01, step 5.2).

    fit() sirf train data se seekhta hai: LotFrontage ka global median
    aur har Neighborhood ka median, MasVnrArea ka median, GarageYrBlt
    ka median, Electrical ka mode. transform() wahi seekhe hue numbers
    use karta hai — isliye test data ki koi jaankari train me nahi aati.
    """

    def fit(self, X, y=None):
        X = X.copy()
        self.lotfrontage_global_median_ = X["LotFrontage"].median()
        self.lotfrontage_by_nbhd_ = (
            X.groupby("Neighborhood")["LotFrontage"].median().to_dict())
        # MasVnrArea ka median: pehle MasVnrType None wali rows me 0 bharo
        # (notebook jaisa order), phir median nikalo.
        mas = X["MasVnrArea"].copy()
        is_none_type = X["MasVnrType"].isna() | (X["MasVnrType"] == "None")
        mas.loc[is_none_type & mas.isna()] = 0
        self.masvnr_median_ = mas.median()
        self.garage_yrblt_median_ = X["GarageYrBlt"].median()
        self.electrical_mode_ = X["Electrical"].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        # Group A: cheez hai hi nahi -> "None" (ye khud ek information hai)
        for c in NONE_COLS:
            X[c] = X[c].fillna("None")
        # LotFrontage: apne Neighborhood ka median; naya area ho to global median
        med = X["Neighborhood"].map(self.lotfrontage_by_nbhd_)
        X["LotFrontage"] = (X["LotFrontage"].fillna(med)
                            .fillna(self.lotfrontage_global_median_))
        # MasVnrArea: MasVnrType None ho to 0, warna median
        mask = (X["MasVnrType"] == "None") & X["MasVnrArea"].isna()
        X.loc[mask, "MasVnrArea"] = 0
        X["MasVnrArea"] = X["MasVnrArea"].fillna(self.masvnr_median_)
        X["Electrical"] = X["Electrical"].fillna(self.electrical_mode_)
        X["GarageYrBlt"] = X["GarageYrBlt"].fillna(self.garage_yrblt_median_)
        return X


def build_preprocessor():
    """Poora preprocessing pipeline, sahi order me.

    add_features -> missing bharo -> log1p / ordinal / one-hot
    (ek hi ColumnTransformer me) -> StandardScaler.
    """
    ordinal = OrdinalEncoder(
        categories=[ORDINAL_MAPS[c] for c in ORD_COLS],
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    onehot = OneHotEncoder(handle_unknown="ignore", drop="first",
                           sparse_output=False)
    encode = ColumnTransformer([
        ("log", FunctionTransformer(np.log1p, validate=False), SKEWED),
        ("ord", ordinal, ORD_COLS),
        ("onehot", onehot, NOMINAL_COLS),
    ], remainder="passthrough")

    scaler = StandardScaler()
    # NOTE: poori pipeline pe set_output nahi kar sakte — custom
    # HandleMissing transformer ko sklearn ka set_output API nahi aata.
    # Isliye sirf un steps pe pandas output lagate hain jo numpy dete
    # hain (ColumnTransformer aur StandardScaler). add_features aur
    # HandleMissing waise bhi DataFrame hi lautate hain.
    encode.set_output(transform="pandas")
    scaler.set_output(transform="pandas")
    pipe = Pipeline([
        ("add_features", FunctionTransformer(add_features, validate=False)),
        ("missing", HandleMissing()),
        ("encode", encode),
        ("scaler", scaler),
    ])
    return pipe


def load_data(path="data/train.csv"):
    return pd.read_csv(path)


def drop_outliers(df):
    """Id 1299 aur 524 hatana — sirf training data se, pipeline ke bahar.

    Pipeline ke andar row drop karna theek nahi hota (prediction pe koi
    row drop nahi hoti), isliye ye step alag hai.
    """
    return df[~df["Id"].isin(OUTLIER_IDS)].copy()


def split_data(df, test_size=0.2, random_state=RANDOM_STATE):
    """80/20 split. Target ka log1p yahin karte hain (skew 1.88 -> 0.12)."""
    X = df.drop(columns=["Id", "SalePrice"])
    y = df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, np.log1p(y_train), np.log1p(y_test), y_train, y_test


if __name__ == "__main__":
    # Phase 4 verification: project root se chalao
    # .venv/bin/python src/preprocessing.py
    df = load_data()
    print("Raw:", df.shape)
    df = drop_outliers(df)
    print("Outliers hatane ke baad:", df.shape)
    X_train, X_test, y_train_log, y_test_log, y_train, y_test = split_data(df)
    print("Train:", X_train.shape, "| Test:", X_test.shape)

    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)   # fit SIRF train pe (no leakage)
    X_test_t = pre.transform(X_test)
    print("Transform ke baad train:", X_train_t.shape,
          "| test:", X_test_t.shape)
    print("Train me NaN:", int(X_train_t.isna().sum().sum()),
          "| Test me NaN:", int(X_test_t.isna().sum().sum()))
    print("Train/test columns same:",
          list(X_train_t.columns) == list(X_test_t.columns))

    # Notebook 01 ke engineered CSV se tulna (features = total - SalePrice - SalePrice_log)
    eng = pd.read_csv("data/train_engineered.csv")
    print("Engineered CSV ke features:", eng.shape[1] - 2,
          "| Pipeline ke features:", X_train_t.shape[1])

    # Ek value check: TotalSF sahi banta hai ya nahi
    row = df.iloc[[0]]
    check = add_features(row)
    expected = (row["TotalBsmtSF"].iloc[0] + row["1stFlrSF"].iloc[0]
                + row["2ndFlrSF"].iloc[0])
    print("TotalSF check (pehli row):", check["TotalSF"].iloc[0],
          "==", expected)
