"""Ames Housing — EDA + Feature Engineering (Step 1 of the project).
Har step ke saath uska 'kyun' comment mein likha hai.
Output: figures/*.png, data/train_engineered.csv, summary.json
"""
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42

df = pd.read_csv("data/train.csv")
print("Raw shape:", df.shape)

# ---------------------------------------------------------------- 1. TARGET
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["SalePrice"], bins=40, kde=True, ax=ax[0], color="#2E86AB")
ax[0].set_title("SalePrice — right-skewed (skew = 1.88)")
sns.histplot(np.log1p(df["SalePrice"]), bins=40, kde=True, ax=ax[1], color="#2E86AB")
ax[1].set_title("log1p(SalePrice) — almost normal (skew = 0.12)")
fig.tight_layout()
fig.savefig("figures/fig_target_dist.png")
plt.close(fig)

# ---------------------------------------------------------------- 2. MISSING
miss = df.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
miss.head(15).sort_values().plot(kind="barh", ax=ax, color="#E07A5F")
ax.set_title("Missing values — top 15 columns (count)")
ax.set_xlabel("Missing rows (out of 1460)")
fig.tight_layout()
fig.savefig("figures/fig_missing.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. CORRELATION
num = df.select_dtypes("number")
corr = num.corr(numeric_only=True)["SalePrice"].drop("SalePrice").sort_values(key=abs, ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
corr.head(12).sort_values().plot(kind="barh", ax=ax, color="#6BCB77")
ax.set_title("Top 12 numeric features correlated with SalePrice")
ax.set_xlabel("Correlation")
fig.tight_layout()
fig.savefig("figures/fig_corr.png")
plt.close(fig)

# ---------------------------------------------------------------- 4. SCATTER + OUTLIERS
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(df["GrLivArea"], df["SalePrice"], alpha=0.5, s=18, color="#2E86AB")
for _id in (1299, 524):
    r = df[df["Id"] == _id].iloc[0]
    ax[0].annotate(f"Id {_id}", (r["GrLivArea"], r["SalePrice"]), fontsize=9, color="red")
ax[0].set_xlabel("GrLivArea (sq ft)"); ax[0].set_ylabel("SalePrice")
ax[0].set_title("GrLivArea vs SalePrice — 2 outliers")
sns.boxplot(data=df, x="OverallQual", y="SalePrice", ax=ax[1], color="#FFD93D")
ax[1].set_title("OverallQual vs SalePrice — quality drives price")
fig.tight_layout()
fig.savefig("figures/fig_scatter.png")
plt.close(fig)

# ---------------------------------------------------------------- 5. NEIGHBORHOOD
nb = df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
nb.head(10).sort_values().plot(kind="barh", ax=ax, color="#9B59D0")
ax.set_title("Median SalePrice by Neighborhood — top 10")
ax.set_xlabel("Median SalePrice ($)")
fig.tight_layout()
fig.savefig("figures/fig_neighborhood.png")
plt.close(fig)

# ================================================================ FEATURE ENGINEERING
fe = df.copy()
fe_log = {}

# --- 5.1 Outliers hatana: Id 1299 aur 524 ---
# Soch: GrLivArea 4600+ sq ft lekin price sirf 1.6-1.8 lakh, OverallQual 10.
# Itna bada + itna achha ghar itna sasta nahi ho sakta — ye partial sale / data issue hai.
# Aise 2 points poori regression line ko khinch lete hain, isliye drop.
fe = fe[~fe["Id"].isin([1299, 524])].copy()
fe_log["outliers_dropped"] = [1299, 524]

# --- 5.2 Missing values: pehle KYUN poochho, phir bharo ---
# Group A — "missing matlab cheez hai hi nahi": NaN ko "None" bharo.
# Soch: PoolQC khali hai kyunki 99.5% gharon me pool hai hi nahi. Median/mode bharna
# galat hota — "None" khud ek information hai (pool nahi hai -> price par asar).
none_cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
             "GarageType", "GarageFinish", "GarageQual", "GarageCond",
             "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
             "MasVnrType"]
for c in none_cols:
    fe[c] = fe[c].fillna("None")
fe_log["none_filled"] = none_cols

# Group B — LotFrontage (17.7% missing): Neighborhood-wise median.
# Soch: ek hi area ke gharon ki frontage aas-paas hoti hai, poore dataset ka median
# galat hota (ameer area ki frontage badi hoti hai). Isliye group median.
fe["LotFrontage"] = fe.groupby("Neighborhood")["LotFrontage"].transform(
    lambda s: s.fillna(s.median()))
fe_log["lotfrontage_method"] = "median by Neighborhood"

# Group C — chhoti missing: MasVnrArea (8) -> 0 jab MasVnrType None, warna median.
# Electrical (1 row) -> mode (SBrkr).
fe.loc[fe["MasVnrType"] == "None", "MasVnrArea"] = fe.loc[
    fe["MasVnrType"] == "None", "MasVnrArea"].fillna(0)
fe["MasVnrArea"] = fe["MasVnrArea"].fillna(fe["MasVnrArea"].median())
fe["Electrical"] = fe["Electrical"].fillna(fe["Electrical"].mode()[0])
fe["GarageYrBlt"] = fe["GarageYrBlt"].fillna(fe["GarageYrBlt"].median())
fe_log["remaining_missing"] = int(fe.isna().sum().sum())

# --- 5.3 Naye features — domain knowledge se ---
# Soch: model ko alag-alag tukde dene se behtar hai ek meaningful number do.
# Kharidne wala "kul kitna area, kul kitne bathroom, ghar kitna purana" sochta hai.
fe["TotalSF"] = fe["TotalBsmtSF"] + fe["1stFlrSF"] + fe["2ndFlrSF"]
fe["TotalBath"] = (fe["FullBath"] + 0.5 * fe["HalfBath"]
                   + fe["BsmtFullBath"] + 0.5 * fe["BsmtHalfBath"])
fe["TotalPorchSF"] = (fe["OpenPorchSF"] + fe["EnclosedPorch"]
                      + fe["3SsnPorch"] + fe["ScreenPorch"])
fe["HouseAge"] = fe["YrSold"] - fe["YearBuilt"]        # ghar kitna purana
fe["RemodAge"] = fe["YrSold"] - fe["YearRemodAdd"]     # remodel ko kitna time
fe["HasPool"] = (fe["PoolArea"] > 0).astype(int)
fe["HasGarage"] = (fe["GarageArea"] > 0).astype(int)
fe["HasFireplace"] = (fe["Fireplaces"] > 0).astype(int)
fe["HasBsmt"] = (fe["TotalBsmtSF"] > 0).astype(int)
fe_log["new_features"] = ["TotalSF", "TotalBath", "TotalPorchSF", "HouseAge",
                          "RemodAge", "HasPool", "HasGarage", "HasFireplace", "HasBsmt"]

# --- 5.4 Ordinal encoding — order wali categories ko number do ---
# Soch: ExterQual me Ex < Gd < TA < Fa < Po ek order hai. One-hot karne se ye order
# kho jata; number dene se model ko pata chalta hai "Gd, TA se behtar hai".
ordinal_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ord_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
            "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
for c in ord_cols:
    fe[c] = fe[c].map(ordinal_map)
fe["BsmtExposure"] = fe["BsmtExposure"].map(
    {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4})
fe["BsmtFinType1"] = fe["BsmtFinType1"].map(
    {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6})
fe["BsmtFinType2"] = fe["BsmtFinType2"].map(
    {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6})
fe["Functional"] = fe["Functional"].map(
    {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7})
fe["Fence"] = fe["Fence"].map(
    {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4})
fe["LotShape"] = fe["LotShape"].map({"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3})
fe_log["ordinal_encoded"] = ord_cols + ["BsmtExposure", "BsmtFinType1",
                                        "BsmtFinType2", "Functional", "Fence", "LotShape"]

# --- 5.5 Skew theek karna — log1p ---
# Soch: Linear Regression assume karta hai residuals normal honge. Right-skewed
# features (LotArea skew 12.21) aur target (skew 1.88) is assumption ko todte hain.
# log1p skew ko dabata hai; 0 wale values bhi safe rehte hain (log(0) error nahi).
skewed = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
          "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea",
          "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
          "ScreenPorch", "MiscVal", "PoolArea", "LowQualFinSF", "TotalSF"]
for c in skewed:
    fe[c] = np.log1p(fe[c])
fe["SalePrice_log"] = np.log1p(fe["SalePrice"])
fe_log["log1p_features"] = skewed

# --- 5.6 One-hot — bina order wali categories ---
# Soch: Neighborhood, MSZoning jaise features me koi order nahi hai (NridgHt > OldTown
# likhna bekaar hai). Inhe 0/1 columns me todo, warna model galat order seekh lega.
nom_cols = fe.select_dtypes(include="object").columns.tolist()
fe = pd.get_dummies(fe, columns=nom_cols, drop_first=True, dtype=int)
fe_log["one_hot_cols"] = nom_cols

# --- 5.7 Id hatana ---
fe = fe.drop(columns=["Id"])
fe_log["dropped"] = ["Id"]

fe.to_csv("data/train_engineered.csv", index=False)
print("Engineered shape:", fe.shape)
print("Remaining missing:", int(fe.isna().sum().sum()))

summary = {
    "raw_shape": [int(x) for x in df.shape],
    "numeric_cols": 38, "categorical_cols": 43, "duplicates": 0,
    "target_mean": round(float(df["SalePrice"].mean()), 1),
    "target_median": round(float(df["SalePrice"].median()), 1),
    "target_min": int(df["SalePrice"].min()), "target_max": int(df["SalePrice"].max()),
    "target_skew": round(float(df["SalePrice"].skew()), 2),
    "target_log_skew": round(float(np.log1p(df["SalePrice"]).skew()), 2),
    "total_missing_cells": 7829,
    "top_corr": {k: round(float(v), 3) for k, v in corr.head(12).items()},
    "garage_corr": 0.882,
    "neighborhood_top3": {k: round(float(v), 0) for k, v in nb.head(3).items()},
    "neighborhood_count": 25,
    "engineered_shape": [int(x) for x in fe.shape],
    "fe_log": fe_log,
}
with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("summary.json + train_engineered.csv saved.")
