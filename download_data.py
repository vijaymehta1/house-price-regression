from sklearn.datasets import fetch_openml
import pandas as pd

# Kaggle competition "House Prices: Advanced Regression Techniques" ka Ames Housing data
# OpenML id 42165 = wahi dataset (train + test combined, 2930 rows)
bunch = fetch_openml(data_id=42165, as_frame=True, parser="auto")
df = bunch.frame
print("Shape:", df.shape)
print("Columns:", df.shape[1])
df.to_csv("data/ames_housing_full.csv", index=False)

# Kaggle train.csv = Id 1..1460, test.csv = Id 1461..2919
train = df[df["Id"] <= 1460].copy()
test = df[df["Id"] > 1460].copy()
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
print("Train:", train.shape, "| Test:", test.shape)
print("SalePrice missing in test (expected):", test["SalePrice"].isna().sum())
# Kaggle train.csv ke known values se verification
print("Id=1 SalePrice (expect 208500):", train.loc[train["Id"]==1, "SalePrice"].values)
print("Id=2 SalePrice (expect 181500):", train.loc[train["Id"]==2, "SalePrice"].values)
print("Id=3 SalePrice (expect 223500):", train.loc[train["Id"]==3, "SalePrice"].values)
print("Target nulls in train:", train["SalePrice"].isna().sum())
