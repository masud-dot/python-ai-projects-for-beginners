import numpy as np
import pandas as pd

df = pd.DataFrame({
    "MedInc": [8.0, 7.0, np.nan, 5.0, 4.0],
    "HouseAge": [41.0, 21.0, 52.0, np.nan, 52.0],
    "Region": ["coast", "coast", "inland", None, "inland"],
})
print(df.isna().sum())
print(df["Region"].dtype)
print("rows kept by dropna:", len(df.dropna()))

filled = df.fillna({"MedInc": df["MedInc"].median()})
print(filled["MedInc"].tolist())

df.loc[df["HouseAge"] > 50, "HouseAge"] = np.nan
print(df["HouseAge"].tolist())
