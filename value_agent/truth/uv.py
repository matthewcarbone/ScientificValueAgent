from functools import cache
from pathlib import Path

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


@cache
def _get_uv_model():

    p = Path(__file__).absolute().parent / "uv_data.csv"
    df = pd.read_csv(p)
    X = df[["NCit", "pH", "HA"]].to_numpy()
    X[:, 1] += 16.0
    Y = df.iloc[:, 4:].to_numpy()
    knn = KNeighborsRegressor(n_neighbors=2, weights="distance")
    knn.fit(X, Y)
    return knn


def truth_uv(x):
    model = _get_uv_model()
    return model.predict(x)
