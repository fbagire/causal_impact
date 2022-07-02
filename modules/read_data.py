import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

le = LabelEncoder()


def read_data(path):
    data = pd.read_csv(path)
    try:
        data.drop('Unnamed: 32', axis=1, inplace=True)
    except KeyError:
        pass
    non_numeric_columns = list(data.select_dtypes(exclude=[np.number]).columns)

    for col in non_numeric_columns:
        data[col] = le.fit_transform(data[col])

    return data
