import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols: list, categorical_cols: list) -> None:
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

        self.log_cols = [
            # "fault_size_in",
            "kurtosis",
            "feat_avg_power",
            "feat_narrowband_power_rate",
        ]

        self.scaler_cols = self.numerical_cols
        self.scalers = {col:StandardScaler() for col in self.scaler_cols} 
        self.onehot_cols = categorical_cols
        self.onehot_encoder = OneHotEncoder()

    def fit(self, x: pd.DataFrame, y: pd.Series = None):
        if len(self.onehot_cols) > 0:
            self.onehot_encoder.fit(x[self.onehot_cols].to_numpy().reshape(-1, len(self.onehot_cols)))
        for col in self.scaler_cols:
            if col in self.log_cols:
                self.scalers[col].fit(x[col].apply(np.log1p).to_numpy().reshape(-1, 1))
            else:
                self.scalers[col].fit(x[col].to_numpy().reshape(-1, 1))
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        if len(self.onehot_cols) > 0:
            onehot_array = self.onehot_encoder.transform(x[self.onehot_cols])
            onehot_dataframe = pd.DataFrame(
                onehot_array, columns=self.onehot_encoder.get_feature_names_out()
            ).reset_index(drop=True)
            x = pd.concat(
                (x.filter(self.onehot_cols).reset_index(drop=True), onehot_dataframe),
                axis=1,
            )
        for col in self.scaler_cols:
            if col in self.log_cols:
                x[col] = self.scalers[col].transform(x[col].apply(np.log1p).to_numpy().reshape(-1, 1))
            else:
                x[col] = self.scalers[col].transform(x[col].to_numpy().reshape(-1, 1))
        #x[self.log_cols] = x[self.log_cols].apply(np.log1p)
        return x
