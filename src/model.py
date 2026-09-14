from enum import Enum
from hashlib import md5
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


class ModelTypes(Enum):
    """
        Models available to be used with the API.
    """
    fault_binary=1
    fault_multiclass=2
    position_multiclass=3


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_cols: list, categorical_cols: list) -> None:
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

        self.log_cols = [
            "kurtosis",
            "feat_avg_power",
            "feat_narrowband_power_rate",
        ]

        self.scaler_cols = self.numerical_cols
        self.scalers = {col: StandardScaler() for col in self.scaler_cols}
        self.onehot_cols = categorical_cols
        self.onehot_encoder = OneHotEncoder()

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None):
        if len(self.onehot_cols) > 0:
            self.onehot_encoder.fit(
                x[self.onehot_cols].to_numpy().reshape(-1, len(self.onehot_cols))
            )
        for col in self.scaler_cols:
            if col in self.log_cols:
                self.scalers[col].fit(x[col].apply(np.log1p).to_numpy().reshape(-1, 1))
            else:
                self.scalers[col].fit(x[col].to_numpy().reshape(-1, 1))
        return self

    def transform(self, x: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        x = x.copy()
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
                x[col] = self.scalers[col].transform(
                    x[col].apply(np.log1p).to_numpy().reshape(-1, 1)
                )
            else:
                x[col] = self.scalers[col].transform(x[col].to_numpy().reshape(-1, 1))
        # x[self.log_cols] = x[self.log_cols].apply(np.log1p)
        return x


class InferenceModel:
    """
    Template for all classification models sharing the previously defined preprocessor.
    """
    def __init__(self, input_columns: list[str]|None = None, label: ModelTypes|None = None, xgb_params: dict = {}, file: str = "") -> None:
        self.label = label
        if file != "":
            self.load_pipeline(file)
        else:
            if input_columns is None or label is None:
                raise Exception("input_columns or label parameters missing.")
            preprocessor = Preprocessor(numerical_cols=input_columns, categorical_cols=[])
            classifier = XGBClassifier(**xgb_params)
            self.pipeline = make_pipeline(
                preprocessor,
                classifier
            )
   
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict = {}) -> None:
        self.pipeline.fit(X_train,y_train,**params)
    
    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        if any(col not in X.columns for col in self.pipeline.named_steps['preprocessor'].numerical_cols):
            raise ValueError("Input DataFrame is missing required columns: " + ", ".join(
                col for col in self.pipeline.named_steps['preprocessor'].numerical_cols if col not in X.columns
            ))
        return self.pipeline.predict(X)

    def load_pipeline(self, file:str):
       self.pipeline = joblib.load(file)

    def save_pipeline(self) -> str:
        model_dir = Path().cwd().parent/"model"
        model_hash = md5(self.pipeline.__str__().encode()).hexdigest()
        model_file = f"pipeline_{self.label.name}_{model_hash}.joblib"
        model_path = model_dir/model_file
        joblib.dump(self.pipeline,model_path)

        return str(model_path)

    @staticmethod
    def model_inference(model_file:str, features: dict) -> pd.DataFrame:
        model = InferenceModel(file=model_file)
        return model(pd.DataFrame([features]))
