import os
from enum import Enum
from hashlib import md5
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
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


class InferencePipeline:
    """
    Template for all classification models sharing the previously defined preprocessor. The design consists of the preprocessor, followed by a gradient boosting classifier and optional label encoding for the target variable.
    """
    def __init__(self, file: str = "", 
            preprocessor_columns: list[str]|None = None, 
            label: ModelTypes|None = None, 
            xgb_params: dict = {}) -> None:

        self.label = label
        if file != "":
            self.load_pipeline(file)
        else:
            if preprocessor_columns is None or label is None:
                raise ValueError("Parameters 'preprocessor_columns' and 'label' are required when no pipeline file is provided.")
            preprocessor = Preprocessor(numerical_cols=preprocessor_columns, categorical_cols=[])
            classifier = XGBClassifier(**xgb_params)
            self.pipeline = make_pipeline(
                preprocessor,
                classifier
            )
            self.preprocessor_columns = preprocessor_columns
            self.label_encoder: LabelEncoder | None = None


    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, params: dict = {}) -> None:
        """
        Fit the inference model with the given training data and parameters. It also handles label encoding for categorical target columns.
        """
        if y_train.shape[1] != 1:
            raise ValueError("InferenceModel supports exactly one target column.")

        target = y_train.iloc[:, 0]
        fit_params = params.copy()
        if target.dtype not in [np.int64, np.float64]:
            self.label_encoder = LabelEncoder()
            target = self.label_encoder.fit_transform(target)
            if "xgbclassifier__eval_set" in fit_params:
                fit_params["xgbclassifier__eval_set"] = [
                    (features, self.label_encoder.transform(labels.iloc[:, 0]))
                    for features, labels in fit_params["xgbclassifier__eval_set"]
                ]

        self.pipeline.fit(X_train, target, **fit_params)
    

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        if any(col not in X.columns for col in self.pipeline.named_steps['preprocessor'].numerical_cols):
            raise ValueError("Input DataFrame is missing required columns: " + ", ".join(
                col for col in self.pipeline.named_steps['preprocessor'].numerical_cols if col not in X.columns
            ))
        predictions = self.pipeline.predict(X)
        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(predictions)
        return predictions


    def load_pipeline(self, file:str):
        if not os.path.exists(file):
            raise FileNotFoundError(f"Pipeline file '{file}' does not exist.")
        self.pipeline = joblib.load(file)

        model_dir = Path(file).parent
        tags = os.path.basename(file).split("_")
        tags[0] = "encoder"
        label_encoder_file = model_dir/"_".join(tags)
        if os.path.exists(label_encoder_file):
            self.label_encoder = joblib.load(label_encoder_file)
        else:
            self.label_encoder = None
        self.preprocessor_columns = self.pipeline.named_steps['preprocessor'].numerical_cols


    def save_pipeline(self) -> str:
        model_dir = Path().cwd().parent/"model"
        model_hash = md5(self.pipeline.__str__().encode()).hexdigest()
        model_file = f"pipeline_{self.label.name}_{model_hash}.joblib"
        model_path = model_dir/model_file
        joblib.dump(self.pipeline,model_path)

        if self.label_encoder is not None:
            label_encoder_file = f"encoder_{self.label.name}_{model_hash}.joblib"
            label_encoder_path = model_dir/label_encoder_file
            joblib.dump(self.label_encoder,label_encoder_path)

        return str(model_path)


    @staticmethod
    def inference_from_file(model_file:str, features: dict) -> np.ndarray:
        model = InferencePipeline(file=model_file)
        input_df = pd.DataFrame.from_dict([features])
        output = model(input_df)
        return output
