import os
from pathlib import Path
import pandas as pd
from scipy.io import loadmat


def get_dataframe_from_mat(file_path: Path) -> pd.DataFrame:
    if not os.path.exists(file_path) or file_path.suffix != ".mat":
        raise Exception(f"[ERROR] Coundn't open mat file at {file_path}.")
    
    mat_data = loadmat(file_path)
    telemetry_positions = ["BA","DE","FE"]
    df = pd.DataFrame()

    for pos in telemetry_positions:
        valid_cols = [col for col in mat_data.keys() if f"_{pos}_" in col]
        for index,col in enumerate(valid_cols):
            try:
                df[f"{pos}_{index}"] = mat_data[col].flatten()
            except Exception as e:
                print(f"[ERROR] Failed to process column {col} in {os.path.basename(file_path)}: {e}")
    return df
        



