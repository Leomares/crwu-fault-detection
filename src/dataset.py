import os
import numpy as np
import pandas as pd
from collections import defaultdict

from scipy.stats import kurtosis

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from .utils import get_dataframe_from_mat


@dataclass
class DatasetIndex:
    mat_file:str
    sampling_rate: int
    motor_load_hp: int
    motor_speed_rpm: int
    fault_position: str
    fault_type: str
    fault_size_in: float
    telemetry_position: str


DEFAULT_DATASET_CONFIG = {
    "time_per_datapoint_s": 1,
    "filters": [("sampling_rate",[12_000])]
}


DATA_FOLDER = "data"


class _Dataset:
    """
    Class responsible for creating the dataset used to train the model. 
    It loads the database time series, group them into windows and compute features for each window. 
    """

    def __init__(self):
        self._data_dir = Path(__file__).resolve().parent.parent/DATA_FOLDER
        database_path = self._data_dir/"database.csv"
        if os.path.exists(database_path):
            self.database = pd.read_csv(database_path)
        else:
            raise Exception("Database file not found. Build it with the scraper class before creating the dataset.")
        dataset_path = self._data_dir/"dataset.csv"
        if os.path.exists(dataset_path):
            self.data = pd.read_csv(dataset_path)
        else:
            self.data = None


    def get_dataset(self, time_per_datapoint_s: float, rebuild: float = False) -> pd.DataFrame:
        if self.data is not None and not rebuild:
            print("Dataset already exists. Loading from disk.")
            return self.data

        df = pd.DataFrame()
        database = self.database.copy()
        database["sampling_rate"] = database["bearing_position"].apply(lambda x: 48_000 if "48k" in x else 12_000)
        database["fault_position"] = database["bearing_position"].apply(lambda x: "fan" if "fan" in x else "drive" if "drive" in x else "")

        for _, row in tqdm(database.iterrows()):
            sampling_rate = row["sampling_rate"]
            motor_load_hp = row["motor_load_hp"]
            motor_speed_rpm = row["motor_speed_rpm"]
            fault_position = row["fault_position"]
            fault_type = row["fault_type"]
            fault_size_in = row["fault_diam_in"]
            
            if not isinstance(row["mat_filename"],str):
                continue

            mat_filename = self._data_dir/"mat_files"/os.path.basename(row["mat_filename"])
            mat_data = get_dataframe_from_mat(mat_filename)
            
            for col,ts in mat_data.items():
                telemetry_position = col
                dataset_index = DatasetIndex(
                    mat_file=mat_filename,
                    sampling_rate=sampling_rate,
                    motor_load_hp=motor_load_hp,
                    motor_speed_rpm=motor_speed_rpm,
                    fault_position=fault_position,
                    fault_type=fault_type,
                    fault_size_in=fault_size_in,
                    telemetry_position=telemetry_position
                )
                dataset_index = pd.DataFrame([dataset_index.__dict__])
                features = self._compute_features(pd.Series(ts), sampling_rate, time_per_datapoint_s)
                features = features.add_prefix("feat_")
                features_indexed = features.copy()
                for col in dataset_index.columns:
                    features_indexed[col] = dataset_index[col].iloc[0]
                # features_indexed = pd.concat([dataset_index,features], axis=1)
                # print(f"[DEBUG] Size check: {dataset_index.shape} + {features.shape} = {features_indexed.shape}")
                df = pd.concat([df, features_indexed], ignore_index=True)
                # print(f"[DEBUG] Processed {col} from {mat_filename}. Current dataset size: {df.shape}")
        
        self.data = df
        self.data.to_csv(self._data_dir/"dataset.csv", index=False)
        return df


    def _compute_features(self, ts: pd.Series, sampling_rate: int, time_per_datapoint_s: float) -> pd.DataFrame:
        features = defaultdict(list)
        bin_size = int(sampling_rate * time_per_datapoint_s)
        n_bins = int(ts.size // bin_size) 

        for bin_index in range(n_bins):
            window = ts[bin_index*bin_size:(bin_index+1)*bin_size]
            
            # average power 
            avg_power = window.pow(2).mean()
            features["avg_power"].append(avg_power)

            #power spectrum between 2kHz and 5kHz 
            fft = np.fft.fft(window)
            freqs = np.fft.fftfreq(len(window), d=1/sampling_rate)
            fft_power = np.abs(fft[:len(fft)//2])**2
            freqs = freqs[:len(freqs)//2]
            
            mask = (freqs > 2_000) & (freqs < 5_000)
            rate = fft_power[mask].sum() / fft_power.sum() if fft_power.sum() > 0 else 0
            features["narrowband_power_rate"].append(rate)

            #most relevant binned frequency components
            n_bins = 10
            binned_freqs = np.array_split(freqs, n_bins)
            binned_fft_power = np.array_split(fft_power, n_bins)
            summed_binned_fft_power = [bp.sum() for bp in binned_fft_power]
            most_relevant_binned_freq = binned_freqs[np.argmax(summed_binned_fft_power)].mean() if fft_power.sum() > 0 else 0
            features["most_relevant_binned_freq"].append(most_relevant_binned_freq)
            features["most_relevant_binned_power"].append(np.max(summed_binned_fft_power))
            for i, bp in enumerate(summed_binned_fft_power):
                features[f"binned_fft_power_bin_{i}"].append(np.sum(bp)/fft_power.sum() if fft_power.sum() > 0 else 0)

            #most relevant frequency component
            dominant_freq = freqs[np.argmax(fft_power)] if fft_power.sum() > 0 else 0
            features["dominant_freq"].append(dominant_freq)

            #kurtosis
            k = kurtosis(window,fisher=False)
            features["kurtosis"].append(k)

        return pd.DataFrame(features)


class Dataloader:
    """
    Class that manages the creation and yielding of batches of data used in training/inference.
    """
   
    def __init__(self, user_config: dict = {}):
        # self.data:pd.DataFrame = None
        self.config = DEFAULT_DATASET_CONFIG | user_config
        self.dataset = _Dataset()


    def get_data(self) -> pd.DataFrame:
        curr_data = self.dataset.get_dataset(self.config["time_per_datapoint_s"])
        if "filters" in self.config:
           for col, values in self.config["filters"]:
               curr_data = curr_data.loc[curr_data[col].isin(values),:]
        return curr_data


    def get_batch(self, batch_size: int, randomize:bool=True) -> pd.DataFrame:
        curr_data = self.dataset.get_dataset(self.config["time_per_datapoint_s"])
        if "filters" in self.config:
           for col, values in self.config["filters"]:
               curr_data = curr_data.loc[curr_data[col].isin(values),:]
        
        if randomize:
            curr_data = curr_data.sample(frac=1, random_state=42).reset_index(drop=True)

        n_batches = len(curr_data) // batch_size
        for i in range(n_batches):
            yield curr_data.iloc[i*batch_size:(i+1)*batch_size]

