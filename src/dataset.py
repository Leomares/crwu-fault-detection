import os
from pathlib import Path

from tqdm import tqdm
from time import time, sleep
import requests
from bs4 import BeautifulSoup
import pandas as pd


MIN_REQUEST_INTERVAL_SECONDS = 0.5


WEBPAGE_PER_FAULT_POSITION = {
    "12k-fan-end-bearing-fault-data": "https://engineering.case.edu/bearingdatacenter/12k-fan-end-bearing-fault-data",
    "12k-drive-end-bearing-fault-data": "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data",
    "48k-drive-end-bearing-fault-data": "https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data",
    "normal-baseline-data": "https://engineering.case.edu/bearingdatacenter/normal-baseline-data"
}


def extract_soup_data(soup:BeautifulSoup, is_faulty_data:bool = True) -> pd.DataFrame|None:
    table = soup.find("table")
    rows = table.find_all("tr")
    
    data = []
    for row in rows:
        cols = row.find_all("td")
        cols = [col for col in cols]
        data.append(cols)

    if len(data) == 0:
        raise ValueError("No data found in the table extracted from soup.")
        return None
    
    numerical_columns = ["fault_diam_in","motor_load_hp","motor_speed_rpm"]
    categorical_columns = ["fault_type","mat_url"]

    if is_faulty_data:
        temp_categorical_columns = ["inner","ball","outer_6","outer_3","outer_12"]
        cols = numerical_columns + temp_categorical_columns
        df = pd.DataFrame(columns=cols)
        
        data = data[2:]
        fault_diam_in = []
        last_fault_diam_in = None
        for row in data:
            curr_fault_diam_in = row[cols.index("fault_diam_in")].text.strip().replace("\"","")
            if len(curr_fault_diam_in) > 0:
                fault_diam_in.append(float(curr_fault_diam_in))            
            else:
                fault_diam_in.append(last_fault_diam_in)
            last_fault_diam_in = fault_diam_in[-1]

        motor_data = []
        for row in data:
            curr_motor_load = row[cols.index("motor_load_hp")].text.strip()
            curr_motor_speed = row[cols.index("motor_speed_rpm")].text.strip()
            motor_data.append([curr_motor_load, curr_motor_speed])

        fault_uri = []
        for row in data:
            for col in temp_categorical_columns:
                curr_fault_type = row[cols.index(col)].find("a")
                if curr_fault_type is not None:
                    row[cols.index(col)] = curr_fault_type.get("href").strip()
                else:
                    row[cols.index(col)] = None
            fault_uri.append(row)

        df[["motor_load_hp","motor_speed_rpm"]] = motor_data
        df["fault_diam_in"] = fault_diam_in
        for i, col in enumerate(temp_categorical_columns):
            df[col] = [row[cols.index(col)] for row in fault_uri]
        
        if len(df) == 0:
            raise ValueError("No data extracted from the table.")
            return None

        df = pd.melt(
            df, 
            id_vars=["motor_load_hp","motor_speed_rpm","fault_diam_in"], 
            value_vars=temp_categorical_columns, 
            var_name="fault_type", 
            value_name="mat_url"
        )
        cols = numerical_columns + categorical_columns
        df = df[[col for col in cols if col not in temp_categorical_columns]]

        if len(df) == 0:
            raise ValueError("No data extracted after melting the dataframe.")
            return None

    else:
        cols = numerical_columns + categorical_columns
        df = pd.DataFrame(columns=cols)
        
        data = data[1:]
        normal_baseline_data = []
        for row in data:
            curr_motor_load = row[0].text.strip()
            curr_motor_speed = row[1].text.strip()
            curr_uri = row[2].find("a").get("href").strip()
            normal_baseline_data.append([curr_motor_load, curr_motor_speed, curr_uri])
        df[["motor_load_hp","motor_speed_rpm","mat_url"]] = normal_baseline_data

    return df


class DatasetScraper():
    """
    Dataset wrapper that handles webscrapping, metadata exctraction and persistent local storage of the raw time series data.
    """

    def __init__(self):
        self.current_path = Path(__file__).resolve()
        self.dataset_path = self.current_path.parent.parent/"dataset"
        os.makedirs(self.dataset_path, exist_ok=True)
        if os.path.exists(self.dataset_path/"intervensions.csv"):
            self.database = pd.read_csv(self.dataset_path/"database.csv",ignore_index=True)
        else:
            self.database = pd.DataFrame()
        self.last_request_time = 0
        return


    def _rate_limited_request(self,url):
        while (time() - self.last_request_time) < MIN_REQUEST_INTERVAL_SECONDS:
            sleep(0.1)
        response = requests.get(url)
        self.last_request_time = time()
        return response


    def fetch_database(self) -> None:
        if len(self.database) > 0:
            print("Resetting existing database.")
            self.database = pd.DataFrame()
            
        for fault_position,url in WEBPAGE_PER_FAULT_POSITION.items():
            response = self._rate_limited_request(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                is_faulty_data = "normal" not in url
                try:
                    df = extract_soup_data(soup, is_faulty_data)
                except ValueError as e:
                    print(f"Error processing data from {url}: {e}")
                    continue
                if df is not None:
                    df["bearing_position"] = fault_position
                    self.database = pd.concat([self.database, df], ignore_index=True)
                else:
                    print(f"No valid data extracted from {url}.")
            else:
                print(f"Failed to fetch data from {url}. Status code: {response.status_code}")

        if len(self.database) > 0:
            self.database.to_csv(self.dataset_path/"database.csv", index=False)


    def resolve_mat_data(self, uri:str) -> str|None:
        raw_data_filename = self.dataset_path/"mat_files"/uri.split("/")[-1]
        if os.path.exists(raw_data_filename):
            return raw_data_filename

        response = self._rate_limited_request(uri)
        if response.status_code == 200:
            raw_data = response.content
            if not os.path.exists(raw_data_filename.parent):
                os.makedirs(raw_data_filename.parent, exist_ok=True)
            with open(raw_data_filename, "wb") as f:
                f.write(raw_data)

            self.database.loc[self.database["mat_url"] == uri, "mat_filename"] = raw_data_filename
            self.database.to_csv(self.dataset_path/"database.csv", index=False)
            return raw_data_filename
        
        print(f"Failed to fetch .mat data from {uri}. Status code: {response.status_code}")
        return None

    
    def resolve_all_mat_data(self) -> None:
        for idx, row in tqdm(self.database.iterrows()):
            mat_url = row["mat_url"]
            if pd.notna(mat_url):
                _ = self.resolve_mat_data(mat_url)