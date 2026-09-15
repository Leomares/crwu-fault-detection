from fastapi.testclient import TestClient

from .main import app, SensorResponse


client = TestClient(app)


example_entry = {
    "feat_avg_power": 0.030597,
    "feat_narrowband_power_rate": 0.563379,
    "feat_most_relevant_binned_freq": 3299.5,
    "feat_most_relevant_binned_power": 471528.555255,
    "feat_binned_fft_power_bin_0": 0.142834,
    "feat_binned_fft_power_bin_1": 0.13092,
    "feat_binned_fft_power_bin_2": 0.05645,
    "feat_binned_fft_power_bin_3": 0.068233,
    "feat_binned_fft_power_bin_4": 0.124684,
    "feat_binned_fft_power_bin_5": 0.206198,
    "feat_binned_fft_power_bin_6": 0.122136,
    "feat_binned_fft_power_bin_7": 0.084743,
    "feat_binned_fft_power_bin_8": 0.062271,
    "feat_binned_fft_power_bin_9": 0.001533,
    "feat_dominant_freq": 0.0,
    "feat_kurtosis": 3.776068,
    "mat_file": "/home/lmres/storage/projects/cwru-fault-detection/data/mat_files/210.mat",
    "sampling_rate": 12000,
    "motor_load_hp": 1,
    "motor_speed_rpm": 1772,
    "fault_size_in": 0.021,
    "telemetry_position": "FE_0",
    "has_fault": 1,
    "fault_type_ball": 0.0,
    "fault_type_inner": 1.0,
    "fault_type_none": 0.0,
    "fault_type_outer": 0.0,
    "fault_position_drive": 1.0,
    "fault_position_fan": 0.0,
    "fault_position_none": 0.0
}

feature_columns = [key for key in SensorResponse.__annotations__.keys() if key not in ["sensor_id", "timestamp"]]
# feature_columns = [
#     "feat_avg_power",
#     "feat_narrowband_power_rate",
#     "feat_most_relevant_binned_freq",
#     "feat_most_relevant_binned_power",
#     "feat_binned_fft_power_bin_0",
#     "feat_binned_fft_power_bin_1",
#     "feat_binned_fft_power_bin_2",
#     "feat_binned_fft_power_bin_3",
#     "feat_binned_fft_power_bin_4",
#     "feat_binned_fft_power_bin_5",
#     "feat_binned_fft_power_bin_6",
#     "feat_binned_fft_power_bin_7",
#     "feat_binned_fft_power_bin_8",
#     "feat_binned_fft_power_bin_9",
#     "feat_dominant_freq",
#     "feat_kurtosis"
# ]


def test_binary_prediction():
    sensor_response = SensorResponse(sensor_id="sensor_1", timestamp="2026-09-14T12:00:00Z", **{col: example_entry[col] for col in feature_columns})
    response = client.post("/predict/fault_binary", json=sensor_response.model_dump())
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == [1]


def test_missing_features_fault_binary():
    sensor_response = SensorResponse(sensor_id="sensor_1", timestamp="2026-09-14T12:00:00Z", **{col: example_entry[col] for col in feature_columns if col != "feat_avg_power"})
    response = client.post("/predict/fault_binary", json=sensor_response.model_dump())
    assert response.status_code == 500  # Unprocessable Entity due to missing required feature
    assert "detail" in response.json()  # Ensure the response contains error details


def test_multiclass_prediction():
    sensor_response = SensorResponse(sensor_id="sensor_1", timestamp="2026-09-14T12:00:00Z", **{col: example_entry[col] for col in feature_columns})
    response = client.post("/predict/fault_multiclass", json=sensor_response.model_dump())
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == ["inner"]


def test_missing_features_multiclass():
    sensor_response = SensorResponse(sensor_id="sensor_1", timestamp="2026-09-14T12:00:00Z", **{col: example_entry[col] for col in feature_columns if col != "feat_avg_power"})
    response = client.post("/predict/fault_multiclass", json=sensor_response.model_dump())
    assert response.status_code == 500  # Unprocessable Entity due to missing required feature


def test_position_multiclass_prediction():
    sensor_response = SensorResponse(sensor_id="sensor_1", timestamp="2026-09-14T12:00:00Z", **{col: example_entry[col] for col in feature_columns})
    response = client.post("/predict/position_multiclass", json=sensor_response.model_dump())
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == ["drive"]


def test_missing_features_position_multiclass():
    sensor_response = SensorResponse(sensor_id="sensor_1", timestamp="2026-09-14T12:00:00Z", **{col: example_entry[col] for col in feature_columns if col != "feat_avg_power"})
    response = client.post("/predict/position_multiclass", json=sensor_response.model_dump())
    assert response.status_code == 500  # Unprocessable Entity due to missing required feature


def test_wrong_endpoint():
    sensor_response = SensorResponse(sensor_id="sensor_1", timestamp="2026-09-14T12:00:00Z", **{col: example_entry[col] for col in feature_columns})
    response = client.post("/predict/non_existent_model", json=sensor_response.model_dump())
    assert response.status_code == 422  # Unprocessable Entity due to wrong model type