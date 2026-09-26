This repository concerns the study of the CWRU bearing dataset and techniques for detecting and predicting ball bearing faults.

# Usage

## Installation and tests

```bash
# install and lock dependencies
uv sync 

# run tests related to the API endpoints
uv run pytest 

# start the local FastAPI backend
uv run fastapi dev 
```

## Endpoints

The API exposes the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| /predict/fault_binary | POST | Predicts whether a fault is present (binary classification) |
| /predict/fault_multiclass | POST | Predicts the type of fault (multiclass classification) |
| /predict/position_multiclass | POST | Predicts the fault position (multiclass classification) |

The request body for these endpoints should contain the features extracted from the accelerometer measurements segment in JSON format.

| Variable | Description |
|---------|-------------|
| sensor_id | The ID of the sensor that recorded the measurements |
| timestamp | The timestamp of the measurements segment |
| feat_avg_power | The average power of the segment |
| feat_narrowband_power_rate | The narrowband (2 kHz - 5 kHz) power rate of the segment |
| feat_kurtosis | The kurtosis of the segment |
| feat_dominant_freq | The dominant frequency of the segment |
| feat_most_relevant_binned_freq | The most relevant binned frequency of the segment |
| feat_most_relevant_binned_power | The most relevant binned power of the segment |
| feat_binned_fft_power_bin_0 | The FFT power in 1st 600 Hz bin of the segment |
| feat_binned_fft_power_bin_1 | The FFT power in 2nd 600 Hz bin of the segment |
| feat_binned_fft_power_bin_2 | The FFT power in 3rd 600 Hz bin of the segment |
| feat_binned_fft_power_bin_3 | The FFT power in 4th 600 Hz bin of the segment |
| feat_binned_fft_power_bin_4 | The FFT power in 5th 600 Hz bin of the segment |
| feat_binned_fft_power_bin_5 | The FFT power in 6th 600 Hz bin of the segment |
| feat_binned_fft_power_bin_6 | The FFT power in 7th 600 Hz bin of the segment |
| feat_binned_fft_power_bin_7 | The FFT power in 8th 600 Hz bin of the segment |
| feat_binned_fft_power_bin_8 | The FFT power in 9th 600 Hz bin of the segment |
| feat_binned_fft_power_bin_9 | The FFT power in 10th 600 Hz bin of the segment |

The implementation of each feature can be found in the database class defined in `src/dataset.py`.

## Response

Example response:
```json
{
    "model_type": "fault_binary",
    "model_name":"model/pipeline_fault_binary_c25cc66bf776c3afc4de19be9909ff23.joblib",
    "prediction": "has_fault"
}
```
# General information

## Dataset

The overview of the experiment design that produced the dataset can be found in the website linked below and it describes the following factors:
- Motor Load: 0, 1, 2 and 3 horsepower.
- Bearing Manufacturer: SKF and NTN (this is a block factor, conditioning fault diameter/depth).
- Fault Location: Inner Raceway, Ball, Outer Raceway 12, 6 and 3 o'clock positions. 
- Fault diameter: 7, 14 and 21 mills for SKF; 28 and 40 mills for NTN. 

Treatment response:
- Accelerometer measurements at different positions: Fan End, Drive End and Base.

## Study Design

The study design involves collecting the accelerometer measurements under several conditions, visualizing patterns on the raw data, building a processed dataset with features from segments of the time series, and training classifiers to mainly two problems: detect the fault and where it is.

- Local database with all accelerometers time series available in order to produce a training dataset on demand
- Comphreensive features of the signals on time and frequency domain
- Model training and evaluation pipeline for each inference type (binary fault detection, fault position and fault type) stored as joblib
- API to run inference on the trained pipelines

## Model Evaluation
Binary Fault Detection Evaluation:

              precision    recall  f1-score   support

           0       1.00      1.00      1.00        54
           1       1.00      1.00      1.00       600
    accuracy                           1.00       654

Fault Type Detection Evaluation:

              precision    recall  f1-score   support

        ball       0.92      0.76      0.83       156
       inner       0.89      0.87      0.88       170
        none       1.00      1.00      1.00        54
       outer       0.85      0.95      0.89       274

    accuracy                           0.89       654


Fault Position Detection Evaluation:

              precision    recall  f1-score   support

       drive       0.99      0.98      0.99       312
         fan       0.98      0.99      0.99       288
        none       1.00      1.00      1.00        54

    accuracy                           0.99       654

# Resources

Bearing Data Center from the Case Western Reserve University: https://engineering.case.edu/bearingdatacenter.
