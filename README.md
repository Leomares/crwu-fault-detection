# Introduction

This repository concerns the study of the CWRU bearing dataset and techniques for detecting and predicting ball bearing faults.

The overview of the experiment design that produced the dataset can be found in the website linked below and it describes the following factors:
- Motor Load: 0, 1, 2 and 3 horsepower.
- Bearing Manufacturer: SKF and NTN (this is a block factor, conditioning fault diameter/depth).
- Fault Location: Inner Raceway, Ball, Outer Raceway 12, 6 and 3 o'clock positions. 
- Fault diameter: 7, 14 and 21 mills for SKF; 28 and 40 mills for NTN. 

Treatment response:
- Accelerometer measurements at different positions: Fan End, Drive End and Base.

# Study Design

The study design involves collecting the accelerometer measurements under several conditions, visualizing patterns on the raw data, building a dataset with features from segments of the time series, and training classifiers to mainly two problems: detect the fault and where it is.

- Local database with all accelerometers time series available in order to produce a training dataset on demand
- Comphreensive features of the signals on time and frequency domain
- API to run inference on the models trained [TODO] 

# Resources

Bearing Data Center from the Case Western Reserve University: https://engineering.case.edu/bearingdatacenter.
