# rf_phase_anomaly_detection
phase anomaly detection for rf stations

This repo holds the backend code for RF anomaly detection using Jason's algorithm ([arXiv paper here](https://arxiv.org/abs/2505.16052)).

This section to be updated while code is implemented:

process_a - handles k2eg snapshots
process_b - does data cleaning and accelerator health inspection; generates anomaly candidates
process_c - runs CoAD to confirm candidates

As of 06/02/2025 this repo only requires python 3.9.18 or higher.