# rf_phase_anomaly_detection
phase anomaly detection for rf stations

This repo holds the backend code for RF anomaly detection using Jason's algorithm ([arXiv paper here](https://arxiv.org/abs/2505.16052)).

This section to be updated while code is implemented:

k2eg_process - handles k2eg snapshots  
process_b - does data cleaning and accelerator health inspection; generates anomaly candidates  
process_c - runs CoAD to confirm candidates  

## Installation instructions on S3DF

This package requires k2eg to function on S3DF.
To install k2eg:

```
conda create --name rf_phase_ad python=3.10
conda activate rf_phase_ad
mkdir phase_ad
cd phase_ad
git clone https://github.com/slaclab/k2eg-python.git
cd k2eg-python
pip install -r requirements.txt
pip install -e .
cd ..
git clone git@github.com:slaclab/rf_phase_anomaly_detection.git
```

To [set/unset](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux) the required environment variable:
```
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
```

then edit `./etc/conda/activate.d/env_vars.sh` to include
```
export K2EG_PYTHON_CONFIGURATION_PATH_FOLDER=/sdf/sw/k2eg/configuration
```

and edit `/etc/conda/deactivate.d/env_vars.sh` to include
```
unset K2EG_PYTHON_CONFIGURATION_PATH_FOLDER
```
