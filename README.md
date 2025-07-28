# rf_phase_anomaly_detection
phase anomaly detection for rf stations

This repo holds the backend code for RF anomaly detection using Jason's algorithm ([arXiv paper here](https://arxiv.org/abs/2505.16052)).

This section to be updated while code is implemented:

k2eg_process - handles k2eg snapshots
process_b - does data cleaning and accelerator health inspection; generates anomaly candidates
process_c - runs CoAD to confirm candidates

## Installation instructions on S3DF

This package requires k2eg to function on S3DF.
To install k2eg along with this current repo:

```
conda create --name rf_phase_ad python=3.10
conda activate rf_phase_ad
mkdir phase_ad
cd phase_ad
git clone git@github.com:slaclab/rf_phase_anomaly_detection.git
cd rf_phase_anomaly_detection
pip install -r requirements.txt
pip install -r dev-requirements.txt
```

It is recommended you set up the pre-commit tool to run before each commit you make.
This will auto-format your code and tidy things up by removing trailing spaces, extra new-lines, etc.
```
# you should be in the phase_ad dir
cd rf_phase_anomaly_detection
pip install pre-commit
pre-commit install
```

You can also run pre-commit before actually making a commit with:
```
pre-commit run --all-files
```

To [set/unset](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux) the required environment variables:
```
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
```

then edit `./etc/conda/activate.d/env_vars.sh` to include:
```
export K2EG_PYTHON_CONFIGURATION_PATH_FOLDER=/sdf/sw/k2eg/configuration
```

and edit `./etc/conda/deactivate.d/env_vars.sh` to include
```
unset K2EG_PYTHON_CONFIGURATION_PATH_FOLDER
```

then reactivate the conda environment to have the environment variables load:
```
conda activate rf_phase_ad
```

and then be sure to cd back to the rf_phase_anomaly_detection dir to finally start running the detection pipeline.

## Running the pipeline

To run the pipeline on real PV data from k2eg, simply run:
```
python main.py
```

To use spoofed (random) data produced locally, you can set the cmdline flag:
```
python main.py --spoof_k2eg_data
```

And log files will be written each run unless disabled with the cmdline flag:
```
python main.py --disable_file_logging
```

## Process B
#### Part 1 - Beam Checks
1. Is the beam rate 120 Hz? We require IOC∶BSY0∶MP01∶PCRATE == 8.
2. Is the entire beam being delivered to the hard x-ray line? We require IOC∶IN20∶EV01∶RG02ACTRATE == 10.
3. Is the beam stopper being used? We require STPR∶BSYH∶2∶STD2INA == 0, indicating the beam stopper is out.
4. Does the beam have a standard charge at the injector? We require BPMS∶IN20∶221∶TMITCUH >
0.5 × 10**9. A lower charge or the charge not being logged can indicate the beam is not being operated in a standard operational mode.

In the future, we do allow temporary (<90 s) violations of these conditions to not disallow short periods of “nonstandard” beam operation caused by automatic feedback or protection-based control mechanisms.

### Part 2 - Candidate Generation
[TODO]


## Docker image deployment on Kubernetes (S3DF)
For more detailed instructions, refer to this [documentation](https://github.com/slaclab/lcls_cu_injector_ml_model?tab=readme-ov-file#containerization-steps).

To deploy the Docker image on Kubernetes, follow these steps. If you have updated the tag,
make sure to replace `<tag>` with the new tag in the commands below, **and in the deployment YAML file**.

0. Ensure you have Docker installed and running on your machine.
1. Update the `deployment.yaml` file with the correct image tag and registry information, if that information has changed.
2. Build the Docker image. The `platform` tag is necessary if you are developing on a machine with a different architecture. 
If you are NOT building on a MacOS machine, you can skip the `--provenance` flag.
   ```bash
   docker build -t rf-phase-anomaly-detection:<tag> . --platform=linux/amd64 --provenance=false
   ```
2. Push the Docker image to the Stanford Container Registry (replace `<your-username>` with your actual username):
    ```bash
    cat ~/.scr-token | docker login --username $USER --password-stdin http://scr.svc.stanford.edu
    docker tag rf-phase-anomaly-detection:<tag> scr.svc.stanford.edu/<your-username>/rf-phase-anomaly-detection:<tag>
    docker push scr.svc.stanford.edu/<your-username>/rf-phase-anomaly-detection:<tag>
    ```
3. To set up kubectl, follow this [link](https://k8s.slac.stanford.edu/ad-accel-online-ml). Update the Kubernetes deployment with the new image:
    ```bash
    kubectl apply -f deployment.yaml
    ```
   
## Accessing the deployed image
Once the image is deployed, you can access following these steps:
1. List all pods to find the name of the pod running the image:
   ```bash
   kubectl get pods
   ```
2. Use the following command to access the pod:
   ```bash
   kubectl exec -ti <pod-name> -- bash
    ```
3. Once inside the pod, you can run the Python scripts or other commands as needed. To run the 
tests, you can run for example:
   ```bash
   python -m tests.anom_table_k2eg_test
   ```
    or
    
    ```bash
    pytest tests/inference/test_predict.py  -s -rsx -v
    ```
    If you want to edit the code, you can use a text editor like `vim` to modify the files directly within the pod. 
    For example if you want to edit the `anom_table_k2eg_test.py` file to write/not write to PV:
   ```bash
   apt update && apt install vim
   vi tests/anom_table_k2eg_test.py
   ```
