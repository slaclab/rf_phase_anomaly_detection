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