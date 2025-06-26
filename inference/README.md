This is the subdirectory for the inference models used to detect the anomalies. 
It contains the models and necessary files/modules to run inference on the model.

## Model Files
The model files are stored in the `models` subdirectory. The model files include
a YAML and a PT file for each of the RF and BPM neural networks. These are generated 
using the `lume-model` package (as `TorchModule` instances) and are used to load the 
model for inference. Each model should have the following files:
- `rf_model.pt`: The PyTorch model file for the RF neural network.
- `bpm_model.pt`: The PyTorch model file for the BPM neural network.
- `rf_model.yml`: The YAML configuration file for the RF neural network.
- `bpm_model.yml`: The YAML configuration file for the BPM neural network.

When updating the models (e.g., retraining), ensure that the new model files are 
placed in the `models` subdirectory and that the YAML files are updated accordingly.

If in the future we want to load the models from MLflow for example, we would have to 
update the `load_models` function in `predict.py` to load the models from MLflow instead of the local files.

## NN Module (AnomNet)
The `AnomNet` class in `anomnet.py` is the NN source code. It is a pared-down version of the AnomNet model from 
the [original code](https://github.com/SLAC-ML/CoincAD/blob/phase/core/CoincAD_NN_utils.py), where only the 1D 
convolutional layers and fully connected layers are retained. 

Note that the inputs for this model are the RF and BPM data, which are expected to be in of the shape
`(1, 1066)` for RF and `(8, 1066)` for BPM, otherwise the model will fail to compute the output (note
we are only using single batches here).

If after retraining, the model class is adjusted/updated, ensure that the `AnomNet` class is updated 
accordingly to reflect the new architecture and the new expected input shapes.

The `config.yml` file contains the configuration for the trained AnomNet models under the "network" key. 
The first element is the RF model and the second element is the BPM model. These are only included here
for reference and are not used in the inference code. These should also be updated if the model 
changes after retraining.

## Inference Code
The inference code is in `predict.py`. If after retraining we need to adjust any logic with generating the 
labels, we can do so in the `predict_label` function. The `config.yml` file contains the configuration for 
the predictions, under the `predict` key. Any changes to the prediction logic should be reflected here as well
(e.g., thresholds or standardization).

## Updating the Models in LUME-model
Below is an example of how to update/generate the model files using `lume-model`. 

Note that you should adjust
any changes to the variables (defaults, value ranges, etc.) in the `ScalarVariable` definitions as needed.

Additionally, ensure that the `networks` variable contains the trained models (RF and BPM, respectively) before 
running this code and that the model class is available. When loading the LUME-model code in this inference module,
the source code for the models should be available (e.g., in the same directory or properly imported as `anomnet.py`
is here). There is also the option to save the models as TorchScript models to avoid the need for the source code, 
but this is not shown here.

```python
from lume_model.models import TorchModel, TorchModule
from lume_model.variables import ScalarVariable

## Make sure to import the trained networks, and that the source code the models is available
# networks = [rf_network, bpm_network]  # Example, replace with actual trained models

# Save model 1 (RF)
# variable specification
input_variables = [
    ScalarVariable(name="klys_phase", default_value=0.0, value_range=[-400, 400]),
]
output_variables = [
    ScalarVariable(name="anom_score"),
]

# creation of TorchModel
rf_model = TorchModel(
    model=networks[0],
    input_variables=input_variables,
    output_variables=output_variables,
    fixed_model=True,
)

# wrap in TorchModule and dump to YAML and PT files
rf_module = TorchModule(model=rf_model)
rf_module.dump("rf_module.yml")


# Save model 2 (BPM)
bpm_list = ['BPMS:LTUH:250:X',    'BPMS:LTUH:450:X',    'BPMS:DMPH:502:Y',    'BPMS:DMPH:693:Y',    
            'BPMS:LTUH:250:TMIT', 'BPMS:LTUH:450:TMIT', 'BPMS:DMPH:502:TMIT', 'BPMS:DMPH:693:TMIT']
# variable specification
input_variables = [
    ScalarVariable(name=bpm_list[0], default_value=0.0, value_range=[-200, 200]),
    ScalarVariable(name=bpm_list[1], default_value=0.0, value_range=[-200, 200]),
    ScalarVariable(name=bpm_list[2], default_value=0.0, value_range=[-200, 200]),
    ScalarVariable(name=bpm_list[3], default_value=0.0, value_range=[-200, 200]),
    ScalarVariable(name=bpm_list[4], default_value=0.0, value_range=[-200, 200]),
    ScalarVariable(name=bpm_list[5], default_value=0.0, value_range=[-200, 200]),
    ScalarVariable(name=bpm_list[6], default_value=0.0, value_range=[-200, 200]),
    ScalarVariable(name=bpm_list[7], default_value=0.0, value_range=[-200, 200]),
]
output_variables = [
    ScalarVariable(name="anom_score"),
]

# creation of TorchModel
bpm_model = TorchModel(
    model=networks[1],
    input_variables=input_variables,
    output_variables=output_variables,
    fixed_model=True,
)

# wrap in TorchModule and dump to YAML and PT files
bpm_module = TorchModule(model=bpm_model)
bpm_module.dump("bpm_module.yml")
```