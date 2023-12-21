# Federated BioNeMo with NVFlare

## Requirements
Download and run the [BioNeMo Framework](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework) container.
Inside the container, install nvflare: `pip install nvflare>=2.4.0rc6`

## Examples
### 1. Federated Protein Embeddings and Task Model Fitting with BioNeMo

This example notebook shows how to obtain protein learned representations in the form of embeddings using the ESM-1nv pre-trained model. The model is trained with NVIDIA's BioNeMo framework for Large Language Model training and inference. For more details, please visit NVIDIA BioNeMo Service at https://www.nvidia.com/en-us/gpu-cloud/bionemo/.

#### Run
Open and run [task_fitting.ipynb](./task_fitting.ipynb)

### 2. Cross-endpoint multi-task fitting

#### Data: “Five computational developability guidelines for therapeutic antibody profiling”
See https://tdcommons.ai/single_pred_tasks/develop/#tap
- 241 Antibodies (both chains)

#### Task Description: *Regression*. 
Given the antibody's heavy chain and light chain sequence, predict its developability. The input X is a list of two sequences where the first is the heavy chain and the second light chain.

Includes five metrics measuring developability of an antibody: 
 - Complementarity-determining regions (CDR) length - Trivial (excluded)
 - patches of surface hydrophobicity (PSH)
 - patches of positive charge (PPC)
 - patches of negative charge (PNC)
 - structural Fv charge symmetry parameter (SFvCSP)

#### Download and prepare the data
```commandline
python prepare_tap_data.py
```

#### Run training (central, local, & FL)
```commandline
python run_sim_tap.py
```

### 3. Cross-compound task fitting

#### Data: “Predicting Antibody Developability from Sequence using Machine Learning”
See https://tdcommons.ai/single_pred_tasks/develop/#sabdab-chen-et-al
- 2,409 Antibodies (both chains)

#### Task Description: *Binary classification*. 
Given the antibody's heavy chain and light chain sequence, predict its developability. The input X is a list of two sequences where the first is the heavy chain and the second light chain.

#### Download and prepare the data
```commandline
python prepare_sabdab_data.py
```

#### Run training (central, local, & FL)
```commandline
python run_sim_sabdab.py
```

### 4. Subcellular location prediction with ESM2nv 650M
Follow the data download and preparation in [task_fitting.ipynb](./task_fitting.ipynb).

#### Run training (local FL)
```commandline
python run_sim_scl.py
```
