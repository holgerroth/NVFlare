# Split Learning with CIFAR-10

This example includes instructions on how to run [split learning](https://arxiv.org/abs/1810.06060) (SL) using the CIFAR-10 dataset and the FL-simulator.

For instructions of how to run CIFAR-10 in real-world deployment settings, 
see the example on ["Real-world Federated Learning with CIFAR-10"](../cifar10-real-world/README.md).

## (Optional) 1. Set up a virtual environment
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```
(If needed) make all shell scripts executable using
```
find . -name ".sh" -exec chmod +x {} \;
```
initialize virtual environment.
```
source ./virtualenv/set_env.sh
```
install required packages for training
```
pip install --upgrade pip
pip install -r ./virtualenv/min-requirements.txt
```
Set `PYTHONPATH` to include custom files of this example:
```
export PYTHONPATH=${PWD}/..
```

### 2. Download the CIFAR-10 dataset 
To speed up the following experiments, first download the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset:
```
python3 ../pt/utils/cifar10_download_data.py
```

## 3. Run simulated SL experiments

We are using NVFlare's [FL simulator](https://nvflare.readthedocs.io/en/latest/user_guide/fl_simulator.html) to run the following experiments. 

To run the experiment, execute:
```
nvflare simulator job_configs/cifar10_splitnn --workspace /tmp/nvflare/splitnn_cifar10 --threads 2 --n_clients 2
```
