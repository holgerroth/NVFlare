# newton-raphson-nvflare

## Set Up Environment & Install Dependencies

Create virtual environment
```
virtualenv -p python3 ./venv
```

Clone the NVFlare repo
```
git clone https://github.com/NVIDIA/NVFlare.git
```

Activate virtual environment
```
source ./venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```
This will install main dev branch of `NVFlare` and `FLamby`

## Download and prepare data

```
bash ./prepare_heart_disease_data.sh
```
This downloads the heart disease dataset under
`/tmp/flare/dataset/heart_disease_data/`

## Centralized training

Centralized logistic regression using either custom implementation
or `sklearn.LogisticRegression` with `newton-cholesky` solver. Both
gitve the same result.

```
./train_centralized.py --solver custom
```

Output:
```
using solver: custom
loading training data.
training data X loaded. shape: (486, 13)
training data y loaded. shape: (486, 1)

site - 1
validation set n_samples:  104
accuracy: 0.75
precision: 0.7115384615384616

site - 2
validation set n_samples:  89
accuracy: 0.7528089887640449
precision: 0.6122448979591837

site - 3
validation set n_samples:  16
accuracy: 0.75
precision: 1.0

site - 4
validation set n_samples:  45
accuracy: 0.6
precision: 0.9047619047619048
```

## Run Federated Newton Raphson in simulator mode
```
nvflare simulator -w ./workspace -n 4 -t 4 job/newton_raphson/
```

Accuracy and precision for each site can be viewed in Tensorboard:
```
tensorboard --logdir=./workspace/simulate_job/tb_events
```
