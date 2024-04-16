# Federated Logistic Regression with Second-Order Newton Raphson optimization

This example show-cases how to implement a federated binary
classification via logistic regression with second-order Newton
Raphson optimization.

## Introduction

The optimization problem can be described as follows.

In a binary classification task with logistic regression, the
probability of a data sample $x$ classified as positive is formulated
as:
$$p(x) = \sigma(\beta \cdot x + \beta_{0})$$
where $\sigma(.)$ denotes the sigmoid function. We can incorporate
$\beta_{0}$ and $\beta$ into a single parameter vector \$theta =
\(\beta_{0}, \beta\)$. Let $d$ be the number
of features for each data sample $x$ and let $N$ be the number of data
samples. We then have the matrix version of the above probability
equation:
$$p(X) = X \theta$$
Here $X$ is the matrix of all samples, with shape $N x (d+1)$, having
it's first column filled with value 1 to account for the intercept
$\theta_{0}$.

The goal is to compute parameter vector \$theta$ that maximizes the
below likelihood function:
$$L(\theta) = \prod_{1}^{N} p(x_i)^{y_i} (1 - p(x_i)^{1-y_i})$$

The Newton Raphson method optimizes the likelihood function via
quadratic approximation. Omitting the maths, the theoretical update
formula for parameter vector $\theta$ is:
$$\theta^{n+1} = \theta^{n} - H^{-1}_{\theta} \nabla L(\theta^{n})$$
where
$$\nabla L(\theta^{n}) = X^{T}(y - p(X))$$
is the gradient of the likelihood function, and
$$H^{-1}_{\theta} = -X^{T} D X$$
is the Hessian of the likelihood function, with $D$ a diagonal matrix
where diagonal value at $(i,i)$ is $D(i,i) = p(x_i) (1 - p(x_i))$.

This example uses the (UCI Heart Disease
dataset)[https://archive.ics.uci.edu/dataset/45/heart+disease],
downloaded and processed as described
(here)[https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_heart_disease]. this
dataset contains samples from 4 sites, splitted into training and
testing sets as described below:
|site         | sample split                          |
|-------------|---------------------------------------|
|Cleveland    | train: 199 samples, test: 104 samples |
|hungary      | train: 172 samples, test: 89 samples  |
|Switzerland  | train: 30 samples, test: 16 samples   |
|Long Beach V | train: 85 samples, test: 45 samples   |
The number of features in each sample is 13.

Using `nvflare`, this example was implemented as follows:
- On the server side, a (custom
  workflow)[./job/newton_raphson/app/custom/newton_raphson_workflow.py]
  was implemented, inheriting the
  (`BaseFedAvg`)[https://github.com/NVIDIA/NVFlare/blob/main/nvflare/app_common/workflows/base_fedavg.py]
  class. The custom workflow recieves gradient and Hessian from each
  client computed during local training, then performs aggregation
  using a custom aggregation function, and finally updates the global
  model based on the theoretical update formula. The implementation of
  server side workflow was based on the new recommanded **Workflow
  Controller**
  (`WFController`)[https://github.com/NVIDIA/NVFlare/blob/main/nvflare/app_common/workflows/wf_controller.py],
  which decouples communication logic from workflow logic.
- On the client side, the implementation was based on the (`Client
  API`)[https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type.html#client-api]. This
  allows user to add minimum `nvflare`-specific codes to turn a
  typical centralized training script to a federated client side
  local training script. During local training, each client receives a
  copy of the global model, sent by the server. Then each client
  computes it's gradient and Hessian based on local training data,
  using their respective theoretical formula described above, and send
  the computed results to server for aggregation. Each client site
  corresponds to a site listed in the data table above.
- The global model is merely a numpy array representing parameter
  vector $\theta$. All gradients and Hessians are computed and
  aggregated on-the-fly.
- Before each round of local training, accuracy and precision were
  measured on local test data of each client site, and then streamed
  to the server, saved in tensorboard readble format.

A (centralized training script)[./train_centralized.py] is also
provided, which allows for comparing the federated Newton Raphson
optimization versus the centralized version. In the centralized
version, training data samples from all 4 sites were concatenated into a single
matrix, used to optimize the model parameters. The the optimized model
was then tested separately on testing data samples of the 4
sites. Accuracy and precision were measured as  metrics.

## Set Up Environment & Install Dependencies

Create & activate a virtual environment
```
virtualenv -p python3 ./venv
source ./venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```
This will install main dev branch of `NVFlare` and `FLamby`

## Download and prepare data

Execute the following script
```
bash ./prepare_heart_disease_data.sh
```
This will download the heart disease dataset under
`/tmp/flare/dataset/heart_disease_data/`

## Launch Centralized Training

Launch the following script:
```
./train_centralized.py --solver custom
```

Two implementations of logistic regression are provided in the
centralized training script, which can be specified by the `--solver`
argument:
- One is using `sklearn.LogisticRegression` with `newton-cholesky`
  solver
- the other one is manually implemented using the theoretical update
  formulas described above.

Both implementations were tested to converge in 4 iterations and to
give the same result.

Example output:
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
