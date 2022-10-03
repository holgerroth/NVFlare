# FL Hub POC

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
```

Install requirements
```
export NVFLARE_HOME=${PWD}/../..
export HUB_EXAMPLE=${NVFLARE_HOME}/examples/flhub
pip install -r ./virtualenv/requirements.txt
pip install -e ${NVFLARE_HOME}
cd ${NVFLARE_HOME}/integration/monai
pip install -e .
cd ${HUB_EXAMPLE}
export PYTHONPATH=${NVFLARE_HOME}/examples
```

## 2. Create your FL workspaces and start all FL systems

### 2.1 Prepare workspaces
```
cd ./workspaces
for system in t1 t2a t2b; do
  python3 -m nvflare.lighter.provision -p ./${system}_project.yml
  cp -r ./workspace/${system}_project/prod_00 ./${system}_workspace
done
cd ..
```

### 2.2 Start FL systems

#### 2.2.1 T1 system

#### 2.2.2 T2a system

#### 2.2.3 T2b system

### 3. Submit job

