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
export PYTHONPATH=${NVFLARE_HOME}:${NVFLARE_HOME}/examples:${NVFLARE_HOME}/integration/monai

pip install -r ./virtualenv/requirements.txt
pip install -r ${NVFLARE_HOME}/requirements-min.txt
```

## 2. Create your FL workspaces and start all FL systems

### 2.1 Prepare workspaces
```
cd ./workspaces
for SYSTEM in "t1" "t2a" "t2b"; do
  nvflare provision -p ./${SYSTEM}_project.yml
  cp -r ./workspace/${SYSTEM}_project/prod_00 ./${SYSTEM}_workspace
done
cd ..
```

### 2.2 Adjust hub configs

Modify hub clients:
```
cp -r ./config/site_a/* ./workspaces/t1_workspace/t1_client_a/local/.
cp -r ./config/site_b/* ./workspaces/t1_workspace/t1_client_b/local/.
```

Modify t2 server configs TODO!
```
for SYSTEM in "t2a" "t2b"; do
    export T2_SERVER_LOCAL=./workspaces/${SYSTEM}_workspace/localhost/local
    mv ${T2_SERVER_LOCAL}/resources.json.default ${T2_SERVER_LOCAL}/resources.json
    sed -i "s|/tmp/nvflare/snapshot-storage|/tmp/nvflare/snapshot-storage_${SYSTEM}|g" ${T2_SERVER_LOCAL}/resources.json
    sed -i "s|/tmp/nvflare/jobs-storage|/tmp/flare/jobs/${SYSTEM}|g" ${T2_SERVER_LOCAL}/resources.json
done 
```

### 2.3 Start FL systems

```
# T1 system
./workspaces/t1_workspace/localhost/startup/start.sh
./workspaces/t1_workspace/t1_client_a/startup/start.sh
./workspaces/t1_workspace/t1_client_b/startup/start.sh

# T2a system
./workspaces/t2a_workspace/localhost/startup/start.sh
./workspaces/t2a_workspace/site_a-1/startup/start.sh

# T2b system
./workspaces/t2b_workspace/localhost/startup/start.sh
./workspaces/t2b_workspace/site_b-1/startup/start.sh
```

### 3. Submit job

Open admin for hub. Provide admin username: `admin@nvflare.com`
```
./workspaces/t1_workspace/admin@nvflare.com/startup/fl_admin.sh
```

Submit job in console. Replace `[HUB_EXAMPLE]` with your local path of this folder
```
submit_job [HUB_EXAMPLE]/job
```

For a simple example, run
```
submit_job /home/hroth/Code2/nvflare/flhub_hroth/examples/hello-numpy-sag
```

For a MonaiAlgo example, run
```
submit_job /home/hroth/Code2/nvflare/flhub_hroth/examples/flhub/job
```

### 4. (Optional) Clean-up

Shutdown all FL systems
```
./workspaces/t1_workspace/t1_client_a/startup/stop_fl.sh <<< "y"
./workspaces/t1_workspace/t1_client_b/startup/stop_fl.sh <<< "y"
./workspaces/t1_workspace/localhost/startup/stop_fl.sh <<< "y"

./workspaces/t2a_workspace/site_a-1/startup/stop_fl.sh <<< "y"
./workspaces/t2a_workspace/localhost/startup/stop_fl.sh <<< "y"

./workspaces/t2b_workspace/site_b-1/startup/stop_fl.sh <<< "y"
./workspaces/t2b_workspace/localhost/startup/stop_fl.sh <<< "y"
```

Delete workspaces & temp folders
```
rm -r workspaces/workspace
rm -r workspaces/*_workspace

rm -r /tmp/nvflare
rm -r /tmp/flare
```