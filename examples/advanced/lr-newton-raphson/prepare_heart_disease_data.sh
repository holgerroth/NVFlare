#! /bin/bash
## ---------------------------------------------------------------------------
##
## File: prepare_heart_disease_data.sh for Newton Raphson
##
## Created by Zhijin Li
## E-mail:   <zhijinl@nvidia.com>
##
## Started on  Fri Apr 12 17:43:39 2024 Zhijin Li
## Last update Sun Apr 14 23:18:03 2024 Zhijin Li
## ---------------------------------------------------------------------------


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATA_DIR=/tmp/flare/dataset/heart_disease_data

# Install dependencies
#pip install wget
FLAMBY_INSTALL_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")
# git clone https://github.com/owkin/FLamby.git && cd FLamby && pip install -e .

# Download data using FLamby
mkdir -p ${DATA_DIR}
python3 ${FLAMBY_INSTALL_DIR}/flamby/datasets/fed_heart_disease/dataset_creation_scripts/download.py --output-folder ${DATA_DIR}

# Convert data to numpy files
${SCRIPT_DIR}/utils/convert_data_to_np.py ${DATA_DIR}
