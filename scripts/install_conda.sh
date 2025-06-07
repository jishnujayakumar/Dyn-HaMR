#!/usr/bin/env bash
set -e

export CONDA_ENV_NAME=jp_dynhamr

conda create -n $CONDA_ENV_NAME python=3.10 -y

conda activate $CONDA_ENV_NAME

# install pytorch using pip, update with appropriate cuda drivers if necessary
# pip install torch torchvision
# uncomment if pip installation isn't working
conda install pytorch torchvision pytorch-cuda -c pytorch -c nvidia -y

# install pytorch scatter using pip, update with appropriate cuda drivers if necessary
# pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html --use-pep517

# uncomment if pip installation isn't working
conda install pytorch-scatter -c pyg -y

# install remaining requirements
pip install -r requirements.txt

# install source
pip install -e .


# install DROID-SLAM/DPVO
cd third-party/DROID-SLAM
python setup.py install
cd ../..

# install HaMeR
cd third-party/HaMeR
pip install -e .[all]
pip install -v -e third-party/ViTPose
