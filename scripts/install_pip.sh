#!/usr/bin/env bash
set -e

echo "Creating virtual environment"
python -m venv .dynhamr
echo "Activating virtual environment"
export CUDA_HOME=/usr/local/cuda-11.3/
source $PWD/.dynhamr/bin/activate

# install pytorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# torch-scatter
$PWD/.dynhamr/bin/pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

# install source
$PWD/.dynhamr/bin/pip install -e .

# install remaining requirements
$PWD/.dynhamr/bin/pip install -r requirements.txt

# install DROID-SLAM/DPVO
cd third-party/DROID-SLAM
python setup.py install
cd ../..

# install HaMeR
cd third-party/HaMeR
pip install -e .[all]
pip install -v -e third-party/ViTPose
