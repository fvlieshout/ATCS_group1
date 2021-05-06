#!/bin/bash

module purge
module load 2019
module load Anaconda3/2018.12

conda env remove -y -n gnn-env
conda create -y -n gnn-env python=3.7.5
source activate gnn-env

pip3 install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
pip3 install torch-geometric

pip3 install -r requirements.txt
