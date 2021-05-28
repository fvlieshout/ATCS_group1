# UvA ATCS Research project

This project has been conducted during the ATCS course at the University of Amsterdam. It involves the topic: **Transfer
learning with Graph Neural Networks for text classification**.

Authors:\
Rahel Habacker (rahel.habacker@student.uva.nl)\
Andrew Harrison (andrew.harrison@student.uva.nl)\
Mathias Parisot (parisot.mathias@student.uva.nl)\
Mátyás Schubert (matyas.schubert@student.uva.nl)\
Floor van Lieshout (floor.vanlieshout@student.uva.nl)

# Repository and Settings

The codebase is mainly contained in the folders [`models`](models), [`data_prep`](data_prep) as well as
files [`train.py`](train.py) and [`evaluate.py`](evaluate.py).

# Running the Experiments

## Environment Configuration

In order to run train and/or evaluate the models clone this repo from github:

```
git clone https://github.com/fvlieshout/ATCS_group1.git
cd ATCS_group1/
```

All dependencies are provided in the environment files [`env.yaml`](env.yaml) and [`requirements.txt`](requirements.txt).
To properly create an Anaconda environment from these two files run:

```
bash install-gnn-env.sh
```

After that, activate it with

```
conda activate gnn-env
```

## Train the models

The different models can be trained using the [train.py](train.py) script. It provides various arguments for running the
experiments with different configurations. At the end of the training, the best model is tested on a test and validation
set.

```
python train.py
```

The default values are the following:

```
epochs: 50
patience: 10
batch-size: 8
lr-enc: 0.01
lr-cl: -1
w-decay-enc: 2e-3
w-decay-cl: -1
warmup: 500
max-iters: -1

dataset: R8
model: roberta_pretrained_gnn
gnn-layer-name: GraphConv
seed: 1234
cf-hid-dim: 512
checkpoint: None
roberta-model: None
transfer: False
h-search: False
```

## Evaluate the models

Similar to the training, all models can also be evaluated separately using the [evaluate.py](evaluate.py)
script. It also provides arguments for running the experiments with different configurations.

```
python evaluate.py
```

The default values are the following:

```
batch-size: 8
dataset: R8
model: roberta_pretrained_gnn
seed: 1234
checkpoint: None
transfer: False
```