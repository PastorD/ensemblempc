# Ensemble Model Predictive Control (EnMPC)
Python library for simulating dynamics and ensemble model predictive control.

The code in this repository was prepared to implement the methodology described in 

1. C. Folkestad, D. Pastor, J. Burdick, "Ensemble Model Predictive Control", in *Proc. Conf on Decision and Control Control*, (submitted) 2020 

The simulation framework of this repository is adapted from the [Learning and Control Core Library](https://github.com/learning-and-control/core).

## Setup using virtual env (outdated)
Set up virtual environment 
```
python3 -m venv .venv
```
Activate virtual environment
```
source .venv/bin/activate
```
Upgrade package installer for Python
```
pip install --upgrade pip
```
Install requirements
```
pip3 install -r requirements.txt
```

## Setup using conda (use at least Python 3.7)
Create conda environment
```bash
conda create --name ensemblempc
conda activate ensemblempc
```
```install pacakges
conda install matplotlib numpy pyqtgraph
pip install torch cvxpy % not available in conda
```

## Running the code
To run the code, run one of the examples in 
```
core/examples
```
Run the example scripts as a module with the root folder of repository as the working directory. For example, in a Python 3 environment run
```
python -m core.examples.1d_drone_landing
```

To visualize use
```
python -m core.examples.1d_pyqtgraph
```
