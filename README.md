# CMPC: Constrianed Model Predictive Control [![Build Status](https://travis-ci.com/vlad17/mpc-bootstrap.svg?token=9HHJycCztSrS3mCpqQ9s&branch=master)](https://travis-ci.com/vlad17/mpc-bootstrap)

Account for model inaccuracies in MPC.

## Requirements

Here are setup-specific requirements that you really, really have to do yourself:

* MuJoCo 1.31, with the appropriate key available - [MuJoCo downloads](https://www.roboti.us/index.html)
* Python 3.5 (`scripts/` assume this is the `python` and `pip` in `PATH`)

Other system dependencies:

* System dependencies for `gym` - [gym README](https://github.com/openai/gym/blob/master/README.rst).

Example conda installation:

    # GPU version
    conda create -y -n gpu-py3.5 python=3.5
    source activate gpu-py3.5
    pip install -r <( sed 's/tensorflow/tensorflow-gpu/' requirements.txt )
    # CPU version
    conda create -y -n cpu-py3.5 python=3.5
    source activate cpu-py3.5
    pip install -r requirements.txt
    
## Scripts

All scripts are available in `scripts/`, and should be run from the repo root.

| script | purpose |
| ------ | ------- |
| `lint.sh` | invokes `pylint` with the appropriate flags for this repo |
| `ubuntu-install.sh` | installs all deps except MuJoCo/python on Ubuntu 14.04 or Ubuntu 16.04 |
| `tests.sh` | runs tests |
| `mkparams.py` | generate parameters for grid search |

## Recreating the Report

TODO

## Adding MuJoCo key to CI securely

Just use the [manual encryption instructions](https://docs.travis-ci.com/user/encrypting-files/#Manual-Encryption) here. `.travis.yml` is already configured to securely unencrypt mjkey.txt.gpg.
