# CMPC: Constrianed Model Predictive Control [![Build Status](https://travis-ci.com/vlad17/cmpc.svg?token=xAqzxKFpxN3pG4om3z4n&branch=master)](https://travis-ci.com/vlad17/cmpc)

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
| `fake-display.sh` | create a dummy X11 display (to render on a server) |

## Recreating the Report

The report and all scripts related to its creation in `report/` are currently out-dated with respect to the current code. They correspond to a report written up when the code base was at commit `cda6b4de42d`.

## Adding MuJoCo key to CI securely

Just use the [manual encryption instructions](https://docs.travis-ci.com/user/encrypting-files/#Manual-Encryption) here. `.travis.yml` is already configured to securely unencrypt mjkey.txt.gpg.
