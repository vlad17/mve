# MPC bootstrap [![Build Status](https://travis-ci.com/vlad17/mpc-bootstrap.svg?token=9HHJycCztSrS3mCpqQ9s&branch=master)](https://travis-ci.com/vlad17/mpc-bootstrap)

Improve expert controllers by using a learned policy within the MPC expansion.

## Requirements

Here are setup-specific requirements that you really, really have to do yourself:

* MuJoCo 1.31, with the appropriate key available - [MuJoCo downloads](https://www.roboti.us/index.html)
* Python 3.5 (`scripts/` assume this is the `python` and `pip` in `PATH`)

Other system dependencies:

* System dependencies for `gym` - [gym README](https://github.com/openai/gym/blob/master/README.rst)

Example conda installation with GPU (for CPU, modify `tensorflow-gpu` to `tensorflow`)

    conda create -y -n gpu-tfgpu-py35 python=3.5
    source activate gpu-tfgpu-py35
    conda install -y numpy
    conda install -y scipy
    pip install tensorflow-gpu
    conda install -y -c anaconda pandas
    conda install -y -c anaconda seaborn
    pip install gym[all] # make sure to install system deps first!
    pip install ray
    pip install cloudpickle==0.4.1

Or, using `requirements.txt`:

    conda create -y -n gpu-tfgpu-py35 python=3.5
    source activate gpu-tfgpu-py35
    pip install -r requirements.txt

## Scripts

All scripts are available in `scripts/`, and should be run from the repo root.

| script | purpose |
| ------ | ------- |
| `lint.sh` | invokes `pylint` with the appropriate flags for this repo |
| `ubuntu-install.sh` | installs all deps except MuJoCo/python on Ubuntu 14.04 or Ubuntu 16.04 |

## Recreating the Report

    ./scripts/easy-cost.sh
    ./scripts/hard-cost.sh
    ./scripts/hard2-cost.sh

    # dagger attempt out-of-date, should re-run later

    cd report
    pdflatex report.tex
    pdflatex report.tex
    cd ..

## Adding MuJoCo key to CI securely

Just use the [manual encryption instructions](https://docs.travis-ci.com/user/encrypting-files/#Manual-Encryption) here. `.travis.yml` is already configured to securely unencrypt mjkey.txt.gpg.
