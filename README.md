# CMPC: Constrianed Model Predictive Control [![Build Status](https://travis-ci.com/vlad17/cmpc.svg?token=xAqzxKFpxN3pG4om3z4n&branch=master)](https://travis-ci.com/vlad17/cmpc)

Account for model inaccuracies in MPC.

## Requirements

Here are setup-specific requirements that you really, really have to do yourself:

* MuJoCo 1.50, with the appropriate key available - [MuJoCo downloads](https://www.roboti.us/index.html)
* Both MuJoCo installations are expected in `~/.mujoco` or the environment variable `MUJOCO_DIRECTORY`, if defined.
* Python 3.5 (`scripts/` assume this is the `python` and `pip` in `PATH`)
* If you get any error messages relating to `glfw3`, then reinstall everything in a shell where the following environment variables are set (and for good measure in the shell where you're launching experiments):

```
    export LD_LIBRARY_PATH=~/.mujoco/mjpro150/bin
    export LIBRARY_PATH=~/.mujoco/mjpro150/bin
```

Other system dependencies:

* System dependencies for `gym` - [gym README](https://github.com/openai/gym/blob/master/README.rst)
* System dependencies for `gym2` - [gym2 README](https://github.com/vlad17/gym2/blob/master/README.md)

Example installation:

    # GPU version
    # ... you install system packages here
    conda create -y -n gpu-py3.5 python=3.5
    source activate gpu-py3.5
    pip install -r <( sed 's/tensorflow/tensorflow-gpu/' requirements.txt )
    
    # CPU version
    # ... you install system packages here
    conda create -y -n cpu-py3.5 python=3.5
    source activate cpu-py3.5
    pip install -r requirements.txt
    
    # Lazy version (defaults to CPU)
    ./scripts/ubuntu-install.sh
    
    # Lazy version (GPU)
    sed -i 's/tensorflow/tensorflow-gpu/' requirements.txt
    ./scripts/ubuntu-install.sh
    
## Scripts

All scripts are available in `scripts/`, and should be run from the repo root.

| script | purpose |
| ------ | ------- |
| `lint.sh` | invokes `pylint` with the appropriate flags for this repo |
| `ubuntu-install.sh` | installs all deps except MuJoCo/python on Ubuntu 14.04 or Ubuntu 16.04 |
| `tests.sh` | runs tests |
| `fake-display.sh` | create a dummy X11 display (to render on a server) |
| `launch-ray-aws.sh` | launch an AWS ray cluster at the current branch |
| `teardown-ray-aws.sh` | tear down a cluster |

## Adding MuJoCo key to CI securely

Just use the [manual encryption instructions](https://docs.travis-ci.com/user/encrypting-files/#Manual-Encryption) here. `.travis.yml` is already configured to securely unencrypt mjkey.txt.gpg.

## Adding New Fully Observable Environment

1. Create a new `FullyObservable<env>` under `cmpc/envs`. See `FullyObservableHalfCheetah.py` for a link to the Open AI Gym commit that contains the source code you should adapt.
2. Expose the new environment by importing it in `cmpc/envs/__init__.py`
3. Make it accessible through `cmpc/env_info.py`'s `_env_class` function.

Make sure to test that your environment works and amend the tests in `scripts/tests.sh`.

## Outdated things

The report and all scripts related to its creation in `report/` are currently out-dated with respect to the current code. They correspond to a report written up when the code base was at commit `cda6b4de42d`.

The poster in `poster/` is also out-of-date.
