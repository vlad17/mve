# MVE: Model-based Value Estimation [![Build Status](https://travis-ci.com/vlad17/mve.svg?token=xAqzxKFpxN3pG4om3z4n&branch=master)](https://travis-ci.com/vlad17/mve)

Using short-horizon nonlinear dynamics for on-policy simulation to improve value estimation. See the [paper](https://arxiv.org/abs/1803.00101) for algorithmic details.

## Requirements

Here are setup-specific requirements that you really, really have to do yourself:

* MuJoCo 1.50, with the appropriate key available - [MuJoCo downloads](https://www.roboti.us/index.html). On Linux, you can run `~/install-mujoco.sh` with the key `mjkey.txt` in the CWD.
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
    conda create -y -n cpu-py3.5 python=3.5
    ./scripts/install-mujoco.sh
    ./scripts/ubuntu-install.sh
    
    # Lazy version (GPU)
    conda create -y -n gpu-py3.5 python=3.5
    ./scripts/install-mujoco.sh
    sed -i 's/tensorflow/tensorflow-gpu/' requirements.txt
    ./scripts/ubuntu-install.sh
    
## Scripts

All scripts are available in `scripts/`, and should be run from the repo root.

| script | purpose |
| ------ | ------- |
| `lint.sh` | invokes `pylint` with the appropriate flags for this repo |
| `ubuntu-install.sh` | installs all deps except MuJoCo/python on Ubuntu 14.04 or Ubuntu 16.04 |
| `install-mujoco.sh` | add MuJoCo 1.50 on linux, assuming a key |
| `tests.sh` | runs tests |
| `fake-display.sh` | create a dummy X11 display (to render on a server) |
| `launch-ray-aws.sh` | launch an AWS ray cluster at the current branch |
| `teardown-ray-aws.sh` | tear down a cluster |

## Running experiments

To run experiments locally, use `main_ray.py` (note the resources requirements specified here are only used for scheduling trials; the actual processes are free to create as many threads as they want if you'd like to oversubscribe the machines by setting `tf_parallelism` within the YAML config to larger than the number of guaranteed CPUs). To multiplex the GPUs locally, set `self_host` to the total number of virtual GPUs.

    python mve/main_ray.py --experiment_name hc0 --config experiments/hc0.yaml --ncpus 1 --ngpus 0 --self_host 1

Experiments can also be run in a distributed manner by connecting to a live ray cluster by changing `--self_host` to `--port RAY_REDIS_PORT` above. Multiple experiments can be run by the same driver. See `python mve/main_ray.py --help` for details.

## Parallelism

Multiple components of this code run in parallel.

* Evaluation on environments is parallelized by multiple processes or threads (which one depends on the environment), and uses `--env_parallelism` workers. I have found that peak performance is reached when there is some batching; i.e., the number of workers is less than half the number of environments used for evaluation (default 8).
* If running on CPUs, TensorFlow internal thread pool parallelism (for each of intra-operation and inter-operation thread pools) is set by the value of `--tf_parallelism` (default `nproc`).
* If running on GPUs, then TensorFlow GPU count used is automatically set to those GPUs specified by the environment variable `CUDA_VISIBLE_DEVICES`, which if left empty uses the first GPU available on the machine. There is currently no support for actually using multiple GPUs in a single experiment.
* The `OMP_NUM_THREADS` is overriden; there's no need to set it.

## Adding MuJoCo key to CI securely

Just use the [manual encryption instructions](https://docs.travis-ci.com/user/encrypting-files/#Manual-Encryption) here. `.travis.yml` is already configured to securely unencrypt mjkey.txt.gpg.

## Adding New Fully Observable Environment

1. Create a new `FullyObservable<env>` under `mve/envs`. See `FullyObservableHalfCheetah.py` for a link to the Open AI Gym commit that contains the source code you should adapt.
2. Expose the new environment by importing it in `mve/envs/__init__.py`
3. Make it accessible through `mve/env_info.py`'s `_env_class` function.

Make sure to test that your environment works and amend the tests in `scripts/tests.sh` to include a check that it runs correctly.

