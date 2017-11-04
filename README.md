# MPC bootstrap

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

    # Commands for figure 1, easy cost
    python mpc_bootstrap/main.py --onpol_iters 15 --agent mpc --exp_name mpc-easy \
        --seed 1 5 10 --time
    python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name bootstrap-easy \
        --seed 1 5 10 --time \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --explore_std 0 --deterministic_learner
    python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name normboot-easy \
        --seed 1 5 10 --time \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --explore_std 1 --deterministic_learner
    python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name stochastic-bmpc-easy \
        --seed 1 5 10 --time \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512
    python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name onlystoch-bmpc-easy \
        --seed 1 5 10 --time \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --no_extra_explore 

    # Figure 2, hard cost (setting 1)
    python mpc_bootstrap/main.py --onpol_iters 15 --agent mpc --exp_name mpc-hard \
        --seed 1 5 10 --time --hard_cost
    python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name bootstrap-hard \
        --seed 1 5 10 --time --hard_cost \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --explore_std 0 --hard_cost --deterministic_learner
    python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name normboot-hard \
        --seed 1 5 10 --time --hard_cost \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --explore_std 1 --hard_cost --deterministic_learner
    python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name stochastic-bmpc-hard \
        --seed 1 5 10 --time \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --hard_cost
    python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name onlystoch-bmpc-hard \
        --seed 1 5 10 --time --hard_cost \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --no_extra_explore 

    # Figure 3, hard cost (setting 2)
    python mpc_bootstrap/main.py --onpol_iters 15 --agent mpc --exp_name mpc-hard2 \
        --seed 1 5 10 --time --hard_cost --mpc_horizon 50 --simulated_paths 200
    python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name bootstrap-hard2 \
        --seed 1 5 10 --time --hard_cost --mpc_horizon 50 --simulated_paths 200 \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --explore_std 0  --deterministic_learner
    python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name normboot-hard2 \
        --seed 1 5 10 --time --hard_cost --mpc_horizon 50 --simulated_paths 200 \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --explore_std 1 --deterministic_learner
    python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name stochastic-bmpc-hard2 \
        --seed 1 5 10 --time --mpc_horizon 50 --simulated_paths 200 \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --hard_cost
    python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name onlystoch-bmpc-hard2 \
        --seed 1 5 10 --time --mpc_horizon 50 --simulated_paths 200 \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --hard_cost --no_extra_explore

    # Figure 4, dagger attempt (no delay)
    python mpc_bootstrap/main.py --onpol_iters 15 --agent dagger --exp_name stochastic-dag-hard \
        --seed 1 5 10 --time  \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --hard_cost
    python mpc_bootstrap/main.py --onpol_iters 15 --agent dagger --exp_name onlystoch-dag-hard \
        --seed 1 5 10 --time  --no_extra_explore \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --hard_cost
    python mpc_bootstrap/main.py --onpol_iters 15 --agent dagger --exp_name bootstrap-dag-hard \
        --seed 1 5 10 --time  --deterministic_learner \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --hard_cost

    # Figure 5, dagger attempt (with delay)
    python mpc_bootstrap/main.py --onpol_iters 15 --agent dagger --exp_name stochastic-dagdelay-hard \
        --seed 1 5 10 --time --delay 5 \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --hard_cost
    python mpc_bootstrap/main.py --onpol_iters 15 --agent dagger --exp_name onlystoch-dagdelay-hard \
        --seed 1 5 10 --time  --no_extra_explore  --delay 5 \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --hard_cost
    python mpc_bootstrap/main.py --onpol_iters 15 --agent dagger --exp_name bootstrap-dagdelay-hard \
        --seed 1 5 10 --time  --deterministic_learner  --delay 5 \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --hard_cost

    # plotting for all figures in order
    python mpc_bootstrap/plot.py \
        "data/mpc-easy_HalfCheetah-v1:AverageReturn:MPC" \
        "data/bootstrap-easy_HalfCheetah-v1:AverageReturn:uniform BMPC" \
        "data/normboot-easy_HalfCheetah-v1:AverageReturn:normal BMPC" \
        "data/stochastic-bmpc-easy_HalfCheetah-v1:AverageReturn:stochastic BMPC" \
        "data/onlystoch-bmpc-easy_HalfCheetah-v1:AverageReturn:stoch,no-exp BMPC" \
        --outfile report/easy-AverageReturn.pdf --yaxis "average rollout return"
    python mpc_bootstrap/plot.py \
        "data/mpc-hard_HalfCheetah-v1:AverageReturn:MPC" \
        "data/bootstrap-hard_HalfCheetah-v1:AverageReturn:uniform BMPC" \
        "data/normboot-hard_HalfCheetah-v1:AverageReturn:normal BMPC" \
        "data/stochastic-bmpc-hard_HalfCheetah-v1:AverageReturn:stochastic BMPC" \
        "data/onlystoch-bmpc-hard_HalfCheetah-v1:AverageReturn:stoch,no-exp BMPC" \
        --outfile report/hard-AverageReturn.pdf --yaxis "average rollout return"
    python mpc_bootstrap/plot.py \
        "data/mpc-hard2_HalfCheetah-v1:AverageReturn:MPC" \
        "data/bootstrap-hard2_HalfCheetah-v1:AverageReturn:uniform BMPC" \
        "data/normboot-hard2_HalfCheetah-v1:AverageReturn:normal BMPC" \
        "data/stochastic-bmpc-hard2_HalfCheetah-v1:AverageReturn:stochastic BMPC" \
        "data/onlystoch-bmpc-hard2_HalfCheetah-v1:AverageReturn:stoch,no-exp BMPC" \
        --outfile report/hard2-AverageReturn.pdf --yaxis "average rollout return"
    python mpc_bootstrap/plot.py \
        "data/stochastic-bmpc-hard_HalfCheetah-v1:AverageReturn:stochastic BMPC" \
        "data/stochastic-dag-hard_HalfCheetah-v1:AverageReturn:dagger stochastic BMPC" \
        "data/onlystoch-dag-hard_HalfCheetah-v1:AverageReturn:dagger stoch,no-exp BMPC" \
        "data/bootstrap-dag-hard_HalfCheetah-v1:AverageReturn:dagger BMPC" \
        --outfile report/dag-AverageReturn.pdf --yaxis "average rollout return"
    python mpc_bootstrap/plot.py \
        "data/stochastic-bmpc-hard_HalfCheetah-v1:AverageReturn:stochastic BMPC" \
        "data/stochastic-dagdelay-hard_HalfCheetah-v1:AverageReturn:dagger stochastic BMPC" \
        "data/onlystoch-dagdelay-hard_HalfCheetah-v1:AverageReturn:dagger stoch,no-exp BMPC" \
        "data/bootstrap-dagdelay-hard_HalfCheetah-v1:AverageReturn:dagger BMPC" \
        --outfile report/dag-AverageReturn.pdf --yaxis "average rollout return"

    cd report
    pdflatex report.tex
    pdflatex report.tex
    cd ..

## Adding MuJoCo key to CI securely

Just use the [manual encryption instructions](https://docs.travis-ci.com/user/encrypting-files/#Manual-Encryption) here. `.travis.yml` is already configured to securely unencrypt mjkey.txt.gpg.
