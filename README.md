# MPC bootstrap

Improve expert controllers by using a learned policy within the MPC expansion.

## Requirements

Here are setup-specific requirements that you really, really have to do yourself:

* MuJoCo 1.31, with the appropriate key available - [MuJoCo downloads](https://www.roboti.us/index.html)
* Python 3.5 (`scripts/` assume this is the `python` and `pip` in `PATH`)

Other system dependencies:

* System dependencies for `gym` - [gym README](https://github.com/openai/gym/blob/master/README.rst)
* System dependencies for `universe` - [universe README](https://github.com/openai/universe/blob/master/README.rst)

Example conda installation with GPU (for CPU, modify `tensorflow-gpu` to `tensorflow`)

    conda create -y -n gpu-tfgpu-py35 python=3.5
    source activate gpu-tfgpu-py35
    conda install -y numpy
    conda install -y scipy
    pip install tensorflow-gpu
    conda install -y -c anaconda pandas
    conda install -y -c anaconda seaborn
    pip install gym[all] # make sure to install system deps first!
    pip install universe # make sure to install system deps first!

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

    # Figure 1, easy cost
    python main.py --onpol_iters 30 --agent mpc --exp_name mpc-easy \
        --seed 1 5 10 15 20 --time
    python main.py --onpol_iters 30 --agent bootstrap --exp_name bootstrap-easy \
        --seed 1 5 10 15 20 --time \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --explore_std 0
    python main.py --onpol_iters 30 --agent bootstrap --exp_name normboot-easy \
        --seed 1 5 10 15 20 --time \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --explore_std 1
    python plot.py \
        "../data/mpc-easy_HalfCheetah-v1:AverageReturn:MPC" \
        "../data/bootstrap-easy_HalfCheetah-v1:AverageReturn:uniform BMPC" \
        "../data/normboot-easy_HalfCheetah-v1:AverageReturn:normal BMPC" \
        --outfile ../report/easy-AverageReturn.pdf --yaxis "average rollout return"

    # Figure 2, hard cost
    python main.py --onpol_iters 30 --agent mpc --exp_name mpc-hard \
        --seed 1 5 10 15 20 --time --hard_cost
    python main.py --onpol_iters 30 --agent bootstrap --exp_name bootstrap-hard \
        --seed 1 5 10 15 20 --time --hard_cost \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --explore_std 0 --hard_cost
    python main.py --onpol_iters 30 --agent bootstrap --exp_name normboot-hard \
        --seed 1 5 10 15 20 --time --hard_cost \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --explore_std 1 --hard_cost
    python plot.py \
        "../data/mpc-hard_HalfCheetah-v1:AverageReturn:MPC" \
        "../data/bootstrap-hard_HalfCheetah-v1:AverageReturn:uniform BMPC" \
        "../data/normboot-hard_HalfCheetah-v1:AverageReturn:normal BMPC" \
        --outfile ../report/hard-AverageReturn.pdf --yaxis "average rollout return"

    # Figure 3, dagger attempt
    python main.py --onpol_iters 30 --agent dagger --exp_name norm-dag-hard \
        --seed 1 5 10 15 20 --time  \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --explore_std 1 --hard_cost
    python plot.py \
        "../data/normboot-hard_HalfCheetah-v1:AverageReturn:normal BMPC" \
        "../data/norm-dag-hard_HalfCheetah-v1:AverageReturn:dagger BMPC" \
        --outfile ../report/dag-AverageReturn.pdf --yaxis "average rollout return"

    cd ../report
    pdflatex report.tex
    pdflatex report.tex
    cd ..

## Adding MuJoCo key to CI securely

Just use the [manual encryption instructions](https://docs.travis-ci.com/user/encrypting-files/#Manual-Encryption) here. `.travis.yml` is already configured to securely unencrypt mjkey.txt.gpg.
