# MPC bootstrap

Improve expert controllers by using a learned policy within the MPC expansion.

## Requirements

* MuJoCo 1.31, with the appropriate key available - [MuJoCo downloads](https://www.roboti.us/index.html)
* Python 3.5
* Python packages, mentioned below.
* `xvfb` - only necessary if on server to generate images, else just remove `./fake-display.sh` calls below when re-creating the report.
* System dependencies for `gym` - [gym README](https://github.com/openai/gym/blob/master/README.rst)
* System dependencies for `universe` - [universe README](https://github.com/openai/universe/blob/master/README.rst)

Example installation with GPU (for CPU, modify `tensorflow-gpu` to `tensorflow`)

    conda create -y -n gpu-tfgpu-py35 python=3.5
    conda install -y numpy
    conda install -y scipy
    pip install tensorflow-gpu
    conda install -y -c anaconda pandas 
    conda install -y -c anaconda seaborn 
    pip install gym[all] # make sure to install system deps first!
    pip install universe # make sure to install system deps first!

## Recreating the Report

    # extra stuff
    # --dagger is very bad, TODO try plato instead
    
    python main.py --onpol_iters 20 --exp_name rand-mpc --exp_name randmpc --seed 5
    python main.py --onpol_iters 20 --agent mpcmf --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate "1e-3" --con_batch_size 512 --exp_name uniflmpc --seed 5 --time --explore_std 0 
    python main.py --onpol_iters 20 --agent mpcmf --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate "1e-3" --con_batch_size 512 --exp_name norm0.01lmpc --seed 5 --time --explore_std 0.01
    python main.py --onpol_iters 20 --agent mpcmf --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate "1e-3" --con_batch_size 512 --exp_name norm0.10lmpc --seed 5 --time --explore_std 0.1
    python main.py --onpol_iters 20 --agent mpcmf --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate "1e-3" --con_batch_size 512 --exp_name norm1.00lmpc --seed 5 --time --explore_std 1.00

    function dn {
    echo "data/$1_HalfCheetah-v1"
    }
    ./fake-display.sh python plot.py \
    $(dn randmpc) $(dn uniflmpc) $(dn norm0.01lmpc) $(dn norm0.10lmpc) $(dn norm1.00lmpc) \
    --value AverageReturn --outprefix lmpc- \
    --legend "random sampling MPC" "learned MPC, Unif" "learned MPC,  N(pi, 0.01w)" "learned MPC, N(pi, 0.10w)" "learned MPC, N(pi, 1.00w)"

    python main.py --onpol_iters 30 --agent mpcmf --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate "1e-3" --con_batch_size 512 --exp_name mpcmf-unif --seed 1 5 10 15 20 --time
    python main.py --onpol_iters 30 --agent mpcmf --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate "1e-3" --con_batch_size 512 --exp_name mpcmf-norm --seed 1 5 10 15 20 --time --explore_std 1.00
    python main.py --onpol_iters 30 --agent mpc --exp_name mpc-long --seed 1 5 10 15 20 --time
    function dn {
    echo "data/$1_HalfCheetah-v1"
    }
    ./fake-display.sh python plot.py \
        $(dn mpc-long) $(dn mpcmf-unif) $(dn mpcmf-norm) \
        --value AverageReturn --outprefix long- \
        --legend "random MPC" "learned MPC, Unif" "learned MPC, N(pi, 1w)"

    python main.py --onpol_iters 30 --agent mpcmf --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate "1e-3" --con_batch_size 512 --exp_name mpcmf-unif-hard --seed 1 5 10 15 20 --time --hard_cost
    python main.py --onpol_iters 30 --agent mpcmf --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate "1e-3" --con_batch_size 512 --exp_name mpcmf-norm-hard --seed 1 5 10 15 20 --time --explore_std 1.00 --hard_cost
    python main.py --onpol_iters 30 --agent mpc --exp_name mpc-long-hard --seed 1 5 10 15 20 --time --hard_cost
    function dn {
    echo "data/$1_HalfCheetah-v1"
    }
    ./fake-display.sh python plot.py \
        $(dn mpc-long-hard) $(dn mpcmf-unif-hard) $(dn mpcmf-norm-hard) \
        --value AverageReturn --outprefix hard- \
        --legend "random MPC" "learned MPC, Unif" "learned MPC, N(pi, 1w)"

    pdflatex report.tex
    pdflatex report.tex
