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

    cd mpc_bootstrap
    
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
    ../fake-display.sh python plot.py \
        data/mpc-easy_HalfCheetah-v1 data/bootstrap-easy_HalfCheetah-v1 \
        data/normboot-easy_HalfCheetah-v1 --value AverageReturn \
        --outprefix easy- --legend "MPC" "uniform BMPC" "normal BMPC"
    mv easy-AverageReturn.pdf ../report
        
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
    ../fake-display.sh python plot.py \
        data/mpc-hard_HalfCheetah-v1 data/bootstrap-hard_HalfCheetah-v1 \
        data/normboot-hard_HalfCheetah-v1 --value AverageReturn \
        --outprefix hard- --legend "MPC" "uniform BMPC" "normal BMPC"
    mv hard-AverageReturn.pdf ../report

    # Figure 3, dagger attempt
    python main.py --onpol_iters 30 --agent dagger --exp_name norm-dag-hard \
        --seed 1 5 10 15 20 --time  \
        --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 \
        --con_batch_size 512 --explore_std 1 --hard_cost
    ../fake-display.sh python plot.py \
        data/norm-dag-hard_HalfCheetah-v1 data/normboot-hard_HalfCheetah-v1 \
        --value AverageReturn \
        --outprefix dag- --legend "dagger BMPC" "normal BMPC"
    mv dag-AverageReturn.pdf ../report
    
    cd ../report
    pdflatex report.tex
    pdflatex report.tex
    cd .. 
