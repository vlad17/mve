# MPC bootstrap

Improve expert controllers by using a learned policy within the MPC expansion.

## Requirements

* MuJoCo 1.31, with the appropriate key available.
* Python 3.5
* Python packages `gym[all] numpy matplotlib seaborn pandas`, tensorflow installed separately.
* `xvfb` - only necessary if on server, else just remove `./fake-display.sh` below.

TODO: below, old stuff from my hw4 report.

## Recreating the Report

    /usr/bin/time --output iter-time.tex --format "%e" \
        python main.py | tee iter-output.tex
    echo "$(tail -n +2 iter-output.tex)" > iter-output.tex # rm logging message
    echo "$(tail -n +2 iter-output.tex)" > iter-output.tex # rm logging message    
    /usr/bin/time --output bptt-time.tex --format "%e" \
        python main.py \
        --agent bptt \
        --mpc_horizon 50 --con_epochs 1 \
        --con_depth 2 --con_width 32 \
        --con_learning_rate="1e-5"
    python main.py \
        --agent random --onpol_iters 15 --no_aggregate \
        --exp_name random-no-agg --seed 1 5 10 15 20
    python main.py \
        --agent random --onpol_iters 15 \
        --exp_name random-agg --seed 1 5 10 15 20
    python main.py --onpol_iters 15 --no_aggregate \
        --exp_name mpc-no-agg --seed 1 5 10 15 20
    python main.py --onpol_iters 15 --exp_name mpc-agg \
        --seed 1 5 10 15 20
    python main.py --onpol_iters 15 --no_delta_norm \
        --exp_name mpc-no-delta --seed 1 5 10 15 20
    python main.py --onpol_iters 15 --agent bptt \
        --mpc_horizon 50 --con_epochs 1 \
        --con_depth 2 --con_width 32 \
        --seed 1 5 10 15 20 --con_learning_rate="1e-5" \
        --exp_name bptt
    python main.py --onpol_iters 15 --agent mpcmf \
        --con_depth 5 --con_width 32 --con_epochs 100 \
        --con_learning_rate "1e-3" --con_batch_size 512 \
        --exp_name mpcmf --seed 1 5 10 15 20
        
    function dn {
        echo "data/$1_HalfCheetah-v1"
    }
    ./fake-display.sh python plot.py \
        $(dn random-no-agg) $(dn random-agg) \
        --value DynamicsMSE \
        --legend "no aggregation" "yes aggregation"
    ./fake-display.sh python plot.py \
        $(dn mpc-no-agg) $(dn mpc-agg) \
        --value AverageReturn \
        --legend "no aggregation" "yes aggregation"
    ./fake-display.sh python plot.py \
        $(dn mpc-no-agg) $(dn mpc-agg) \
        --value AverageCost \
        --legend "no aggregation" "yes aggregation"
    ./fake-display.sh python plot.py \
        $(dn mpc-agg) $(dn mpc-no-delta) \
        --value AverageReturn --outprefix delta- \
        --legend "delta normalization" "no delta normalization"
    ./fake-display.sh python plot.py \
        $(dn bptt) \
        --value AverageReturn --outprefix bptt- \
        --legend "BPTT"
    ./fake-display.sh python plot.py \
        $(dn mpc-agg) $(dn mpcmf) \
        --value AverageReturn --outprefix mpcmf- \
        --legend "random sampling MPC" "learned MPC"

    pdflatex report.tex
    pdflatex report.tex

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

    # friendlier settings for hard cost picked by anusha
    python main.py --onpol_iters 30 --agent mpc --exp_name mpc2 --seed 1 5 10 15 20 --time --hard_cost --onpol_paths 10 --random_paths 10 --mpc_horizon 20 --dyn_depth 1
    python main.py --onpol_iters 30 --agent mpcmf --exp_name unif2 --seed 1 5 10 15 20 --time --hard_cost --onpol_paths 10 --random_paths 10 --mpc_horizon 20 --dyn_depth 1 --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate "1e-3" --con_batch_size 512
    python main.py --onpol_iters 30 --agent mpcmf --exp_name norm2 --seed 1 5 10 15 20 --time --hard_cost --onpol_paths 10 --random_paths 10 --mpc_horizon 20 --dyn_depth 1 --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate "1e-3" --con_batch_size 512 --explore_std 1.00
    function dn {
    echo "data/$1_HalfCheetah-v1"
    }
    ./fake-display.sh python plot.py \
        $(dn mpc-long-hard) $(dn mpcmf-unif-hard) $(dn mpcmf-norm-hard) \
        --value AverageReturn --outprefix hard- \
        --legend "random MPC" "learned MPC, Unif" "learned MPC, N(pi, 1w)"
