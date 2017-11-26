#! /usr/bin/env bash
# Runs overoptimization experiments for MPC on walker2d.
# Should be run from repo root as follows:
#
# ./scripts/walker-mpc-overopt.sh
#
# Data from running this experiment can be found in report/walker-mpc-overopt.tgz

# Warning: don't actually run all these serially
for i in 100 1000 10000 ; do 
    python mpc_bootstrap/main_mpc.py --verbose --exp_name mpc-$i --onpol_iters 50 --seed 1 5 10 15 --mpc_simulated_paths $i --env_name walker2d --mpc_horizon 10
    python mpc_bootstrap/main_bootstrapped_mpc.py zero --verbose --exp_name zero-$i --onpol_iters 50 --seed 1 5 10 15 --mpc_simulated_paths $i --env_name walker2d --mpc_horizon 10
done


for i in 100 1000 10000 ; do 
    python mpc_bootstrap/plot.py "data/mpc-${i}_walker2d:avg-return:${i} Uniform RS MPC" "data/zero-${i}_walker2d:avg-return:${i} 0 RS MPC" --outfile "report/mpc-return-walker2d-${i}.pdf" --yaxis "10-episode avg return MA(5)" --smoothing 5
done
