#! /usr/bin/env bash
# Runs overoptimization experiments for MPC on ant.
# Should be run from repo root as follows:
#
# ./scripts/ant-mpc-overopt.sh
#
# Data from running this experiment can be found in report/ant-mpc-overopt.tgz

# Warning: don't actually run all these serially
for i in 100 1000 10000 ; do 
    python mpc_bootstrap/main_mpc.py --verbose --exp_name mpc-$i --onpol_iters 50 --seed 1 5 10 15 --mpc_simulated_paths $i --env_name ant --mpc_horizon 10
    python mpc_bootstrap/main_bootstrapped_mpc.py zero --verbose --exp_name zero-$i --onpol_iters 50 --seed 1 5 10 15 --mpc_simulated_paths $i --env_name ant --mpc_horizon 10
done

for i in 100 1000 10000 ; do 
    python mpc_bootstrap/plot.py "data/mpc-${i}_ant:avg-return:${i} Uniform RS MPC" "data/zero-${i}_ant:avg-return:${i} 0 RS MPC" --outfile "report/mpc-return-ant-${i}.pdf" --yaxis "10-episode avg return MA(5)" --smoothing 5
done
