#! /usr/bin/env bash
# Runs overoptimization experiments for MPC on ant.
# Should be run from repo root as follows:
#
# ./scripts/mpc-overopt.sh
#
# Data from running this experiment can be found in report/mpc-overopt.tgz

# Warning: don't actually run all these serially
for i in hc-hard ant walker2d ; do
    python mpc_bootstrap/main_mpc.py --verbose --exp_name mpc --onpol_iters 50 --seed 3 5 8 --env_name $i
    python mpc_bootstrap/main_bootstrapped_mpc.py zero --verbose --exp_name zero --onpol_iters 50 --seed 3 5 8 --env_name $i
    python mpc_bootstrap/main_mpc.py --verbose --exp_name mpc-long --onpol_iters 50 --seed 3 5 8 --env_name $i --mpc_horizon 30 --mpc_simulated_paths 3000
    python mpc_bootstrap/main_bootstrapped_mpc.py zero --verbose --exp_name zero-long --onpol_iters 50 --seed 3 5 8 --env_name $i  --mpc_horizon 30 --mpc_simulated_paths 3000
done

for i in hc-hard ant walker2d ; do 
    python mpc_bootstrap/plot.py "data/mpc_${i}:avg-return:Uniform RS MPC" "data/zero_${i}:avg-return:0 RS MPC" --outfile "report/return-${i}.pdf" --yaxis "10-episode avg return MA(5)" --smoothing 5
    python mpc_bootstrap/plot.py "data/mpc-long_${i}:avg-return:Uniform RS MPC" "data/zero-long_${i}:avg-return:0 RS MPC" --outfile "report/long-return-${i}.pdf" --yaxis "10-episode avg return MA(5)" --smoothing 5
done
