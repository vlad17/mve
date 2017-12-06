#! /usr/bin/env bash
# Runs overoptimization experiments for MPC on hard half-cheetah.
# Should be run from repo root as follows:
#
# ./scripts/hc-mpc-overopt.sh
#
# Data from running this experiment can be found in report/hc-mpc-overopt.tgz

# Warning: don't actually run all these serially
for i in 100 500 1000 5000 10000 ; do 
    python cmpc/main_mpc.py --verbose --exp_name mpc-$i --onpol_iters 50 --seed 1 5 10 15 --mpc_simulated_paths $i
    python cmpc/main_cmpc.py zero --verbose --exp_name zero-$i --onpol_iters 50 --seed 1 5 10 15 --mpc_simulated_paths $i
done

python cmpc/plot.py data/mpc-100_hc-hard:avg-return:100 data/mpc-500_hc-hard:avg-return:500 data/mpc-1000_hc-hard:avg-return:1000 data/mpc-5000_hc-hard:avg-return:5000 data/mpc-10000_hc-hard:avg-return:10000 --outfile "report/mpc-return.pdf" --yaxis "10-episode avg return MA(5)" --smoothing 5

python cmpc/plot.py data/mpc-100_hc-hard:standardized-reward-bias:100 data/mpc-500_hc-hard:standardized-reward-bias:500 data/mpc-1000_hc-hard:standardized-reward-bias:1000 data/mpc-5000_hc-hard:standardized-reward-bias:5000 data/mpc-10000_hc-hard:standardized-reward-bias:10000 --outfile "report/mpc-reward-bias.pdf" --yaxis "10-episode reward bias MA(5)" --drop_iterations 10 --smoothing 5

python cmpc/plot.py data/mpc-100_hc-hard:standardized-reward-mse:100 data/mpc-500_hc-hard:standardized-reward-mse:500 data/mpc-1000_hc-hard:standardized-reward-mse:1000 data/mpc-5000_hc-hard:standardized-reward-mse:5000 data/mpc-10000_hc-hard:standardized-reward-mse:10000 --outfile "report/mpc-reward-mse.pdf" --yaxis "10-episode reward MSE MA(5)" --smoothing 5 --drop_iterations 10

python cmpc/plot.py data/zero-100_hc-hard:avg-return:100 data/zero-500_hc-hard:avg-return:500 data/zero-1000_hc-hard:avg-return:1000 data/zero-5000_hc-hard:avg-return:5000 data/zero-10000_hc-hard:avg-return:10000 --outfile "report/zero-return.pdf" --yaxis "10-episode avg return MA(5)" --smoothing 5

python cmpc/plot.py data/zero-100_hc-hard:standardized-reward-bias:100 data/zero-500_hc-hard:standardized-reward-bias:500 data/zero-1000_hc-hard:standardized-reward-bias:1000 data/zero-5000_hc-hard:standardized-reward-bias:5000 data/zero-10000_hc-hard:standardized-reward-bias:10000 --outfile "report/zero-reward-bias.pdf" --yaxis "10-episode reward bias MA(5)" --smoothing 5 --drop_iterations 10

python cmpc/plot.py data/zero-100_hc-hard:standardized-reward-mse:100 data/zero-500_hc-hard:standardized-reward-mse:500 data/zero-1000_hc-hard:standardized-reward-mse:1000 data/zero-5000_hc-hard:standardized-reward-mse:5000 data/zero-10000_hc-hard:standardized-reward-mse:10000 --outfile "report/zero-reward-mse.pdf" --yaxis "10-episode reward MSE MA(5)" --smoothing 5 --drop_iterations 10
