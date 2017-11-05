#!/bin/bash

python mpc_bootstrap/main.py --onpol_iters 15 --agent mpc --exp_name mpc-easy --seed 1 5 10 --time --easy_cost
python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name bmpc-easy --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --deterministic_learner --easy_cost
python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name sbmpc-easy --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --easy_cost

python mpc_bootstrap/main.py --onpol_iters 15 --agent mpc --exp_name mpc-com --seed 1 5 10 --time --com_pos --com_vel
python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name bmpc-com --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --deterministic_learner --com_pos --com_vel
python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name sbmpc-com --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --com_pos --com_vel

python mpc_bootstrap/main.py --onpol_iters 15 --agent mpc --exp_name mpc-com-reg --seed 1 5 10 --time --com_pos --com_vel --action_regularization 1.0
python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name bmpc-com-reg --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --deterministic_learner --com_pos --com_vel --action_regularization 1.0
python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name sbmpc-com-reg --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --com_pos --com_vel --action_regularization 1.0

python mpc_bootstrap/main.py --onpol_iters 15 --agent mpc --exp_name mpc-hard --seed 1 5 10 --time
python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name bmpc-hard --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512 --deterministic_learner
python mpc_bootstrap/main.py --onpol_iters 15 --agent bootstrap --exp_name sbmpc-hard --seed 1 5 10 --time --con_depth 5 --con_width 32 --con_epochs 100 --con_learning_rate 1e-3 --con_batch_size 512
