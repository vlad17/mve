#!/usr/bin/env python3
"""
Generates a file called params.json in the cwd with parameters set
in json, ready for consumption by cmpc/tune.py
"""

import json
import itertools

# mutually exclusive options ----------------------------------------

# Every item in the following lists is a dictionary of flag
# values which is MUTUALLY EXCLUSIVE with other dictionaries
# in the same list; in other words, all settings will be generated
# with every combination of items between the lists

# Note that all the usual DDPG-BMPC flag params can be given here,
# but we also need to specify the "smoothing" value, which
# says that the last "smoothing" iterations' rewards
# are averaged to give the final reward on which the parameter
# set is being evaluated.

paths_iters = [
    {"smoothing": 3,
     "warmup_paths_mpc": 25,
     "onpol_paths": 1,
     "onpol_iters": 300}
]

sims = [{"mpc_simulated_paths": x} for x in [1000]]
mpc_horiz = [{"mpc_horizon": x} for x in [15]]
con_train_time = [
    {"con_epochs": 10},
    {"con_epochs": 1},
    {"training_batches": 4000}]
nn_shape = [
    {"con_depth": 2, "con_width": 64},
    {"con_depth": 5, "con_width": 32}]
con_lr = [{"con_learning_rate": x, "critic_lr": x} for x in
          [1e-4, 5e-5, 1e-5]]

constants = [{
    "action_noise_exploration": 0.1,
    "dyn_depth": 2,
    "dyn_width": 500,
    "dyn_epochs": 60}]

# create all combos of the above  -----------------------------------
all_items = [
    paths_iters, sims, mpc_horiz, con_train_time, nn_shape, con_lr, constants]

params = []
for prod in itertools.product(*all_items):
    params.append({k: v for d in prod for k, v in d.items()})
print(len(params))
with open('params.json', 'w') as f:
    json.dump(params, f)
