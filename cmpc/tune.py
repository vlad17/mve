"""
Search for best hyperparameters of CMPC.

Adapted from:
https://github.com/ray-project/ray/blob/ray-0.2.2/python/ray/tune/examples/tune_mnist_ray.py
"""

import collections
import datetime
import os
import json

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import
import numpy as np
import ray
from ray.tune.result import TrainingResult
from ray.tune.trial import (Trial, Resources)
from ray.tune.trial_runner import TrialRunner

from experiment import experiment_main
from flags import (ArgSpec, Flags, parse_args)
from main_cmpc import train as cmpc_train
from main_cmpc import flags_to_parse as cmpc_flags
import reporter


class TuneFlags(Flags):
    """
    These flags define parameters for the ray tune script.
    """

    @staticmethod
    def _generate_arguments():
        """Adds flags to an argparse parser."""
        yield ArgSpec(
            name='ray_addr',
            type=str,
            default='',
            required=True,
            help='ray head node redis ip:port address')
        yield ArgSpec(
            name='tunefile',
            type=str,
            required=True,
            help='json file with grid to search over')
        yield ArgSpec(
            name='result_dir',
            type=str,
            default='data',
            help='directory to store results in')

    def __init__(self):
        super().__init__('tune', 'ray tuning',
                         list(TuneFlags._generate_arguments()))


def train(config, status_reporter):
    """
    Entry point called by ray hypers tuning, remotely.

    config should be a dictionary with the usual BMPC flags.

    This basically runs

    python main_cmpc.py <config dictionary as flags>

    The other arguments are determined by their keys (flag) and values
    (argument for the flag). If the value is None then that flag gets no
    argument.

    Note that config should also have a smoothing argument (not related to
    flags, but related to reporting).

    This will take care of setting annoying stuff like seeds and exp name.
    """
    # Make sure we only ever use one GPU
    # per https://ray.readthedocs.io/en/latest/using-ray-with-gpus.html
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, ray.get_gpu_ids()))

    args = []
    smoothing = config['smoothing']

    assert 'exp_name' not in config['hypers'], 'exp_name in config'
    assert 'seed' not in config['hypers'], 'seed in config'

    for k, v in config['hypers'].items():
        args.append('--' + k)
        if v is not None:
            if isinstance(v, str):
                args.append(v)
            else:
                args.append(repr(v))

    exp_name = datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')
    args += ['--exp_name', exp_name, '--seed', '7']

    flags = cmpc_flags()
    parsed_flags = parse_args(flags, args)

    ctr = 0
    historical_returns = collections.deque()

    def _report_hook(_, statistics):
        nonlocal ctr, historical_returns
        if 'current policy reward mean' not in statistics:
            return
        result = np.mean(statistics['current policy reward mean'])
        if len(historical_returns) == smoothing:
            historical_returns.popleft()
        historical_returns.append(result)
        smoothed_result = np.mean(historical_returns)
        status_reporter.report(TrainingResult(
            timesteps_total=ctr,
            episode_reward_mean=smoothed_result))
        ctr += 1

    with reporter.report_hook(_report_hook):
        experiment_main(parsed_flags, cmpc_train)


def _search_hypers(all_hypers, tune):
    runner = TrialRunner()

    for hyp in all_hypers:
        config = {
            'script_file_path': os.path.abspath(__file__),
            'script_min_iter_time_s': 0,
            'smoothing': hyp['smoothing'],
        }
        del hyp['smoothing']
        config['hypers'] = hyp
        nit = hyp['onpol_iters'] - config['smoothing']
        runner.add_trial(
            Trial(
                'tune',
                'script',
                stopping_criterion={'timesteps_total': nit},
                local_dir=tune.result_dir,
                config=config,
                resources=Resources(cpu=1, gpu=1)))

    while not runner.is_finished():
        runner.step()
        print(runner.debug_string())

    errored = []
    for trial in runner.get_trials():
        if trial.status == Trial.ERROR:
            errored.append(trial.config)

    if errored:
        raise ValueError('tasks errored: {}'.format(errored))


def _main(args):
    ray.init(redis_address=(args.tune.ray_addr))
    with open(args.tune.tunefile) as f:
        hypers = json.load(f)
    np.random.shuffle(hypers)
    _search_hypers(hypers, args.tune)


if __name__ == "__main__":
    _args = parse_args([TuneFlags()])
    _main(_args)
