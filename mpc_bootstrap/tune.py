"""
Search for best hyperparameters of BMPC.

Adapted from:
https://github.com/ray-project/ray/blob/ray-0.2.2/python/ray/tune/examples/tune_mnist_ray.py
"""

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

from flags import (Flags, parse_args, parse_args_with_subcmds)
from main_bootstrapped_mpc import main as bmpc_main
from main_bootstrapped_mpc import flags_to_parse as bmpc_flags


class TuneFlags(Flags):
    """
    These flags define parameters for the ray tune script.
    """

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        tuner = parser.add_argument_group('tune')
        tuner.add_argument(
            '--ray_addr',
            type=str,
            default='',
            required=True,
            help='ray head node redis ip:port address'
        )
        tuner.add_argument(
            '--tunefile',
            type=str,
            required=True,
            help='json file with grid to search over'
        )
        tuner.add_argument(
            '--env_name',
            type=str,
            default='hc-hard',
            help='environment name for testing')
        tuner.add_argument(
            '--result_dir',
            type=str,
            default='data',
            help='directory to store results in')

    @staticmethod
    def name():
        return 'tune'

    def __init__(self, args):
        self.ray_addr = args.ray_addr
        self.tunefile = args.tunefile
        self.env_name = args.env_name
        self.result_dir = args.result_dir


def train(config, status_reporter):
    """
    Entry point called by ray hypers tuning, remotely.

    config should be a dictionary with the usual BMPC flags.

    This basically runs
    python main_bootstrapped_mpc.py config['bmpc_type'] <other config args>

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

    args = [config['bmpc_type']]
    smoothing = config['smoothing']

    assert 'exp_name' not in config, 'exp_name in config'
    assert 'seed' not in config, 'seed in config'

    for k, v in config['hypers'].items():
        args.append('--' + k)
        if v is not None:
            args.append(repr(v))

    exp_name = datetime.datetime.now().strftime('%b-%d-%I%M%p-%G')
    args += ['--exp_name', exp_name, '--seed', '7']

    flags, subflags = bmpc_flags()
    parsed_flags, parsed_subflags = parse_args_with_subcmds(
        flags, subflags, args)
    ctr = 0

    def _reporter(result):
        nonlocal ctr
        status_reporter.report(TrainingResult(
            timesteps_total=ctr,
            episode_reward_mean=result))
        ctr += 1

    bmpc_main(parsed_flags, parsed_subflags, smoothing, _reporter)


def _search_hypers(all_hypers, tune):
    runner = TrialRunner()

    for hyp in all_hypers:
        config = {
            'script_file_path': os.path.abspath(__file__),
            'script_min_iter_time_s': 0,
            'smoothing': hyp['smoothing'],
            'bmpc_type': 'ddpg',
        }
        del hyp['smoothing']
        config['hypers'] = hyp
        nit = hyp['onpol_iters'] - config['smoothing']
        runner.add_trial(
            Trial(
                tune.env_name,
                'script',
                stopping_criterion={'timesteps_total': nit},
                local_dir=tune.result_dir,
                config=config,
                resources=Resources(cpu=1, gpu=1)))

    while not runner.is_finished():
        runner.step()
        print(runner.debug_string())


def _main(args):
    ray.init(redis_address=(args.tune.ray_addr))
    with open(args.tune.tunefile) as f:
        hypers = json.load(f)
    np.random.shuffle(hypers)
    _search_hypers(hypers, args.tune)


if __name__ == "__main__":
    _args = parse_args([TuneFlags])
    _main(_args)
