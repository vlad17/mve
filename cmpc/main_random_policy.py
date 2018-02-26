"""Generate random rollouts."""

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import

import env_info
from experiment import ExperimentFlags, experiment_main
from flags import (Flags, parse_args, ArgSpec)
from random_policy import RandomPolicy
from utils import timesteps
from sample import sample_venv
import reporter


class RandomPolicyFlags(Flags):
    """Random policy flags."""

    def __init__(self):
        super().__init__('random', 'random policy evaluation flags',
                         [ArgSpec(name='num_paths', type=int, default=100,
                                  help='number of paths to evaluate')])


def _train(args):
    venv = env_info.make_venv(args.random.num_paths)
    random_policy = RandomPolicy(venv)
    paths = sample_venv(venv, random_policy)
    rewards = [path.rewards.sum() for path in paths]
    reporter.advance(timesteps(paths), len(paths))
    reporter.add_summary_statistics('reward', rewards)
    reporter.report()


if __name__ == "__main__":
    _args = parse_args([
        ExperimentFlags(), RandomPolicyFlags()])
    experiment_main(_args, _train)
