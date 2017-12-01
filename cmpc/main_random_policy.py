"""Generate random rollouts."""

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import

from dataset import one_shot_dataset
from experiment import ExperimentFlags, experiment_main
from flags import (Flags, parse_args)
from multiprocessing_env import mk_venv
from random_policy import RandomPolicy
from sample import sample_venv
import reporter


class RandomPolicyFlags(Flags):
    """Random policy flags."""

    @staticmethod
    def add_flags(parser):
        """Adds flags to an argparse parser."""
        random = parser.add_argument_group('random')
        random.add_argument(
            '--num_paths',
            type=int,
            default=100,
            help='number of paths',
        )

    @staticmethod
    def name():
        return "random"

    def __init__(self, args):
        self.num_paths = args.num_paths


def _train(args):
    venv = mk_venv(args.experiment.mk_env, args.random.num_paths)
    random_policy = RandomPolicy(venv)
    paths = sample_venv(venv, random_policy, args.experiment.horizon)
    data = one_shot_dataset(paths)
    returns = data.per_episode_rewards()
    reporter.add_summary_statistics('return', returns)
    reporter.advance_iteration()

if __name__ == "__main__":
    _args = parse_args([ExperimentFlags, RandomPolicyFlags])
    experiment_main(_args, _train)
