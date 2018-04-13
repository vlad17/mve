"""Generate random rollouts."""

from context import flags
from contextlib import closing
import env_info
from experiment import ExperimentFlags, setup_experiment_context
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


def _train():
    with closing(env_info.make_venv(flags().random.num_paths)) as venv:
        random_policy = RandomPolicy(venv)
        paths = sample_venv(venv, random_policy.exploit_act)
        rewards = [path.rewards.sum() for path in paths]
        reporter.advance(timesteps(paths), len(paths))
        reporter.add_summary_statistics('reward', rewards)
        reporter.report()

if __name__ == "__main__":
    _args = parse_args([
        ExperimentFlags(), RandomPolicyFlags()])
    with setup_experiment_context(_args):
        _train()
