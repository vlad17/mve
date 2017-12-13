"""Generate random rollouts."""

# import mujoco for weird dlopen reasons
import mujoco_py  # pylint: disable=unused-import

from dataset import one_shot_dataset
from experiment import ExperimentFlags, experiment_main
from flags import (Flags, parse_args, ArgSpec)
from multiprocessing_env import make_venv
from random_policy import RandomPolicy
from sample import sample_venv
import reporter


class RandomPolicyFlags(Flags):
    """Random policy flags."""

    def __init__(self):
        super().__init__('random', 'random policy evaluation flags',
                         [ArgSpec(name='num_paths', type=int, default=100,
                                  help='number of paths to evaluate')])


def _train(args):
    venv = make_venv(args.experiment.make_env, args.random.num_paths)
    random_policy = RandomPolicy(venv)
    paths = sample_venv(venv, random_policy, args.experiment.horizon)
    data = one_shot_dataset(paths)
    data.log_rewards()
    reporter.advance_iteration()


if __name__ == "__main__":
    _args = parse_args([ExperimentFlags(), RandomPolicyFlags()])
    experiment_main(_args, _train)
