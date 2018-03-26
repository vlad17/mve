"""Generate SAC rollouts and train on them"""

from context import flags
from sac_learner import SACLearner, SACFlags
from dynamics import DynamicsFlags, NNDynamicsModel
from experiment import ExperimentFlags, setup_experiment_context
from persistable_dataset import PersistableDatasetFlags
from rl_loop import rl_loop, RLLoopFlags
from flags import parse_args
from sample import SamplerFlags


def train():
    """
    Runs the SAC + MVE training procedure, reading from the global flags.
    """
    need_dynamics = (
        flags().sac.sac_mve or
        flags().sac.imaginary_buffer > 0)
    if need_dynamics:
        dynamics = NNDynamicsModel()
    else:
        dynamics = None

    learner = SACLearner(dynamics=dynamics)
    rl_loop(learner, dynamics)


ALL_SAC_FLAGS = [ExperimentFlags(), PersistableDatasetFlags(),
                 SACFlags(), DynamicsFlags(), RLLoopFlags(),
                 SamplerFlags()]

if __name__ == "__main__":
    _args = parse_args(ALL_SAC_FLAGS)
    with setup_experiment_context(_args):
        train()
