"""Generate SAC rollouts and train on them"""

from context import flags
from sac_learner import SACLearner, SACFlags
from dynamics import DynamicsFlags, NNDynamicsModel
from experiment import ExperimentFlags, setup_experiment_context
from memory import Normalizer, NormalizationFlags
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
    normalizer = Normalizer()
    if need_dynamics:
        dynamics = NNDynamicsModel(normalizer)
    else:
        dynamics = None

    learner = SACLearner(dynamics=dynamics, normalizer=normalizer)
    rl_loop(learner, normalizer, dynamics)


ALL_SAC_FLAGS = [ExperimentFlags(), PersistableDatasetFlags(),
                 SACFlags(), DynamicsFlags(), RLLoopFlags(),
                 SamplerFlags(), NormalizationFlags()]

if __name__ == "__main__":
    _args = parse_args(ALL_SAC_FLAGS)
    with setup_experiment_context(_args):
        train()
