"""Generate DDPG rollouts and train on them"""

from context import flags
from ddpg_learner import DDPGLearner, DDPGFlags
from dynamics import DynamicsFlags, NNDynamicsModel
from experiment import ExperimentFlags, setup_experiment_context
from persistable_dataset import PersistableDatasetFlags
from rl_loop import rl_loop, RLLoopFlags
from flags import parse_args
from sample import SamplerFlags


def train():
    """
    Runs the DDPG + MVE training procedure, reading from the global flags.
    """
    need_dynamics = (
        flags().ddpg.dynamics_type == 'learned' or
        flags().ddpg.imaginary_buffer > 0)
    if need_dynamics:
        dynamics = NNDynamicsModel()
    else:
        dynamics = None

    learner = DDPGLearner(dynamics=dynamics)
    rl_loop(learner, dynamics)


ALL_DDPG_FLAGS = [ExperimentFlags(), PersistableDatasetFlags(),
                  DDPGFlags(), DynamicsFlags(), RLLoopFlags(),
                  SamplerFlags()]

if __name__ == "__main__":
    _args = parse_args(ALL_DDPG_FLAGS)
    with setup_experiment_context(_args):
        train()
