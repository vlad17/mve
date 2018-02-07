"""Import into envs automatically"""

from .fully_observable_half_cheetah import FullyObservableHalfCheetah
from .fully_observable_ant import FullyObservableAnt
from .fully_observable_walker2d import FullyObservableWalker2d
from .fully_observable_pusher import FullyObservablePusher
from .fully_observable_hopper import FullyObservableHopper
from .fully_observable_swimmer import FullyObservableSwimmer
from .parallel_gym_venv import ParallelGymVenv
from .numpy_reward import NumpyReward

# acrobot is intentionally NOT imported by defualt because its numba jit
# dependence creates an OpenMP thread pool which creates too many threads
# after forks. So users should import envs.acrobot on demand (knowing that
# doing so will create lots of threads).
