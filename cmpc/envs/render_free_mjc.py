"""
Remove some window-creating code from the gym MuJoCo environment.
We still need an X display even to render off-screen, but this lets
us run tests without creating actual windows when running locally
on one's laptop.
"""

import gym.envs.mujoco
import mujoco_py


class RenderFreeMJC(gym.envs.mujoco.mujoco_env.MujocoEnv):
    """
    Don't render in a windowing environment.
    """
    def _render(self, mode='human', close=False):
        if mode == 'human':
            return None # don't make a window
        return super()._render(mode, close)

    def _step(self, action):
        raise NotImplementedError

    def reset_model(self):
        raise NotImplementedError

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(visible=False)
            self.viewer.start()
            self.viewer.set_model(self.model)
            self.viewer_setup()
        return self.viewer
