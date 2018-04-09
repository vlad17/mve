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
        if close:
            if not hasattr(self, 'viewer'):
                self.viewer = None
            if self.viewer is not None:
                self.viewer = None  # pylint: disable=attribute-defined-outside-init
            return None

        if mode == 'rgb_array':
            width, height = 640, 480
            self._setup_render()
            data = self.sim.render(width, height)
            return data.reshape(height, width, 3)[::-1, :, :]
        elif mode == 'human':
            # avoid literally all GL pain by using off-screen renders
            return None
        return None

    def _setup_render(self):
        if self.sim._render_context_offscreen is None:  # pylint: disable=protected-access
            self.sim.render(640, 480)
            assert self.sim._render_context_offscreen is not None  # pylint: disable=protected-access
            ctx = self.sim._render_context_offscreen  # pylint: disable=protected-access
            ctx.cam.distance = self.sim.model.stat.extent * 0.5
            ctx.cam.type = mujoco_py.const.CAMERA_TRACKING
            ctx.cam.trackbodyid = 0

    def _step(self, action):
        raise NotImplementedError

    def reset_model(self):
        raise NotImplementedError

    def viewer_setup(self):
        pass
