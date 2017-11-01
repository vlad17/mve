"""White-box HalfCheetah environment which reveals relevant physical info"""

from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np

from utils import inherit_doc


@inherit_doc
class HalfCheetahEnvNew(mujoco_env.MujocoEnv, utils.EzPickle):
    """Provided code from HW4 for creating a white-box HalfCheetah env"""

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 1)
        utils.EzPickle.__init__(self)

    def _step(self, action):
        xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.model.data.qpos[0, 0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        info = dict(reward_run=reward_run, reward_ctrl=reward_ctrl)
        return ob, reward, done, info

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
            # TODO: find out why the below was filtered by HW4
            # self.get_body_comvel("torso").flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
