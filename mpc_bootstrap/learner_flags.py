"""Learner flags."""

from flags import Flags

class LearnerFlags(Flags):
    """Learner flags."""

    def make_learner(self, venv, sess, data):
        """
        Makes a learner.

        Parameters
        ----------
        venv: multiprocessing_env.MultiprocessingEnv
        sess: tensorflow.Session
        data: utils.Dataset

        Returns
        -------
        learner.Learner
        """
        raise NotImplementedError()
