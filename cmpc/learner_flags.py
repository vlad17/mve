"""Learner flags."""

from flags import Flags


class LearnerFlags(Flags):
    """Learner flags."""

    def make_learner(self, venv, sess):
        """
        Makes a learner.

        Parameters
        ----------
        venv: multiprocessing_env.MultiprocessingEnv
        sess: tensorflow.Session

        Returns
        -------
        learner.Learner
        """
        raise NotImplementedError()
