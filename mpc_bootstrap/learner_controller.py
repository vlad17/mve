"""
A learner learns to imitate a controller. A LearnerController lets the learner
_be_ the controller.
"""

from controller import Controller

class LearnerController(Controller):
    """
    Only use the learner follow actions.
    """

    def __init__(self, learner):
        self.learner = learner

    def act(self, states_ns):
        return self.learner.act(states_ns)

    def fit(self, data):
        self.learner.fit(data)
