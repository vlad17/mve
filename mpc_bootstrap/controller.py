"""Base controller class."""


from policy import Policy


class Controller(Policy):  # pylint: disable=abstract-method
    """
    A possibly stateful controller, which decides which actions to take.
    A controller might choose to label the dataset.
    """

    def fit(self, data):
        """A controller might fit internal learners here."""
        pass

    def label(self, states_ns):
        """Optionally label a dataset's existing states with new actions."""
        pass
