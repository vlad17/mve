"""Parameter-space noise spec for DDPG"""


class AdaptiveParamNoiseSpec(object):
    """
    Add noise to parameters, adaptively resizing the amount of noise
    until action space noise is at the desired level.
    """

    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1,
                 adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        """
        Modify noise (per the specified coefficient) so that the desired
        magnitude of action-space noise is reached.
        """
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        """get noise statistics"""
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, '
        fmt += 'desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev,
                          self.adoption_coefficient)
