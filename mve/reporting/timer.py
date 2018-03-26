"""
This creates a timer utility which facilitates easy periodic
logging.

The reporter's timestep is a montonically increasing value, and
the timers "time out" relative to this value.
"""

import reporter

class Timer:
    """
    A logging timer keeps track of the last time it was snoozed.

    A timer times out when a period of time has passed since it
    was last snoozed.

    When a timer has timed out, it remains timed out until it
    is again snoozed.

    A period of 0 means that the timer should never time out.
    """

    def __init__(self, period):
        self._period = period
        self._last_snoozed = -period

    def has_timed_out(self):
        """Returns if the period has elapsed since last snoozed"""
        if self._period == 0:
            return False
        return reporter.timestep() >= self._last_snoozed + self._period

    def snooze(self):
        """Snooze the timer"""
        self._last_snoozed = reporter.timestep()
