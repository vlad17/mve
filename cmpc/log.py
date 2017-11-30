"""
Imagine you want to log stuff in multiple files of a Python project. You'd
probably write some code that looks like this:

    logging.debug("Hello, World!")

Pretty simple. Unfortunately, some packages like gym seem to clobber the
default logger that is used by logging.debug. No big deal, just use a different
logger:

    logging.getLogger("mylogger").debug("Hello, World!")

Two problems stick out though. First, "mylogger" is a bit of a magic constant.
Second, that's a lot of typing for a print statement! This file makes it a
tinier bit more convenient to log stuff. In one file, run this:

    import log
    log.init(verbose=True)

And then from any other file, run something this:

    from log import debug
    debug("Iteration {} of {}", 1, num_iters)
"""

import inspect
import logging


class _StackCrawlingFormatter(logging.Formatter):
    """
    If we configure a python logger with the format string
    "%(pathname):%(lineno): %(message)", messages logged via `log.debug` will
    be prefixed with the path name and line number of the code that called
    `log.debug`. Unfortunately, when a `log.debug` call is wrapped in a helper
    function (e.g. debug below), the path name and line number is always that
    of the helper function, not the function which called the helper function.

    A _StackCrawlingFormatter is a hack to log a different pathname and line
    number. Simply set the `pathname` and `lineno` attributes of the formatter
    before you call `log.debug`. See `debug` below for an example.
    """
    def __init__(self, format_str):
        super().__init__(format_str)
        self.pathname = None
        self.lineno = None

    def format(self, record):
        s = super().format(record)
        if self.pathname is not None:
            s = s.replace('{pathname}', self.pathname)
        if self.lineno is not None:
            s = s.replace('{lineno}', str(self.lineno))
        return s


_LOGGER = logging.getLogger("cmpc")
_FORMAT_STRING = "[%(asctime)-15s {pathname}:{lineno}] %(message)s"
_FORMATTER = _StackCrawlingFormatter(_FORMAT_STRING)


def init(verbose):
    """Initialize the logger."""
    handler = logging.StreamHandler()
    handler.setFormatter(_FORMATTER)
    _LOGGER.propagate = False
    _LOGGER.addHandler(handler)
    if verbose:
        _LOGGER.setLevel(logging.DEBUG)


def debug(s, *args):
    """debug(s, x1, ..., xn) logs s.format(x1, ..., xn)."""
    # Get the path name and line number of the function which called us.
    previous_frame = inspect.currentframe().f_back
    pathname, lineno, _, _, _ = inspect.getframeinfo(previous_frame)
    _FORMATTER.pathname = pathname
    _FORMATTER.lineno = lineno
    _LOGGER.debug(s.format(*args))
