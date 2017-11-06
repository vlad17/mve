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

import logging

logger = logging.getLogger("mpc_bootstrap")

def init(verbose):
    """Initialize the logger."""
    format_ = "[%(asctime)-15s %(pathname)s:%(lineno)-3s] %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(format_))
    logger.propagate = False
    logger.addHandler(handler)
    if verbose:
        logger.setLevel(logging.DEBUG)

def debug(s, *args):
    """debug(s, x1, ..., xn) logs s.format(x1, ..., xn)."""
    logger.debug(s.format(*args))
