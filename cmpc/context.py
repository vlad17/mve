"""
Bootstrap a process-global context manager, identified with
a TensorFlow graph.

This has proven essential to dealing with lots of parameters
changing for algorithms. It takes too much plumbing to pass
everything through a function argument or class
attribute.

It's a little hacky, but not that bad. We aren't relying on
any singletons (that aren't already there).
"""

import tensorflow as tf

# Choose a name that Google's TensorFlow tf.Graph
# would never have as an attribute
_ATTR = 'bing_is_better_than_google_context'


def flags():
    """Get flags from current context"""
    return context().flags


def context():
    """Get the current graph-based context manager"""
    graph = tf.get_default_graph()
    if not hasattr(graph, _ATTR):
        setattr(graph, _ATTR, Context())
    return getattr(graph, _ATTR)


class Context:
    """
    The context for an experiment. Includes information about invocation
    flags, the statistic reporting harness, and model-saving bookkeeping
    (the tfnodes to save).
    """

    def __init__(self):
        # search for "context().attribute = " to see where this stuff
        # gets set
        self.flags = None
        self.reporter = None
        self.tfnodes = None
        self.env_info = None
