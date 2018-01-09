"""
Another context (see context.py) that manages the child servers created during
graph construction.

The server registry tracks a simple heierarchy for our two-tier distributed
TensorFlow graph. There is a parent job, run on a parent process, and
child jobs, run on child processes (currently on the same machine) (also duh).

The workflow is like this:

1. The server registry is created in the parent process (attached to its
   default TF graph) via:

       with server_registry.create(child_index=None):
           # do stuff with server_registry

   The parent process builds up its TensorFlow graph. Parts of its graph
   that are private don't need a device specification. Those TF ops and
   variables can be created as usual. Any part of the TF graph that
   should be available to children should be publicized with the parent
   device specification. E.g.:

       with tf.device(server_registry.parent_device()):
           tf.get_variable('x', 0)
           # do stuff with x

   Once the parent is done building its graph, it can create a distributed
   TF session to use the graph in a manner that will be visible to other
   distributed TF nodes via:

       with server_registry.make_default_session():
           # do stuff with tf.get_default_session()

   Note that the make_default_session() call finalizes the TF graph.

2. Children (which can run in different processes, or just on different
   default graphs) have similar workflows, first registering under their
   child_index, then constructing their own graphs, and finally creating
   a session on them. Children can access parent variables.

       with server_registry.create(child_index=<user supplied>):
           with tf.device(server_registry.parent_device()):
               x = tf.get_variable('x', 0)
               # do stuff with x
           with tf.device(server_registry.child_device()):
               y = x + 1 # complex computation on child
           with server_registry.make_default_session():
               tf.get_default_session().run(y)

3. Steps 1 and 2 can happen concurrently, but the first run() call won't
   go through until make_default_session() has been called on every child
   and parent.

The parent and children take up server ports starting at
flags().experiment.port and going up contiguously by the number of children.

Note that the flags must be set before any server registry function can be
called.
"""

from contextlib import contextmanager

import tensorflow as tf

from context import flags, context
from utils import create_tf_config


def reserve_child():
    """
    Reserve a port for a child worker, and returns its ID (required for
    session initialization on the worker node). Can only be done on parent.

    Within a given context, this must be called before make_default_session().
    """
    is_parent = context().servers.is_parent
    assert is_parent, 'cannot reserve a child on child {}'.format(
        context().servers.child_index)
    server = context().servers.own_server
    assert server is None, 'server {} has already been created'.format(
        server.target)
    children = context().servers.children
    child_index = len(children)
    children.append(_child_host(child_index))
    return child_index


def parent_device():
    """
    Return the device identifying the parent subgraph. May be called
    on either the parent or the child.
    """
    return '/job:parent'


def child_device():
    """
    Return the device identifying the child subgraph. Can only be called
    on a child process, and returns only the device for that child process.
    """
    is_parent = context().servers.is_parent
    assert not is_parent, 'cannot access child device on parent'
    child_index = context().servers.child_index
    return '/job:child/task:{}'.format(child_index)


def get_child_index():
    """Makes sure this a child process and returns the child index."""
    is_parent = context().servers.is_parent
    assert not is_parent, 'cannot access child index on parent'
    child_index = context().servers.child_index
    return child_index


@contextmanager
def make_default_session():
    """
    Creates a TensorFlow session with GPU access and the appropriate cluster
    specification. Should only be called once per server registry context.

    After calling this, you may not call reserve_child() in the current
    server registry context.

    To enforce the workflow suggested in the module documentation, this
    method also finalizes the TF graph.
    """
    cluster = tf.train.ClusterSpec({
        'parent': [_parent_host()],
        'child': context().servers.children})
    config = create_tf_config(gpu=True)
    if context().servers.is_parent:
        server = tf.train.Server(cluster, job_name='parent', config=config)
    else:
        child_index = context().servers.child_index
        server = tf.train.Server(cluster, job_name='child',
                                 task_index=child_index, config=config)
    context().servers.own_server = server
    tf.get_default_graph().finalize()
    with tf.Session(target=server.target, config=config) as sess:
        with sess.as_default():
            yield


@contextmanager
def create(child_index=None):
    """
    Create a server registry within the current TF graph context.

    If child_index is None, then this is created as a parent registry, which
    can dynamically add child nodes (until session creation). Otherwise, this
    is created as a child registry.
    """
    assert context().servers is None, 'server registry already exists'
    context().servers = _ServerRegistry(child_index)
    yield
    context().env_info = None


class _ServerRegistry:
    def __init__(self, child_index):
        self.is_parent = child_index is None
        self.child_index = child_index
        if self.is_parent:
            self.children = []
        else:
            self.children = {child_index: _child_host(child_index)}
        self.own_server = None


def _child_host(child_index):
    base_port = flags().experiment.port
    return 'localhost:{}'.format(base_port + child_index + 1)


def _parent_host():
    base_port = flags().experiment.port
    return 'localhost:{}'.format(base_port)
