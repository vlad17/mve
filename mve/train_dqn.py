import gym
import itertools, pickle, sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
import deepq
# from baselines import deepq
from deepq import EnvSim
from deepq.replay_buffer import ReplayBuffer
from deepq.utils import BatchInput
from baselines.common.schedules import LinearSchedule
from deepq.simple import run_experiment


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out

if __name__ == '__main__':
    run_experiment(model, horizon=int(sys.argv[1]))
