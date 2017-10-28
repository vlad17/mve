import numpy as np
import tensorflow as tf


#========================================================
#
# Environment-specific cost functions:
#

def cheetah_cost_fn(state, action, next_state):
    assert len(state.shape) == 2
    heading_penalty_factor = 10
    scores = np.zeros((state.shape[0],))

    # dont move front shin back so far that you tilt forward
    front_leg = state[:, 5]
    my_range = 0.2
    scores[front_leg >= my_range] += heading_penalty_factor

    front_shin = state[:, 6]
    my_range = 0
    scores[front_shin >= my_range] += heading_penalty_factor

    front_foot = state[:, 7]
    my_range = 0
    scores[front_foot >= my_range] += heading_penalty_factor

    scores -= (next_state[:, 17] - state[:, 17]) / 0.01
    return scores
    # scores += 0.1 * (np.sum(action**2))

def tf_cheetah_cost_fn(state, action, next_state, cost):
    heading_penalty_factor = tf.constant(10.)

    front_leg = state[:, 5]
    my_range = 0.2
    cost += tf.cast(front_leg >= my_range, tf.float32) * heading_penalty_factor

    front_shin = state[:, 6]
    my_range = 0
    cost += tf.cast(front_shin >= my_range, tf.float32) * heading_penalty_factor

    front_foot = state[:, 7]
    my_range = 0
    cost += tf.cast(front_foot >= my_range, tf.float32) * heading_penalty_factor

    cost -= (next_state[:, 17] - state[:, 17]) / 0.01
    return cost


#========================================================
# A harder, less supervised version of the cheetah_cost_fn

def hard_cheetah_cost_fn(state, action, next_state):
    assert len(state.shape) == 2
    scores = np.zeros((state.shape[0],))
    scores += 0.1 * np.square(action).sum(axis=1)
    scores -= (next_state[:, 17] - state[:, 17]) / 0.01
    return scores

def hard_tf_cheetah_cost_fn(state, action, next_state, cost):
    cost += tf.reduce_sum(tf.square(action), axis=1)
    cost -= (next_state[:, 17] - state[:, 17]) / 0.01
    return cost

#========================================================
#
# Cost function for a whole trajectory:
#


def trajectory_cost_fn(cost_fn, states, actions, next_states):
    # assumes axes are (time, batch, state/action dim)
    # can handle     # assumes axes are (time, batch, state/action dim)
    assert len(states.shape) >= 2, states.shape
    if len(states.shape) == 2:
        return trajectory_cost_fn(
            cost_fn,
            states[:, np.newaxis, :],
            actions[:, np.newaxis, :],
            next_states[:, np.newaxis, :])[0]
    assert len(states.shape) == 3, states.shape
    trajectory_costs = 0
    for t in range(len(actions)):
        trajectory_costs += cost_fn(states[t], actions[t], next_states[t])
    return trajectory_costs
