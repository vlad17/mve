"""
Helper functions for calculating the discounted cumulative rewards of
trajectories. This entire file assumes deterministic dynamics.

To set the record straight and avoid off-by-one errors,
starting at a given state s_0 and action a_0, for a given
critic and actor and dynamics:

* h-step reward - the discounted sum of the next h rewards

  sum from t=0 to h-1 of discount^t * reward(s_t, a_t, s_t+1)
  where
    s_t+1 = dynamics(s_t, a_t)
    a_t = actor(s_t)

* h-step Q estimate - the discounted sum of the next h rewards

  h-step reward + discount^h critic(s_h, a_h)
  where the same definitions hold

Note if there is an early termination, then the critic and reward
values are taken to be 0 in the h-step reward calculation.
"""

import numpy as np

from context import flags
import log
from utils import rate_limit


def qvals(paths, discount):
    """
    Return a list of the Q-values at each point along several
    trajectories according to a given discount.

    This is just a sample of the "true net present value" of the reward
    for a given path.
    """
    rewards = [path.rewards for path in paths]
    all_qvals = []
    for reward_trajectory in rewards:
        # numerically stable cumulative reward sum
        path_qvals = np.empty_like(reward_trajectory, dtype=np.float64)
        future_qval = 0
        for i in range(len(reward_trajectory) - 1, -1, -1):
            path_qvals[i] = reward_trajectory[i] + discount * future_qval
            future_qval = path_qvals[i]
        all_qvals.append(path_qvals)
    return all_qvals


def hstep(paths, discount, h):
    """
    Return the h-step reward, which is similar to the Q-value but truncated
    to h steps (for any given timestep, the h-step reward is the Q-value,
    assuming we no longer receive any rewards h steps later).

    The 1-step reward is just the original reward array.
    The the inifinity-step reward is qvals(...)

    Like qvals, gives a list of the h-step rewards at each point along several
    trajectories according to a given discount.
    """
    assert h > 0, h
    all_hstep = qvals(paths, discount)
    for hsteps in all_hstep:
        hsteps[:-h] -= discount ** h * hsteps[h:]
    return all_hstep


def corrected_horizon(paths, h):
    """
    Given a list of paths and a maximum horizon to consider, this returns the
    concatenated list of horizons which we can look ahead with from every
    state, action pair in the trajectories in paths.

    Most of these values will be the maximum horizon h, except the last h-1
    transitions in each path, which will have progressively smaller horizons
    for consideration (e.g., the horizon for the last transition in a
    trajectory is at most 1, expanded to include the transition's own
    reward).


    Returns all horizons concatenated.
    """
    if h == 0:
        totlen = sum(len(path.rewards) for path in paths)
        return np.zeros(totlen, dtype=int)
    horizons = []
    for path in paths:
        hs = np.full(len(path.rewards), h, dtype=int)
        hs[-h:] -= np.arange(h)[-len(hs[-h:]):]
        horizons.append(hs)
    return np.concatenate(horizons)


def oracle_q(critic, actor, states_ns, acs_na, venv, h_n):
    """
    Given a critic and actor, which should be functions of type
    (states, actions) -> (Q estimates) and (states) -> (actions) respectively,
    and given a venv (the oracle) in which to project h steps ahead, this
    produces the h-step Q value estimates.

    The h used in "h-step" can be state dependent (i.e., the caller can request
    the 100-step Q value estimate for states_ns[0] and acs_na[0] but only a
    10-step Q value estimate for states_ns[1] and acs_na[1]. This flexibility
    is useful for terminating episodic MDPs.
    """
    q_n, early_term_n = rate_limit(
        venv.n, lambda s, a, h: _oracle_q(critic, actor, s, a, venv, h),
        states_ns, acs_na, h_n)
    early_terms = early_term_n.sum()
    if early_terms > 0:
        log.debug('warning: {} of {} oracle simulations terminated early',
                  early_terms, len(q_n))
    return q_n


def _oracle_q(critic, actor, states_ns, acs_na, venv, h_n):
    qest = np.zeros(len(states_ns))
    venv.reset()
    venv.set_state_from_ob(states_ns)
    discount = flags().experiment.discount
    active_n = np.ones(len(states_ns), dtype=bool)
    terminated_early_n = np.zeros(len(states_ns), dtype=bool)
    active_n[h_n == 0] = False
    final_states_ns = np.copy(states_ns)
    final_acs_na = np.copy(acs_na)
    maxh = h_n.max()
    end_time = h_n.copy()  # could be earlier if env becomes done
    for i in range(maxh):
        states_ns, reward_n, done_n, _ = venv.step(acs_na)
        states_ns = np.asarray(states_ns, dtype=float)
        reward_n = np.asarray(reward_n, dtype=float)
        done_n = np.asarray(done_n, dtype=bool)
        acs_na = actor(states_ns)
        qest[active_n] += discount ** i * reward_n[active_n]
        done_n &= active_n
        terminated_early_n |= done_n
        done_n |= i + 1 == h_n
        active_n[done_n] = False
        final_states_ns[done_n] = states_ns[done_n]
        final_acs_na[done_n] = acs_na[done_n]
        end_time[done_n] = i + 1
    assert maxh == 0 or not np.any(active_n), (maxh, active_n.sum())
    assert np.all(end_time <= h_n), np.sum(end_time > h_n)
    final_critic = critic(final_states_ns, final_acs_na)
    final_critic *= np.power(discount, end_time)
    final_critic *= 1 - terminated_early_n
    qest += final_critic
    return (qest, end_time != h_n)


def offline_oracle_q(paths, critic_qs, h):
    """
    This offers the results the oracle would have in oracle_q, but in an
    offline manner. critic_qs should be a single tensor of all the critic
    values at the corresponding state and action along each trajectory,
    concatenated along the 0-th axis.

    If h is at least as large as the horizon, this recovers the exact Q-value.
    This method returns a concatenation of the ideal Q-value estimates for
    all timesteps in the given paths.

    If h is equal to 0, then the critic values are directly returned.

    For the same, deterministic actor, offline_oracle_q and oracle_q result
    in identical values.
    """
    if h == 0:
        return critic_qs
    discount = flags().experiment.discount
    hsteps = hstep(paths, discount, h)
    hsteps = np.concatenate(hsteps)
    pathlens = np.cumsum([len(path.rewards) for path in paths])[:-1]
    per_path_critic_qs = np.split(critic_qs, pathlens)
    # need a little finessing to deal with end-of-episode bias (in those cases
    # we just use the m-step oracle, where m is the largest value that's at
    # most h but keeps us within the trajectory still
    per_path_h_step_critic_qs = []
    for q in per_path_critic_qs:
        q = np.roll(q, -h) * discount ** h
        q[-h:] = 0
        per_path_h_step_critic_qs.append(q)
    h_step_critic_qs = np.concatenate(per_path_h_step_critic_qs)
    return hsteps + h_step_critic_qs
