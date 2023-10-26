"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

import policy
from common import Batch, InfoDict, Model, PRNGKey, Params



ACTION_MIN, ACTION_MAX = -1, 1


def safe_norm(x, **kwargs):
    # l2 norm with gradient set to 0 when norm is 0
    return jnp.linalg.norm(jnp.where(x == 0, 0, x), **kwargs)


def update_actor(key: PRNGKey, actor: Model, batch: Batch, lambd: float,
                      dist_temperature: float) -> Tuple[Model, InfoDict]:
    
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # calculate distance
        policy_actions = actor.apply({"params": actor_params},
                           batch.observations,
                           training=True,
                           rngs={"dropout": key},
                           )
        actions = batch.actions
        scores = batch.scores

        # normalize action differences
        step_diffs = (policy_actions - actions) / (ACTION_MAX - ACTION_MIN)

        step_distances = safe_norm(step_diffs, axis=2)
        traj_distances = (step_distances * batch.masks).sum(axis=1) / batch.masks.sum(axis=1)
        
        # calculate score
        indices = jnp.argsort(scores, axis=0)
        indices = jnp.flip(indices) # to descending order

        distances_sorted = traj_distances[indices] / dist_temperature
        distances_sum = jnp.exp(-distances_sorted) + jnp.exp(-lambd * distances_sorted).reshape(-1)
        pair_score = jnp.exp(-distances_sorted) / distances_sum

        log_score_triu = jnp.triu(jnp.log(pair_score), k=1)

        mask = log_score_triu != 0
        score = (log_score_triu).sum() / mask.sum()
        actor_loss = - score
        return actor_loss, {'actor_loss': actor_loss, 'score': score}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info

@jax.jit
def _update_jit(
    rng: PRNGKey, actor: Model, batch: Batch, lambd: float, dist_temperature: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, batch, lambd, dist_temperature)

    return rng, new_actor, {
        **actor_info
    }

class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 opt_decay_schedule: str = "",
                  
                 lambd: float = 1.0,
                 dist_temperature: float = 1.0,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """
        self.lambd = lambd
        self.dist_temperature = dist_temperature

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng, 2)

        action_dim = actions.shape[-1]
        actor_def = policy.DeterministicPolicy(hidden_dims,
                                               action_dim,
                                               dropout_rate=dropout_rate)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimizer = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimizer = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimizer)

        self.actor = actor
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       **kwargs,
                       ) -> jnp.ndarray:
        actions = policy.sample_actions_det(self.actor.apply_fn,
                                             self.actor.params, observations)

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, info = _update_jit(
            self.rng, self.actor, batch, self.lambd, self.dist_temperature)

        self.rng = new_rng
        self.actor = new_actor

        return info
