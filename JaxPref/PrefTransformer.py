from functools import partial

from ml_collections import ConfigDict

import jax
import jax.numpy as jnp

import optax
import numpy as np
from flax.training.train_state import TrainState

from .jax_utils import next_rng, value_and_multi_grad, mse_loss, cross_ent_loss, kld_loss

class PrefTransformer(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.trans_lr = 1e-4
        config.optimizer_type = 'adamw'
        config.scheduler_type = 'CosineDecay'
        config.vocab_size = 1
        config.n_layer = 1
        config.embd_dim = 256
        config.n_embd = config.embd_dim
        config.n_head = 4
        config.n_positions = 1024
        config.resid_pdrop = 0.1
        config.attn_pdrop = 0.1
        config.pref_attn_embd_dim = 256

        config.train_type = "mean"
        config.causal_mask = "False"

        config.smooth_w = 0.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())

        return config

    def __init__(self, config, trans):
        self.config = config
        self.trans = trans
        self.observation_dim = trans.observation_dim
        self.action_dim = trans.action_dim

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'adamw': optax.adamw,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]

        scheduler_class = {
            'CosineDecay': optax.warmup_cosine_decay_schedule(
                init_value=self.config.trans_lr,
                peak_value=self.config.trans_lr * 10,
                warmup_steps=self.config.warmup_steps,
                decay_steps=self.config.total_steps,
                end_value=self.config.trans_lr
            ),
            "OnlyWarmup": optax.join_schedules(
                [
                    optax.linear_schedule(
                        init_value=0.0,
                        end_value=self.config.trans_lr,
                        transition_steps=self.config.warmup_steps,
                    ),
                    optax.constant_schedule(
                        value=self.config.trans_lr
                    )
                ],
                [self.config.warmup_steps]
            ),
            'none': None
        }[self.config.scheduler_type]

        if scheduler_class:
            tx = optimizer_class(scheduler_class)
        else:
            tx = optimizer_class(learning_rate=self.config.trans_lr)

        trans_params = self.trans.init(
            {"params": next_rng(), "dropout": next_rng()},
            jnp.zeros((10, 25, self.observation_dim)),
            jnp.zeros((10, 25, self.action_dim)),
            jnp.ones((10, 25), dtype=jnp.int32)
        )
        self._train_states['trans'] = TrainState.create(
            params=trans_params,
            tx=tx,
            apply_fn=None
        )

        model_keys = ['trans']
        self._model_keys = tuple(model_keys)
        self._total_steps = 0      

    def evaluation(self, batch_id, batch_ood):
        metrics = self._eval_pref_step(
            self._train_states, next_rng(), batch_id, batch_ood
        )
        return metrics

    def get_score(self, batch):
        return self._get_score_step(self._train_states, batch)

    @partial(jax.jit, static_argnames=('self'))
    def _get_score_step(self, train_states, batch):
        obs = batch['observations']
        act = batch['actions']
        timestep = batch['timestep']
        attn_mask = batch['attn_mask']

        train_params = {key: train_states[key].params for key in self.model_keys}

        trans_pred, attn_weights = self.trans.apply(train_params['trans'], obs, act, timestep, attn_mask=attn_mask)
        return trans_pred["value"], attn_weights[-1]
  
    @partial(jax.jit, static_argnames=('self'))
    def _eval_pref_step(self, train_states, rng, batch_id, batch_ood):

        def loss_fn(train_params, rng):
            # score
            in_obs_1 = batch_id['observations_1']
            in_act_1 = batch_id['actions_1']
            in_obs_2 = batch_id['observations_2']
            in_act_2 = batch_id['actions_2']
            in_timestep_1 = batch_id['timestep_1']
            in_timestep_2 = batch_id['timestep_2']
            labels = batch_id['labels']
          
            B, T, _ = batch_id['observations_1'].shape
            B, T, _ = batch_id['actions_1'].shape

            rng, _ = jax.random.split(rng)
           
            in_trans_pred_1, _ = self.trans.apply(train_params['trans'], in_obs_1, in_act_1, in_timestep_1, training=False, attn_mask=None, rngs={"dropout": rng})
            in_trans_pred_2, _ = self.trans.apply(train_params['trans'], in_obs_2, in_act_2, in_timestep_2, training=False, attn_mask=None, rngs={"dropout": rng})

            in_trans_val_1 = in_trans_pred_1["value"]
            in_trans_val_2 = in_trans_pred_2["value"]

            in_logits = jnp.concatenate([in_trans_val_1, in_trans_val_2], axis=1)
           
            label_target = jax.lax.stop_gradient(labels)
            xent_loss = cross_ent_loss(in_logits, label_target)
            draw_mask = label_target[:, 0] == 0.5
            acc_raw = jnp.argmax(in_logits, axis=-1) == jnp.argmax(label_target, axis=-1)
            corr = jnp.where(draw_mask, 0, acc_raw)
            all = jnp.where(draw_mask, 0, 1)
            acc = corr.sum() / all.sum()

            # smooth
            out_obs_1 = batch_ood['observations_1']
            out_act_1 = batch_ood['actions_1']
            out_obs_2 = batch_ood['observations_2']
            out_act_2 = batch_ood['actions_2']
            out_timestep_1 = batch_ood['timestep_1']
            out_timestep_2 = batch_ood['timestep_2']
            out_masks_1 = batch_ood['masks_1']
            out_masks_2 = batch_ood['masks_2']
            
            out_trans_pred_1, _ = self.trans.apply(train_params['trans'], out_obs_1, out_act_1, out_timestep_1, training=False, attn_mask=out_masks_1, rngs={"dropout": rng})
            out_trans_pred_2, _ = self.trans.apply(train_params['trans'], out_obs_2, out_act_2, out_timestep_2, training=False, attn_mask=out_masks_2, rngs={"dropout": rng})

            out_trans_val_1 = out_trans_pred_1["value"]
            out_trans_val_2 = out_trans_pred_2["value"]

            squared_error = (out_trans_val_1 - out_trans_val_2)**2
            smooth_loss = jnp.mean(squared_error) # mse

            loss_collection = {}
            total_loss = xent_loss + self.config.smooth_w * smooth_loss
            loss_collection['trans'] = total_loss

            return tuple(loss_collection[key] for key in self.model_keys), locals()
        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), _ = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        metrics = dict(
            eval_xent_loss=aux_values['xent_loss'],
            eval_smooth_loss=aux_values['smooth_loss'],
            eval_total_loss=aux_values['total_loss'],
            eval_acc=aux_values['acc'],
        )

        return metrics
      
    def train(self, batch_id, batch_ood):
        self._total_steps += 1
        self._train_states, metrics = self._train_pref_step(
            self._train_states, next_rng(), batch_id, batch_ood
        )
        return metrics

    @partial(jax.jit, static_argnames=('self'))
    def _train_pref_step(self, train_states, rng, batch_id, batch_ood):

        def loss_fn(train_params, rng):
            # score
            in_obs_1 = batch_id['observations_1']
            in_act_1 = batch_id['actions_1']
            in_obs_2 = batch_id['observations_2']
            in_act_2 = batch_id['actions_2']
            in_timestep_1 = batch_id['timestep_1']
            in_timestep_2 = batch_id['timestep_2']
            labels = batch_id['labels']
          
            B, T, _ = batch_id['observations_1'].shape
            B, T, _ = batch_id['actions_1'].shape

            key, rng = jax.random.split(rng)
            in_trans_pred_1, _ = self.trans.apply(train_params['trans'], in_obs_1, in_act_1, in_timestep_1, training=True, attn_mask=None, rngs={"dropout": rng})
            in_trans_pred_2, _ = self.trans.apply(train_params['trans'], in_obs_2, in_act_2, in_timestep_2, training=True, attn_mask=None, rngs={"dropout": rng})

            in_trans_val_1 = in_trans_pred_1["value"]
            in_trans_val_2 = in_trans_pred_2["value"]

            in_logits = jnp.concatenate([in_trans_val_1, in_trans_val_2], axis=1)

            label_target = jax.lax.stop_gradient(labels)
            xent_loss = cross_ent_loss(in_logits, label_target)
            draw_mask = label_target[:, 0] == 0.5
            acc_raw = jnp.argmax(in_logits, axis=-1) == jnp.argmax(label_target, axis=-1)
            corr = jnp.where(draw_mask, 0, acc_raw)
            all = jnp.where(draw_mask, 0, 1)
            acc = corr.sum() / all.sum()

            # smooth
            out_obs_1 = batch_ood['observations_1']
            out_act_1 = batch_ood['actions_1']
            out_obs_2 = batch_ood['observations_2']
            out_act_2 = batch_ood['actions_2']
            out_timestep_1 = batch_ood['timestep_1']
            out_timestep_2 = batch_ood['timestep_2']
            out_masks_1 = batch_ood['masks_1']
            out_masks_2 = batch_ood['masks_2']
            
            out_trans_pred_1, _ = self.trans.apply(train_params['trans'], out_obs_1, out_act_1, out_timestep_1, training=True, attn_mask=out_masks_1, rngs={"dropout": rng})
            out_trans_pred_2, _ = self.trans.apply(train_params['trans'], out_obs_2, out_act_2, out_timestep_2, training=True, attn_mask=out_masks_2, rngs={"dropout": rng})

            out_trans_val_1 = out_trans_pred_1["value"]
            out_trans_val_2 = out_trans_pred_2["value"]

            squared_error = (out_trans_val_1 - out_trans_val_2)**2
            smooth_loss = jnp.mean(squared_error) # mse

            loss_collection = {}
            total_loss = xent_loss + self.config.smooth_w * smooth_loss
            loss_collection['trans'] = total_loss

            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            xent_loss=aux_values['xent_loss'],
            smooth_loss=aux_values['smooth_loss'],
            total_loss=aux_values['total_loss'],
            acc=aux_values['acc'],
        )

        return new_train_states, metrics
    
    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps
