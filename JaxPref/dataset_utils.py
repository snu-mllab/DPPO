import os

import pickle
import numpy as np
import collections
from typing import Optional, List, Dict

import gym
import d4rl
import jax
from tqdm import tqdm

from.reward_transform import find_timestep_in_traj

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

BatchSeq = collections.namedtuple(
    'BatchOurs',
    ['observations', 'actions', 'rewards', 'masks'])

BatchOurs = collections.namedtuple(
    'BatchOurs',
    ['observations', 'actions', 'rewards', 'scores', 'masks'])


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)


def get_d4rl_dataset(env):
    # Modify here if you already have D4RL dataset
    dataset = d4rl.qlearning_dataset(env)
    dataset["dones"] = dataset['terminals'].astype(np.float32)
    return dataset


class HumanPrefDataset(object):
    def __init__(self,
                 observations_1, actions_1, rewards_1, timestep_1, mask_1,
                 observations_2, actions_2, rewards_2, timestep_2, mask_2,
                 labels,
                 len_query,
                 ):
        self.observations_1=observations_1
        self.actions_1=actions_1
        self.rewards_1=rewards_1
        self.timestep_1=timestep_1
        self.mask_1=mask_1
        
        self.observations_2=observations_2
        self.actions_2=actions_2
        self.rewards_2=rewards_2
        self.timestep_2=timestep_2
        self.mask_2=mask_2

        self.labels=labels
        self.len_query=len_query

        self.size = len(self.labels)
        self.rng = np.random.default_rng()

    def sample(self, batch_size: Optional[int] = None, indices: Optional[List[int]] = None):
        assert (batch_size is None) ^ (indices is None)
        if batch_size is None:
            batch_size = len(indices)
        if indices is None:
            indices = self.rng.choice(self.size, size=batch_size, replace=False)
        seq_indices_1 = self.sample_seq_indices(self.mask_1[indices])
        seq_indices_2 = self.sample_seq_indices(self.mask_2[indices])
        indices_res = indices[:, None]
        timestep = np.tile(np.arange(1, self.len_query+1), (batch_size, 1))
        return dict(
            observations_1=self.observations_1[indices_res, seq_indices_1],
            actions_1=self.actions_1[indices_res, seq_indices_1],
            timestep_1=timestep,
            observations_2=self.observations_2[indices_res, seq_indices_2],
            actions_2=self.actions_2[indices_res, seq_indices_2],
            timestep_2=timestep,
            labels=self.labels[indices],
        )

    def sample_seq_indices(self, mask):
        len_valid = mask.sum(axis=1)
        max_start = len_valid - self.len_query
        start_indices = np.random.randint(max_start+1)
        return np.stack([start_indices+i for i in range(self.len_query)], axis=1)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: Optional[gym.Env] = None,
                 dataset: Optional[Dict] = None,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 ):
        if dataset is None:
            dataset=get_d4rl_dataset(env)
        self.dataset=dataset

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-5 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


class SeqD4RLDataset(D4RLDataset):
    def __init__(self, seq_len=100, min_seq_len=0, in_indices=[], **kwargs):
        self.seq_len = seq_len
        self.min_seq_len = min_seq_len
        self.in_indices = in_indices
        self.rng = np.random.default_rng()
        super().__init__(**kwargs)

        # build sequence indices
        ending_points = np.where(self.dones_float > 0)[0]
        ending_points = np.concatenate([[-1], ending_points])
        self.traj_lens = np.diff(ending_points)
        self.traj_starting_points = ending_points[:-1] + 1

        self.traj_num = len(self.traj_lens)

        seq_indices = []
        seq_traj_indices = []
        seq_in_indices = []
        seq_indices_cutting_points = [0]
        traj_returns = []
        traj_complete = []
        last_done = -1
        for i, done in tqdm(enumerate(self.dones_float), total=len(self.dones_float), desc="calc seq indices"):
            seq_start = max(last_done+1, i-seq_len+1)
            seq_end = i+1
            if done > 0:
                traj_complete.append(True if self.dataset["terminals"][i] else False)
                traj_returns.append(self.dataset["rewards"][last_done+1:i+1].sum())
                last_done = i
            if seq_end - seq_start < min_seq_len:
                continue
            seq_indices.append([seq_start, seq_end])
            if (seq_start in self.in_indices) and ((seq_end-seq_start) == seq_len):
                seq_in_indices.append(len(seq_indices)-1)
            seq_traj_indices.append(len(seq_indices_cutting_points)-1)
            if done > 0:
                seq_indices_cutting_points.append(len(seq_indices))

        self.seq_indices = np.array(seq_indices)
        self.seq_traj_indices = np.array(seq_traj_indices)
        self.seq_in_indices = np.array(seq_in_indices)
        self.seq_size = len(self.seq_indices)
        self.seq_indices_starting_points = np.array(seq_indices_cutting_points[:-1])
        self.seq_indices_ending_points = np.array(seq_indices_cutting_points[1:])
        self.traj_complete = np.array(traj_complete)
        self.traj_returns = np.array(traj_returns)

        # build sequences
        # Note: self.seq_masks has different meaning from self.masks!
        # self.masks: environment termination mask
        # self.seq_masks: attention mask
        self.seq_observations = np.zeros((self.seq_size, self.seq_len, self.observations.shape[1]), np.float32)
        self.seq_actions = np.zeros((self.seq_size, self.seq_len, self.actions.shape[1]), np.float32)
        self.seq_rewards = np.zeros((self.seq_size, self.seq_len), np.float32) # to record true rewards
        self.seq_masks = np.zeros((self.seq_size, self.seq_len), np.float32)
        self.seq_timesteps = np.zeros((self.seq_size, self.seq_len), np.int32)

        for i in tqdm(range(self.seq_size), total=self.seq_size, desc="build seq data"):
            seq_start, seq_end = self.seq_indices[i]
            seq_len_i = seq_end - seq_start
            self.seq_observations[i, :seq_len_i, :] = self.observations[seq_start:seq_end, :]
            self.seq_actions[i, :seq_len_i, :] = self.actions[seq_start:seq_end, :]
            self.seq_rewards[i, :seq_len_i] = self.rewards[seq_start:seq_end]
            self.seq_masks[i, :seq_len_i] = 1
            timestep_start = 1
            timestep_end = timestep_start + seq_len_i
            self.seq_timesteps[i, :seq_len_i] = np.arange(timestep_start, timestep_end, dtype=np.int32)

    def sample(self, batch_size) -> Batch:
        if batch_size < 0:
            batch_size = self.traj_num
        else:
            batch_size = min(self.traj_num, batch_size)

        indx = self.rng.choice(self.seq_size, size=batch_size, replace=False)

        return dict(
            observations=self.seq_observations[indx],
            actions=self.seq_actions[indx],
            rewards=self.seq_rewards[indx],
            masks=self.seq_masks[indx],
            timestep=self.seq_timesteps[indx],
        )


class DoubleSeqD4RLDataset(SeqD4RLDataset):
    def __init__(self, smooth_sigma, smooth_in, **kwargs):
        self.smooth_sigma = smooth_sigma
        self.smooth_in = smooth_in
        super().__init__(**kwargs)

        seq_indx_low, seq_indx_high = [], []
        for i, traj_idx in enumerate(self.seq_traj_indices):
            if i == 0:
                cur_low = 0
                cur_high = 0
                cur_traj_idx = traj_idx
            else:
                if cur_traj_idx == traj_idx:
                    cur_high = i
                else:
                    seq_indx_low += [cur_low] * (cur_high - cur_low + 1)
                    seq_indx_high += [cur_high] * (cur_high - cur_low + 1)
                    cur_traj_idx = traj_idx
                    cur_low, cur_high = i, i
        seq_indx_low += [cur_low] * (cur_high - cur_low + 1)
        seq_indx_high += [cur_high] * (cur_high - cur_low + 1)

        self.seq_indx_low = np.array(seq_indx_low)
        self.seq_indx_high = np.array(seq_indx_high)

    def disc_gaussian_noise(self, batch_size):
        return np.round(np.random.randn(batch_size) * self.smooth_sigma).astype(int)

    def sample(self, batch_size) -> Batch:
        assert batch_size % 2 == 0

        if self.smooth_in:
            in_batch_size = min(batch_size//2, len(self.seq_in_indices))
            out_batch_size = batch_size - in_batch_size
            in_indx_1 = self.rng.choice(self.seq_in_indices, size=in_batch_size, replace=False) 
            out_indx_1 = self.rng.choice(self.seq_size, size=out_batch_size, replace=False)
            indx_1 = np.concatenate([in_indx_1, out_indx_1])
        else:
            indx_1 = self.rng.choice(self.seq_size, size=batch_size, replace=False)

        indx_2 = indx_1 + self.disc_gaussian_noise(batch_size)
        indx_2 = np.clip(indx_2, self.seq_indx_low[indx_1], self.seq_indx_high[indx_1])

        return dict(
            observations_1=self.seq_observations[indx_1],
            actions_1=self.seq_actions[indx_1],
            rewards_1=self.seq_rewards[indx_1],
            masks_1=self.seq_masks[indx_1],
            timestep_1=self.seq_timesteps[indx_1],
            
            observations_2=self.seq_observations[indx_2],
            actions_2=self.seq_actions[indx_2],
            rewards_2=self.seq_rewards[indx_2],
            masks_2=self.seq_masks[indx_2],
            timestep_2=self.seq_timesteps[indx_2],
        )


class PrefD4RLDataset(SeqD4RLDataset):
    def __init__(self, reward_model=None, score_batch_size=1024, save_dataset=False, **kwargs):
        self.reward_model = reward_model
        self.score_batch_size = score_batch_size
        self.save_dataset = save_dataset
        super().__init__(**kwargs)

        # calculate scores
        self.seq_scores = np.zeros((self.seq_size, 1))
        if self.reward_model is None:
            # scripted (g.t.) score
            self.seq_scores[:] = self.seq_rewards.sum(axis=1).reshape(-1, 1)
        else:
            # estimated human score
            num_batches = int(np.ceil(self.seq_size / self.score_batch_size))
            for i in tqdm(range(num_batches), total=num_batches, desc="calc score"):
                batch_start = i * self.score_batch_size
                batch_end = min((i+1) * self.score_batch_size, self.seq_size)
                input = dict(
                    observations=self.seq_observations[batch_start:batch_end, :, :],
                    actions=self.seq_actions[batch_start:batch_end, :, :],
                    timestep=self.seq_timesteps[batch_start:batch_end, :],
                    attn_mask=self.seq_masks[batch_start:batch_end, :]
                )
                jax_input = batch_to_jax(input)
                score, _  = reward_model.get_score(jax_input)
                score = score.reshape(-1)
                score = np.asarray(list(score))
                self.seq_scores[batch_start:batch_end, :] = score.copy().reshape(-1, 1)
        
        del self.reward_model 
        
        if self.save_dataset:
            self.save_data()

    def sample(self, batch_size: int) -> Batch:
        if batch_size < 0:
            batch_size = self.traj_num
        else:
            max_batch_size = self.seq_size
            batch_size = min(max_batch_size, batch_size)

        indx = self.rng.choice(self.seq_size, size=batch_size, replace=False)

        scores = self.seq_scores[indx]

        return BatchOurs(observations=self.seq_observations[indx],
                     actions=self.seq_actions[indx],
                     rewards=self.seq_rewards[indx],
                     scores=scores,
                     masks=self.seq_masks[indx],
                     )

    # to reduce dataset generation time when debugging 
    def save_data(self, path="temp.pkl"):
        data = dict(
            seq_indices=self.seq_indices,
            seq_size=self.seq_size,
            seq_observations=self.seq_observations,
            seq_actions=self.seq_actions,
            seq_rewards=self.seq_rewards,
            seq_masks=self.seq_masks,
            seq_timesteps=self.seq_timesteps,
            seq_scores=self.seq_scores,
            seq_indices_starting_points=self.seq_indices_starting_points,
            seq_indices_ending_points=self.seq_indices_ending_points,
            traj_num=self.traj_num,
            traj_returns=self.traj_returns,
            traj_complete=self.traj_complete,
        )
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def load_data(self, path="temp.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.seq_indices=data["seq_indices"]
        self.seq_size=data["seq_size"]
        self.seq_observations=data["seq_observations"]
        self.seq_actions=data["seq_actions"]
        self.seq_rewards=data["seq_rewards"]
        self.seq_masks=data["seq_masks"]
        self.seq_timesteps=data["seq_timesteps"]
        self.seq_scores=data["seq_scores"]
        self.seq_indices_starting_points=data["seq_indices_starting_points"]
        self.seq_indices_ending_points=data["seq_indices_ending_points"]
        self.traj_num=data["traj_num"]
        self.traj_returns=data["traj_returns"]
        self.traj_complete=data["traj_complete"]