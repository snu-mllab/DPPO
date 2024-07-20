import os
import pickle

import gym
import imageio
import jax
import numpy as np
from absl import app, flags
from tqdm import tqdm, trange
from dm_control.mujoco import engine

import d4rl
import glfw
glfw.init()

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "antmaze-medium-diverse-v2", "Environment name.")
flags.DEFINE_string("save_dir", "./video/", "saving dir.")
flags.DEFINE_string("query_path", "./human_label/", "query path")
flags.DEFINE_integer("num_query", 1000, "number of query.")
flags.DEFINE_integer("query_len", 100, "length of each query.")
flags.DEFINE_integer("label_type", 1, "label type.")
flags.DEFINE_integer("seed", 3407, "seed for reproducibility.")

video_size = {"medium": (500, 500), "large": (600, 450)}


def set_seed(env, seed):
    np.random.seed(seed)
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)


def qlearning_mujoco_adroit_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    xy_ = []
    done_bef_ = []

    qpos_ = []
    qvel_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i]) or episode_step == env._max_episode_steps - 1
        xy = dataset["infos/qpos"][i][:2].astype(np.float32)

        qpos = dataset["infos/qpos"][i]
        qvel = dataset["infos/qvel"][i]

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
            next_final_timestep = dataset["timeouts"][i + 1]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
            next_final_timestep = episode_step == env._max_episode_steps - 2

        done_bef = bool(next_final_timestep)

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        xy_.append(xy)
        done_bef_.append(done_bef)

        qpos_.append(qpos)
        qvel_.append(qvel)
        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
        "xys": np.array(xy_),
        "dones_bef": np.array(done_bef_),
        "qposes": np.array(qpos_),
        "qvels": np.array(qvel_),
    }

def qlearning_kitchen_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    xy_ = []
    done_bef_ = []

    qpos_ = []
    qvel_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i]) or episode_step == env._max_episode_steps - 1

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
            next_final_timestep = dataset["timeouts"][i + 1]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
            next_final_timestep = episode_step == env._max_episode_steps - 2

        done_bef = bool(next_final_timestep)

        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        done_bef_.append(done_bef)

        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
        "dones_bef": np.array(done_bef_),
    }

def visualize_query_mujoco_adroit(gym_env, dataset, start_indices, query_len, num_query, camera_name, width=500, height=500, save_dir="./video", verbose=False):
    save_dir = os.path.join(save_dir, gym_env.spec.id)
    os.makedirs(save_dir, exist_ok=True)

    for seg_idx in trange(num_query):
        start_1, start_2 = (
            start_indices[0][seg_idx],
            start_indices[1][seg_idx],
        )
        frames = []
        frames_2 = []

        idx1 = range(start_1, start_1 + query_len)
        idx2 = range(start_2, start_2 + query_len)

        gym_env.reset()

        if verbose:
            print(f"start pos of first one: {dataset['qposes'][idx1[0]][:2]}")
            print("=" * 50)
            print(f"start pos of second one: {dataset['qposes'][idx2[0]][:2]}")

        for t in trange(query_len, leave=False):
            gym_env.set_state(dataset["qposes"][idx1[t]], dataset["qvels"][idx1[t]])
            curr_frame = gym_env.sim.render(width=width, height=height, mode="offscreen", camera_name=camera_name)
            frames.append(np.flipud(curr_frame))
        gym_env.reset()
        for t in trange(query_len, leave=False):
            gym_env.set_state(
                dataset["qposes"][idx2[t]],
                dataset["qvels"][idx2[t]],
            )
            curr_frame = gym_env.sim.render(width=width, height=height, mode="offscreen", camera_name=camera_name)
            frames_2.append(np.flipud(curr_frame))

        video = np.concatenate((np.array(frames), np.array(frames_2)), axis=2)

        writer = imageio.get_writer(os.path.join(save_dir, f"./idx{seg_idx}.mp4"), fps=30)
        for frame in tqdm(video, leave=False):
            writer.append_data(frame)
        writer.close()

def visualize_query_kitchen(gym_env, dataset, start_indices, query_len, num_query, camera_name, width=500, height=500, save_dir="./video", verbose=False):
    save_dir = os.path.join(save_dir, gym_env.spec.id)
    os.makedirs(save_dir, exist_ok=True)

    for seg_idx in trange(num_query):
        start_1, start_2 = (
            start_indices[0][seg_idx],
            start_indices[1][seg_idx],
        )
        frames = []
        frames_2 = []

        idx1 = range(start_1, start_1 + query_len)
        idx2 = range(start_2, start_2 + query_len)

        gym_env.reset()

        if verbose:
            print(f"start pos of first one: {dataset['qposes'][idx1[0]][:2]}")
            print("=" * 50)
            print(f"start pos of second one: {dataset['qposes'][idx2[0]][:2]}")

        nq = gym_env.model.nq
        nv = gym_env.model.nv

        for t in trange(query_len, leave=False):
            gym_env.sim.set_state(dataset["observations"][idx1[t]][:-1])
            gym_env.sim.forward()
            
            camera = engine.MovableCamera(gym_env.sim, 1920, 2560)
            camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
            curr_frame = camera.render()
            curr_frame = np.flip(curr_frame, 0)
            frames.append(np.flipud(curr_frame))
        gym_env.reset()
        for t in trange(query_len, leave=False):
            gym_env.sim.set_state(dataset["observations"][idx2[t]][:-1])
            gym_env.sim.forward()
            
            camera = engine.MovableCamera(gym_env.sim, 1920, 2560)
            camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
            curr_frame = camera.render()
            curr_frame = np.flip(curr_frame, 0)
            frames_2.append(np.flipud(curr_frame))

        video = np.concatenate((np.array(frames), np.array(frames_2)), axis=2)

        writer = imageio.get_writer(os.path.join(save_dir, f"./idx{seg_idx}.mp4"), fps=30)
        for frame in tqdm(video, leave=False):
            writer.append_data(frame)
        writer.close()

def main(_):
    gym_env = gym.make(FLAGS.env_name)
    if "medium" in FLAGS.env_name:
        width, height = video_size["medium"]
    else:
        width, height = video_size["large"]
    set_seed(gym_env, FLAGS.seed)
    if "kitchen" in FLAGS.env_name:
        ds = qlearning_kitchen_dataset(gym_env)
    else:
        ds = qlearning_mujoco_adroit_dataset(gym_env)

    if "kitchen" in FLAGS.env_name:
        camera = None
    elif "pen" in FLAGS.env_name:
        camera = 'fixed'
    else: # mujoco
        camera = 'track'

    base_path = os.path.join(FLAGS.query_path, FLAGS.env_name)
    target_file = os.path.join(base_path, f"indices_num{FLAGS.num_query}")
    if not os.path.exists(target_file):
        cuts = np.where(ds['terminals'] == 1)[0]
        starts = np.concatenate([[0], cuts[:-1] + 1])
        ends = cuts
        print(starts[:10], ends[:10])
        lens = ends - starts
        traj_num = len(starts)
        human_indices, human_indices_2 = [], []

        for _ in range(FLAGS.num_query):
            while True:
                ti1 = np.random.choice(traj_num)
                if lens[ti1] >= FLAGS.query_len:
                    break
            i1 = np.random.randint(starts[ti1], ends[ti1] - FLAGS.query_len + 1)
            human_indices.append(i1)
            
            while True:
                ti2 = np.random.choice(traj_num)
                if lens[ti2] >= FLAGS.query_len:
                    break
            i2 = np.random.randint(starts[ti2], ends[ti2] - FLAGS.query_len + 1)
            human_indices_2.append(i2)

        base_path = os.path.join(FLAGS.query_path, FLAGS.env_name)
        os.makedirs(base_path, exist_ok=True)
        with open(os.path.join(base_path, f"indices_num{FLAGS.num_query}"), "wb") as fp:   # Unpickling
            pickle.dump(human_indices, fp)
        with open(os.path.join(base_path, f"indices2_num{FLAGS.num_query}"), "wb") as fp:   # Unpickling
            pickle.dump(human_indices_2, fp)
    with open(os.path.join(base_path, f"indices_num{FLAGS.num_query}"), "rb") as fp:   # Unpickling
        human_indices = pickle.load(fp)
    with open(os.path.join(base_path, f"indices2_num{FLAGS.num_query}"), "rb") as fp:   # Unpickling
        human_indices_2 = pickle.load(fp)

    if "kitchen" in FLAGS.env_name:
        visualize_query_kitchen(
            gym_env, ds, [human_indices, human_indices_2], FLAGS.query_len, FLAGS.num_query, camera, width=width, height=height, save_dir=FLAGS.save_dir
        )
    else:
        visualize_query_mujoco_adroit(
            gym_env, ds, [human_indices, human_indices_2], FLAGS.query_len, FLAGS.num_query, camera, width=width, height=height, save_dir=FLAGS.save_dir
        )


if __name__ == "__main__":
    app.run(main)
