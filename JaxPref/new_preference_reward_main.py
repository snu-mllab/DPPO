import os
import pickle
from collections import defaultdict

import numpy as np

import transformers

import gym
import wrappers as wrappers

import absl.app
import absl.flags
from flaxmodels.flaxmodels.gpt2.trajectory_gpt2 import TransRewardModel

from .sampler import TrajSampler
from .jax_utils import batch_to_jax
import JaxPref.reward_transform as r_tf
from viskit.logging import logger, setup_logger
from .PrefTransformer import PrefTransformer
from .utils import Timer, define_flags_with_default, set_random_seed, get_user_flags, prefix_metrics, WandBLogger, save_pickle
from .dataset_utils import HumanPrefDataset, get_d4rl_dataset, DoubleSeqD4RLDataset

# Jax memory
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'

FLAGS_DEF = define_flags_with_default(
    env_name='halfcheetah-medium-v2',
    model_type='PrefTransformer',
    max_traj_length=1000,
    seed=42,
    data_seed=42,
    save_model=True,
    batch_size=256,

    activations='relu',
    activation_final='none',

    n_epochs=10000,
    eval_period=5,

    data_dir='./human_label',
    seq_len=100,
    min_seq_len=0,
    use_human_label=True,

    smooth_sigma=0.0,
    smooth_in=True,

    comment='base',

    transformer=PrefTransformer.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(_):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    
    FLAGS.logging.output_dir = os.path.join(FLAGS.logging.output_dir, "pref")

    FLAGS.logging.group = "".join([s[0] for j, s in enumerate(FLAGS.env_name.split("-")) if j <= 2])

    assert FLAGS.comment, "You must leave your comment for logging experiment."
    # automatic comment according to hyperparameters
    comment = FLAGS.comment
    if FLAGS.smooth_sigma > 0:
        comment += f"_sm{FLAGS.smooth_sigma:.1f}_{FLAGS.transformer.smooth_w:.1f}"
    FLAGS.logging.group += f"_{comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    save_dir = os.path.join(FLAGS.logging.output_dir, FLAGS.env_name,
                            str(FLAGS.model_type), comment, f"s{FLAGS.seed}")
    

    setup_logger(
        variant=variant,
        seed=FLAGS.seed,
        base_log_dir=save_dir,
        include_exp_prefix_sub_dir=False
    )

    FLAGS.logging.output_dir = save_dir
    wb_logger = WandBLogger(FLAGS.logging, variant=variant)

    set_random_seed(FLAGS.seed)

    gym_env = gym.make(FLAGS.env_name)
    eval_sampler = TrajSampler(gym_env.unwrapped, FLAGS.max_traj_length)
    dataset = get_d4rl_dataset(eval_sampler.env)

    # use fixed seed for collecting segments.
    set_random_seed(FLAGS.data_seed)

    print("load saved indices.")
    base_path = os.path.join(FLAGS.data_dir, FLAGS.env_name)
    human_indices_2_file, human_indices_1_file, human_labels_file = sorted(os.listdir(base_path))
    with open(os.path.join(base_path, human_indices_1_file), "rb") as fp:   # Unpickling
        human_indices = pickle.load(fp)
    with open(os.path.join(base_path, human_indices_2_file), "rb") as fp:   # Unpickling
        human_indices_2 = pickle.load(fp)
    with open(os.path.join(base_path, human_labels_file), "rb") as fp:   # Unpickling
        human_labels = pickle.load(fp)

    if not isinstance(human_indices, np.ndarray):
        human_indices = np.array(human_indices)
    if not isinstance(human_indices_2, np.ndarray):
        human_indices_2 = np.array(human_indices_2)
    if not isinstance(human_labels, np.ndarray):
        human_labels = np.array(human_labels)

    pref_dataset = r_tf.load_queries_with_indices(
        dataset, FLAGS.seq_len,
        saved_indices=[human_indices, human_indices_2],
        saved_labels=human_labels, scripted_teacher=not FLAGS.use_human_label)

    pref_dataset = HumanPrefDataset(len_query=FLAGS.seq_len,
                                    **pref_dataset)
    
    ood_dataset = DoubleSeqD4RLDataset(
        dataset=dataset,
        seq_len=FLAGS.seq_len,
        min_seq_len=FLAGS.min_seq_len,
        smooth_sigma=FLAGS.smooth_sigma,
        smooth_in=FLAGS.smooth_in,
        in_indices=np.concatenate([human_indices, human_indices_2]),
    )

    set_random_seed(FLAGS.seed)
    observation_dim = gym_env.observation_space.shape[0]
    action_dim = gym_env.action_space.shape[0]

    train_data_size = pref_dataset.size
    interval = int(train_data_size / FLAGS.batch_size) + 1

    if FLAGS.model_type == "PrefTransformer":
        total_epochs = FLAGS.n_epochs
        config = transformers.GPT2Config(
            **FLAGS.transformer
        )
        config.warmup_steps = int(total_epochs * 0.1 * interval)
        config.total_steps = total_epochs * interval

        trans = TransRewardModel(config=config, observation_dim=observation_dim,
                                 action_dim=action_dim, activation=FLAGS.activations,
                                 activation_final=FLAGS.activation_final)
        reward_model = PrefTransformer(config, trans)
    else:
        raise NotImplementedError(FLAGS.model_type)

    if FLAGS.model_type == "PrefTransformer":
        train_loss = "reward/total_loss"

    criteria_key = None
    for epoch in range(FLAGS.n_epochs + 1):
        metrics = defaultdict(list)
        metrics['epoch'] = epoch
        if epoch:
            # train phase
            shuffled_idx = np.random.permutation(train_data_size)
            for i in range(interval):
                start_pt = i * FLAGS.batch_size
                end_pt = min((i + 1) * FLAGS.batch_size, train_data_size)
                with Timer() as train_timer:
                    # train
                    batch_id = batch_to_jax(pref_dataset.sample(indices=shuffled_idx[start_pt:end_pt]))
                    batch_ood = batch_to_jax(ood_dataset.sample(batch_size=(end_pt-start_pt)))
                    for key, val in prefix_metrics(reward_model.train(batch_id, batch_ood), 'reward').items():
                        metrics[key].append(val)
            metrics['train_time'] = train_timer()
        else:
            # for using early stopping with train loss.
            metrics[train_loss] = [float(FLAGS.seq_len)]

        for key, val in metrics.items():
            if isinstance(val, list):
                metrics[key] = np.mean(val)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        wb_logger.log(metrics, step=epoch)

    if FLAGS.save_model:
        save_data = {'reward_model': reward_model, 'variant': variant, 'epoch': epoch}
        save_pickle(save_data, 'model.pkl', save_dir)


if __name__ == '__main__':
    absl.app.run(main)
