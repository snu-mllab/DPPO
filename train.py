import datetime
import os
import pickle
from typing import Tuple

import gym
import numpy as np
import absl

import wrappers
from evaluation import evaluate
from learner import Learner

from viskit.logging import logger, setup_logger
from JaxPref.utils import WandBLogger, define_flags_with_default, get_user_flags, \
    set_random_seed, Timer, prefix_metrics

from JaxPref.dataset_utils import PrefD4RLDataset

from JaxPref.PrefTransformer import PrefTransformer

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'

FLAGS_DEF = define_flags_with_default(
    env_name='halfcheetah-medium-v2',
    seed=42,
    tqdm=True,
    eval_episodes=10,
    log_interval=1000,
    eval_interval=5000,
    batch_size=256,
    max_steps=int(1e6),
    model_type="PrefTransformer",
    comment="base",
    seq_len=100,
    min_seq_len=0,
    dropout=0.0,

    lambd=1.0,
    dist_temperature=0.1,
    logging=WandBLogger.get_default_config(),

    # params for loading preference transformer
    ckpt_base_dir="./logs/pref",
    ckpt_type="last",
    pref_comment="base",
    transformer=PrefTransformer.get_default_config(),
    smooth_sigma=0.0,
    smooth_in=True,
)
    
FLAGS = absl.flags.FLAGS


def initialize_model(pref_comment):
    ckpt_dir = os.path.join(FLAGS.ckpt_base_dir, FLAGS.env_name, FLAGS.model_type, pref_comment, f"s{FLAGS.seed}")
    if FLAGS.ckpt_type == "best":
        model_path = os.path.join(ckpt_dir, "best_model.pkl")
    elif FLAGS.ckpt_type == "last":
        model_path = os.path.join(ckpt_dir, "model.pkl")
    else:
        raise NotImplementedError

    print("Loading score model from", model_path)
    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    reward_model = ckpt['reward_model']
    return reward_model

def make_env_and_dataset(env_name: str,
                         seed: int,
                         pref_comment: str,
                         ) -> Tuple[gym.Env, PrefD4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    reward_model = initialize_model(pref_comment)

    dataset = PrefD4RLDataset(
        env=env,
        seq_len=FLAGS.seq_len,
        min_seq_len=FLAGS.min_seq_len,
        reward_model=reward_model,
    )

    return env, dataset


def main(_):
    VARIANT = get_user_flags(FLAGS, FLAGS_DEF)

    FLAGS.logging.output_dir = os.path.join(FLAGS.logging.output_dir, "policy")

    FLAGS.logging.group = "".join([s[0] for j, s in enumerate(FLAGS.env_name.split("-")) if j <= 2])

    pref_comment = FLAGS.pref_comment
    if FLAGS.smooth_sigma > 0:
        pref_comment += f"_sm{FLAGS.smooth_sigma:.1f}_{FLAGS.transformer.smooth_w:.1f}"

    comment = FLAGS.comment
    comment += f"_lam{FLAGS.lambd:.2f}"
    if FLAGS.dropout > 0:
        comment += f"_do{FLAGS.dropout:.1f}"

    comment = "_".join([pref_comment, comment])

    FLAGS.logging.group += f"_{comment}"
    FLAGS.logging.experiment_id = FLAGS.logging.group + f"_s{FLAGS.seed}"
    
    save_dir = os.path.join(FLAGS.logging.output_dir, FLAGS.env_name,
                            FLAGS.model_type, comment, f"s{FLAGS.seed}")

    setup_logger(
        variant=VARIANT,
        seed=FLAGS.seed,
        base_log_dir=save_dir,
        include_exp_prefix_sub_dir=False
    )

    FLAGS.logging.output_dir = save_dir
    wb_logger = WandBLogger(FLAGS.logging, variant=VARIANT)

    set_random_seed(int(FLAGS.seed))

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed, pref_comment)

    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    lambd=FLAGS.lambd,
                    dist_temperature=FLAGS.dist_temperature,
                    dropout_rate=FLAGS.dropout if (FLAGS.dropout > 0) else None,
                    )

    for i in range(FLAGS.max_steps + 1):
        metrics = dict()
        metrics["step"] = i
        with Timer() as timer:
            batch = dataset.sample(FLAGS.batch_size)
            train_info = prefix_metrics(agent.update(batch), 'train')

            if i % FLAGS.log_interval == 0:
                for k, v in train_info.items():
                    metrics[k] = v

            if i % FLAGS.eval_interval == 0:
                eval_info = prefix_metrics(evaluate(agent, env, FLAGS.eval_episodes), 'eval')

                for k, v in eval_info.items():
                    metrics[k] = v
        
        if len(metrics) > 1: # has something to log
            metrics["time"] = timer()
            logger.record_dict(metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=True)
            wb_logger.log(metrics, step=i)

    # save model
    agent.actor.save(os.path.join(save_dir, "model.pkl"))


if __name__ == '__main__':
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    absl.app.run(main)
