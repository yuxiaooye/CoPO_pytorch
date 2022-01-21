import sys
import os.path as osp
from copo.algo_copo.constants import *
from copo.algo_yyx_copo_tc.copo_tc import CoPOTorchTrainer
from copo.algo_svo.svo_env import get_svo_env
from copo.callbacks import MultiAgentDrivingCallbacks
from copo.ccenv import get_ccenv
from copo.train.train import train
from copo.train.utils import get_train_parser
from copo.utils import get_rllib_compatible_env
from metadrive.envs.marl_envs import MultiAgentRoundaboutEnv
from ray import tune
import os
import logging
import sys


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    args = get_train_parser().parse_args()
    exp_name = args.exp_name or "TEST"

    # Setup config
    # We set the stop criterion to 2M environmental steps! Since PPO in single OurEnvironment converges at around 20k steps.
    stop = int(100_0000)
    if args.debug:
        stop = 1

    config = dict(
        # ===== Environmental Setting =====
        # We can grid-search the environmental parameters!
        env=get_rllib_compatible_env(get_svo_env(get_ccenv(MultiAgentRoundaboutEnv), return_env_class=True)),
        env_config=dict(
            # start_seed=tune.grid_search([5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]),
            start_seed=5000,
            neighbours_distance=40,
        ),

        # ===== Resource =====
        # So we need 0.2(num_cpus_per_worker) * 5(num_workers) + 1(num_cpus_for_driver) = 2 CPUs per trial!
        num_gpus=0.5 if args.num_gpus != 0 else 0,
        num_cpus_per_worker=0.1,

        # ===== Meta SVO =====
        initial_svo_std=0.1,
        **{USE_DISTRIBUTIONAL_SVO: True},
        svo_lr=1e-4,
        svo_num_iters=5,
        use_global_value=True,
        **{USE_CENTRALIZED_CRITIC: False},
        num_workers=1,  # TODO uncomment when local mode
        framework="torch",  # in order to use PPOTorchPolicy instead of PPOTFPolicy

    )

    # Launch training
    train(
        CoPOTorchTrainer,
        exp_name=exp_name,
        keep_checkpoints_num=1,
        stop=stop,
        config=config,  # Don't call get_ippo_config here!
        num_gpus=args.num_gpus,
        num_seeds=1,
        # test_mode=True,
        custom_callback=MultiAgentDrivingCallbacks,
        # fail_fast='raise',
        local_mode=True  # TODO uncomment when local mode
    )
