import datetime
import os
from os.path import dirname, abspath
import pprint
import shutil
import time
import threading
from types import SimpleNamespace as SN
from os.path import dirname, abspath

import torch as th

from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.episode_buffer import Prioritized_ReplayBuffer
from components.transforms import OneHot
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from utils.general_reward_support import test_alg_config_supports_reward
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from components.episodic_memory_buffer import Episodic_memory_buffer

from ea import mod_neuro_evo as utils_ne

import numpy as np
import copy as cp
import random


def rl_to_evo(rl_agent, evo_net, index):
    for target_param, param in zip(evo_net.agent.parameters(), rl_agent.agent.parameters()):
        alpha = 0.1
        target_param.data.copy_(alpha * target_param.data + (1 - alpha) * param.data)


def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    assert test_alg_config_supports_reward(
        args
    ), "The specified algorithm does not support the general reward setup. Please choose a different algorithm or set `common_reward=True`."

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]
    unique_token = (
        f"{_config['name']}_seed{_config['seed']}_{map_name}_{datetime.datetime.now()}"
    )

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_logs_direc)

    if args.use_wandb:
        logger.setup_wandb(
            _config, args.wandb_team, args.wandb_project, args.wandb_mode
        )

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Finish logging
    logger.finish()

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def save_one_buffer(args, save_buffer, env_name, from_start=False):
    x_env_name = env_name
    if from_start:
        x_env_name += '_from_start/'
    path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.save_buffer_id) + '/'
    if os.path.exists(path_name):
        random_name = '../../buffer/' + x_env_name + '/buffer_' + str(random.randint(10, 1000)) + '/'
        os.rename(path_name, random_name)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    save_buffer.save(path_name)


def run_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    env_name = args.env
    if env_name == 'sc2':
        args.unit_dim = env_info["unit_dim"]
    else:
        args.unit_dim = 1
    args.obs_shape = env_info["obs_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    # For individual rewards in gymmai reward is of shape (1, n_agents)
    if args.common_reward:
        scheme["reward"] = {"vshape": (1,)}
    else:
        scheme["reward"] = {"vshape": (args.n_agents,)}
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    env_name = args.env
    if env_name == 'sc2':
        env_name += '/' + args.env_args['map_name']

    if args.is_prioritized_buffer:
        buffer = Prioritized_ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                                          args.prioritized_buffer_alpha,
                                          preprocess=preprocess,
                                          device="cpu" if args.buffer_cpu_only else args.device)
    else:
        buffer = ReplayBuffer(
            scheme,
            groups,
            args.buffer_size,
            env_info["episode_limit"] + 1,
            args.burn_in_period,
            preprocess=preprocess,
            device="cpu" if args.buffer_cpu_only else args.device,
        )

    if args.is_save_buffer:
        save_buffer = ReplayBuffer(scheme, groups, args.save_buffer_size, env_info["episode_limit"] + 1,
                                   args.burn_in_period,
                                   preprocess=preprocess,
                                   device="cpu" if args.buffer_cpu_only else args.device)

    if args.is_batch_rl:
        assert (args.is_save_buffer == False)
        x_env_name = env_name
        if args.is_from_start:
            x_env_name += '_from_start/'
        path_name = '../../buffer/' + x_env_name + '/buffer_' + str(args.load_buffer_id) + '/'
        assert (os.path.exists(path_name) == True)
        buffer.load(path_name)

    if getattr(args, "use_emdqn", False):
        ec_buffer = Episodic_memory_buffer(args, scheme)

    ###############################进化#######################################################
    evolver = utils_ne.SSNE(args)

    # Setup multiagent controller here
    if args.EA:
        pop = []
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
        for i in range(args.pop_size):
            pop.append(mac_REGISTRY[args.mac](buffer.scheme, groups, args))
    else:
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
        pop = []
    ###############################进化#######################################################
    fitness = []

    # Give runner the scheme
    if args.runner != 'offpolicy':
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    if args.learner == "fast_QLearner" or args.learner == "qplex_curiosity_vdn_learner" or args.learner == "EA_fast_QLearner" or args.learner == "EA_qplex_curiosity_vdn_learner":
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args, groups=groups)
    else:
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.runner == 'offpolicy':
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, test_mac=learner.extrinsic_mac)

    if hasattr(args, "save_buffer") and args.save_buffer:
        learner.buffer = buffer

    if args.checkpoint_path != "":
        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        if args.EA and runner.t_env > args.start_timesteps and episode % args.EA_freq == 0:
            fitness = []
            for ea_mac in pop:
                with th.no_grad():
                    episode_batch, episode_return, _ = runner.run(ea_mac, test_mode=False, EA=True,
                                                                  MultiObject=args.Pareto)
                fitness.append(episode_return)
                # print("EA ", episode_batch.batch_size)
                if getattr(args, "use_emdqn", False):
                    ec_buffer.update_ec(episode_batch)
                buffer.insert_episode_batch(episode_batch)

                if args.is_save_buffer:
                    save_buffer.insert_episode_batch(episode_batch)
                    if save_buffer.is_from_start and save_buffer.episodes_in_buffer == save_buffer.buffer_size:
                        save_buffer.is_from_start = False
                        save_one_buffer(args, save_buffer, env_name, from_start=True)
                        break
                    if save_buffer.buffer_index % args.save_buffer_interval == 0:
                        print('current episodes_in_buffer: ', save_buffer.episodes_in_buffer)

            with th.no_grad():
                episode_batch, _, _ = runner.run(mac, EA=False, test_mode=False)
            if getattr(args, "use_emdqn", False):
                ec_buffer.update_ec(episode_batch)
            buffer.insert_episode_batch(episode_batch)

            if args.is_save_buffer:
                save_buffer.insert_episode_batch(episode_batch)
                if save_buffer.is_from_start and save_buffer.episodes_in_buffer == save_buffer.buffer_size:
                    save_buffer.is_from_start = False
                    save_one_buffer(args, save_buffer, env_name, from_start=True)
                    break
                if save_buffer.buffer_index % args.save_buffer_interval == 0:
                    print('current episodes_in_buffer: ', save_buffer.episodes_in_buffer)

            print("EA evolution start ...")

            if args.SAME:
                elite_index = evolver.epoch(pop, fitness, 0, agent_level=True)
            else:
                for agent_index in range(args.n_agents):
                    elite_index = evolver.epoch(pop, fitness, agent_index, agent_level=True)
                # elite_index = evolver.epoch(pop, fitness, 0, agent_level=True)
            print("EA evolution end.")
        else:
            if args.Pareto:
                fitness = np.zeros((args.pop_size, 3))
            else:
                fitness = np.zeros(args.pop_size)
            elite_index = 0
            with th.no_grad():
                episode_batch, _, _ = runner.run(mac, test_mode=False, EA=False)
            if getattr(args, "use_emdqn", False):
                ec_buffer.update_ec(episode_batch)
            buffer.insert_episode_batch(episode_batch)

            if args.is_save_buffer:
                save_buffer.insert_episode_batch(episode_batch)
                if save_buffer.is_from_start and save_buffer.episodes_in_buffer == save_buffer.buffer_size:
                    save_buffer.is_from_start = False
                    save_one_buffer(args, save_buffer, env_name, from_start=True)
                    break
                if save_buffer.buffer_index % args.save_buffer_interval == 0:
                    print('current episodes_in_buffer: ', save_buffer.episodes_in_buffer)

        for _ in range(args.num_circle):
            if buffer.can_sample(args.batch_size):
                if args.is_prioritized_buffer:
                    sample_indices, episode_sample = buffer.sample(args.batch_size)
                else:
                    episode_sample = buffer.sample(args.batch_size)

                if args.is_batch_rl:
                    runner.t_env += int(
                        th.sum(episode_sample['filled']).cpu().clone().detach().numpy()) // args.batch_size

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                if args.EA:
                    all_teams = []
                    for index_n in range(args.pop_size):
                        all_teams.append(pop[index_n])
                    all_teams.append(mac)
                else:
                    all_teams=[]

                if args.is_prioritized_buffer:
                    if getattr(args, "use_emdqn", False):
                        td_error = learner.train(episode_sample, all_teams, runner.t_env,
                                                 episode,
                                                 ec_buffer=ec_buffer)
                    else:
                        td_error = learner.train(episode_sample, all_teams, runner.t_env,
                                                 episode)
                        buffer.update_priority(sample_indices, td_error)
                else:
                    if getattr(args, "use_emdqn", False):
                        td_error = learner.train(episode_sample, all_teams, runner.t_env,
                                                 episode,
                                                 ec_buffer=ec_buffer)
                    else:
                        if args.EA:
                            learner.train(episode_sample, all_teams, runner.t_env, episode)
                        else:
                            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                episode_sample, _, _ = runner.run(mac, test_mode=True)

            if args.EA and runner.t_env > args.start_timesteps and episode % args.EA_freq == 0:
                import copy
                # Replace any index different from the new elite
                if args.Pareto:
                    fitness = np.array(fitness)
                    replace_index = np.argmin(fitness[:, 0])
                else:
                    replace_index = np.argmin(fitness)
                if replace_index == elite_index:
                    replace_index = (replace_index + 1) % args.pop_size
                    while replace_index == elite_index:
                        replace_index = (replace_index + 1) % args.pop_size
                if replace_index != elite_index:
                    prev_state = copy.deepcopy(pop[replace_index].agent.state_dict())
                    for index in range(args.n_agents):
                        rl_to_evo(mac, pop[replace_index], index)
                # if evaluate_new_population(pop) < evaluate_previous_population(prev_state):
                #     pop[replace_index].agent.load_state_dict(prev_state)
                evolver.rl_policy = replace_index
                print('Sync from RL --> Nevo')

        if args.save_model and (
                runner.t_env - model_save_time >= args.save_model_interval
                or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

            if args.use_wandb and args.wandb_save_model:
                wandb_save_dir = os.path.join(
                    logger.wandb.dir, "models", args.unique_token, str(runner.t_env)
                )
                os.makedirs(wandb_save_dir, exist_ok=True)
                for f in os.listdir(save_path):
                    shutil.copyfile(
                        os.path.join(save_path, f), os.path.join(wandb_save_dir, f)
                    )

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
                                          config["test_nepisode"] // config["batch_size_run"]
                                  ) * config["batch_size_run"]

    return config
