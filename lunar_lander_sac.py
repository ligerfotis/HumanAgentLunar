import csv
from datetime import timedelta
import time
import gym
import numpy as np
import torch
from sac_agent import Agent
from utils import plot_learning_curve, plot_actions, get_config, get_plot_and_chkpt_dir, reward_function, plot


def main():
    # get configuration
    config = get_config()
    # creating environment
    env = gym.make(config['game']['env_name'])
    if not config['SAC']['use_custom_reward']:
        print("Reward Threshold: " + str(env.spec.reward_threshold))

    random_seed = None
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    chkpt_dir, plot_dir, timestamp = get_plot_and_chkpt_dir(config['game']['load_checkpoint'],
                                                            config['game']['checkpoint_name'])
    # sac = Agent(input_dims=env.observation_space.shape, env=env,
    #             n_actions=env.action_space.shape[0], max_size=buffer_memory_size, gamma=gamma, tau=tau,
    #             update_interval=learn_every_n_steps, layer1_size=layer1_size, layer2_size=layer2_size,
    #             batch_size=batch_size, reward_scale=reward_scale, chkpt_dir=chkpt_dir)
    sac = Agent(config=config, env=env, input_dims=env.observation_space.shape, n_actions=env.action_space.shape[0],
                chkpt_dir=chkpt_dir)

    # best_score = env.reward_range[0]
    best_score = -100
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    total_steps = 0

    training_epochs_per_update = 128
    action_history = []
    score_history = []
    episode_duration_list = []
    length_list = []
    info = {}
    if config['game']['load_checkpoint']:
        sac.load_models()
        # env.render(mode='human')

    max_episodes = config['Experiment']['max_episodes']
    max_timesteps = config['Experiment']['max_timesteps']
    start_experiment = time.time()
    # training loop
    for i_episode in range(1, max_episodes + 1):
        observation = env.reset()
        timedout = False
        episode_reward = 0
        start = time.time()
        for timestep in range(max_timesteps):
            total_steps += 1

            # if total_steps < start_training_step:  # Pure exploration
            #     action = random.randint(0, action_dim - 1)
            # else:  # Explore with actions_prob
            #     action = sac.choose_action(observation)
            action = sac.choose_action(observation)
            """
            Add the human part here
            """
            action_history.append(action)
            observation_, reward, done, info = env.step(action)

            if timestep == max_timesteps:
                timedout = True

            if config['SAC']['use_custom_reward']:
                reward, done = reward_function(env, observation_, timedout)
            sac.remember(observation, action, reward, observation_, done)
            if not config['game']['test_model']:
                sac.learn()
            observation = observation_

            # clipped reward in [-1.0, 1.0]
            # clipped_reward = max(min(reward, 1.0), -1.0)
            # clipped_reward = reward
            if config['game']['render']:
                env.render()

            n_epoch_every_update = config['Experiment']['n_epoch_every_update']
            if total_steps >= config['Experiment']['start_training_step'] and total_steps % sac.update_interval == 0:
                for e in range(n_epoch_every_update):
                    sac.learn()
                    # sac.soft_update_target()

            # if total_steps >= start_training_step and total_steps % sac.target_update_interval == 0:
            #     sac.soft_update_target()

            running_reward += reward
            episode_reward += reward
            # if render:
            #     env.render()
            if done:
                break

        end = time.time()
        episode_duration = end - start
        episode_duration_list.append(episode_duration)
        score_history.append(episode_reward)

        avg_ep_duration = np.mean(episode_duration_list[-100:])
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not config['game']['test_model']:
                sac.save_models()
        # for e in range(training_epochs_per_update):
        #     sac.learn()
        #     sac.soft_update_target()
        length_list.append(timestep)
        avg_length += timestep

        # logging
        log_interval = config['Experiment']['log_interval']
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t Total reward(last {} episodes): {} \t Best Score: {} \t avg '
                  'episode duration: {}'.format(i_episode, avg_length, log_interval, running_reward, best_score,
                                                timedelta(seconds=avg_ep_duration)))
            running_reward = 0
            avg_length = 0
    end_experiment = time.time()
    experiment_duration = timedelta(seconds=end_experiment - start_experiment)
    info['experiment_duration'] = experiment_duration
    print('Total Experiment time: {}'.format(experiment_duration))

    if not config['game']['test_model']:
        x = [i + 1 for i in range(len(score_history))]
        np.savetxt('tmp/sac_' + timestamp + '/scores.csv', np.asarray(score_history), delimiter=',')

        actions = np.hsplit(np.asarray(action_history), 2)
        action_main = actions[0].flatten()
        action_side = actions[1].flatten()
        x_actions = [i + 1 for i in range(len(action_side))]
        # Save logs in files
        np.savetxt('tmp/sac_' + timestamp + '/action_main.csv', action_main, delimiter=',')
        np.savetxt('tmp/sac_' + timestamp + '/action_side.csv', action_side, delimiter=',')
        np.savetxt('tmp/sac_' + timestamp + '/epidode_durations.csv', np.asarray(episode_duration_list), delimiter=',')
        np.savetxt('tmp/sac_' + timestamp + '/avg_length_list.csv', np.asarray(length_list), delimiter=',')
        w = csv.writer(open('tmp/sac_' + timestamp + '/rest_info.csv', "w"))
        for key, val in info.items():
            w.writerow([key, val])

        plot_learning_curve(x, score_history, plot_dir + "/scores.png")
        # plot_actions(x_actions, action_main, plot_dir + "/action_main.png")
        # plot_actions(x_actions, action_side, plot_dir + "/action_side.png")
        plot(length_list, plot_dir + "/length_list.png", x=[i + 1 for i in range(max_episodes)])
        plot(episode_duration_list, plot_dir + "/epidode_durations.png", x=[i + 1 for i in range(max_episodes)])


if __name__ == '__main__':
    main()
    exit(0)
