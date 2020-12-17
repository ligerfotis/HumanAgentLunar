# python-sudo.sh

import sys, gym, time
from pynput.keyboard import Listener
import keyboard

#
# Action space Size:2 | (Main Engine, Side Engines):= ( [-1, 1], [-1, 1])
#

env = gym.make('LunarLanderContinuous-v2' if len(sys.argv) < 2 else sys.argv[1])

ACTIONS = env.action_space.shape
SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you
# can test what skip is still usable.

human_agent_action = 0
sac_agent_action = 0
human_wants_restart = False
human_sets_pause = False

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high
print("State dim: " + str(state_dim))
print("Action Dim: " + str(action_dim))
print("Reward Threshold: " + str(env.spec.reward_threshold))
print("Action Bound: " + str(action_bound))

a = 0


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, sac_agent_action
    if key == 0xff0d: human_wants_restart = True
    if key == 32: human_sets_pause = not human_sets_pause
    # a = int(key - ord('0'))
    # if a <= 0 or a > 4: return
    if key not in [ord("w"), ord("s"), ord("a"), ord("d")]: return
    if key == ord("w"):
        # print("pressed w")
        human_agent_action = 1
    if key == ord("s"):
        # print("pressed s")
        human_agent_action = -1
    if key == ord("a"):
        # print("pressed a")
        sac_agent_action = -1
    if key == ord("d"):
        # print("pressed d")
        sac_agent_action = 1


def key_release(key, mod):
    global human_agent_action, sac_agent_action
    if key not in [ord("w"), ord("s"), ord("a"), ord("d")]: return

    if key == ord("w"):
        # print("released w")
        human_agent_action = 0
    if key == ord("s"):
        # print("released s")
        human_agent_action = 0
    if key == ord("a"):
        # print("released a")
        sac_agent_action = 0
    if key == ord("d"):
        # print("released d")
        sac_agent_action = 0


env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    a_x, a_y = [0, 0]
    while 1:
        if not skip:
            a_x, a_y = [human_agent_action, sac_agent_action]
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step([a_x, a_y])
        # if r != 0:
        # print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open == False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))


print("ACTIONS={}".format(ACTIONS))
print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    if window_still_open == False: break
