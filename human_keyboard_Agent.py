import time
import pybullet_envs
import gym
import pybullet as p
import numpy as np

env = gym.make('InvertedPendulumBulletEnv-v0')
client = p.connect(p.GUI)
p.setGravity(0, 0, -10, physicsClientId=client)


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, sac_agent_action
    if key == 0xff0d: human_wants_restart = True
    if key == 32: human_sets_pause = not human_sets_pause
    # a = int(key - ord('0'))
    # if a <= 0 or a > 4: return
    if key not in [ord("w"), ord("s"), ord("a"), ord("d")]: return
    if key == ord("d"):
        human_agent_action = 1
    if key == ord("a"):
        human_agent_action = -1
    if key == ord("s"):
        sac_agent_action = -1
    if key == ord("w"):
        sac_agent_action = 1


def key_release(key, mod):
    global human_agent_action, sac_agent_action
    if key not in [ord("w"), ord("s"), ord("a"), ord("d")]: return

    if key == ord("a"):
        human_agent_action = 0
    if key == ord("d"):
        human_agent_action = 0
    if key == ord("w"):
        sac_agent_action = 0
    if key == ord("s"):
        sac_agent_action = 0


human_agent_action = 0
sac_agent_action = 0
human_wants_restart = False
human_sets_pause = False

ACTIONS = env.action_space.shape
SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you


# can test what skip is still usable.

def get_human_action():
    keys = p.getKeyboardEvents()
    for k, v in keys.items():

        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
            turn = -0.5
        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)):
            turn = 0
        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED)):
            turn = 0.5
        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)):
            turn = 0

        if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED)):
            forward = 1
        if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED)):
            forward = 0
        if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED)):
            forward = -1
        if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED)):
            forward = 0
    return 0


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    a = 0.0
    while 1:
        if not skip:
            a = get_human_action()
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1
        action = np.asarray([a])
        obser, r, done, info = env.step(action)
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
    if not window_still_open: break
