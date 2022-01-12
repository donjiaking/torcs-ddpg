import numpy as np

# TODO: try other rewards here

def reward_function1(obs):
    """
    obs: current full-observation (raw)
    """
    v_x = obs['speedX']
    angle = obs['angle']
    trackPos = obs['trackPos']

    reward = v_x*np.cos(angle) - np.abs(v_x*np.sin(angle)) - np.abs(v_x*trackPos)

    return reward


def reward_function2(obs):
    """
    obs: current full-observation (raw)
    """
    v_x = obs['speedX']
    v_y =obs['speedY']
    angle = obs['angle']
    trackPos = obs['trackPos']

    reward = v_x*np.cos(angle) - np.abs(v_x*np.sin(angle)) \
            - 2*v_x*np.abs(trackPos*np.sin(angle)) - v_y*np.cos(angle)

    return reward


reward_func = reward_function1
