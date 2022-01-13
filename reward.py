import numpy as np

# TODO: try other rewards here

def reward_function1(obs):
    """
    obs: current full-observation (raw)
    """
    v_x = obs['speedX']
    angle = obs['angle']
    trackPos = obs['trackPos']

    # reward = v_x*np.cos(angle) - np.abs(v_x*np.sin(angle)) - np.abs(max(v_x,1)*trackPos)
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

    reward = 1*v_x*np.cos(angle) - np.abs(v_x*np.sin(angle)) \
            - 0.5*v_x*np.abs(trackPos)-0.5*trackPos

    return reward


reward_func = reward_function1
