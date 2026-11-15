import numpy as np

def get_fixed_length_vector(vector, length, default_element=0.0):
    """
    Get a fixed length vector from the end of a given numpy vector. If the given vector is shorter than the desired length, the missing
    elements are filled with the default element. If the given vector is longer than the desired length, it is truncated.
    :param vector: The given vector
    :param length: The desired length
    :param default_element: The default element to fill the missing elements
    :return: The fixed length vector
    """

    # assert len(vector.shape) == 1
    if vector.shape[0] >= length:
        return vector[-length:]
    else:
        return np.concatenate([np.full((length - vector.shape[0], *vector.shape[1:]), default_element), vector], axis=0)

# Function that returns a moving average of the input list
def moving_average(input_list, window_size):
    """
    Compute the moving average of a list.
    :param input_list: The input list
    :param window_size: The window size
    :return: The moving average list
    """
    cumsum = np.cumsum(input_list)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]

    return cumsum[window_size - 1:] / window_size

def reward_fn(state, action, env):
    distance = np.linalg.norm(state[...,:2] - state[...,4:6], axis=-1)
    out = np.where(distance<=0.45, 0.0, -0.01*np.exp(distance/10))

    # if distance <= 0.45:
    #     return 0.0
    return out
