import numpy as np

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

class FeatureExtractor:
    def __init__(self):
        self.num_features = 3

    def get_features(self, obs):
        return obs


class ThetaExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()
        self.num_features = 4

    def get_features(self, obs):
        features = np.concatenate([obs, [np.arccos(obs[0])]])
        return features

class ThetaPolynomialExtractor(FeatureExtractor):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        self.num_features = 3 + degree + 1

    def get_features(self, obs):
        features = np.array([np.arccos(obs[0])**i for i in range(self.degree + 1)])
        features = np.concatenate([obs, features])

        return features

class ThetaAndPolynomialExtractor(FeatureExtractor):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        self.num_features = 4 * degree

    def get_features(self, obs):
        obs = np.concatenate([obs, [np.arccos(obs[0])]])
        features = np.copy(obs)

        for i in range(2, self.degree + 1):
            features = np.concatenate([features, obs**i])

        return features

# Reward function for the pendulum that uses current (state, event) and action
def reward_fn(feature_event, action, env):

    th = feature_event[3]
    thdot = feature_event[2]

    num_features = env.unwrapped.feature_extractor.num_features
    event = feature_event[num_features]

    u = action + event * env.unwrapped.event_scale
    u = np.clip(u, -env.unwrapped.max_torque, env.unwrapped.max_torque)[0]

    costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

    return -costs
