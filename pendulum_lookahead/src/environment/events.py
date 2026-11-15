import numpy as np
import matplotlib.pyplot as plt

# Class that simulates random Gaussian temporal events
class EventProcess:
    def __init__(self, params: dict={'event_dim': 1, 'time_horizon': 15000, 'base_intensity': 0.1,
                                                        'alpha_coef': 1, 'beta_coef': 1, 'lamda_intensity': 0.02, 'lamda_means': 0.2,
                                                        'alpha_exp_coef': 1, 'beta_decay': 'poly', 'beta_exp_coef': 1,
                                                        'beta_poly_degree': 2, 'event_std': 1}, debug=False):

        self.params = params
        self.debug = debug

        self.event_dim = params.get('event_dim', 1)
        self.time_horizon = params.get('time_horizon', 1000)

        self.base_intensity = params.get('base_intensity', 0.1)

        self.alpha_coef = params.get('alpha_coef', 1)
        self.alpha_exp_coef = params.get('alpha_exp_coef', 1)
        # self.alpha = self.alpha_coef / (1 + np.array(range(1, self.time_horizon + 1)) ** 3)
        self.alpha = self.alpha_coef * np.exp(- self.alpha_exp_coef * np.array(range(1, self.time_horizon + 2)))

        self.beta_coef = params.get('beta_coef', 1)
        self.beta_decay = params.get('beta_decay', 'poly')
        self.beta_exp_coef = params.get('beta_exp_coef', 1)
        self.beta_poly_degree = params.get('beta_poly_degree', 2)

        if self.beta_decay == 'exp':
            self.beta = self.beta_coef * np.exp(- self.beta_exp_coef * np.array(range(1, self.time_horizon + 2)))
        else:
            self.beta = self.beta_coef / (1 + np.array(range(1, self.time_horizon + 2)) ** self.beta_poly_degree)


        self.lamda_intensity = params.get('lamda_intensity', 0.02)
        self.lamda_means = params.get('lamda_means', 0.2)

        self.event_std = params.get('event_std', 1)


    def reset(self):
        self.event_history = np.zeros((1))
        self.event = np.zeros((self.event_dim))
        self.time_step = 0
        self.intensity_history = np.zeros((1))
        self.intensity_history[0] = 0
        self.jumped = np.zeros((1), dtype=np.bool_)
        self.jumped_history = np.zeros((1), dtype=np.bool_)

        return self.event

    def step(self):
        self.time_step += 1

        # Update each intensity based on its own jump history

        if self.time_step == 1:
            self.intensity = np.array([self.base_intensity])
        else:
            self.intensity = self.base_intensity + (self.alpha[:self.time_step][::-1].reshape(1, -1) @ self.jumped_history[:self.time_step].reshape(-1, 1))[0]


        # Each element jumps with Bernoulli(intensity)
        self.jumped = np.random.binomial(1, self.intensity)

        # Sample Gaussian event with mean based on history
        self.means = np.zeros((self.event_dim))

        for t in range(1, self.time_step):
            self.means += self.beta[self.time_step - t - 1] * self.jumped_history[t] * self.event_history[t]

        # No covariances
        self.event = np.random.normal(self.means, self.event_std, size=None) * np.float32(self.jumped)

        # Add intensity, jumped and event to history
        self.intensity_history = np.concatenate((self.intensity_history, self.intensity), axis=0)
        self.jumped_history = np.concatenate((self.jumped_history, self.jumped), axis=0)
        self.event_history = np.concatenate((self.event_history, self.event), axis=0)

        if self.debug and self.time_step % 1 == 0:
            print('Time step: ', self.time_step)
            print('Intensity: ', self.intensity)
            print('Number of events: ', np.sum(self.jumped_history))
            print('Cumulative event frequency: ', np.mean(self.jumped_history, axis=0))
            print('Largest event size:', np.max(np.abs(self.event_history)))
            print('Average event size (counting only when event occurs):', np.mean(np.abs(self.event_history[self.jumped_history == 1])))

        return self.event
