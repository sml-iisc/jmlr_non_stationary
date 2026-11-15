import numpy as np
import random


# TODO:
#   Separate alphas and betas for intensities and events means
#   Separate lambdas for intensities and events means

# Class that simulates random temporal events on the nodes and edges of a graph
class EventProcess:
    def __init__(self, params: dict={}, debug=False):
        self.params = params
        self.debug = debug

        self.event_dim = params.get('event_dim', 2)
        self.time_horizon = params.get('time_horizon', 1000)

        self.gravity_centers = params.get('gravity_centers')

        base_intensity = params.get('base_intensity', 0.1)
        self.base_intensity = np.ones((self.gravity_centers,)) * base_intensity
        self.intensity = np.ones((self.gravity_centers,)) * base_intensity

        self.alpha_coef = params.get('alpha_coef', 1)
        self.alpha = self.alpha_coef / (1 + np.array(range(1, self.time_horizon + 1)) ** 2)

        self.beta_coef = params.get('beta_coef', 1)
        self.beta = self.beta_coef / (1 + np.array(range(1, self.time_horizon + 1)) ** 2)

        self.lamda_intensity = params.get('lamda_intensity', 0.02)
        self.lamda_means = params.get('lamda_means', 0.2)

    def reset(self):
        self.event_history = np.zeros((1, self.gravity_centers, self.event_dim))
        self.event = np.zeros((self.gravity_centers, self.event_dim))
        self.time_step = 0
        self.intensity_history = np.zeros((1, self.gravity_centers))
        self.intensity_history[0] = 0
        self.jumped = np.zeros((self.gravity_centers,), dtype=np.bool_)
        self.jumped_history = np.zeros((1, self.gravity_centers), dtype=np.bool_)

    def step(self):
        self.time_step += 1

        # Update each intensity based on its own jump history

        if self.time_step == 1:
            self.intensity = self.base_intensity
        else:
            self.intensity = self.base_intensity + (self.alpha[1:self.time_step][::-1].reshape(1, -1) @ self.jumped_history[1:self.time_step])[0]


        # Add intensity due to neighbors' previous jumps
        # nb_intensity = np.zeros_like(self.intensity)
        # for idx in range(self.gravity_centers):
        #     for neighbor_idx in self.event_graph_adj_list[idx]:
        #         if self.jumped_history[self.time_step - 1][neighbor_idx]:
        #             # nb_intensity[idx] += 1
        #             self.intensity[idx] += self.lamda_intensity
        # self.intensity = (1 - self.lamda) * self.intensity + self.lamda * nb_intensity

        # nb_intensity = np.zeros_like(self.intensity)
        # Regress intensity towards neighbors previous jumps as a convex combination
        # for idx in range(self.graph.num_nodes + self.graph.num_edges):
        #     for neighbor_idx in self.event_graph_adj_list[idx]:
        #         if self.jumped_history[self.time_step - 1][neighbor_idx]:
        #             nb_intensity[idx] += self.intensity[neighbor_idx]
        # self.intensity = (1 - self.lamda) * self.intensity + self.lamda * nb_intensity

        # Each element jumps with Bernoulli(intensity)
        self.jumped = np.random.binomial(1, self.intensity)

        # Sample standard Gaussian event
        # self.event = np.random.normal(0, 1, size=(self.graph.num_nodes + self.graph.num_edges, self.event_dim))

        # Sample Gaussian event with mean based on history
        self.means = np.zeros((self.gravity_centers, self.event_dim))

        for t in range(1, self.time_step):
            self.means += self.beta[self.time_step - t] * self.jumped_history[t, None].reshape(-1, 1) * self.event_history[t]


        # Convexly add mean due to neighbors' previous jumps
        # nb_means = np.zeros_like(self.means)
        # for idx in range(self.graph.num_nodes + self.graph.num_edges):
        #     for neighbor_idx in self.event_graph_adj_list[idx]:
        #         if self.jumped_history[self.time_step - 1][neighbor_idx]:
        #             nb_means[idx] += self.event_history[self.time_step - 1, neighbor_idx]
        # self.means = (1 - self.lamda) * self.means + self.lamda * nb_means

        # No covariances
        self.event = np.random.normal(self.means, 1, size=None) * self.jumped[:, None]

        # Add event due to neighbors' previous jumps
        # for idx in range(self.graph.num_nodes + self.graph.num_edges):
        #     for neighbor_idx in self.event_graph_adj_list[idx]:
        #         if self.jumped_history[self.time_step - 1][neighbor_idx]:
        #             self.event[idx] += self.lamda_means * self.event_history[self.time_step - 1, neighbor_idx]

        # Add intensity, jumped and event to history
        self.intensity_history = np.concatenate((self.intensity_history, np.expand_dims(self.intensity, axis=0)), axis=0)
        self.jumped_history = np.concatenate((self.jumped_history, np.expand_dims(self.jumped, axis=0)), axis=0)
        self.event_history = np.concatenate((self.event_history, np.expand_dims(self.event, axis=0)), axis=0)

        if self.debug and self.time_step % 100 == 0:
            print('Time step: ', self.time_step)
            print('Number of events: ', np.sum(self.jumped))
            print('Cumulative event frequency: ', np.mean(self.jumped_history, axis=0))
            # print('Events: ', self.event)
            print('Largest event size ', np.max((self.event ** 2).sum(axis=1) ** 0.5))

    def init_history(self, history):
        self.time_step = history.shape[0] - 1
        self.event_history = np.copy(history)
        self.intensity_history = np.zeros((self.time_step + 1, self.gravity_centers))

        # Jumped history is 1 (non-zero) where event is non-zero
        self.jumped_history = (np.sum(np.abs(self.event_history) > 1e-6, axis=-1)).astype(np.bool_)
