import numpy as np
import matplotlib.pyplot as plt

class DiscreteHawkesProcess:
    def __init__(self):
        self.events = []
        self.delta = 0.1 # seconds
        self.baseline_intensity = 0.1
        self.alpha = 0.1
        self.decay = 1.0

        self.total_events = 0

    def step(self):
        if not self.events:
            intensity = self.baseline_intensity
        else:
            intensity = self.baseline_intensity
            current_time = self.delta * len(self.events)

            for idx, event in enumerate(self.events):
                if event != 0:
                    event_time = self.delta * (idx)

                    intensity += self.alpha * self.decay * np.exp(-self.decay * (current_time - event_time))

        probability = 1 - np.exp(-intensity * self.delta)
        event = np.random.binomial(1, probability)

        # if event == 1:
        #     print(f"Event occurred at time {len(self.events) * self.delta:.2f} seconds")

        self.total_events += event

        self.events.append(event)

        return event

    def simulate(self, num_events):
        while self.total_events < num_events:
            self.step()

        return self.events

    def plot_events(self):
        plt.figure(figsize=(10, 5))
        plt.scatter(np.arange(len(self.events))*self.delta, self.events, marker='o')
        plt.title('Discrete Hawkes Process Events')
        plt.xlabel('Time Step')
        plt.ylabel('Event Occurrence (0 or 1)')
        plt.grid()
        plt.show()

    def plot_intensity(self):
        times = np.arange(len(self.events)) * self.delta
        intensities = []

        for t in times:
            intensity = self.baseline_intensity

            for idx, event in enumerate(self.events):
                if event != 0:
                    event_time = self.delta * (idx + 1)

                    if event_time < t:
                        intensity += self.alpha * self.decay * np.exp(-self.decay * (t - event_time))

            intensities.append(intensity)

        plt.figure(figsize=(10, 5))
        plt.plot(times, intensities)
        plt.title('Discrete Hawkes Process Intensity')
        plt.xlabel('Time Step')
        plt.ylabel('Intensity')
        plt.grid()
        plt.show()

    def get_events(self):
        events = []

        for idx, event in enumerate(self.events):
            if event != 0:
                events.append(self.delta * (idx + 1))

        return events

    def reset(self):
        self.events = []
        self.total_events = 0
