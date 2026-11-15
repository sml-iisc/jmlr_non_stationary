import numpy as np
from src.environment.events import EventProcess as BaseEventProcess

# Class that simulates random Gaussian temporal events
class EventProcess(BaseEventProcess):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Initialize the process with a partially observed history
    def init_history(self, history):
        self.time_step = history.shape[0] - 1
        self.event_history = np.copy(history)
        self.intensity_history = np.zeros((self.time_step + 1))

        # Jumped history is 1 (non-zero) where event is non-zero
        self.jumped_history = (np.abs(self.event_history) > 1e-6).reshape(-1).astype(np.bool_)
