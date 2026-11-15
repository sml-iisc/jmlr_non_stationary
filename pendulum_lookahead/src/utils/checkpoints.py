"""Load the archived actor after moving its network definition into src/models."""
import pickle
import types


class ActorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "net":
            module = "src.models.networks"

        return super().find_class(module, name)


def load_actor(path, device="cpu"):
    import torch

    checkpoint_pickle = types.ModuleType("checkpoint_pickle")
    checkpoint_pickle.Unpickler = ActorUnpickler
    checkpoint_pickle.load = pickle.load
    checkpoint_pickle.dump = pickle.dump
    actor = torch.load(path, map_location=device, pickle_module=checkpoint_pickle, weights_only=False)
    actor.to(device)

    return actor
