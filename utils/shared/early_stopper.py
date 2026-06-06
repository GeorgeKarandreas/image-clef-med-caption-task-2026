"""Early stopping utility."""

from copy import deepcopy


class EarlyStopper:
    """Stop training when a validation metric stops improving."""

    def __init__(self, patience=5, min_delta=0.0, mode="max"):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'.")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_state_dict = None
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, value, model=None):
        """Update early-stopping state from the latest validation metric."""
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            if model is not None:
                self.best_state_dict = deepcopy(model.state_dict())
            self.bad_epochs = 0
            self.should_stop = False
        else:
            self.bad_epochs += 1
            self.should_stop = self.bad_epochs >= self.patience

        if self.should_stop and model is not None:
            self.restore_best_weights(model)

        return self.should_stop

    def restore_best_weights(self, model):
        """Restore the best recorded model weights."""
        if self.best_state_dict is None:
            raise ValueError("No best weights have been recorded yet.")

        model.load_state_dict(self.best_state_dict)
