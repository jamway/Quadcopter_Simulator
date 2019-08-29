import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.scale = 0.
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.scale = 0.
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state

        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        self.scale = np.mean(dx/(x+dx)) if (x+dx).any() != 0 else 0
        return self.state
