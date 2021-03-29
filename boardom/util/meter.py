class Average:
    """Keeps an average of values."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the average to 0."""
        self.count = 0
        self.value = 0

    def add(self, value, count=1):
        """Adds a new value

        Args:
            value: Value to be added
            count (optional): Number of summed values that make the value given.
                Can be used to register multiple (summed) values at once (default 1).
        """
        self.count += count
        self.value += value * count

    def get(self):
        """Returns the current average"""
        if self.count == 0:
            return None
        return self.value / self.count


# Welford algorithm for mean and variance
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
class MeanVar:
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.m2 = 0

    def add(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = x - self.mean
        self.m2 = self.m2 + delta * delta2

    @property
    def variance(self):
        return self.m2 / self.count

    def sample_variance(self):
        return self.m2 / (self.count - 1)
