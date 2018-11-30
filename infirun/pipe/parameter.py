import abc

import numpy as np

from infirun.util.reflect import ensure_basic_data_type

__all__ = ['Uniform', 'Normal', 'UniformInt', 'Choice', 'RandomParameter']


class RandomParameter(abc.ABC):
    def __init__(self):
        self._last_value = None

    @abc.abstractmethod
    def calculate_value(self):
        pass

    def value(self):
        v = self.calculate_value()
        self._last_value = v
        return v

    def last_value(self):
        return self._last_value

    def get_constructor_kwargs(self):
        return {k: ensure_basic_data_type(v)['value'] for k, v in self.__dict__.items()
                if k[0] != '_' and k not in ('calculate_value',
                                             'value',
                                             'last_value',
                                             'get_constructor_kwargs')}


class Uniform(RandomParameter):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def calculate_value(self):
        return np.random.uniform(self.low, self.high)


class Normal(RandomParameter):
    def __init__(self, mu, sigma, low_clip=None, high_clip=None):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.low_clip = low_clip
        self.high_clip = high_clip

    def calculate_value(self):
        while True:
            v = np.random.normal(self.mu, self.sigma)
            if self.low_clip is not None and v < self.low_clip:
                continue
            elif self.high_clip is not None and v > self.high_clip:
                continue
            else:
                break
        return v


class UniformInt(RandomParameter):
    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def calculate_value(self):
        return np.random.randint(low=self.low, high=self.high)


class Choice(RandomParameter):
    def __init__(self, choices, probs=None):
        super().__init__()
        self._orig_choices = choices
        self._orig_probs = probs
        if probs is None:
            probs = [1] * len(choices)
        _choices = []
        _probs = []
        for c, p in zip(choices, probs):
            if p == 0:
                continue
            _choices.append(c)
            _probs.append(p)
        _probs = np.asarray(_probs, dtype='float')
        _probs /= _probs.sum()
        self.choices = _choices
        self.probs = _probs

    def calculate_value(self):
        return np.random.choice(self.choices, p=self.probs)

    def get_constructor_kwargs(self):
        return {'choices': self._orig_choices, 'probs': self._orig_probs}
