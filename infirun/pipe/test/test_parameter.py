import numpy as np
import pytest

from ..parameter import *


def test_uniform_parameter():
    v = Uniform(0, 20)
    for _ in range(100):
        assert 0 <= v.value() <= 20


def test_normal_parameter():
    v = Normal(0, 10)
    outside = 0
    for _ in range(1000):
        value = v.value()
        if np.abs(value) > 30:
            outside += 1

    assert outside < 20


def test_normal_with_clip():
    v = Normal(0, 10, low_clip=-1, high_clip=100)
    for _ in range(1000):
        assert -1 <= v.value() <= 100


def test_uniform_int():
    from collections import Counter
    v = UniformInt(0, 20)
    ct = Counter()
    for _ in range(1000):
        ct[v.value()] += 1
    for v in ct.values():
        assert 20 <= v <= 80
    assert len(ct) == 20


def test_choice():
    v = Choice('abc', [0, 0, 1])
    for _ in range(10):
        assert v.value() == 'c'

    v = Choice('abc', [0, 1, 2])
    ct = {'a': 0, 'b': 0, 'c': 0}
    for _ in range(3000):
        ct[v.value()] += 1
    assert ct['a'] == 0
    assert 700 <= ct['b'] <= 1300
    assert ct['b'] + ct['c'] == 3000
