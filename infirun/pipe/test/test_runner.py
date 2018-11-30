import pytest
import time
import multiprocessing.dummy

from infirun.pipe.step import Step
from infirun.pipe.sink import *
from infirun.pipe.runner import *
from infirun.pipe.parameter import *

DEFAULT_TIMEOUT = 2


def tf_one():
    return 1


def tf_dbl(x):
    return x * 2


def tf_sum(x, y):
    return x + y


@pytest.mark.timeout(DEFAULT_TIMEOUT)
@pytest.mark.parametrize('ctx', [multiprocessing, multiprocessing.dummy])
def test_process(ctx):
    pipeline = Step(tf_one)()
    pipeline.set_runner(ProcessRunner, concurrency=1, ctx=ctx)
    with QueueSink(pipeline) as q_get:
        for _ in range(10):
            assert q_get() == 1


@pytest.mark.timeout(DEFAULT_TIMEOUT)
@pytest.mark.parametrize('ctx', [multiprocessing, multiprocessing.dummy])
def test_step_chain_functions(ctx):
    rp = Uniform(0, 1)
    const = Step(tf_one)  # 1
    dblrp = Step(tf_dbl)(rp)  # 0 ~ 2
    dblrp.set_runner(ProcessRunner, concurrency=2, ctx=ctx)
    result = Step(tf_sum)(const, dblrp)
    with QueueSink(result) as q_get:
        for _ in range(100):
            assert 1 <= q_get() <= 3


mark_count = 0


def tf_mark_one():
    global mark_count
    mark_count += 1
    return 1


@pytest.mark.timeout(DEFAULT_TIMEOUT)
@pytest.mark.parametrize('ctx', [multiprocessing, multiprocessing.dummy])
def test_backpressure(ctx):
    global mark_count
    mark_count = 0
    const = Step(tf_mark_one)()
    dbl = Step(tf_dbl)(const)
    dbl.set_runner(ProcessRunner, concurrency=2, ctx=ctx)
    with QueueSink(dbl) as q_get:
        for _ in range(10):
            time.sleep(0.01)
            assert q_get() == 2
            time.sleep(0.01)
    assert 10 <= mark_count <= 20


class TfCallableClass:
    def __init__(self, ret_val, *, multiplier):
        self.ret_val = ret_val
        self.multiplier = multiplier

    def __call__(self, in_v):
        return (self.ret_val + in_v) * self.multiplier


class TfConstCallableClass:
    def __init__(self, ret_val):
        self.ret_val = ret_val

    def __call__(self):
        return self.ret_val


@pytest.mark.timeout(DEFAULT_TIMEOUT)
@pytest.mark.parametrize('ctx', [multiprocessing, multiprocessing.dummy])
def test_function_class(ctx):
    const = Step(TfConstCallableClass, 4)()
    pipeline = Step(TfCallableClass, 10, multiplier=3)(const)
    const.set_runner(ProcessRunner, concurrency=1, ctx=ctx)
    pipeline.set_runner(ProcessRunner, concurrency=2, ctx=ctx)
    with QueueSink(pipeline) as q_get:
        for i in range(1000):
            assert q_get() == (4 + 10) * 3


def tf_random_seed():
    import random
    time.sleep(0.05)
    return random.random()


@pytest.mark.timeout(DEFAULT_TIMEOUT)
@pytest.mark.parametrize('ctx', [multiprocessing, multiprocessing.dummy])
def test_randomization(ctx):
    rd = Step(tf_random_seed)()
    rd.set_runner(ProcessRunner, concurrency=10, ctx=ctx)
    res = set()
    with QueueSink(rd) as q_get:
        for _ in range(100):
            res.add(q_get())
    assert 100 == len(res)


def tf_random_seed_np():
    import numpy.random
    time.sleep(0.05)
    return numpy.random.random()


@pytest.mark.timeout(DEFAULT_TIMEOUT)
@pytest.mark.parametrize('ctx', [multiprocessing, multiprocessing.dummy])
def test_randomization_np(ctx):
    rd = Step(tf_random_seed_np)()
    rd.set_runner(ProcessRunner, concurrency=10, ctx=ctx)
    res = set()
    with QueueSink(rd) as q_get:
        for _ in range(100):
            res.add(q_get())
    assert 100 == len(res)
