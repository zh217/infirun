import collections

import pytest
import time
import itertools
import multiprocessing.dummy

from ..step import *
from ..parameter import *
from ..sink import *
from ..runner import *

DEFAULT_TIMEOUT = 2


def tf_one():
    return 1


def tf_dbl(x):
    return x * 2


def tf_sum(x, y):
    return x + y


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_step_single_function():
    pipeline = Step(tf_one)()
    out_q_sink = QueueSink(pipeline)
    with out_q_sink as q_get:
        for _ in range(100):
            assert q_get() == 1


tf_count_counter = 0


def tf_count():
    global tf_count_counter
    tf_count_counter += 1


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_buffer():
    global tf_count_counter
    tf_count_counter = 0
    pipeline = Step(tf_count).buffer(10)()
    with QueueSink(pipeline):
        time.sleep(0.01)
    assert 10 <= tf_count_counter <= 15


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_step_single_function_iter():
    pipeline = Step(tf_one)()
    out_q_sink = IterSink(pipeline)
    with out_q_sink as it:
        for v in itertools.islice(it, 100):
            assert v == 1


def tf_arr():
    return [1] * 10


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_manual_flatten():
    pipeline = Step(tf_arr)().flatten()
    with QueueSink(pipeline) as q_get:
        for _ in range(100):
            assert q_get() == 1


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_step_single_function_random_param():
    rp = Uniform(0, 1)
    pipeline = Step(tf_dbl)(rp)
    with QueueSink(pipeline) as q_get:
        for _ in range(100):
            assert 0 <= q_get() <= 2


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_step_chain_functions():
    rp = Uniform(0, 1)
    const = Step(tf_one)  # 1
    dblrp = Step(tf_dbl)(rp)  # 0 ~ 2
    result = Step(tf_sum)(const, dblrp)
    with QueueSink(result) as q_get:
        for _ in range(100):
            assert 1 <= q_get() <= 3


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_step_diamond():
    const = Step(tf_one)()
    sum = Step(tf_sum)(const, const)
    with QueueSink(sum) as q_get:
        for _ in range(10):
            assert q_get() == 2


mark_count = 0


def tf_mark_one():
    global mark_count
    mark_count += 1
    return 1


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_backpressure():
    const = Step(tf_mark_one)()
    dbl = Step(tf_dbl)(const)
    with QueueSink(dbl) as q_get:
        for _ in range(10):
            time.sleep(0.01)
            assert q_get() == 2
            time.sleep(0.01)
    assert 10 <= mark_count <= 20


def tf_gen():
    for i in range(10):
        yield i


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_generator_step():
    pipeline = Step(tf_gen)()
    with IterSink(pipeline) as it:
        assert list(range(10)) * 3 == list(itertools.islice(it, 30))


class TfCallableClass:
    def __init__(self, ret_val, *, multiplier):
        self.ret_val = ret_val
        self.multiplier = multiplier

    def __call__(self, in_v):
        return (self.ret_val + in_v) * self.multiplier


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_function_class():
    pipeline = Step(TfCallableClass, 10, multiplier=3)(4)
    with QueueSink(pipeline) as q_get:
        for _ in range(10):
            assert q_get() == (4 + 10) * 3


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_switch_step():
    a_step = Step(lambda: 'a')()
    b_step = Step(lambda: 'b')()
    c_step = Step(lambda: 'c')()
    switch = UniformInt(0, 3)
    pipeline = Switch(a_step, b_step, c_step)(switch)
    ct = collections.Counter()
    with QueueSink(pipeline) as q_get:
        for _ in range(6000):
            ct[q_get()] += 1
    assert ct['a'] + ct['b'] + ct['c'] == 6000
    for ch in 'abc':
        assert 1500 <= ct[ch] <= 2500


@pytest.mark.timeout(DEFAULT_TIMEOUT * 10)
def test_switch_step_intermediate():
    a_step = Step(lambda: 'a')()
    b_step = Step(lambda: 'b')()
    c_step = Step(lambda: 'c')()
    switch = UniformInt(0, 3)
    pipeline = Switch(a_step, b_step, c_step)(switch)
    d_step = Step(lambda x: x)(pipeline)
    ct = collections.Counter()
    with QueueSink(d_step) as q_get:
        for _ in range(6000):
            ct[q_get()] += 1
    assert ct['a'] + ct['b'] + ct['c'] == 6000
    for ch in 'abc':
        assert 1500 <= ct[ch] <= 2500


def tf_none():
    return None


class TfClassNone():
    pass


def tf_except(msg):
    raise Exception(msg)


def test_exception_propagation():
    pipeline = Step(tf_except)(msg='stopped')
    with QueueSink(pipeline) as q_get:
        with pytest.raises(Exception, message='stopped'):
            q_get()


def tf_except_gen(msg):
    for i in range(10):
        yield i
    raise Exception(msg)


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_exception_propagation_gen():
    pipeline = Step(tf_except_gen)(msg='stopped')
    with QueueSink(pipeline) as q_get:
        for i in range(10):
            assert q_get() == i
        with pytest.raises(Exception, message='stopped'):
            q_get()
        for i in range(10):
            assert q_get() == i
        with pytest.raises(Exception, message='stopped'):
            q_get()


class TfExceptClass:
    def __init__(self, msg):
        self.msg = msg
        self.count = 0

    def __call__(self):
        c = self.count
        self.count += 1
        if self.count <= 10:
            return c
        else:
            self.count = 0
            raise Exception(self.msg)


@pytest.mark.timeout(DEFAULT_TIMEOUT)
def test_exception_propagation_cls():
    pipeline = Step(TfExceptClass, msg='stopped')()
    with QueueSink(pipeline) as q_get:
        for i in range(10):
            assert q_get() == i
        with pytest.raises(Exception, message='stopped'):
            q_get()


@pytest.mark.timeout(DEFAULT_TIMEOUT)
@pytest.mark.parametrize('ctx', [multiprocessing, multiprocessing.dummy])
def test_exception_propagation_cls_proc(ctx):
    pipeline = Step(TfExceptClass, msg='stopped')()
    pipeline.set_runner(ProcessRunner, concurrency=1, ctx=ctx)
    with QueueSink(pipeline) as q_get:
        for i in range(10):
            assert q_get() == i
        with pytest.raises(Exception, message='stopped'):
            q_get()
        for i in range(10):
            assert q_get() == i
        with pytest.raises(Exception, message='stopped'):
            q_get()
        for i in range(10):
            assert q_get() == i
