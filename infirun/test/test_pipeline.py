from ..pipeline import *
import pytest


@pipeline
def my_randn():
    return 42


def test_pipeline_no_args():
    invocation = my_randn()
    serialized = invocation.serialize()
    assert serialized['type'] == 'func_invoke'
    assert serialized['func']['type'] == 'func'
    assert not serialized['func']['flatten_output']
    assert my_randn.raw() == 42
    assert next(invocation.build()) == 42

    restored = deserialize(serialized)
    restored.build()
    assert next(restored) == 42


@pipeline(flatten_output=True)
def totally_random():
    return list(range(3))


def test_pipeline_flatten_output():
    invocation = totally_random()
    serialized = invocation.serialize()
    assert serialized['type'] == 'func_invoke'
    assert serialized['func']['type'] == 'func'
    assert serialized['func']['flatten_output']
    assert totally_random.raw() == list(range(3))
    invocation.build()
    assert next(invocation) == 0
    assert next(invocation) == 1
    assert next(invocation) == 2
    assert next(invocation) == 0
    assert next(invocation) == 1
    assert next(invocation) == 2

    restored = deserialize(serialized)
    restored.build()
    assert next(restored) == 0
    assert next(restored) == 1


@pipeline(flatten_output=True, n_epochs=1)
def only_once():
    return list(range(3))


def test_epoch():
    invocation = only_once()
    assert next(invocation) == 0
    assert next(invocation) == 1
    assert next(invocation) == 2

    with pytest.raises(StopIteration):
        next(invocation)

    with pytest.raises(StopIteration):
        next(invocation)


@pipeline
def times_two(n):
    return n * 2


@pipeline
def plus(m, n):
    return m + n


def test_arguments():
    x2_invoke = times_two(1)
    assert next(x2_invoke) == 2

    serialized = x2_invoke.serialize()
    print(serialized)
    assert next(deserialize(serialized).build()) == 2

    x2_invoke_2 = times_two(2)
    plus_invoke = plus(x2_invoke, n=x2_invoke_2)
    assert next(plus_invoke) == 6

    serialized = plus_invoke.serialize()

    restored = deserialize(serialized)
    restored.build()
    assert next(restored) == 6
