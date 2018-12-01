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
    invocation.start()
    assert next(invocation) == 42

    restored = deserialize(serialized)
    restored.start()
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
    invocation.start()
    assert next(invocation) == 0
    assert next(invocation) == 1
    assert next(invocation) == 2
    assert next(invocation) == 0
    assert next(invocation) == 1
    assert next(invocation) == 2

    restored = deserialize(serialized)
    restored.start()
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
    s_invoke = deserialize(serialized)
    s_invoke.start()
    assert next(s_invoke) == 2

    x2_invoke_2 = times_two(2)
    plus_invoke = plus(x2_invoke, n=x2_invoke_2)
    assert next(plus_invoke) == 6

    serialized = plus_invoke.serialize()

    restored = deserialize(serialized)
    restored.start()
    assert next(restored) == 6


@pipeline(flatten_output=True)
def counter():
    return (i % 2 for i in range(10))


def test_switch_case():
    a = times_two(1)
    b = times_two(0)
    x = SwitchCase(counter(),
                   {0: a,
                    1: b})
    y = times_two(x)
    assert next(y) == 4
    assert next(y) == 0
    assert next(y) == 4
    assert next(y) == 0

    assert x.serialize()['type'] == 'switch'

    serialized = y.serialize()
    restored = deserialize(serialized)
    restored.start()

    assert next(restored) == 4
    assert next(restored) == 0
    assert next(restored) == 4


@pipeline
class SimpleClass:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.k = 0

    def __call__(self, p):
        self.k += 1
        return self.m + self.n + self.k + p

    def n_calls(self, p):
        self.k += 1
        return self.k + p


def test_simple_class():
    inst = SimpleClass(1, 2)
    invoke1 = inst(1)
    invoke1.start()
    assert next(invoke1) == 5
    assert next(invoke1) == 6

    invoke2 = inst.n_calls(0)
    invoke2.start()

    assert next(invoke2) == 1
    assert next(invoke2) == 2

    assert next(invoke1) == 7
    assert next(invoke1) == 8

    assert next(invoke2) == 3
    assert next(invoke2) == 4


@component
class StateClass:
    def __init__(self, n):
        self.n = n


@pipeline(flatten_output=True, n_epochs=1)
class FlatClass:
    def __init__(self, max_obj):
        self.values = iter(range(max_obj.n))

    def my_values(self):
        return self.values


def test_flatclass():
    comp = StateClass(3)
    inst = FlatClass(max_obj=comp)
    invoke = inst.my_values()
    invoke.start()
    assert next(invoke) == 0
    assert next(invoke) == 1
    assert next(invoke) == 2
    with pytest.raises(StopIteration):
        next(invoke)


@component
class TestComponent:
    def __init__(self, m, n):
        self.m = m
        self.n = n

    def __call__(self, p, q):
        return [self.m, self.n, p, q]


def test_component():
    inst = TestComponent(1, 2)
    inst.start()
    assert inst(3, 4) == [1, 2, 3, 4]


@component
class MegaComponent:
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2
        self._n = 0

    def __call__(self, p, q):
        self._n += 1
        c1r = self.c1(p, q)
        c2r = self.c2(p, q)
        return [c1r[0] + c2r[0],
                c1r[1] + c2r[1],
                c1r[2] + c2r[2],
                c1r[3] + c2r[3]]

    def n_called(self):
        return self._n


def test_mega_component():
    inst1 = TestComponent(1, 2)
    inst2 = TestComponent(3, 4)
    instm = MegaComponent(inst1, inst2)
    instm.start()
    assert instm(5, 6) == [4, 6, 10, 12]
    assert instm.n_called() == 1

    serialized = instm.serialize()
    restored = deserialize(serialized)
    restored.start()
    assert restored(5, 6) == [4, 6, 10, 12]
    assert restored(1, 2) == [4, 6, 2, 4]
    assert restored.n_called() == 2
    with pytest.raises(KeyError):
        restored.xxx()
