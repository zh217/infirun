import pytest
import pprint as pp

from infirun.runner import *
from ..pipeline import *


@pipeline(iter_output=True, n_epochs=1)
def number_gen(n=10):
    return range(n)


@pipeline
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, n):
        return self.factor * n


@pipeline
def multiply(m, n):
    return m * n


@pytest.mark.timeout(5)
def test_simplest_threading():
    n_gen = number_gen()
    n_gen.set_upstream_runner(ProcessRunner, n_process=2)
    it = run_with_runner(n_gen.serialize(), return_iter=True)
    assert len(list(it)) > 10


@pytest.mark.timeout(5)
def test_simplest_threading_2():
    n_gen = number_gen()
    n_gen.set_upstream_runner(ProcessRunner, n_process=10)
    mult1 = Multiplier(2)
    mult1_res = mult1(n_gen).set_name('mult1')
    it = run_with_runner(mult1_res.serialize(), return_iter=True)
    assert len(list(it)) > 20


@pytest.mark.timeout(5)
def test_runner_example():
    n_gen = number_gen().set_name('n_gen')
    mult1 = Multiplier(2)
    mult1_res = mult1(n_gen).set_name('mult1')
    mult2 = Multiplier(3)
    mult2_res = mult2(mult1_res).set_name('mult2')
    mult3 = Multiplier(5)
    mult3_res = mult3(mult2_res).set_name('mult3')
    n_gen2 = number_gen().set_name('n_gen2')
    mult4 = Multiplier(10)
    mult4_res = mult4(n_gen2).set_name('mult4_res')
    final = multiply(m=mult3_res, n=mult4_res).set_name('final')

    mult3_res.set_upstream_runner(ProcessRunner, n_process=10)
    mult1_res.set_upstream_runner(ProcessRunner)
    n_gen2.set_upstream_runner(ProcessRunner)

    it = run_with_runner(final.serialize(), return_iter=True)
    assert len(list(it)) == 10


@pipeline
def combiner(m, n):
    return [m, n]


@pytest.mark.timeout(5)
def test_runner_example_2():
    n = 1000
    n_gen = number_gen(n).set_name('n_gen')
    mult1 = Multiplier(2)
    mult1_res = mult1(n_gen).set_name('mult1')
    mult2 = Multiplier(3)
    mult2_res = mult2(mult1_res).set_name('mult2')
    mult3 = Multiplier(5)
    mult3_res = mult3(mult2_res).set_name('mult3')
    n_gen2 = number_gen(n).set_name('n_gen2')
    mult4 = Multiplier(10)
    mult4_res = mult4(n_gen2).set_name('mult4_res')
    final = combiner(m=mult3_res, n=mult4_res).set_name('final')

    mult3_res.set_upstream_runner(ProcessRunner)
    mult1_res.set_upstream_runner(ProcessRunner, n_process=2, process_type='process')
    n_gen.set_upstream_runner(ProcessRunner)
    n_gen2.set_upstream_runner(ProcessRunner)
    final.set_upstream_runner(ProcessRunner, process_type='process')

    it = run_with_runner(final.serialize(), return_iter=True)
    assert len(list(it)) == 1000


@pytest.mark.timeout(5)
def test_runner_anonymous():
    n = 1000
    n_gen = number_gen(n)
    mult1 = Multiplier(2)
    mult1_res = mult1(n_gen)
    mult2 = Multiplier(3)
    mult2_res = mult2(mult1_res)
    mult3 = Multiplier(5)
    mult3_res = mult3(mult2_res)
    n_gen2 = number_gen(n)
    mult4 = Multiplier(10)
    mult4_res = mult4(n_gen2)
    final = combiner(m=mult3_res, n=mult4_res)

    mult3_res.set_upstream_runner(ProcessRunner)
    mult1_res.set_upstream_runner(ProcessRunner, n_process=2, process_type='process')
    n_gen.set_upstream_runner(ProcessRunner)
    n_gen2.set_upstream_runner(ProcessRunner)
    final.set_upstream_runner(ProcessRunner, process_type='process')

    it = run_with_runner(final.serialize(), return_iter=True)
    assert len(list(it)) == 1000


@pytest.mark.timeout(5)
def test_runner_name_clash():
    n = 1000
    n_gen = number_gen(n).set_name('bad name')
    mult1 = Multiplier(2)
    mult1_res = mult1(n_gen).set_name('bad name')
    mult2 = Multiplier(3)
    mult2_res = mult2(mult1_res)
    mult3 = Multiplier(5)
    mult3_res = mult3(mult2_res)
    n_gen2 = number_gen(n)
    mult4 = Multiplier(10)
    mult4_res = mult4(n_gen2)
    final = combiner(m=mult3_res, n=mult4_res)

    mult3_res.set_upstream_runner(ProcessRunner)
    mult1_res.set_upstream_runner(ProcessRunner, n_process=2)
    n_gen.set_upstream_runner(ProcessRunner)
    n_gen2.set_upstream_runner(ProcessRunner)
    final.set_upstream_runner(ProcessRunner)

    with pytest.raises(Exception):
        run_with_runner(final.serialize())


def test_params():
    n_gen = number_gen().set_name('n_gen')
    mult1 = Multiplier(2).set_name('mult1_cls')
    mult1_res = mult1(n_gen).set_name('mult1')
    mult2 = Multiplier(3).set_name('mult2_cls')
    mult2_res = mult2(mult1_res).set_name('mult2')
    mult3 = Multiplier(factor=5).set_name('mult3_cls')
    mult3_res = mult3(mult2_res).set_name('mult3')
    n_gen2 = number_gen().set_name('n_gen2')
    mult4 = Multiplier(10).set_name('mult4_cls')
    mult4_res = mult4(n_gen2).set_name('mult4_res')
    penultimate = multiply(m=mult3_res, n=mult4_res).set_name('penultimate')
    branch = multiply(2, n=5).set_name('branch')
    final = combiner(penultimate, branch).set_name('final')
    serialized = final.serialize()
    params = get_const_params(serialized)

    assert params['mult1_cls'][0] == 2
    assert params['mult2_cls'][0] == 3
    assert params['mult3_cls']['factor'] == 5
    assert params['mult4_cls'][0] == 10
    assert 'penultimate' not in params
    assert params['branch'] == {0: 2, 'n': 5}

    params['branch'] = {0: 4, 'n': 8}
    params['mult1_cls'][0] = 20

    new_serialized = override_const_params(serialized, params)
    new_params = get_const_params(new_serialized)

    assert new_params['mult1_cls'][0] == 20
    assert new_params['mult2_cls'][0] == 3
    assert new_params['mult3_cls']['factor'] == 5
    assert new_params['mult4_cls'][0] == 10
    assert 'penultimate' not in new_params
    assert new_params['branch'] == {0: 4, 'n': 8}


@pipeline
class StatefulCounter(PersistentState):
    def __init__(self, init_count=0, will_count=10):
        super().__init__()
        self.count = init_count
        self.remaining = will_count

    def __call__(self):
        if self.remaining == 0:
            raise StopIteration
        self.count += 1
        self.remaining -= 1
        self.persist_state('init_count', self.count)
        return self.count


def test_persistent_state():
    stateful1 = StatefulCounter(init_count=0).set_name('stateful1')
    stateful2 = StatefulCounter(init_count=0).set_name('stateful2')
    invoke1 = stateful1().set_name('invoke1')
    invoke2 = stateful2().set_name('invoke2')
    combined = combiner(invoke1, invoke2).set_name('combined')

    invoke1.set_upstream_runner(ProcessRunner)
    invoke2.set_upstream_runner(ProcessRunner, process_type='process')
    combined.set_upstream_runner(ProcessRunner)

    result = run_with_runner(combined.serialize())
    assert result == {'stateful1': {'init_count': 10},
                      'stateful2': {'init_count': 10}}

    result = run_with_runner(override_const_params(combined.serialize(), result))
    assert result == {'stateful1': {'init_count': 20},
                      'stateful2': {'init_count': 20}}
