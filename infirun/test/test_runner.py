import pytest

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
