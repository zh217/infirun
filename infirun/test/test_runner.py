import pytest
import os

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


def test_pipeline():
    n_gen = number_gen()
    mult1 = Multiplier(2)
    mult1_res = mult1(n_gen)
    mult2 = Multiplier(3)
    mult2_res = mult2(mult1_res)
    mult3 = Multiplier(5)
    mult3_res = mult3(mult2_res)

    vs = []
    sink = PipelineSink(mult3_res, lambda v: vs.append(v))
    sink.start()
    assert vs == list(range(0, 300, 30))

    serialized = mult3_res.serialize()
    restored = deserialize(serialized)

    vs2 = []
    sink2 = PipelineSink(restored, lambda v: vs2.append(v))
    sink2.start()
    assert vs == list(range(0, 300, 30))


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

    # final.start()
    #
    # print()
    #
    # pprint(final.serialize())
    #
    # collected, ordered = generate_dependency_map(final.serialize())
    #
    # for k, v in collected.items():
    #     print()
    #     pprint(v, prefix=f'P[{k}]')
    #
    # print(ordered)
    it = run_with_runner(final.serialize(), return_iter=True)
    assert len(list(it)) == 1000

    # try:
    #     os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tmp_out'))
    # except:
    #     pass
    #
    # serialize_to_file(final,
    #                   os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tmp_out',
    #                                'serialized.json'),
    #                   pretty=True)
