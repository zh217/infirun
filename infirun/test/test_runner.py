from infirun.runner import *
from ..pipeline import *
import pprint as pp


@pipeline(iter_output=True, n_epochs=1)
def number_gen():
    return range(10)


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


def test_runner_example():
    """
    like this:
    ```

    mult1 = Multiplier(2)
    mult1_res = mult1(n_gen)
    mult2 = Multiplier(3)
    mult2_res = mult2(mult1_res)
    mult3 = Multiplier(5)
    mult3_res = mult3(mult2_res)

    mult3_res.set_upstream_runner(RunnerClass, *runner_args, **runner_kwargs)
    mult1_res.set_upstream_runner()

    ```

    :return:
    """
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

    mult3_res.set_upstream_runner(Invocation)
    mult1_res.set_upstream_runner(Invocation)
    n_gen2.set_upstream_runner(Invocation)

    final.start()

    print()

    pprint(final.serialize())

    collected = {}
    n, root = chop_serialized(final.serialize(), 0, collected)

    for k, v in collected.items():
        print()
        pprint(v, prefix=f'P[{k}]')

    print()
    pprint(root)
