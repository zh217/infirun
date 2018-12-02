from infirun.runner import *
from ..pipeline import *


@pipeline(iter_output=True, n_epochs=1)
def number_gen():
    return range(10)


@pipeline
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, n):
        return self.factor * n


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
    print()
    print()
    pprint(mult3_res.serialize())
