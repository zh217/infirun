import time

import pytest
import multiprocessing
import multiprocessing.dummy

from infirun.pipe.sink import *
from infirun.pipe.step import *

DEFAULT_TIMEOUT = 1


def tf_range():
    i = 0
    while True:
        yield i
        i += 1


class SinkClass(SinkRunner):
    def __init__(self, m1, m2):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.counter = 0
        self.total = 0

    def __call__(self, value):
        self.counter += 1
        self.total += value * self.m1 * self.m2
        if self.counter == 10:
            self.halt(self.total)


@pytest.mark.timeout(DEFAULT_TIMEOUT)
@pytest.mark.parametrize('ctx', [multiprocessing, multiprocessing.dummy])
def test_process_sink(ctx):
    rg = Step(tf_range)()
    sink = ClassRunnerSink(rg)(SinkClass, 2, m2=3)
    assert sink.start() == sum(range(10)) * 2 * 3
