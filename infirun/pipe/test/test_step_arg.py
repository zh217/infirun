import asyncio
import pytest
import numpy as np

from infirun.util.schematic import from_schema
from ..step_arg import *
from ..parameter import Uniform


@pytest.mark.parametrize('v', [1, {}, None, []])
def test_const(v):
    p = make_step_arg(v)
    assert isinstance(p, ConstStepArg)
    assert v == asyncio.get_event_loop().run_until_complete(p.get_next_value())
    assert v == asyncio.get_event_loop().run_until_complete(from_schema(p.get_schema()).get_next_value())


def test_random():
    p = make_step_arg(Uniform(0, 1))
    assert isinstance(p, RandomStepArg)
    for _ in range(10):
        assert 0 <= asyncio.get_event_loop().run_until_complete(p.get_next_value()) <= 1

    p2 = from_schema(p.get_schema())

    for _ in range(10):
        assert 0 <= asyncio.get_event_loop().run_until_complete(p2.get_next_value()) <= 1
