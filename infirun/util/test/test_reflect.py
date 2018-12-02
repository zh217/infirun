import numpy as np
import pytest

from ..reflect import *


def f123():
    pass


class X123:
    pass


def test_module():
    assert get_module(f123) == __name__
    assert get_module(X123) == __name__
    assert get_name(f123) == 'f123'
    assert get_name(X123) == 'X123'
    assert get_source_data(f123) == {'type': 'fun',
                                     'mod': __name__,
                                     'name': 'f123'}
    assert get_source_data(X123) == {'type': 'cls',
                                     'mod': __name__,
                                     'name': 'X123'}


def test_basic_data_types():
    ensure_basic_data_type(None)
    ensure_basic_data_type(1)
    ensure_basic_data_type({'aa': [1, 2, 3, 4, None]})

    with pytest.raises(ValueError):
        ensure_basic_data_type(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        ensure_basic_data_type(f123)
    with pytest.raises(ValueError):
        ensure_basic_data_type(X123)
    with pytest.raises(ValueError):
        ensure_basic_data_type(X123())
