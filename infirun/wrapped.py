import abc
import json


class Wrapped(abc.ABC):
    pass


class Constant(Wrapped):
    @staticmethod
    def deserialize(serialized):
        return Constant(serialized['value'])

    def __init__(self, const):
        assert json.loads(json.dumps(const)) == const, 'Constant value must be JSON-serializable'
        self.const = const

    def __next__(self):
        return self.const

    def __iter__(self):
        return self

    def start(self):
        pass

    @property
    def raw(self):
        return self.const

    def serialize(self):
        return {
            'type': 'const',
            'value': self.const
        }


def ensure_wrapped(obj):
    if isinstance(obj, Wrapped):
        return obj
    else:
        return Constant(obj)


def ensure_constant(obj):
    if isinstance(obj, Constant):
        return obj
    else:
        return Constant(obj)


def ensure_unwrapped_constant(obj):
    if isinstance(obj, Constant):
        return obj.raw
    return obj
