import abc

from infirun.util.reflect import ensure_basic_data_type
from infirun.util.schematic import Schematic
from .parameter import RandomParameter


class StepArg(Schematic):
    wrapped_type = None

    def __init__(self, v):
        self.wrapped = v

    @abc.abstractmethod
    async def get_next_value(self):
        pass


class ConstStepArg(StepArg):
    def __init__(self, v):
        ensure_basic_data_type(v)
        super().__init__(v)

    async def get_next_value(self):
        return self.wrapped

    def _get_schema(self):
        return {'value': self.wrapped}

    @staticmethod
    def from_schema(schema):
        return ConstStepArg(schema['value'])


class RandomStepArg(StepArg):
    wrapped_type = RandomParameter

    async def get_next_value(self):
        return self.wrapped.value()

    def _get_schema(self):
        return {'subtype': type(self.wrapped).__name__,
                'kwargs': self.wrapped.get_constructor_kwargs()}

    @staticmethod
    def from_schema(schema):
        import infirun.pipe.parameter as mod
        return RandomStepArg(getattr(mod, schema['subtype'])(**schema['kwargs']))


def make_step_arg(v):
    for Cls in StepArg.__subclasses__():
        if Cls.wrapped_type is None:
            continue
        if isinstance(v, Cls.wrapped_type):
            return Cls(v)
    return ConstStepArg(v)
