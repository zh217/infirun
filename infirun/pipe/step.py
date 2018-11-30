import asyncio
import inspect

import infirun.pipe.step_arg as step_arg
import infirun.pipe.runner as runner

DEFAULT_BUFFER = 1

__all__ = ['Step', 'Switch']


class Step:
    def __init__(self, f, *init_args, **init_kwargs):
        self._f = f
        self._init_args = init_args
        self._init_kwargs = init_kwargs
        self._args_stream = ArgsStream([], {})
        self._buffer = DEFAULT_BUFFER
        self._runner = None
        self._runner_cls = runner.AsyncRunner
        self._runner_args = []
        self._runner_kwargs = {}
        self._force_flatten = None
        self._q = None
        self.running = False
        self.parents = get_parents([])
        self.get_next_value = None

    def __call__(self, *args, **kwargs):
        self._args_stream = ArgsStream(args, kwargs)
        self.parents = get_parents(list(args) + list(kwargs.values()))
        return self

    def set_runner(self, cls, *args, **kwargs):
        self._runner_cls = cls
        self._runner_args = args
        self._runner_kwargs = kwargs or {}

    def flatten(self, flatten=True):
        self._force_flatten = flatten
        return self

    def buffer(self, n):
        assert n >= 1
        self._buffer = n
        return self

    def start(self, loop):
        if self.running:
            return
        assert loop is not None
        self._q = asyncio.Queue(maxsize=self._buffer, loop=loop)
        self.get_next_value = self._q.get
        self.running = True
        self._runner = self._runner_cls(loop=loop,
                                        q=self._q,
                                        f=self._f,
                                        init_args=self._init_args,
                                        init_kwargs=self._init_kwargs,
                                        args_stream=self._args_stream)
        if self._force_flatten is not None:
            self._runner.force_flatten(self._force_flatten)
        self._runner.setup(*self._runner_args, **self._runner_kwargs)
        self._runner.start()
        self._start_parents(loop)

    def _start_parents(self, loop):
        for parent in self.parents:
            parent.start(loop)

    def stop(self):
        if not self.running:
            return
        self._stop_parents()
        self._runner.stop()
        self.running = False

    def _stop_parents(self):
        for parent in self.parents:
            parent.stop()


class ArgsStream:
    def __init__(self, args, kwargs):
        self._args = [step_arg.make_step_arg(a) for a in args]
        self._kwargs = {k: step_arg.make_step_arg(a) for k, a in kwargs.items()}

    def _get_schema(self):
        return [a.get_schema() for a in self._args], {k: a.get_schema() for k, a in self._kwargs.items()}

    @staticmethod
    def from_schema(schema):
        pass

    async def get_next_args(self):
        args = []
        for a in self._args:
            v = await a.get_next_value()
            if isinstance(v, Exception):
                raise v
            args.append(v)
        kwargs = {}
        for k, a in self._kwargs.items():
            v = await a.get_next_value()
            if isinstance(v, Exception):
                raise v
            kwargs[k] = v
        return args, kwargs


class StepAsArg(step_arg.StepArg):
    wrapped_type = Step

    async def get_next_value(self):
        return await self.wrapped.get_next_value()

    def _get_schema(self):
        is_class = inspect.isclass(self.wrapped._f)
        return {'wrapped': self.wrapped._f.__name__,
                'is_class': is_class,
                'init_args': 1 if is_class else None,
                'init_kwargs': 1 if is_class else None,
                'defined_in': self.wrapped._f.__module__,
                'args': self.wrapped._args_stream.get_schema()}

    @staticmethod
    def from_schema(schema):
        pass


class Switch:
    def __init__(self, *args, **kwargs):
        self.running = False
        self.args = list(args)
        self.kwargs = dict(kwargs)
        self.switch = None
        self.switch_fn = None
        self.parents = None
        self.sources = None

    def __call__(self, switch_arg):
        self.parents = get_parents([switch_arg] + self.args + list(self.kwargs.values()))
        self.switch = step_arg.make_step_arg(switch_arg)
        sources = {}
        for k, v in enumerate(self.args):
            sources[k] = step_arg.make_step_arg(v)
        for k, v in self.kwargs.values():
            assert k not in sources
            sources[k] = step_arg.make_step_arg(v)
        self.sources = sources
        return self

    async def get_next_value(self):
        selector = await self.switch.get_next_value()
        return await self.sources[selector].get_next_value()

    def start(self, loop):
        if self.running:
            return
        self.running = True
        self._start_parents(loop)

    def _start_parents(self, loop):
        for parent in self.parents:
            parent.start(loop)

    def stop(self):
        if not self.running:
            return
        self._stop_parents()
        self.running = False

    def _stop_parents(self):
        for parent in self.parents:
            parent.stop()


class SwitchAsArg(step_arg.StepArg):
    wrapped_type = Switch

    async def get_next_value(self):
        return await self.wrapped.get_next_value()

    def _get_schema(self):
        """TODO"""

    @staticmethod
    def from_schema(schema):
        """TODO"""


def get_parents(args):
    return [arg for arg in list(args) if isinstance(arg, Step) or isinstance(arg, Switch)]
