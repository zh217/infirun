import json
import abc
import inspect
import importlib


def _called_with_no_args(args, kwargs):
    return len(args) == 1 and len(kwargs) == 0


class Wrapped(abc.ABC):
    pass


class PipelineFunctionInvocation(Wrapped):
    @staticmethod
    def deserialize(serialized):
        fn_wrapper = PipelineFunctionWrapper.deserialize(serialized['func'])
        args = [deserialize(a) for a in serialized['args']]
        kwargs = {k: deserialize(a) for k, a in serialized['kwargs'].items()}
        return PipelineFunctionInvocation(fn_wrapper,
                                          args=args,
                                          kwargs=kwargs)

    def __init__(self, fn_wrapper, args, kwargs):
        self.fn_wrapper = fn_wrapper
        self.args = [ensure_wrapped(a) for a in args]
        self.kwargs = {k: ensure_wrapped(v) for k, v in kwargs.items()}
        self.flatten_output = fn_wrapper.flatten_output
        self.flatten_output_buffer = iter(())
        self.n_epochs_left = fn_wrapper.n_epochs

    def serialize(self):
        return {
            'type': 'func_invoke',
            'func': self.fn_wrapper.serialize(),
            'args': [a.serialize() for a in self.args],
            'kwargs': {k: v.serialize() for k, v in self.kwargs.items()}
        }

    def build(self):
        return self

    def __next__(self):
        if self.flatten_output:
            try:
                return next(self.flatten_output_buffer)
            except StopIteration:
                if self.n_epochs_left == 0:
                    raise

                if self.n_epochs_left is not None:
                    self.n_epochs_left -= 1

                self.flatten_output_buffer = iter(self._invoke_raw())
                return next(self.flatten_output_buffer)
        else:
            return self._invoke_raw()

    def _invoke_raw(self):
        args = (next(a) for a in self.args)
        kwargs = {k: next(a) for k, a in self.kwargs.items()}
        return self.fn_wrapper.raw(*args, **kwargs)


class PipelineFunctionWrapper:
    @staticmethod
    def deserialize(serialized):
        func = getattr(importlib.import_module(serialized['module']), serialized['name'])

        if isinstance(func, PipelineFunctionWrapper):
            return func

        return PipelineFunctionWrapper(func,
                                       flatten_output=serialized['flatten_output'],
                                       n_epochs=serialized['n_epochs'])

    def __init__(self, wrapped, flatten_output=False, n_epochs=None):
        self.raw = wrapped
        self.flatten_output = flatten_output
        self.n_epochs = n_epochs

    def __call__(self, *args, **kwargs):
        return PipelineFunctionInvocation(self, args, kwargs)

    def serialize(self):
        return {
            'type': 'func',
            'module': self.raw.__module__,
            'name': self.raw.__name__,
            'flatten_output': self.flatten_output,
            'n_epochs': self.n_epochs
        }


def _make_pipeline_decorator(**kwargs):
    def decorator(f):
        if inspect.isclass(f):
            return _decorate_class(f, **kwargs)
        else:
            return PipelineFunctionWrapper(f, **kwargs)

    return decorator


def _decorate_class(cls, **kwargs):
    pass


def pipeline(*args, **kwargs):
    if _called_with_no_args(args, kwargs):
        return pipeline()(*args)

    return _make_pipeline_decorator(**kwargs)


def deserialize(serialized):
    s_type = serialized['type']

    if s_type == 'func_invoke':
        return PipelineFunctionInvocation.deserialize(serialized)
    elif s_type == 'const':
        return Constant.deserialize(serialized)


class Constant(Wrapped):
    @staticmethod
    def deserialize(serialized):
        return Constant(serialized['value'])

    def __init__(self, const):
        assert json.loads(json.dumps(const)) == const, 'Constant value must be JSON-serializable'
        self.const = const

    def __next__(self):
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


class SwitchCase(Wrapped):
    @staticmethod
    def deserialize(serialized):
        pass

    def __init__(self):
        pass

    def __next__(self):
        pass

    def serialize(self):
        pass
