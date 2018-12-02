import inspect
import importlib
import abc

from .wrapped import ensure_wrapped, Wrapped, ensure_constant, Constant


class RunnerWrapper:
    @staticmethod
    def deserialize(serialized):
        if not serialized['has_runner']:
            runner_cls = None
        else:
            runner_cls = getattr(importlib.import_module(serialized['module']), serialized['name'])
        args = [deserialize(a) for a in serialized['args']]
        kwargs = {k: deserialize(a) for k, a in serialized['kwargs']}
        return RunnerWrapper(runner_cls, args, kwargs)

    def serialize(self):
        return {
            'type': 'runner',
            'has_runner': self.module is not None,
            'module': self.module,
            'name': self.name,
            'args': [a.serialize() for a in self.args],
            'kwargs': {k: a.serialize() for k, a in self.kwargs.items()}
        }

    def __init__(self, runner_cls=None, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        self.has_runner = runner_cls is not None
        if self.has_runner:
            self.module = runner_cls.__module__
            self.name = runner_cls.__name__
        else:
            self.module = None
            self.name = None
        self.args = [ensure_constant(a) for a in args]
        self.kwargs = {k: ensure_constant(a) for k, a in kwargs.items()}


class Invocation(abc.ABC):
    def serialize_runner(self):
        return self._runner.serialize()

    def set_upstream_runner(self, runner_cls, *args, **kwargs):
        self._runner = RunnerWrapper(runner_cls, args, kwargs)

    def __init__(self, runner=None):
        if runner is None:
            self._runner = RunnerWrapper()
        else:
            self._runner = runner


class PipelineFunctionWrapper:
    @staticmethod
    def deserialize(serialized):
        fun = getattr(importlib.import_module(serialized['module']), serialized['name'])

        if isinstance(fun, PipelineFunctionWrapper):
            return fun

        return PipelineFunctionWrapper(fun,
                                       iter_output=serialized['iter_output'],
                                       n_epochs=serialized['n_epochs'])

    def serialize(self):
        return {
            'type': 'fun',
            'module': self.raw.__module__,
            'name': self.raw.__name__,
            'iter_output': self.iter_output,
            'n_epochs': self.n_epochs
        }

    def __init__(self, wrapped, iter_output=False, n_epochs=None):
        self.raw = wrapped
        self.iter_output = iter_output
        self.n_epochs = n_epochs

    def __call__(self, *args, **kwargs):
        return PipelineFunctionInvocation(self, args, kwargs)


class PipelineFunctionInvocation(Wrapped, Invocation):
    @staticmethod
    def deserialize(serialized):
        fn_wrapper = PipelineFunctionWrapper.deserialize(serialized['fun'])
        args = [deserialize(a) for a in serialized['args']]
        kwargs = {k: deserialize(a) for k, a in serialized['kwargs'].items()}
        runner = RunnerWrapper.deserialize(serialized['runner'])
        return PipelineFunctionInvocation(fn_wrapper,
                                          args=args,
                                          kwargs=kwargs,
                                          runner=runner)

    def serialize(self):
        return {
            'type': 'fun_invoke',
            'fun': self.fn_wrapper.serialize(),
            'args': [a.serialize() for a in self.args],
            'kwargs': {k: v.serialize() for k, v in self.kwargs.items()},
            'runner': self.serialize_runner()
        }

    def __init__(self, fn_wrapper, args, kwargs, runner=None):
        super().__init__(runner)
        self.fn_wrapper = fn_wrapper
        self.args = [ensure_wrapped(a) for a in args]
        self.kwargs = {k: ensure_wrapped(v) for k, v in kwargs.items()}
        self.iter_output = fn_wrapper.iter_output
        self.iter_output_buffer = iter(())
        self.n_epochs_left = fn_wrapper.n_epochs

    def start(self):
        for a in self.args:
            a.start()

        for a in self.kwargs.values():
            a.start()

    def __next__(self):
        if self.iter_output:
            try:
                return next(self.iter_output_buffer)
            except StopIteration:
                if self.n_epochs_left == 0:
                    raise

                if self.n_epochs_left is not None:
                    self.n_epochs_left -= 1

                self.iter_output_buffer = iter(self._invoke_raw())
                return next(self.iter_output_buffer)
        else:
            return self._invoke_raw()

    def __iter__(self):
        return self

    def _invoke_raw(self):
        args = (next(a) for a in self.args)
        kwargs = {k: next(a) for k, a in self.kwargs.items()}
        return self.fn_wrapper.raw(*args, **kwargs)


class PipelineClassWrapper:
    @staticmethod
    def deserialize(serialized):
        cls = getattr(importlib.import_module(serialized['module']), serialized['name'])

        if isinstance(cls, PipelineClassWrapper):
            return cls

        return PipelineClassWrapper(cls,
                                    iter_output=serialized['iter_output'],
                                    n_epochs=serialized['n_epochs'])

    def serialize(self):
        return {
            'type': 'cls',
            'iter_output': self.iter_output,
            'n_epochs': self.n_epochs,
            'module': self.raw.__module__,
            'name': self.raw.__name__
        }

    def __init__(self, cls, iter_output=False, n_epochs=None):
        self.iter_output = iter_output
        self.n_epochs = n_epochs
        self.raw = cls

    def __call__(self, *args, **kwargs):
        return PipelineClassWrapperInstance(self, args, kwargs)

    def start(self):
        pass


class PipelineClassWrapperInstance:
    @staticmethod
    def deserialize(serialized):
        cls = PipelineClassWrapper.deserialize(serialized['cls'])
        args = [deserialize(a) for a in serialized['args']]
        kwargs = {k: deserialize(a) for k, a in serialized['kwargs'].items()}
        return PipelineClassWrapperInstance(cls, args, kwargs)

    def serialize(self):
        return {
            'type': 'obj',
            'cls': self.cls_wrapper.serialize(),
            'args': [a.serialize() for a in self.args],
            'kwargs': {k: a.serialize() for k, a in self.kwargs.items()}
        }

    def __init__(self, cls_wrapper, args, kwargs):
        self.cls_wrapper = cls_wrapper
        self.args = [ensure_component_argument(a) for a in args]
        self.kwargs = {k: ensure_component_argument(a) for k, a in kwargs.items()}

    def __call__(self, *args, **kwargs):
        return PipelineClassWrapperInstanceInvocation(self, None, args, kwargs)

    def __getattr__(self, method):
        if hasattr(self.cls_wrapper.raw, method) and callable(getattr(self.cls_wrapper.raw, method)):
            def f(*args, **kwargs):
                return PipelineClassWrapperInstanceInvocation(self, method, args, kwargs)

            return f
        raise Exception(f'Class {self.cls_wrapper.raw} has no callable member {method}')

    def start(self):
        for a in self.args:
            a.start()

        for a in self.kwargs.values():
            a.start()

        args = [a.raw for a in self.args]
        kwargs = {k: a.raw for k, a in self.kwargs.items()}
        return self.cls_wrapper.raw(*args, **kwargs)


class PipelineClassWrapperInstanceInvocation(Wrapped, Invocation):
    @staticmethod
    def deserialize(serialized):
        args = [deserialize(a) for a in serialized['args']]
        kwargs = {k: deserialize(a) for k, a in serialized['kwargs'].items()}
        inst = PipelineClassWrapperInstance.deserialize(serialized['inst'])
        runner = RunnerWrapper.deserialize(serialized['runner'])
        return PipelineClassWrapperInstanceInvocation(inst, serialized['method'], args, kwargs, runner)

    def serialize(self):
        return {
            'type': 'obj_invoke',
            'inst': self.inst_wrapper.serialize(),
            'method': self.method,
            'args': [a.serialize() for a in self.args],
            'kwargs': {k: a.serialize() for k, a in self.kwargs.items()},
            'runner': self.serialize_runner()
        }

    def __init__(self, inst_wrapper, method, args, kwargs, runner=None):
        super().__init__(runner)
        self.inst_wrapper = inst_wrapper
        self.method = method
        self.args = [ensure_wrapped(a) for a in args]
        self.kwargs = {k: ensure_wrapped(a) for k, a in kwargs.items()}
        self.inst = None
        self.iter_output_buffer = iter(())
        self.iter_output = inst_wrapper.cls_wrapper.iter_output
        self.n_epochs_left = inst_wrapper.cls_wrapper.n_epochs

    def __next__(self):
        if self.iter_output:
            try:
                return next(self.iter_output_buffer)
            except StopIteration:
                if self.n_epochs_left == 0:
                    raise StopIteration

                if self.n_epochs_left is not None:
                    self.n_epochs_left -= 1

                self.iter_output_buffer = iter(self._invoke_raw())
                return next(self.iter_output_buffer)
        else:
            return self._invoke_raw()

    def __iter__(self):
        return self

    def _invoke_raw(self):
        args = [next(a) for a in self.args]
        kwargs = {k: next(a) for k, a in self.kwargs.items()}
        if self.method is None:
            return self.inst(*args, **kwargs)
        else:
            return getattr(self.inst, self.method)(*args, **kwargs)

    def start(self):
        for a in self.args:
            a.start()

        for a in self.kwargs.values():
            a.start()

        self.inst = self.inst_wrapper.start()


class SwitchCase(Wrapped):
    @staticmethod
    def deserialize(serialized):
        return SwitchCase(deserialize(serialized['selector']),
                          {k: deserialize(v) for k, v in serialized['choices'].items()})

    def serialize(self):
        return {
            'type': 'switch',
            'selector': self.selector.serialize(),
            'choices': {k: v.serialize() for k, v in self.choices.items()}
        }

    def __init__(self, selector, choices):
        self.selector = ensure_wrapped(selector)
        self.choices = {k: ensure_wrapped(v) for k, v in choices.items()}

    def __next__(self):
        selector_value = next(self.selector)
        chosen = self.choices[selector_value]
        return next(chosen)

    def __iter__(self):
        return self

    def start(self):
        self.selector.start()
        for a in self.choices.values():
            a.start()


def switch_case(value, **choice_dict):
    return SwitchCase(value, choice_dict)


def pipeline(*args, iter_output=False, n_epochs=None):
    if len(args) == 1:
        return pipeline()(*args)

    def decorator(f):
        if inspect.isclass(f):
            return PipelineClassWrapper(f, iter_output=iter_output, n_epochs=n_epochs)
        else:
            return PipelineFunctionWrapper(f, iter_output=iter_output, n_epochs=n_epochs)

    return decorator


class ComponentWrapper:
    @staticmethod
    def deserialize(serialized):
        cls = getattr(importlib.import_module(serialized['module']), serialized['name'])

        if isinstance(cls, ComponentWrapper):
            return cls

        return ComponentWrapper(cls)

    def serialize(self):
        return {
            'type': 'comp_cls',
            'module': self.raw.__module__,
            'name': self.raw.__name__
        }

    def __init__(self, cls):
        self.raw = cls

    def __call__(self, *args, **kwargs):
        return ComponentWrapperInstance(self, args, kwargs)


class ComponentWrapperInstance:
    @staticmethod
    def deserialize(serialized):
        comp = ComponentWrapper.deserialize(serialized['comp'])
        args = [deserialize(a) for a in serialized['args']]
        kwargs = {k: deserialize(a) for k, a in serialized['kwargs']}
        return ComponentWrapperInstance(comp, args, kwargs)

    def serialize(self):
        return {
            'type': 'comp_obj',
            'comp': self.comp.serialize(),
            'args': [a.serialize() for a in self.args],
            'kwargs': {k: a.serialize() for k, a in self.kwargs.items()}
        }

    def __init__(self, comp, args, kwargs):
        self.comp = comp
        self.args = [ensure_component_argument(a) for a in args]
        self.kwargs = {k: ensure_component_argument(a) for k, a in kwargs.items()}
        self.inst = None

    def start(self):
        for a in self.args:
            a.start()
        for a in self.kwargs.values():
            a.start()
        args = [a.raw for a in self.args]
        kwargs = {k: a.raw for k, a in self.kwargs.items()}
        self.inst = self.comp.raw(*args, **kwargs)

    @property
    def raw(self):
        return self.inst

    def __call__(self, *args, **kwargs):
        return self.inst(*args, **kwargs)

    def __getattr__(self, method):
        if self.inst and hasattr(self.inst, method) and callable(getattr(self.inst, method)):
            return getattr(self.inst, method)

        raise KeyError(f'Class {self.comp.raw} has no callable member {method}')


def component(*args, **kwargs):
    if len(args) == 1:
        return component()(*args)

    def decorator(o):
        if inspect.isclass(o):
            return ComponentWrapper(o)

    return decorator


def ensure_component_argument(arg):
    if isinstance(arg, ComponentWrapperInstance) or isinstance(arg, Constant):
        return arg
    return Constant(arg)


type_map = {
    'fun_invoke': PipelineFunctionInvocation,
    'obj_invoke': PipelineClassWrapperInstanceInvocation,
    'const': Constant,
    'switch': SwitchCase,
    'comp_obj': ComponentWrapperInstance
}


def deserialize(serialized):
    return type_map[serialized['type']].deserialize(serialized)
