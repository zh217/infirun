import inspect
import importlib
import traceback

from .wrapped import ensure_wrapped, Wrapped, Constant


class PipelineFunctionInvocation(Wrapped):
    @staticmethod
    def deserialize(serialized):
        fn_wrapper = PipelineFunctionWrapper.deserialize(serialized['fun'])
        args = [deserialize(a) for a in serialized['args']]
        kwargs = {k: deserialize(a) for k, a in serialized['kwargs'].items()}
        return PipelineFunctionInvocation(fn_wrapper,
                                          args=args,
                                          kwargs=kwargs)

    def __init__(self, fn_wrapper, args, kwargs):
        self.fn_wrapper = fn_wrapper
        self.args = [ensure_wrapped(a) for a in args]
        self.kwargs = {k: ensure_wrapped(v) for k, v in kwargs.items()}
        self.iter_output = fn_wrapper.iter_output
        self.iter_output_buffer = iter(())
        self.n_epochs_left = fn_wrapper.n_epochs

    def serialize(self):
        return {
            'type': 'fun_invoke',
            'fun': self.fn_wrapper.serialize(),
            'args': [a.serialize() for a in self.args],
            'kwargs': {k: v.serialize() for k, v in self.kwargs.items()}
        }

    def start(self):
        pass

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


class PipelineFunctionWrapper:
    @staticmethod
    def deserialize(serialized):
        fun = getattr(importlib.import_module(serialized['module']), serialized['name'])

        if isinstance(fun, PipelineFunctionWrapper):
            return fun

        return PipelineFunctionWrapper(fun,
                                       iter_output=serialized['iter_output'],
                                       n_epochs=serialized['n_epochs'])

    def __init__(self, wrapped, iter_output=False, n_epochs=None):
        self.raw = wrapped
        self.iter_output = iter_output
        self.n_epochs = n_epochs

    def __call__(self, *args, **kwargs):
        return PipelineFunctionInvocation(self, args, kwargs)

    def serialize(self):
        return {
            'type': 'fun',
            'module': self.raw.__module__,
            'name': self.raw.__name__,
            'iter_output': self.iter_output,
            'n_epochs': self.n_epochs
        }


class PipelineClassWrapperInstanceInvocation(Wrapped):
    @staticmethod
    def deserialize(serialized):
        pass

    def __init__(self, inst_wrapper, method, args, kwargs):
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

    def serialize(self):
        return {
            'type': 'obj_invoke',
            'inst': self.inst_wrapper.serialize(),
            'method': self.method,
            'args': [a.serialize() for a in self.args],
            'kwargs': {k: a.serialize() for k, a in self.kwargs.items()}
        }


class PipelineClassWrapperInstance:
    @staticmethod
    def deserialize(serialized):
        pass

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

    def serialize(self):
        return {
            'type': 'obj',
            'cls': self.cls_wrapper.serialize(),
            'args': [a.serialize() for a in self.args],
            'kwargs': {k: a.serialize() for k, a in self.kwargs.items()}
        }


class PipelineClassWrapper:
    @staticmethod
    def deserialize(serialized):
        cls = getattr(importlib.import_module(serialized['module']), serialized['name'])

        if isinstance(cls, PipelineClassWrapper):
            return cls

        return PipelineClassWrapper(cls,
                                    iter_output=serialized['iter_output'],
                                    n_epochs=serialized['n_epochs'])

    def __init__(self, cls, iter_output=False, n_epochs=None):
        self.iter_output = iter_output
        self.n_epochs = n_epochs
        self.raw = cls

    def __call__(self, *args, **kwargs):
        return PipelineClassWrapperInstance(self, args, kwargs)

    def start(self):
        pass

    def serialize(self):
        return {
            'type': 'cls',
            'iter_output': self.iter_output,
            'n_epochs': self.n_epochs,
            'module': self.raw.__module__,
            'name': self.raw.__name__
        }


def _make_pipeline_decorator(**kwargs):
    def decorator(f):
        if inspect.isclass(f):
            return PipelineClassWrapper(f, **kwargs)
        else:
            return PipelineFunctionWrapper(f, **kwargs)

    return decorator


def pipeline(*args, iter_output=False, n_epochs=None):
    if len(args) == 1:
        return pipeline()(*args)

    return _make_pipeline_decorator(iter_output=iter_output, n_epochs=n_epochs)


def deserialize(serialized):
    s_type = serialized['type']

    if s_type == 'fun_invoke':
        return PipelineFunctionInvocation.deserialize(serialized)
    elif s_type == 'const':
        return Constant.deserialize(serialized)
    elif s_type == 'switch':
        return SwitchCase.deserialize(serialized)
    elif s_type == 'comp_obj':
        return ComponentWrapperInstance.deserialize(serialized)


class SwitchCase(Wrapped):
    @staticmethod
    def deserialize(serialized):
        return SwitchCase(deserialize(serialized['selector']),
                          {k: deserialize(v) for k, v in serialized['choices'].items()})

    def __init__(self, selector, choices):
        self.selector = ensure_wrapped(selector)
        self.choices = {k: ensure_wrapped(v) for k, v in choices.items()}

    def __next__(self):
        selector_value = next(self.selector)
        chosen = self.choices[selector_value]
        return next(chosen)

    def __iter__(self):
        return self

    def serialize(self):
        return {
            'type': 'switch',
            'selector': self.selector.serialize(),
            'choices': {k: v.serialize() for k, v in self.choices.items()}
        }


def switch_case(value, **choice_dict):
    return SwitchCase(value, choice_dict)


class ComponentWrapper:
    @staticmethod
    def deserialize(serialized):
        cls = getattr(importlib.import_module(serialized['module']), serialized['name'])

        if isinstance(cls, ComponentWrapper):
            return cls

        return ComponentWrapper(cls)

    def __init__(self, cls):
        self.raw = cls

    def __call__(self, *args, **kwargs):
        return ComponentWrapperInstance(self, args, kwargs)

    def serialize(self):
        return {
            'type': 'comp_cls',
            'module': self.raw.__module__,
            'name': self.raw.__name__
        }


class ComponentWrapperInstance:
    @staticmethod
    def deserialize(serialized):
        comp = ComponentWrapper.deserialize(serialized['comp'])
        args = [deserialize(a) for a in serialized['args']]
        kwargs = {k: deserialize(a) for k, a in serialized['kwargs']}
        return ComponentWrapperInstance(comp, args, kwargs)

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

    def serialize(self):
        return {
            'type': 'comp_obj',
            'comp': self.comp.serialize(),
            'args': [a.serialize() for a in self.args],
            'kwargs': {k: a.serialize() for k, a in self.kwargs.items()}
        }


def component(*args, **kwargs):
    if len(args) == 1:
        return component()(*args)

    return _make_component_decorator(**kwargs)


def _make_component_decorator(**kwargs):
    def decorator(o):
        if inspect.isclass(o):
            return ComponentWrapper(o)

    return decorator


def ensure_component_argument(arg):
    if isinstance(arg, ComponentWrapperInstance) or isinstance(arg, Constant):
        return arg
    return Constant(arg)
