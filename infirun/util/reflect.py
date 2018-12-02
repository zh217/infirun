import inspect
import json


def get_module(fun):
    return fun.__module__


def get_name(fun):
    return fun.__name__


def get_source_data(fun_or_cls):
    if inspect.isclass(fun_or_cls):
        type_str = 'cls'
    elif inspect.isfunction(fun_or_cls):
        type_str = 'fun'
    else:
        raise ValueError('fun_or_cls not function nor class')
    return {'type': type_str,
            'mod': get_module(fun_or_cls),
            'name': get_name(fun_or_cls)}


def ensure_basic_data_type(arg):
    try:
        if json.loads(json.dumps(arg)) == arg:
            return {'type': 'const', 'value': arg}
        else:
            raise ValueError('Argument is not basic', arg)
    except TypeError:
        raise ValueError('Argument is not basic', arg)
