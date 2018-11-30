import inspect
import json


def get_module(func):
    return func.__module__


def get_name(func):
    return func.__name__


def get_source_data(func_or_cls):
    if inspect.isclass(func_or_cls):
        type_str = 'cls'
    elif inspect.isfunction(func_or_cls):
        type_str = 'func'
    else:
        raise ValueError('func_or_cls not function nor class')
    return {'type': type_str,
            'mod': get_module(func_or_cls),
            'name': get_name(func_or_cls)}


def ensure_basic_data_type(arg):
    try:
        if json.loads(json.dumps(arg)) == arg:
            return {'type': 'const', 'value': arg}
        else:
            raise ValueError('Argument is not basic', arg)
    except TypeError:
        raise ValueError('Argument is not basic', arg)
