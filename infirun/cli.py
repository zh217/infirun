import importlib
import inspect
import sys
import json

from infirun.runner import run_with_runner


def run_serialized(module, name, output_iter=False):
    mod = importlib.import_module(module)
    fun_or_dict = getattr(mod, name)
    if inspect.isfunction(fun_or_dict):
        fun_or_dict = fun_or_dict()
    return run_with_runner(fun_or_dict, output_iter)


def main():
    try:
        if ':' in sys.argv[1]:
            _module, _name = sys.argv[1].split(':')
            it = run_serialized(_module, _name, output_iter=True)
        else:
            with open(sys.argv[1], 'r', encoding='utf-8') as f:
                serialized = json.load(f)
            it = run_with_runner(serialized, True)

        for el in it:
            print(el)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == '__main__':
    main()
