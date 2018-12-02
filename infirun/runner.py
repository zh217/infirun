import abc

from .wrapped import Wrapped


class Runner(abc.ABC):

    def start_upstream_runners(self, serialized):
        pass


class ThreadPoolRunner(Runner):
    @staticmethod
    def deserialize(serialized):
        pass

    def __init__(self):
        pass

    def start(self, pipeline):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def serialize(self):
        pass


class ProcessPoolRunner(Runner):
    @staticmethod
    def deserialize(serialized):
        pass

    def __init__(self):
        pass

    def start(self, serialized):
        total, found_upstream_runner_dict = mark_serialization(serialized)

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def serialize(self):
        pass


def mark_serialization(serialized, n=0):
    if serialized['type'] not in ['fun_invoke', 'obj_invoke']:
        return serialized, n

    current = n
    found = {}

    for arg in serialized['args']:
        if 'runner' in arg:
            current += 1
            arg['runner_mark'] = current
            found[current] = arg
        else:
            current, new_found = mark_serialization(arg, current)
            found.update(new_found)

    for arg in serialized['kwargs'].values():
        if 'runner' in arg:
            current += 1
            arg['runner_mark'] = current
            found[current] = arg
        else:
            current, new_found = mark_serialization(arg, current)
            found.update(new_found)

    return current, found


class PipelineInProxy:
    def __init__(self, queue, pipeline):
        self.queue = queue
        self.pipeline = pipeline

    def start(self):
        try:
            while True:
                self.queue.put(next(self.pipeline))
        except StopIteration:
            pass


class PipelineOutProxy(Wrapped):
    def __init__(self, queue):
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self):
        return self.queue.get()

    def start(self):
        pass


class PipelineSink:
    def __init__(self, pipeline, callback=None):
        self.pipeline = pipeline
        self.callback = callback

    def start(self):
        self.pipeline.start()
        if self.callback:
            for v in self.pipeline:
                self.callback(v)
        else:
            for _ in self.pipeline:
                pass


def pprint(serialized, level=0, prefix='root', outfile=None):
    s_type = serialized['type']

    if s_type not in ['fun_invoke', 'obj_invoke', 'placeholder']:
        return

    name = serialized['name']
    name_str = f' <{name}>' if name else ''

    if s_type == 'placeholder':
        print(f'{"  " * level} {prefix} [P]:{name_str} {serialized["idx"]}')
        return

    runner_str = " [R]" if serialized["runner"]["has_runner"] else ""
    prefix_str = f'{"  " * level} {prefix}{runner_str}:{name_str} '

    if s_type == 'obj_invoke':
        print(prefix_str +
              f'{serialized["inst"]["cls"]["module"]}:'
              f'{serialized["inst"]["cls"]["name"]}().{serialized["method"] or "__call__"}'
              f'{" [I]" if serialized["inst"]["cls"]["iter_output"] else ""}')

    if s_type == 'fun_invoke':
        print(prefix_str +
              f'{serialized["fun"]["module"]}:{serialized["fun"]["name"]}'
              f'{" [I]" if serialized["fun"]["iter_output"] else ""}')

    for i, v in enumerate(serialized['args']):
        pprint(v, level + 1, prefix=str(i), outfile=outfile)

    for k, v in serialized['kwargs'].items():
        pprint(v, level + 1, prefix=k, outfile=outfile)


def should_chop(serialized):
    return serialized['type'] in ['fun_invoke', 'obj_invoke'] and serialized['runner']['has_runner']


def chop_serialized(serialized, n, collected):
    # ret = {k: v for k, v in serialized.items()}
    ret = serialized
    current = n

    for i in range(len(ret['args'])):
        arg = ret['args'][i]
        current, chopped = chop_serialized(arg, current, collected)
        if should_chop(arg):
            current += 1
            ret['args'][i] = {'type': 'placeholder',
                              'name': arg['name'],
                              'idx': current}
            collected[current] = chopped

    for k in ret['kwargs']:
        arg = ret['kwargs'][k]
        current, chopped = chop_serialized(arg, current, collected)
        if should_chop(arg):
            current += 1
            ret['kwargs'][k] = {'type': 'placeholder',
                                'name': arg['name'],
                                'idx': current}
            collected[current] = chopped

    return current, ret
