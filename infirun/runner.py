import abc
import copy
import tarjan

from .wrapped import ensure_unwrapped_constant
from .pipeline import RunnerWrapper, run_in_process, deserialize


class Runner(abc.ABC):
    pass


class ProcessRunner(Runner):

    def __init__(self, queue_size=1, n_process=1, process_type='thread'):
        assert process_type in ['thread', 'pytorch', 'process']
        self.queue_size = ensure_unwrapped_constant(queue_size)
        assert self.queue_size >= 1
        self.n_process = ensure_unwrapped_constant(n_process)
        assert self.n_process >= 1
        self.process_type = ensure_unwrapped_constant(process_type)
        self.started = False
        self.processes = []

    def negotiate_queue_with_downstream(self, other):
        if other is None:
            return None

        if self.process_type == 'pytorch' or other.process_type == 'pytorch':
            import torch.multiprocessing
            return torch.multiprocessing.Queue
        elif self.process_type == 'process' or other.process_type == 'process':
            import multiprocessing
            return multiprocessing.Queue
        elif self.process_type == 'thread' or other.process_type == 'thread':
            import multiprocessing.dummy
            return multiprocessing.dummy.Queue

    def get_process_constructor(self):
        if self.process_type == 'pytorch':
            import torch.multiprocessing
            return torch.multiprocessing.Process
        elif self.process_type == 'process':
            import multiprocessing
            return multiprocessing.Process
        else:
            import multiprocessing.dummy
            return multiprocessing.dummy.Process

    def start(self, serialized, downsteam_q, upstream_qs):
        if self.started:
            raise RuntimeError('Runner already started')

        for i in range(self.n_process):
            Process = self.get_process_constructor()
            p = Process(target=run_in_process, args=(serialized, downsteam_q, upstream_qs, i))
            self.processes.append(p)
            p.daemon = True
            p.start()

        self.started = True


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


def _should_chop(serialized):
    return serialized['type'] in ['fun_invoke', 'obj_invoke'] and serialized['runner']['has_runner']


def _chop_serialized(serialized, n, collected):
    # ret = {k: v for k, v in serialized.items()}
    ret = serialized
    current = n

    if ret['type'] == 'const':
        return current, ret

    for i in range(len(ret['args'])):
        arg = ret['args'][i]
        current, chopped = _chop_serialized(arg, current, collected)
        if _should_chop(arg):
            current += 1
            ret['args'][i] = {'type': 'placeholder',
                              'name': arg['name'],
                              'idx': current}
            collected[current] = chopped

    for k in ret['kwargs']:
        arg = ret['kwargs'][k]
        current, chopped = _chop_serialized(arg, current, collected)
        if _should_chop(arg):
            current += 1
            ret['kwargs'][k] = {'type': 'placeholder',
                                'name': arg['name'],
                                'idx': current}
            collected[current] = chopped

    return current, ret


def _generate_dependency(serialized):
    found = []
    for arg in serialized['args']:
        if arg['type'] == 'placeholder':
            found.append(arg['idx'])
        elif arg['type'] in ['fun_invoke', 'obj_invoke']:
            found.extend(_generate_dependency(arg))

    for arg in serialized['kwargs'].values():
        if arg['type'] == 'placeholder':
            found.append(arg['idx'])
        elif arg['type'] in ['fun_invoke', 'obj_invoke']:
            found.extend(_generate_dependency(arg))
    return found


def generate_dependency_map(serialized):
    serialized = copy.deepcopy(serialized)
    collected = {}
    n, root = _chop_serialized(serialized, 0, collected)
    if not root['runner']['has_runner']:
        root['runner'] = RunnerWrapper(ProcessRunner).serialize()
    collected[0] = root
    deps = {k: _generate_dependency(v) for k, v in collected.items()}
    inv_deps = {}
    for k, vs in deps.items():
        for v in vs:
            inv_deps[v] = k
    _ordered = tarjan.tarjan(deps)
    ordered = []
    for o in _ordered:
        if len(o) > 1:
            raise RuntimeError('Cycle detected in dataflow')
        ordered.append((o[0], deps[o[0]], inv_deps.get(o[0], None)))
    return collected, ordered


def _queue_to_gen(q):
    while True:
        data = q.get()
        if data == StopIteration:
            break
        yield data


def run_with_runner(serialized, return_iter=False):
    dep_maps, order = generate_dependency_map(serialized)
    runner_map = {}
    for k, v in dep_maps.items():
        runner_def = v['runner']
        args = [deserialize(a).raw for a in runner_def['args']]
        kwargs = {k: deserialize(a).raw for k, a in runner_def['kwargs'].items()}
        runner_map[k] = ProcessRunner(*args, **kwargs)

    downstream_queues = {}

    for i, _, downstream_idx in order:
        runner = runner_map[i]
        downstream_runner = runner_map.get(downstream_idx, None)
        Queue = runner.negotiate_queue_with_downstream(downstream_runner)
        downstream_queues[i] = Queue and Queue(maxsize=runner.queue_size)

    root_runner = runner_map[0]

    if return_iter:
        Queue = root_runner.negotiate_queue_with_downstream(root_runner)
        downstream_queues[0] = Queue(maxsize=1)

    for i, upstream_idxs, downstream_idx in order:
        runner = runner_map[i]
        serialized = dep_maps[i]
        down_q = downstream_queues[i]
        up_qs = {k: downstream_queues[k] for k in upstream_idxs}
        runner.start(serialized, down_q, up_qs)

    if return_iter:
        return _queue_to_gen(downstream_queues[0])
    else:
        for p in root_runner.processes:
            p.join()
