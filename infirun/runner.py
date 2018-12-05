import abc
import copy
import multiprocessing.managers
import sys

import tarjan

from infirun.signal import GracefulExit, setup_signal_handlers
from .wrapped import ensure_unwrapped_constant
from .pipeline import RunnerWrapper, run_in_process, deserialize, StateDictProxy


class Runner(abc.ABC):
    pass


class ProcessRunner(Runner):

    def __init__(self, queue_size=1, n_process=1, process_type='thread', state_dict_proxy=None, idx=None):
        self.state_dict_proxy = state_dict_proxy
        self.idx = idx
        assert process_type in ['thread', 'pytorch', 'process']
        self.queue_size = ensure_unwrapped_constant(queue_size)
        assert self.queue_size >= 1
        self.n_process = ensure_unwrapped_constant(n_process)
        assert self.n_process >= 1
        self.process_type = ensure_unwrapped_constant(process_type)
        self.started = False
        self.stop_switches = []
        self.error_switch = None
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

    def get_process_constructor_module(self):
        if self.process_type == 'pytorch':
            import torch.multiprocessing
            return torch.multiprocessing
        elif self.process_type == 'process':
            import multiprocessing
            return multiprocessing
        else:
            import multiprocessing.dummy
            return multiprocessing.dummy

    def start(self, serialized, downsteam_q, upstream_qs):
        if self.started:
            raise RuntimeError('Runner already started')

        mod = self.get_process_constructor_module()
        self.error_switch = mod.Value('b', 0)

        for i in range(self.n_process):
            stop_switch = mod.Value('b', 0, lock=False)
            p = mod.Process(target=run_in_process,
                            args=(serialized, downsteam_q, upstream_qs, i, self.process_type != 'thread', stop_switch,
                                  self.error_switch, self.state_dict_proxy))
            self.stop_switches.append(stop_switch)
            self.processes.append(p)
            p.daemon = True
            p.start()

        self.started = True

    def join(self):
        for p in self.processes:
            p.join()

    def terminate(self):
        for v in self.stop_switches:
            v.value = 1
        for p in self.processes:
            if p.is_alive() and self.process_type != 'thread':
                try:
                    p.join(timeout=1)
                    p.terminate()
                except:
                    pass
            elif p.exitcode != 0:
                raise RuntimeError(f'Worker {self.idx} exited with non-zero status ${p.exitcode}')
            elif self.error_switch.value:
                raise RuntimeError(
                    f'At least {self.error_switch.value} subprocesses of worker {self.idx} exited with exceptions')


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


def _ensure_unique_names(serialized, seen):
    if serialized.get('type', None) not in ['fun_invoke', 'obj_invoke', 'comp_obj', 'obj']:
        return

    if serialized['name'] in seen:
        raise RuntimeError(f'Duplicate name in serialization: {serialized["name"]}')

    seen.add(serialized['name'])

    for a in serialized['args'] + list(serialized['kwargs'].values()):
        _ensure_unique_names(a, seen)

    inst = serialized.get('inst')

    if inst:
        _ensure_unique_names(inst, seen)


class StateManager(multiprocessing.managers.BaseManager):
    pass


StateManager.register('StateDictProxy', StateDictProxy)


def run_with_runner(serialized, return_iter=False, state_change_callback=None):
    setup_signal_handlers()

    state_manager = StateManager()
    state_manager.start()

    state_dict = get_const_params(serialized)
    state_dict_proxy = state_manager.StateDictProxy(state_dict, state_change_callback)

    dep_maps, order = generate_dependency_map(serialized)
    runner_map = {}
    for k, v in dep_maps.items():
        runner_def = v['runner']
        args = [deserialize(a).raw for a in runner_def['args']]
        kwargs = {k: deserialize(a).raw for k, a in runner_def['kwargs'].items()}
        runner_map[k] = ProcessRunner(*args, idx=k, state_dict_proxy=state_dict_proxy, **kwargs)

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

    try:
        for i, upstream_idxs, downstream_idx in order:
            runner = runner_map[i]
            serialized = dep_maps[i]
            down_q = downstream_queues[i]
            up_qs = {k: downstream_queues[k] for k in upstream_idxs}
            runner.start(serialized, down_q, up_qs)

        if return_iter:
            return _queue_to_gen(downstream_queues[0])
        else:
            root_runner.join()
            for i, _, _ in reversed(order):
                runner_map[i].terminate()

        return state_dict_proxy.current_state()
    except GracefulExit:
        print('Exiting due to interrupt')
        for i, _, _ in reversed(order):
            try:
                runner_map[i].terminate()
            except:
                pass
        sys.exit(0)


def get_const_params(serialized, collected=None):
    if collected is None:
        _ensure_unique_names(serialized, set())
        collected = {}

    if serialized.get('type', None) in ['fun_invoke', 'obj_invoke', 'comp_obj', 'obj']:

        for k, a in list(enumerate(serialized['args'])) + list(serialized['kwargs'].items()):
            if a.get('type', None) == 'const':
                arg_map = collected.setdefault(serialized['name'], {})
                arg_map[k] = a['value']
            else:
                get_const_params(a, collected)

    inst = serialized.get('inst', None)
    if inst:
        get_const_params(inst, collected)

    return collected


def _override_const_params(serialized, params_dict):
    if serialized.get('type', None) in ['fun_invoke', 'obj_invoke', 'comp_obj', 'obj']:

        for k, a in list(enumerate(serialized['args'])) + list(serialized['kwargs'].items()):
            if a.get('type', None) == 'const':
                if serialized['name'] in params_dict and k in params_dict[serialized['name']]:
                    a['value'] = params_dict[serialized['name']][k]
            else:
                _override_const_params(a, params_dict)

    inst = serialized.get('inst', None)
    if inst:
        _override_const_params(inst, params_dict)


def override_const_params(serialized, params_dict):
    serialized = copy.deepcopy(serialized)
    _override_const_params(serialized, params_dict)
    return serialized
