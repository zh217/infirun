import abc


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
    if serialized['type'] not in ['func_invoke', 'pipe_inst_invoke']:
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
