import copy

"""
The plan:

* Start with config stating available resources
* Periodically poll server for work to do, offering available resources
* If server state changes, kill subprocess accordingly, and update server

"""


class JobConflict(Exception):
    pass


class InsufficientResource(Exception):
    pass


class ResourceManager:
    def __init__(self, resource_map):
        for v in resource_map.values():
            assert v >= 0

        self.resource_map = copy.deepcopy(resource_map)
        self.active_resources = {}

    def remaining(self):
        ret = copy.deepcopy(self.resource_map)
        for active in self.active_resources.values():
            for k, v in active.items():
                ret[k] -= v
        return ret

    def release_resources(self, handle):
        del self.active_resources[handle]

    def acquire_resources(self, handle, active_resource_map):
        if handle in self.active_resources:
            raise JobConflict
        available = self.remaining()
        for k, v in active_resource_map.items():
            if v > available[k]:
                raise InsufficientResource(f'Resource {k}, want {v}, have {available[k]}')
        self.active_resources[handle] = active_resource_map


class LocalWorkerManager:
    def __init__(self, resource_map, server_url):
        self.resources = ResourceManager(resource_map)
        self.server_url = server_url
        self.run_loop()

    def main_loop(self):
        pass

    def run_loop(self):
        pass

    def poll_for_job(self):
        pass

    def start_job(self):
        pass

    def stop_job(self):
        pass

    def finalize_job(self):
        pass


class JobStateReporter:
    pass


class JobManager:
    def ensure_git_repo(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass
