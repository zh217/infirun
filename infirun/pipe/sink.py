import abc
import threading
import asyncio
import queue
import multiprocessing.dummy as dummy

__all__ = ['QueueSink', 'IterSink', 'ClassRunnerSink', 'SinkRunner']


class Sink(abc.ABC):
    def __init__(self, source):
        self._source = source
        self.running = False
        self._loop = None
        self._loop_thread = None

    @abc.abstractmethod
    async def main_loop(self):
        pass

    @abc.abstractmethod
    def return_value(self):
        pass

    @abc.abstractmethod
    def setup_before_start(self, loop, loop_thread):
        pass

    def start(self):
        self._loop = asyncio.new_event_loop()
        self._source.start(self._loop)
        self._setup_main_loop_thread()
        self.setup_before_start(self._loop, self._loop_thread)
        self.running = True
        self._loop_thread.start()
        return self.return_value()

    def _setup_main_loop_thread(self):
        self._loop_thread = threading.Thread(target=self._run_loop,
                                             args=(self._loop, self.main_loop()))
        self._loop_thread.daemon = True

    @staticmethod
    def _run_loop(loop, coro):
        try:
            loop.run_until_complete(coro)
        except asyncio.CancelledError:
            pass

    def stop(self):
        self._source.stop()
        self.running = False
        self._stop_all_loop_tasks()
        self.cleanup_after_tasks_cancelled()

    def _stop_all_loop_tasks(self):
        for task in asyncio.Task.all_tasks(self._loop):
            task.cancel()

    @abc.abstractmethod
    def cleanup_after_tasks_cancelled(self):
        pass

    def __enter__(self):
        self.start()
        return self.return_value()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class QueueSink(Sink):
    def __init__(self, source, buffer=1):
        super().__init__(source)
        self._buffer = buffer
        self._q = dummy.Queue()
        self._semaphore = None

    async def main_loop(self):
        try:
            while self.running:
                result = await self._source.get_next_value()
                await self._semaphore.acquire()
                self._q.put(result)
        except asyncio.CancelledError:
            pass

    def _get_value(self):
        value = self._q.get()
        self._loop.call_soon_threadsafe(lambda: self._semaphore.release())
        if isinstance(value, Exception):
            raise value
        return value

    def setup_before_start(self, loop, loop_thread):
        self._semaphore = asyncio.Semaphore(loop=loop, value=self._buffer)

    def return_value(self):
        return self._get_value

    def cleanup_after_tasks_cancelled(self):
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break


class IterSink(QueueSink):
    def return_value(self):
        getter = super().return_value()
        while True:
            yield getter()


class ClassRunnerSink(QueueSink):
    def __call__(self, cls, *args, **kwargs):
        self._instance = cls(*args, **kwargs)
        return self

    def return_value(self):
        getter = super().return_value()
        while True:
            value = getter()
            self._instance(value)
            if self._instance.stopped:
                break
        self.stop()
        return self._instance.ret_value


class SinkRunner(abc.ABC):
    def __init__(self):
        self.stopped = False
        self.ret_value = None

    def halt(self, ret_value=None):
        self.stopped = True
        self.ret_value = ret_value
