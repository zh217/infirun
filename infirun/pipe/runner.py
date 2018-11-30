import asyncio
import functools
import abc
import inspect
import multiprocessing
import threading
import math

__all__ = ['AsyncRunner', 'ProcessRunner']


class Runner(abc.ABC):
    def __init__(self, loop, q, f, init_args, init_kwargs, args_stream):
        self.loop = loop
        self.q = q
        self.f = f
        self.init_args = init_args
        self.init_kwargs = init_kwargs
        self.running = False
        self.args_stream = args_stream
        self.flatten = inspect.isgeneratorfunction(f)
        self._main_loop_task = None

    def setup(self, *args, **kwargs):
        pass

    def teardown(self):
        pass

    async def get_next_args(self):
        while True:
            try:
                args, kwargs = await self.args_stream.get_next_args()
                return args, kwargs
            except Exception as e:
                await self.put_to_downstream(e)

    def force_flatten(self, flatten):
        self.flatten = flatten

    async def put_to_downstream(self, value):
        if self.flatten:
            try:
                for v in value:
                    await self.q.put(v)
            except Exception as v:
                await self.q.put(v)

        else:
            await self.q.put(value)

    @abc.abstractmethod
    async def main_loop(self, f, init_args, init_kwargs):
        pass

    def start(self):
        self.running = True
        self._main_loop_task = self.loop.create_task(self.main_loop(self.f,
                                                                    self.init_args,
                                                                    self.init_kwargs))
        return self

    def stop(self):
        self.running = False
        self._main_loop_task.cancel()
        self.teardown()


class AsyncRunner(Runner):
    async def main_loop(self, f, init_args, init_kwargs):
        delegate_f = self._make_delegate_f(f, init_args, init_kwargs)

        try:
            while self.running:
                args, kwargs = await self.get_next_args()
                try:
                    result = delegate_f(*args, **kwargs)
                except Exception as e:
                    result = e
                await self.put_to_downstream(result)
        except asyncio.CancelledError:
            pass
        except RuntimeError:
            pass

    @staticmethod
    def _make_delegate_f(f, init_args, init_kwargs):
        if inspect.isclass(f):
            return f(*init_args, **init_kwargs)
        else:
            return f


class ProcessRunner(Runner):
    _thread_local_store = threading.local()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.concurrency = None
        self.ctx = None
        self.pool_kwargs = None
        self.pool = None
        self.semaphore = None
        self.u_q = None

    def setup(self, concurrency=1, ctx=multiprocessing, **kwargs):
        self.concurrency = concurrency
        self.ctx = self._normalize_ctx(ctx)
        self.pool_kwargs = kwargs

    @staticmethod
    def _normalize_ctx(ctx):
        if ctx is None:
            return multiprocessing
        elif ctx == 'thread':
            return multiprocessing.dummy
        elif ctx == 'torch':
            import torch.multiprocessing
            return torch.multiprocessing
        else:
            return ctx

    def teardown(self):
        self.pool.close()

    async def main_loop(self, f, init_args, init_kwargs):
        delegate_f, init_f, init_f_args = self._make_delegate_f(f, init_args, init_kwargs)

        self.pool = pool = self.ctx.Pool(processes=self.concurrency,
                                         initializer=init_f,
                                         initargs=init_f_args,
                                         **self.pool_kwargs)
        self.semaphore = semaphore = asyncio.Semaphore(value=await self._get_semaphore_limit())
        self.u_q = asyncio.Queue(loop=self.loop)
        self.loop.create_task(self._pipe_queues())

        try:
            while self.running:
                await semaphore.acquire()
                args, kwargs = await self.get_next_args()
                pool.apply_async(delegate_f,
                                 args=args,
                                 kwds=kwargs,
                                 callback=self._put_result,
                                 error_callback=self._pool_error_cb)

        except asyncio.CancelledError:
            pass

    async def _get_semaphore_limit(self):
        return self.concurrency + min(8, math.ceil(self.concurrency * 1.25))

    @staticmethod
    def _make_delegate_f(f, init_args, init_kwargs):
        return ProcessRunner._call_delegate_f, ProcessRunner._init_delegate_f, (f, init_args, init_kwargs)

    @staticmethod
    def _init_delegate_f(cls, init_args, init_kwargs):
        ProcessRunner._seed_random()
        if inspect.isclass(cls):
            ProcessRunner._thread_local_store.instance = cls(*init_args, **init_kwargs)
        else:
            ProcessRunner._thread_local_store.instance = cls

    @staticmethod
    def _seed_random():
        import random
        import numpy.random
        import threading

        MAX = 2 ** 32 - 1
        seed = random.SystemRandom().randint(0, MAX) + threading.get_ident()
        seed = seed % MAX

        random.seed(seed)

        numpy.random.seed(seed)

    @staticmethod
    def _call_delegate_f(*args, **kwargs):
        try:
            return ProcessRunner._thread_local_store.instance(*args, **kwargs)
        except Exception as e:
            return e

    async def _pipe_queues(self):
        while self.running:
            value = await self.u_q.get()
            await self.put_to_downstream(value)
            self.semaphore.release()

    def _put_result(self, v):
        self.loop.call_soon_threadsafe(functools.partial(self.u_q.put_nowait, v))

    def _pool_error_cb(self, e):
        raise e
