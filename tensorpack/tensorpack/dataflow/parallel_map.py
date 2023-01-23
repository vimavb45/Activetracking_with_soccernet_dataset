# -*- coding: utf-8 -*-
# File: parallel_map.py
import copy
import ctypes
import multiprocessing as mp
import numpy as np
import threading
import zmq
from six.moves import queue

from ..utils.concurrency import StoppableThread, enable_death_signal
from ..utils.serialize import dumps, loads
from .base import DataFlow, DataFlowReentrantGuard, ProxyDataFlow
from .common import RepeatedData
from .parallel import _bind_guard, _get_pipe_name, _MultiProcessZMQDataFlow, _repeat_iter, _zmq_catch_error

__all__ = ['ThreadedMapData', 'MultiThreadMapData',
           'MultiProcessMapData', 'MultiProcessMapDataZMQ']


class _ParallelMapData(ProxyDataFlow):
    def __init__(self, ds, buffer_size, strict=False):
        super(_ParallelMapData, self).__init__(ds)
        assert buffer_size > 0, buffer_size
        self._buffer_size = buffer_size
        self._buffer_occupancy = 0  # actual #elements in buffer, only useful in strict mode
        self._strict = strict

    def reset_state(self):
        super(_ParallelMapData, self).reset_state()
        if not self._strict:
            ds = RepeatedData(self.ds, -1)
        else:
            ds = self.ds
        self._iter = ds.__iter__()

    def _recv(self):
        pass

    def _send(self, dp):
        pass

    def _recv_filter_none(self):
        ret = self._recv()
        assert ret is not None, \
            "[{}] Map function cannot return None when strict mode is used.".format(type(self).__name__)
        return ret

    def _fill_buffer(self, cnt=None):
        if cnt is None:
            cnt = self._buffer_size - self._buffer_occupancy
        try:
            for _ in range(cnt):
                dp = next(self._iter)
                self._send(dp)
        except StopIteration:
            raise RuntimeError(
                "[{}] buffer_size cannot be larger than the size of the DataFlow when strict=True!".format(
                    type(self).__name__))
        self._buffer_occupancy += cnt

    def get_data_non_strict(self):
        for dp in self._iter:
            self._send(dp)
            ret = self._recv()
            if ret is not None:
                yield ret

    def get_data_strict(self):
        self._fill_buffer()
        for dp in self._iter:
            self._send(dp)
            yield self._recv_filter_none()
        self._iter = self.ds.__iter__()   # refresh

        # first clear the buffer, then fill
        for k in range(self._buffer_size):
            dp = self._recv_filter_none()
            self._buffer_occupancy -= 1
            if k == self._buffer_size - 1:
                self._fill_buffer()
            yield dp

    def __iter__(self):
        if self._strict:
            for dp in self.get_data_strict():
                yield dp
        else:
            for dp in self.get_data_non_strict():
                yield dp


class MultiThreadMapData(_ParallelMapData):
    """
    Same as :class:`MapData`, but start threads to run the mapping function.
    This is useful when the mapping function is the bottleneck, but you don't
    want to start processes for the entire dataflow pipeline.

    The semantics of this class is __identical__ to :class:`MapData` except for the ordering.
    Threads run in parallel and can take different time to run the
    mapping function. Therefore the order of datapoints won't be preserved.

    When `strict=True`, `MultiThreadMapData(df, ...)`
    is guaranteed to produce the exact set of data as `MapData(df, ...)`,
    if both are iterated until `StopIteration`. But the produced data will have different ordering.
    The behavior of strict mode is undefined if the given dataflow `df` is infinite.

    When `strict=False`, the data that's produced by `MultiThreadMapData(df, ...)`
    is a reordering of the data produced by `RepeatedData(MapData(df, ...), -1)`.
    In other words, first pass of `MultiThreadMapData.__iter__` may contain
    datapoints from the second pass of `df.__iter__`.


    Note:
        1. You should avoid starting many threads in your main process to reduce GIL contention.

           The threads will only start in the process which calls :meth:`reset_state()`.
           Therefore you can use ``PrefetchDataZMQ(MultiThreadMapData(...), 1)``
           to reduce GIL contention.
    """
    class _Worker(StoppableThread):
        def __init__(self, inq, outq, evt, map_func):
            super(MultiThreadMapData._Worker, self).__init__(evt)
            self.inq = inq
            self.outq = outq
            self.func = map_func
            self.daemon = True

        def run(self):
            try:
                while True:
                    dp = self.queue_get_stoppable(self.inq)
                    if self.stopped():
                        return
                    # cannot ignore None here. will lead to unsynced send/recv
                    obj = self.func(dp)
                    self.queue_put_stoppable(self.outq, obj)
            except Exception:
                if self.stopped():
                    pass        # skip duplicated error messages
                else:
                    raise
            finally:
                self.stop()

    def __init__(self, ds, nr_thread, map_func, buffer_size=200, strict=False):
        """
        Args:
            ds (DataFlow): the dataflow to map
            nr_thread (int): number of threads to use
            map_func (callable): datapoint -> datapoint | None. Return None to
                discard/skip the datapoint.
            buffer_size (int): number of datapoints in the buffer
            strict (bool): use "strict mode", see notes above.
        """
        super(MultiThreadMapData, self).__init__(ds, buffer_size, strict)

        self._strict = strict
        self.nr_thread = nr_thread
        self.map_func = map_func
        self._threads = []
        self._evt = None

    def reset_state(self):
        super(MultiThreadMapData, self).reset_state()
        if self._threads:
            self._threads[0].stop()
            for t in self._threads:
                t.join()

        self._in_queue = queue.Queue()
        self._out_queue = queue.Queue()
        self._evt = threading.Event()
        self._threads = [MultiThreadMapData._Worker(
            self._in_queue, self._out_queue, self._evt, self.map_func)
            for _ in range(self.nr_thread)]
        for t in self._threads:
            t.start()

        self._guard = DataFlowReentrantGuard()

        # Call once at the beginning, to ensure inq+outq has a total of buffer_size elements
        self._fill_buffer()

    def _recv(self):
        return self._out_queue.get()

    def _send(self, dp):
        self._in_queue.put(dp)

    def __iter__(self):
        with self._guard:
            for dp in super(MultiThreadMapData, self).__iter__():
                yield dp

    def __del__(self):
        if self._evt is not None:
            self._evt.set()
        for p in self._threads:
            p.stop()
            p.join(timeout=5.0)
            # if p.is_alive():
            #     logger.warn("Cannot join thread {}.".format(p.name))


# TODO deprecated
ThreadedMapData = MultiThreadMapData


class MultiProcessMapDataZMQ(_ParallelMapData, _MultiProcessZMQDataFlow):
    """
    Same as :class:`MapData`, but start processes to run the mapping function,
    and communicate with ZeroMQ pipe.

    The semantics of this class is __identical__ to :class:`MapData` except for the ordering.
    Processes run in parallel and can take different time to run the
    mapping function. Therefore the order of datapoints won't be preserved.

    When `strict=True`, `MultiProcessMapData(df, ...)`
    is guaranteed to produce the exact set of data as `MapData(df, ...)`,
    if both are iterated until `StopIteration`. But the produced data will have different ordering.
    The behavior of strict mode is undefined if the given dataflow `df` is infinite.

    When `strict=False`, the data that's produced by `MultiProcessMapData(df, ...)`
    is a reordering of the data produced by `RepeatedData(MapData(df, ...), -1)`.
    In other words, first pass of `MultiProcessMapData.__iter__` may contain
    datapoints from the second pass of `df.__iter__`.
    """
    class _Worker(mp.Process):
        def __init__(self, identity, map_func, pipename, hwm):
            super(MultiProcessMapDataZMQ._Worker, self).__init__()
            self.identity = identity
            self.map_func = map_func
            self.pipename = pipename
            self.hwm = hwm

        def run(self):
            enable_death_signal(_warn=self.identity == b'0')
            ctx = zmq.Context()
            socket = ctx.socket(zmq.REP)
            socket.setsockopt(zmq.IDENTITY, self.identity)
            socket.set_hwm(self.hwm)
            socket.connect(self.pipename)

            while True:
                dp = loads(socket.recv(copy=False))
                dp = self.map_func(dp)
                socket.send(dumps(dp), copy=False)

    def __init__(self, ds, nr_proc, map_func, buffer_size=200, strict=False):
        """
        Args:
            ds (DataFlow): the dataflow to map
            nr_proc(int): number of threads to use
            map_func (callable): datapoint -> datapoint | None. Return None to
                discard/skip the datapoint.
            buffer_size (int): number of datapoints in the buffer
            strict (bool): use "strict mode", see notes above.
        """
        _ParallelMapData.__init__(self, ds, buffer_size, strict)
        _MultiProcessZMQDataFlow.__init__(self)
        self.nr_proc = nr_proc
        self.map_func = map_func
        self._strict = strict
        self._procs = []
        self._guard = DataFlowReentrantGuard()

    def reset_state(self):
        _MultiProcessZMQDataFlow.reset_state(self)
        _ParallelMapData.reset_state(self)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.socket.set_hwm(self._buffer_size * 2)
        pipename = _get_pipe_name('dataflow-map')
        _bind_guard(self.socket, pipename)

        self._proc_ids = [u'{}'.format(k).encode('utf-8') for k in range(self.nr_proc)]
        worker_hwm = int(self._buffer_size * 2 // self.nr_proc)
        self._procs = [MultiProcessMapDataZMQ._Worker(
            self._proc_ids[k], self.map_func, pipename, worker_hwm)
            for k in range(self.nr_proc)]

        self._start_processes()
        self._fill_buffer()     # pre-fill the bufer

    def _send(self, dp):
        msg = [b"", dumps(dp)]
        self.socket.send_multipart(msg, copy=False)

    def _recv(self):
        msg = self.socket.recv_multipart(copy=False)
        dp = loads(msg[1])
        return dp

    def __iter__(self):
        with self._guard, _zmq_catch_error('MultiProcessMapData'):
            for dp in super(MultiProcessMapDataZMQ, self).__iter__():
                yield dp


MultiProcessMapData = MultiProcessMapDataZMQ  # alias


def _pool_map(data):
    global SHARED_ARR, WORKER_ID, MAP_FUNC
    res = MAP_FUNC(data)
    if res is None:
        return None
    shared = np.reshape(SHARED_ARR, res.shape)
    assert shared.dtype == res.dtype
    shared[:] = res
    return WORKER_ID


# TODO shutdown pool, improve speed.
class MultiProcessMapDataComponentSharedArray(DataFlow):
    """
    Similar to :class:`MapDataComponent`, but perform IPC by shared memory,
    therefore more efficient when data (result of map_func) is large.
    It requires `map_func` to always return a numpy array of fixed shape and dtype, or None.
    """
    def __init__(self, ds, nr_proc, map_func, output_shape, output_dtype, index=0):
        """
        Args:
            ds (DataFlow): the dataflow to map on
            nr_proc(int): number of processes
            map_func (data component -> ndarray | None): the mapping function
            output_shape (tuple): the shape of the output of map_func
            output_dtype (np.dtype): the type of the output of map_func
            index (int): the index of the datapoint component to map on.
        """
        self.ds = ds
        self.nr_proc = nr_proc
        self.map_func = map_func
        self.output_shape = output_shape
        self.output_dtype = np.dtype(output_dtype).type
        self.index = index

        self._shared_mem = [self._create_shared_arr() for k in range(nr_proc)]
        id_queue = mp.Queue()
        for k in range(nr_proc):
            id_queue.put(k)

        def _init_pool(arrs, queue, map_func):
            id = queue.get()
            global SHARED_ARR, WORKER_ID, MAP_FUNC
            SHARED_ARR = arrs[id]
            WORKER_ID = id
            MAP_FUNC = map_func

        self._pool = mp.pool.Pool(
            processes=nr_proc,
            initializer=_init_pool,
            initargs=(self._shared_mem, id_queue, map_func))
        self._guard = DataFlowReentrantGuard()

    def _create_shared_arr(self):
        TYPE = {
            np.float32: ctypes.c_float,
            np.float64: ctypes.c_double,
            np.uint8: ctypes.c_uint8,
            np.int8: ctypes.c_int8,
            np.int32: ctypes.c_int32,
        }
        ctype = TYPE[self.output_dtype]
        arr = mp.RawArray(ctype, int(np.prod(self.output_shape)))
        return arr

    def __len__(self):
        return len(self.ds)

    def reset_state(self):
        self.ds.reset_state()

    def __iter__(self):
        ds_itr = _repeat_iter(self.ds.get_data)
        with self._guard:
            while True:
                dps = []
                for k in range(self.nr_proc):
                    dps.append(copy.copy(next(ds_itr)))
                to_map = [x[self.index] for x in dps]
                res = self._pool.map_async(_pool_map, to_map)

                for index in res.get():
                    if index is None:
                        continue
                    arr = np.reshape(self._shared_mem[index], self.output_shape)
                    dp = dps[index]
                    dp[self.index] = arr.copy()
                    yield dp


if __name__ == '__main__':
    import time

    class Zero(DataFlow):
        def __init__(self, size):
            self._size = size

        def __iter__(self):
            for k in range(self._size):
                yield [k]

        def __len__(self):
            return self._size

    def f(x):
        if x[0] < 10:
            time.sleep(1)
        return x

    ds = Zero(100)
    ds = MultiThreadMapData(ds, 50, f, buffer_size=50, strict=True)
    ds.reset_state()
    for idx, k in enumerate(ds):
        print("Bang!", k)
        if idx == 100:
            break
    print("END!")
