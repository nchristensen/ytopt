import logging
import time
from collections import defaultdict, namedtuple
import sys
from concurrent.futures import Executor, ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Union

from ytopt.evaluator.evaluate import Evaluator

logger = logging.getLogger(__name__)


class SubmittedFuture:
    FAIL_RETURN_VALUE = Evaluator.FAIL_RETURN_VALUE
    
    def __init__(self, submitted_future):
        self.submitted_future = submitted_future
        self._state = 'active'
        self._result = None

    def _poll(self):
        if self._state != 'active': 
            return

        #print("Calling get()")
        #self.submitted_future.get()

        if self.submitted_future.done():
            try:
                self._result = self.submitted_future.result(timeout=0.001)
                print("Result:", self._result)
                self._state = 'done'
            except Exception as e:
                print("Exception:", e)
                self._state = 'failed'
        else:
            self._state = 'active'
        #print(len(self.submitted_future.values), self.submitted_future.nvals, self._state)

    def result(self):
        if not self.done:
            self._result = self.FAIL_RETURN_VALUE
        return self._result

    def cancel(self):
        cancelled = self.submitted_future.cancel()
        if cancelled:
            self._state = 'cancelled'

    @property
    def active(self):
        self._poll()
        return self._state == 'active'

    @property
    def done(self):
        self._poll()
        return self._state == 'done'

    @property
    def failed(self):
        self._poll()
        return self._state == 'failed'

    @property
    def cancelled(self):
        self._poll()
        return self._state == 'cancelled'


# Parent class for evaluators using Executor objects, such as MPI4PyPoolExecutor.
# Subclasses should instantiate self.executor in __init__.    

@dataclass
class ExecutorEvaluator(Evaluator):

    """Evaluator using subprocess.

        The ``SubprocessEvaluator`` use the ``subprocess`` package. The generated processes have a fresh memory independant from their parent process. All the imports are going to be repeated.

        Args:
            run_function (func): takes one parameter of type dict and returns a scalar value.
            cache_key (func): takes one parameter of type dict and returns a hashable type, used as the key for caching evaluations. Multiple inputs that map to the same hashable key will only be evaluated once. If ``None``, then cache_key defaults to a lossless (identity) encoding of the input dict.
    """
    WaitResult = namedtuple(
        'WaitResult', ['active', 'done', 'failed', 'cancelled'])
    
    executor: Union[Executor, None]
    num_workers: int

    def __init__(self, problem, executor, cache_key=None, output_file_base="results", num_workers=None):
        super().__init__(problem, cache_key, output_file_base=output_file_base)
        self.executor = executor
        if num_workers is not None:
            self.num_workers = num_workers
        elif self.executor is not None and hasattr(self.executor, "_max_workers"):
            # Should work for MPIPoolExecutor, ThreadPool, and ProcessPool
            # Won't work for MPICommExecutor
            self.num_workers = self.executor._max_workers
        else:
            self.num_workers = self.WORKERS_PER_NODE

        logger.info(
            f"Executor Evaluator will execute {self.problem.objective.__name__}() from module {self.problem.objective.__module__}")

    def _eval_exec(self, x: dict):
        assert isinstance(x, dict)
        #print(type(self.executor))
        if self.executor is not None: # Needed for MPICommExecutor
            executor_future = self.executor.submit(self.problem.objective, x)
            print("Submitted", x)
            evaluator_future = SubmittedFuture(executor_future)
            return evaluator_future
        else:
            return None

    @staticmethod
    def _timer(timeout):
        if timeout is None:
            return lambda: True
        else:
            timeout = max(float(timeout), 0.01)
            start = time.time()
            return lambda: (time.time()-start) < timeout

    def wait(self, futures, timeout=None, return_when='ANY_COMPLETED'):
        assert return_when.strip() in ['ANY_COMPLETED', 'ALL_COMPLETED']
        waitall = bool(return_when.strip() == 'ALL_COMPLETED')

        num_futures = len(futures)
        active_futures = [f for f in futures if f.active]
        time_isLeft = self._timer(timeout)

        if waitall:
            def can_exit(): return len(active_futures) == 0
        else:
            def can_exit(): return len(active_futures) < num_futures

        while time_isLeft():
            if can_exit():
                break
            else:
                active_futures = [f for f in futures if f.active]
                time.sleep(0.04)

        if not can_exit():
            raise TimeoutError(f'{timeout} sec timeout expired while '
                               f'waiting on {len(futures)} tasks until {return_when}')

        results = defaultdict(list)
        for f in futures:
            results[f._state].append(f)


        waitresult = self.WaitResult(
            active=results['active'],
            done=results['done'],
            failed=results['failed'],
            cancelled=results['cancelled']
        )

        return waitresult

    def __del__(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False)


class ThreadPoolExecutorEvaluator(ExecutorEvaluator):

    # Assumes MPI has been initialized at some point before the constructor is called
    # Should probably add some way for it to obey maximum workers per node
    def __init__(self, problem, cache_key=None, output_file_base="results", num_workers=None):
        if num_workers is None:
            num_workers = self.WORKERS_PER_NODE

        executor = ThreadPoolExecutor(max_workers=num_workers) 
        super().__init__(problem, executor, cache_key=cache_key, output_file_base=output_file_base, num_workers=num_workers)


class ProcessPoolExecutorEvaluator(ExecutorEvaluator):

    # Assumes MPI has been initialized at some point before the constructor is called
    # Should probably add some way for it to obey maximum workers per node
    def __init__(self, problem, cache_key=None, output_file_base="results", num_workers=None):
        if num_workers is None:
            num_workers = self.WORKERS_PER_NODE
        
        executor = ProcessPoolExecutor(max_workers=num_workers) 
        super().__init__(problem, executor, cache_key=cache_key, output_file_base=output_file_base, num_workers=num_workers)


class MPICommExecutorEvaluator(ExecutorEvaluator):

    # Assumes MPI has been initialized at some point before the constructor is called
    # Should probably add some way for it to obey maximum workers per node
    def __init__(self, problem, cache_key=None, output_file_base="results", comm=None):
        from mpi4py.futures import MPICommExecutor
        from mpi4py.MPI import COMM_WORLD
        executor = MPICommExecutor(comm=comm, root=0).__enter__()
        comm_size = COMM_WORLD.Get_size() if comm is None else comm.Get_size()
        super().__init__(problem, executor, cache_key=cache_key, output_file_base=output_file_base, num_workers=comm_size - 1)


class MPIPoolExecutorEvaluator(ExecutorEvaluator):

    # Assumes MPI has been initialized at some point before the constructor is called
    # Should probably add some way for it to obey maximum workers per node
    def __init__(self, problem, cache_key=None, output_file_base="results", num_workers=None):
        from mpi4py.futures import MPIPoolExecutor
        executor = MPIPoolExecutor(max_workers=num_workers)
        super().__init__(problem, executor, cache_key=cache_key, output_file_base=output_file_base)



class Charm4pyPoolExecutorEvaluator(ExecutorEvaluator):

    # Assumes MPI has been initialized at some point before the constructor is called
    # Should probably add some way for it to obey maximum workers per node
    def __init__(self, problem, cache_key=None, output_file_base="results"):
        from charm4py.pool import PoolExecutor, PoolScheduler, Pool
        from charm4py.chare import Chare
        pool_proxy = Chare(PoolScheduler, onPE=0)
        executor = PoolExecutor(pool_proxy) 
        super().__init__(problem, executor, cache_key=cache_key, output_file_base=output_file_base)
