from typing import *
from queue import Empty, Full
from threading import Thread
from queue import Queue
from threading import Thread, Event
import threading
import inspect
import time
import random
import itertools
import functools

__all__ = [
    'Worker', 
    'Source',
    'Batch',
    'Unbatch',
    'Buffer',
    'Sequential',
    'Parallel',
    'Distribute',
    'Broadcast',
    'Switch',
    'Router',
    'Filter'
]


TERMINATE_CHECK_INTERVAL = 0.5
DEFAULT_QUEUE_SIZE = 1


class Terminate(Exception):
    pass


class EndOfInput:
    pass


def _get_queue(queue: Queue, terminate_flag: Event, timeout: float = None):
    while True:
        try:
            item = queue.get(block=True, timeout=TERMINATE_CHECK_INTERVAL if timeout is None else min(timeout, TERMINATE_CHECK_INTERVAL))
            if terminate_flag.is_set():
                raise Terminate()
            return item
        except Empty:
            if terminate_flag.is_set():
                raise Terminate()
            
        if timeout is not None:
            timeout -= TERMINATE_CHECK_INTERVAL
            if timeout <= 0:
                raise Empty()


def _put_queue(queue: Queue, item: Any, terminate_flag: Event, timeout: float = None):
    while True:
        try:
            queue.put(item, block=True, timeout=TERMINATE_CHECK_INTERVAL if timeout is None else min(timeout, TERMINATE_CHECK_INTERVAL))
            if terminate_flag.is_set():
                raise Terminate()
            return
        except Full:
            if terminate_flag.is_set():
                raise Terminate()
            
        if timeout is not None:
            timeout -= TERMINATE_CHECK_INTERVAL
            if timeout <= 0:
                raise Full()


class Node:
    def __init__(self):
        self._in_queue_lock = threading.Lock()
        self._out_queue_lock = threading.Lock()
        self._terminate_flag = threading.Event()
        self._input = None
        self._output = None
        self._is_started = False

    @property
    def input(self) -> Queue:
        with self._in_queue_lock:
            if self._input is None:
                self._input = Queue(maxsize=DEFAULT_QUEUE_SIZE)
        return self._input

    @input.setter
    def input(self, value: Queue):
        with self._in_queue_lock:
            if self._input is not None:
                raise AttributeError("Node input is already set.")
            self._input = value

    @property
    def output(self) -> Queue:
        with self._out_queue_lock:
            if self._output is None:
                self._output = Queue(maxsize=DEFAULT_QUEUE_SIZE)
        return self._output
    
    @output.setter
    def output(self, value: Queue):
        with self._out_queue_lock:
            if self._output is not None:
                raise AttributeError("Node output is already set.")
            self._output = value

    def start(self):
        self._is_started = True
        self._terminate_flag = threading.Event()

    def terminate(self):
        self._is_started = False
        self._terminate_flag.set()

    def stop(self):
        self.terminate()
    
    def put(self, data: Any, timeout: float = None) -> None:
        assert self._is_started, "Node is not started."
        _put_queue(self.input, data, self._terminate_flag, timeout)
    
    def get(self, timeout: float = None) -> Any:
        assert self._is_started, "Node is not started."
        return _get_queue(self.output, self._terminate_flag, timeout)

    def put_nowait(self, data: Any) -> None:
        assert self._is_started, "Node is not started."
        self.input.put_nowait(data)
    
    def get_nowait(self) -> Any:
        assert self._is_started, "Node is not started."
        return self.output.get_nowait()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def __call__(self, iterator: Iterable):
        assert self._is_started, "Node is not started."
        return NodeIterator(self, iterator)


class NodeIterator:
    def __init__(self, node: Node, iterator: Iterable):
        self.node = node
        self.iterator = iterator
        self.source_thread = Thread(target=self._source_thread_fn)
        self.source_thread.start()
    
    def __iter__(self):
        return self
    
    def _source_thread_fn(self):
        for item in self.iterator:
            _put_queue(self.node.input, item, self.node._terminate_flag)
        _put_queue(self.node.input, EndOfInput(), self.node._terminate_flag)
    
    def __next__(self):
        item = self.node.get()
        if isinstance(item, EndOfInput):
            raise StopIteration()
        return item


class ThreadingNode(Node):
    thread_functions: List[Callable]
    threads: List[Thread]
    terminate_flag: Event

    def start(self):
        super().start()
        self.threads = [Thread(target=fn) for fn in self.thread_functions]
        for thread in self.threads:
            thread.start()

    def stop(self):
        super().stop()
        for thread in self.threads:
            thread.join()


class Worker(ThreadingNode):
    def __init__(self, work: Callable = None):
        super().__init__()
        self.work_fn = work
        self.thread_functions = [self.loop]

    def init(self) -> None:
        """
        This method is called the the thread is started, to initialize any resources that is only held in the thread.
        """
        pass

    def work(self, *args, **kwargs) -> Union[Any, Dict[str, Any]]:
        """
        This method defines the job that the node should do for each input item. 
        A item obtained from the input queue is passed as arguments to this method, and the result is placed in the output queue.
        The method is executed concurrently with other nodes.
        """
        return self.work_fn(*args, **kwargs)

    def loop(self):
        self.init()
        try:
            while True:
                item = _get_queue(self.input, self._terminate_flag)
                if isinstance(item, EndOfInput):
                    _put_queue(self.output, EndOfInput(), self._terminate_flag)
                    continue
                result = self.work(item)
                _put_queue(self.output, result, self._terminate_flag)
        except Terminate:
            return


class Source(ThreadingNode):
    """
    A node that provides data to successive nodes. It takes no input and provides data to the output queue.
    """
    def __init__(self, provide: Callable = None):
        super().__init__()
        self.provide_fn = provide
        self.thread_functions = [self.loop]

    def init(self) -> None:
        """
        This method is called the the thread or process is started, to initialize any resources that is only held in the thread or process.
        """
        pass

    def provide(self) -> Generator[Any, None, None]:
        for item in self.provide_fn():
            yield item

    def loop(self):
        self.init()
        try:
            for data in self.provide():
                _put_queue(self.output, data, self._terminate_flag)
        except Terminate:
            return


class Batch(ThreadingNode):
    """
    Groups every `batch_size` items into a batch (a list of items) and passes the batch to successive nodes.
    The `patience` parameter specifies the maximum time to wait for a batch to be filled before sending it to the next node,
    i.e., when the earliest item in the batch is out of `patience` seconds, the batch is sent regardless of its size.
    """
    def __init__(self, batch_size: int, patience: float = None):
        assert batch_size > 0, "Batch size must be greater than 0."
        super().__init__()
        self.batch_size = batch_size
        self.patience = patience
        self.thread_functions = [self.loop]

    def loop(self):
        try:
            while True:
                batch = []
                # Try to fill the batch
                for i in range(self.batch_size):
                    if i == 0 or self.patience is None:
                        # Wait forever for the first item or if patience is not set
                        timeout = None
                    else:
                        # Calculate the remaining time for the batch
                        timeout = self.patience - (time.time() - earliest_time)
                        if timeout < 0:
                            break
                    # Try to get an item within the remaining time
                    try:
                        item = _get_queue(self.input, self._terminate_flag, timeout)
                    except Empty:
                        break
                    # If the item is EndOfInput, break the loop
                    if isinstance(item, EndOfInput):
                        break
                    # If the first item, start timing
                    if i == 0:
                        earliest_time = time.time()
                    batch.append(item)

                if len(batch) > 0:
                    _put_queue(self.output, batch, self._terminate_flag)

                if isinstance(item, EndOfInput):
                    _put_queue(self.output, EndOfInput(), self._terminate_flag)
                    continue
        except Terminate:
            return


class Unbatch(ThreadingNode):
    """
    Ungroups every batch (a list of items) into individual items and passes them to successive nodes.
    """
    def __init__(self):
        super().__init__()
        self.thread_functions = [self.loop]

    def loop(self):
        try:
            while True:
                batch = _get_queue(self.input, self._terminate_flag)
                
                if isinstance(batch, EndOfInput):
                    _put_queue(self.output, EndOfInput(), self._terminate_flag)
                    continue
                
                for item in batch:
                    _put_queue(self.output, item, self._terminate_flag)
        except Terminate:
            return


class Buffer(ThreadingNode):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.thread_functions = [self.loop]
    
    @property
    def output(self):
        if self._output is None:
            self._output = Queue(maxsize=self.size)
        return self._output

    def loop(self):
        try:
            while True:
                item = _get_queue(self.input, self._terminate_flag)
                _put_queue(self.output, item, self._terminate_flag)
        except Terminate:
            return


class Sequential(Node):
    """
    Pipeline of nodes in sequential order, where each node takes the output of the previous node as input.
    The order of input and output items is preserved (FIFO)
    """
    nodes: List[Node]
    def __init__(self, nodes: List[Union[Node, Callable]]):
        super().__init__()
        self.nodes = []
        for node in nodes:
            if isinstance(node, Node):
                pass
            elif isinstance(node, Callable):
                if inspect.isgeneratorfunction(node):
                    node = Source(node)
                else:
                    node = Worker(node)
            else:
                raise ValueError(f"Invalid node type: {type(node)}")
            self.nodes.append(node)
        
        for node_pre, node_suc in zip(self.nodes[:-1], self.nodes[1:]):
            node_suc.input = node_pre.output
    
    @property
    def input(self) -> Queue:
        return self.nodes[0].input    

    @input.setter
    def input(self, value: Queue):
        self.nodes[0].input = value

    @property
    def output(self) -> Queue:
        return self.nodes[-1].output

    def start(self):
        super().start()
        for node in self.nodes:
            node.start()

    def stop(self):
        super().stop()
        for node in self.nodes:
            node.stop()


class Parallel(ThreadingNode):
    """
    A FIFO node that runs multiple nodes in parallel to process the input items. Each input item is handed to one of the nodes whoever is available.
    NOTE: It is FIFO if and only if all the nested nodes are FIFO.
    """
    nodes: List[Node]

    def __init__(self, nodes_or_callable: Union[Callable, Sequence[Node]], num_duplicates: int = None):
        super().__init__()
        if isinstance(nodes_or_callable, Callable):
            assert num_duplicates is not None, "Duplicates count must be specified for callable"
            self.nodes = [Worker(nodes_or_callable) for _ in range(num_duplicates)]
        else:
            self.nodes = []
            for node in nodes_or_callable:
                if isinstance(node, Node):
                    pass
                elif isinstance(node, Callable):
                    if inspect.isgeneratorfunction(node):
                        node = Source(node)
                    else:
                        node = Worker(node)
                else:
                    raise ValueError(f"Invalid node type: {type(node)}")
                self.nodes.append(node)
        self.working_nodes = Queue()
        self.idle_nodes = Queue()
        for node in self.nodes:
            self.idle_nodes.put(node)
        self.thread_functions = [self._in_thread_fn, self._out_thread_fn]

    def _in_thread_fn(self):
        try:
            while True:
                item = _get_queue(self.input, self._terminate_flag)
                node = _get_queue(self.idle_nodes, self._terminate_flag)
                self.working_nodes.put(node)
                _put_queue(node.input, item, self._terminate_flag)
        except Terminate:
            return
    
    def _out_thread_fn(self):
        try:
            while True:
                node = _get_queue(self.working_nodes, self._terminate_flag)
                item = _get_queue(node.output, self._terminate_flag)
                _put_queue(self.output, item, self._terminate_flag)
                _put_queue(self.idle_nodes, node, self._terminate_flag)
        except Terminate:
            return

    def start(self):
        super().start()
        for node in self.nodes:
            node.start()
    
    def terminate(self):
        super().terminate()
        for node in self.nodes:
            node.terminate()

    def stop(self):
        super().stop()
        for node in self.nodes:
            node.stop()


class Distribute(ThreadingNode):
    branches: Dict[str, Node]

    def __init__(self, branches: Dict[str, Node]):
        super().__init__()
        self.branches = {}
        for key, node in branches.items():
            if isinstance(node, Node):
                pass
            elif isinstance(node, Callable):
                if inspect.isgeneratorfunction(node):
                    raise ValueError("Source node is not allowed in Distribute block")
                else:
                    node = Worker(node)
            else:
                raise ValueError(f"Invalid node type: {type(node)}")
            self.branches[key] = node
        self.thread_functions = [self._in_thread_fn] + [self._out_thread_fn]
    
    def _in_thread_fn(self):
        try:
            while True:
                item = _get_queue(self.input, self._terminate_flag)

                if isinstance(item, EndOfInput):
                    for node in self.branches.values():
                        _put_queue(node.input, EndOfInput(), self._terminate_flag)
                    continue

                if any(k not in self.branches for k in item) or any(k not in item for k in self.branches):
                    raise ValueError(f"Distribute keys mismatch. Input keys: {list(item.keys())}. Required keys: {list(self.branches.keys())}.")
                for k, v in item.items():
                    _put_queue(self.branches[k].input, v, self._terminate_flag)
        except Terminate:
            return

    def _out_thread_fn(self):
        try:
            while True:
                item = {k: _get_queue(node.output, self._terminate_flag) for k, node in self.branches.items()}
                if all(isinstance(v, EndOfInput) for v in item.values()):
                    _put_queue(self.output, EndOfInput(), self._terminate_flag)
                    continue
                _put_queue(self.output, item, self._terminate_flag)
        except Terminate:
            return

    def start(self):
        for node in self.branches.values():
            node.start()
        super().start()

    def terminate(self):
        for node in self.branches.values():
            node.terminate()
        super().terminate()  

    def stop(self):
        for node in self.branches.values():
            node.stop()
        super().stop()


class Switch(ThreadingNode):
    branches: Dict[str, Node]

    def __init__(self, predicate: Callable[[Any], str], branches: Dict[str, Node]):
        self.predicate = predicate
        self.branches = {}
        for key, node in branches.items():
            if isinstance(node, Node):
                pass
            elif isinstance(node, Callable):
                if inspect.isgeneratorfunction(node):
                    raise ValueError("Source node is not allowed in Dispatch block")
                else:
                    node = Worker(node)
            else:
                raise ValueError(f"Invalid node type: {type(node)}")
            self.branches[key] = node
        self.fifo_order = Queue()

    def _in_thread_fn(self):
        try:
            while True:
                item = _get_queue(self.input, self._terminate_flag)
                
                if isinstance(item, EndOfInput):
                    self.fifo_order.put(EndOfInput)
                    continue
                
                key = self.predicate(item)
                if key not in self.branches:
                    raise ValueError(f"Switch block key mismatches. \"{key}\" not in found in {list(self.branches.keys())}.")
                _put_queue(self.branches[key].input, item, self._terminate_flag)
                self.fifo_order.put(key)
        except Terminate:
            return

    def _out_thread_fn(self):
        try:
            while True:
                key = _get_queue(self.fifo_order, self._terminate_flag)
                if isinstance(key, EndOfInput):
                    _put_queue(self.output, EndOfInput(), self._terminate_flag)
                    continue
                item = _get_queue(self.branches[key], self._terminate_flag)
                _put_queue(self.output, item, self._terminate_flag)
        except Terminate:
            return
        
    def start(self):
        super().start()
        for node in self.branches.values():
            node.start()

    def terminate(self):
        super().terminate()
        for node in self.branches.values():
            node.terminate()

    def stop(self):
        super().stop()
        for node in self.branches.values():
            node.stop()
        

class Router(ThreadingNode):
    branches: Dict[str, Node]

    def __init__(self, predicate: Callable[[Any], List[str]], branches: Dict[str, Node]):
        self.predicate = predicate
        self.branches = {}
        for key, node in branches.items():
            if isinstance(node, Node):
                pass
            elif isinstance(node, Callable):
                if inspect.isgeneratorfunction(node):
                    raise ValueError("Source node is not allowed in Dispatch block")
                else:
                    node = Worker(node)
            else:
                raise ValueError(f"Invalid node type: {type(node)}")
            self.branches[key] = node
        self.fifo_order = Queue()

    def _in_thread_fn(self):
        try:
            while True:
                item = _get_queue(self.input, self._terminate_flag)
                
                if isinstance(item, EndOfInput):
                    self.fifo_order.put(EndOfInput())
                    continue
                
                keys = self.predicate(item)
                if any(k not in self.branches for k in keys) or any(k not in keys for k in self.branches):
                    raise ValueError(f"Switch block key mismatches. Input keys: {list(keys)}. Expected keys: {list(self.branches.keys())}.")
                for key in keys:
                    _put_queue(self.branches[key].input, item, self._terminate_flag)
                self.fifo_order.put(keys)
        except Terminate:
            return

    def _out_thread_fn(self):
        try:
            while True:
                keys = _get_queue(self.fifo_order, self._terminate_flag)
                if isinstance(keys, EndOfInput):
                    _put_queue(self.output, EndOfInput(), self._terminate_flag)
                    continue
                item = {k: _get_queue(self.branches[k], self._terminate_flag) for k in keys}
                _put_queue(self.output, item, self._terminate_flag)
        except Terminate:
            return
        
    def start(self):
        super().start()
        for node in self.branches.values():
            node.start()

    def stop(self):
        super().stop()
        for node in self.branches.values():
            node.stop()
        

class Broadcast(ThreadingNode):
    branches: Union[List[Node], Dict[str, Node]]

    def __init__(self, branches: Union[List[Node], Dict[str, Node]]):
        
        if isinstance(branches, list):
            self.branches = []
            for node in branches:
                if isinstance(node, Node):
                    pass
                elif isinstance(node, Callable):
                    if inspect.isgeneratorfunction(node):
                        raise ValueError("Source node is not allowed in Broadcast block")
                    else:
                        node = Worker(node)
                else:
                    raise ValueError(f"Invalid node type: {type(node)}")
                self.branches.append(node)
        elif isinstance(branches, dict):
            self.branches = {}
            for key, node in branches.items():
                if isinstance(node, Node):
                    pass
                elif isinstance(node, Callable):
                    if inspect.isgeneratorfunction(node):
                        raise ValueError("Source node is not allowed in Broadcast block")
                    else:
                        node = Worker(node)
                else:
                    raise ValueError(f"Invalid node type: {type(node)}")
                self.branches[key] = node

    def _in_thread_fn(self):
        try:
            while True:
                item = _get_queue(self.input, self._terminate_flag)
                if isinstance(self.branches, list):
                    for node in self.branches:
                        _put_queue(node.input, item, self._terminate_flag)
                else:
                    for key, node in self.branches.items():
                        _put_queue(node.input, item, self._terminate_flag)
        except Terminate:
            return

    def _out_thread_fn(self):
        try:
            while True:
                if isinstance(self.branches, list):
                    item = [_get_queue(node.output, item, self._terminate_flag) for node in self.branches]
                else:
                    item = {k: _get_queue(node.output, self._terminate_flag) for k, node in self.branches.items()}
                
                if (isinstance(self.branches, list) and all(isinstance(v, EndOfInput) for v in item)) \
                    or (isinstance(self.branches, dict) and all(isinstance(v, EndOfInput) for v in item.values())):
                    _put_queue(self.output, EndOfInput(), self._terminate_flag)
                    continue
                
                _put_queue(self.output, item, self._terminate_flag)
        except Terminate:
            return
        
    def start(self):
        super().start()
        for node in self.branches.values():
            node.start()

    def stop(self):
        super().stop()
        for node in self.branches.values():
            node.stop()
        

class Filter(ThreadingNode):
    """
    A node that filters items based on a predicate function. 
    If the predicate returns True, the item is passed to the output queue, otherwise it is discarded.
    """
    def __init__(self, predicate: Optional[Callable[[Any], bool]] = None):
        """
        ### Parameters
        - `predicate`: A function that takes an item and returns True if the item should be passed to the output queue. Default to pass items that are not None.
        """
        super().__init__()
        self.predicate = predicate
        self.thread_functions = [self.loop]

    def loop(self):
        try:
            while True:
                item = _get_queue(self.input, self._terminate_flag)
                if isinstance(item, EndOfInput):
                    _put_queue(self.output, EndOfInput(), self._terminate_flag)
                    continue
                if (self.predicate is None and item is not None) or (self.predicate is not None and self.predicate(item)):
                    _put_queue(self.output, item, self._terminate_flag)
        except Terminate:
            return