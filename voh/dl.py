import multiprocessing as mp
import threading
import time
from queue import Empty, Full

from .utils import *


class _dataset:
    """Iterable dataset for audio triplets (Log Mel-filterbank preprocessing).

    - Loads audio index data from JSON file
    - Applies `filterbank`, log of Mel-filterbank engergies
    - Generates batches of anchor-positive-negative triplets
    """

    def __init__(
        self,
        path,
        n_mels=80,
        sr=16000,
        size_batch=1,
        p=None,
        num_aug=1,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.db = read_json(path)
        self.size_batch = size_batch
        self.processor = cf_(  # log Mel-filterbanks
            filterbank(n_mels=n_mels, sr=sr, from_ysr=bool(p)),
            (  # data augmentor
                id  # never augment data when validation set
                if not p
                else augwav(augmentor=perturb(p=p, num_aug=num_aug), wav=False)
            ),
        )

    def __iter__(self):
        while True:
            anchors, positives, negatives = zip(
                *(
                    map(
                        self.processor,
                        triplet(self.db),
                    )
                    for _ in range(self.size_batch)
                )
            )
            yield (
                pad_(anchors),
                pad_(positives),
                pad_(negatives),
            )


class _safeiter:
    """Thread-safe iterator wrapper for synchronized access."""

    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


class _dataloader:
    """Multi-process data loader with thread-safe iteration.
    - Asynchronous data loading using multiple worker processes
    - Thread-safe data retrieval from a shared queue
    - 'pause' and 'resume' to enhance efficiency in each phase
    """

    def __init__(self, dataset, num_workers=1, size_queue=32):
        self.dataset = dataset
        self.num_workers = num_workers
        self.size_queue = size_queue
        self.queue = mp.Queue(maxsize=size_queue)
        self.abort = mp.Event()
        self.hold = mp.Event()
        self.process = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.abort.is_set() and self.queue.empty():
            raise StopIteration
        try:
            data = self.queue.get()
            if data is None:
                raise StopIteration
            return data
        except Empty:
            return self.__next__()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def worker(self):
        safe_iterator = _safeiter(iter(self.dataset))
        threads = []

        def thread_worker():
            while not self.abort.is_set():
                if self.hold.is_set():
                    time.sleep(0.1)
                    continue
                try:
                    data = next(safe_iterator)
                    self.queue.put(data, block=False)
                except StopIteration:
                    break
                except Full:
                    continue

        for _ in range(self.num_workers):
            t = threading.Thread(target=thread_worker)
            t.daemon = True
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        self.queue.put(None)

    def start(self):
        self.process = mp.Process(target=self.worker)
        self.process.start()
        return self

    def stop(self):
        self.abort.set()
        if self.process:
            self.process.join(timeout=5)
        while not self.queue.empty():
            try:
                self.queue.get(block=False)
            except Empty:
                break

    def pause(self):
        self.hold.set()

    def resume(self):
        self.hold.clear()
