import multiprocessing as mp
import threading
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
        max_frames=400,
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
            filterbank(
                n_mels=n_mels,
                sr=sr,
                max_frames=max_frames,
                from_ysr=bool(p),
            ),
            (  # data augmentor
                id  # never augment data when validation set
                if not p
                else augwav(augmentor=perturb(p=p, num_aug=num_aug), wav=False)
            ),
        )

    def __iter__(self):
        while True:
            anchors, positives, negatives = map(
                cf_(pad_, mapl(self.processor)),
                triplet(self.db, size=self.size_batch),
            )
            yield anchors, positives, negatives


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
    """A multiprocessing-based data loader for training and evaluation datasets.
    This supports dynamics switching between training and validation modes.

    start | initialize and start the data loading process.
     stop | terminate the data loading process.
    train | switch to training mode, loading data from 'training queue.
     eval | switch to validation mode, loading data from 'evaluation queue'.
    """

    def __init__(self, trainset, evalset=None, num_workers=1, size_queue=32):
        self.trainset = trainset
        self.evalset = evalset
        self.num_workers = num_workers
        self.trainq = mp.Queue(maxsize=size_queue)
        self.evalq = mp.Queue(maxsize=size_queue)
        self.activeq = self.trainq
        self.abort = mp.Event()
        self.mode = mp.Value("b", True)  # True for 'train', False for 'eval'
        self.process = None

    def __iter__(self):
        return self

    def __next__(self):
        while not self.abort.is_set() or not self.activeq.empty():
            try:
                return self.activeq.get()
            except Empty:
                continue
        raise StopIteration

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.stop()

    def worker(self):
        def worker_thread():
            while not self.abort.is_set():
                mode = self.mode.value
                dataset = self.trainset if mode else self.evalset
                queue = self.trainq if mode else self.evalq
                for data in _safeiter(iter(dataset)):
                    if self.abort.is_set():
                        return
                    if self.mode.value != mode:
                        break
                    try:
                        queue.put(data, block=False)
                    except Full:
                        continue

        threads = [
            threading.Thread(target=worker_thread) for _ in range(self.num_workers)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def start(self):
        self.process = mp.Process(target=self.worker)
        self.process.start()
        return self

    def stop(self):
        self.abort.set()
        if self.process:
            self.process.join(timeout=5)
        for q in (self.trainq, self.evalq):
            while not q.empty():
                try:
                    q.get(block=False)
                except Empty:
                    break

    def train(self):
        with self.mode.get_lock():
            self.mode.value = True
        self.activeq = self.trainq

    def eval(self):
        with self.mode.get_lock():
            self.mode.value = False
        self.activeq = self.evalq
