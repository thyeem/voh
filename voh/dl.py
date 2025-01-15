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
        augment=True,
        ipad=0,
    ):
        super().__init__()
        self.db = read_json(path)
        self.n_mels = n_mels
        self.sr = sr
        self.max_frames = max_frames
        self.size_batch = size_batch
        self.augment = augment
        self.ipad = ipad

    def __iter__(self):
        while True:
            anchors, positives, negatives = map(
                f_(pad_, ipad=self.ipad),  # collate-fn
                self.triads(),  # list of triplets with batch size
            )
            yield anchors, positives, negatives

    def triads(self):
        return zip(
            *map(
                self.processor,  # filterbank + norm-channel + perturb
                zip(*triplet(self.db, size=self.size_batch)),
            ),
        )

    def processor(self, triad):
        augmetor = perturb_tiny if self.augment else id
        anchor, positive, negative = map(readwav, triad)
        return map(
            filterbank(  # log Mel-filterbank energies
                n_mels=self.n_mels,
                sr=self.sr,
                max_frames=self.max_frames,
            ),
            (anchor, augmetor(positive), augmetor(negative)),
        )


class _dataloader:
    """A multiprocessing-based data loader for training and evaluation datasets.
    This supports dynamics switching between training and validation modes.

    start | initialize and start the data loading process.
     stop | terminate the data loading process.
    train | switch to training mode, loading data from 'training queue.
     eval | switch to validation mode, loading data from 'evaluation queue'.
    """

    def __init__(self, tset, vset, num_workers=1, size_queue=32):
        self.tset = tset  # training set builder
        self.vset = vset  # validation set builder
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
            trainset = iter(self.tset())
            valset = iter(self.vset())
            while not self.abort.is_set():
                mode = self.mode.value
                dataset = trainset if mode else valset
                queue = self.trainq if mode else self.evalq
                for data in dataset:
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
