import logging
import queue
import random
from concurrent.futures import ThreadPoolExecutor

import torch
from foc import *
from ouch import *
from torch import nn
from torch.nn import functional as F

from .model import *
from .utils import *


class _dataset:
    def __init__(self, path, n_mels=80, sr=16000, size_batch=1):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.db = read_json(path)
        self.preprocessor = filterbank(n_mels=n_mels, sr=sr)
        self.size_batch = size_batch

    def __iter__(self):
        while True:
            with torch.no_grad():
                anchors, positives, negatives = zip(
                    *[
                        tuple(map(self.preprocessor, triplet(self.db)))
                        for _ in range(self.size_batch)
                    ]
                )
                yield (
                    pad_(anchors),
                    pad_(positives),
                    pad_(negatives),
                )


class _dataloader:
    def __init__(self, datasets=None, size_buffer=400, num_workers=2):
        self.datasets = datasets  # (train_set, val_set)
        self.dataset = None
        self.size_buffer = size_buffer
        self.num_workers = num_workers
        self.queue = queue.Queue(maxsize=size_buffer)
        self.executor = None

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            self.reset()
            item = self.queue.get()
        return item

    def train(self):
        self.dataset = self.datasets[0]
        self.reset()
        return self

    def eval(self):
        self.dataset = self.datasets[1]
        self.reset()
        return self

    def worker(self):
        if self.dataset is not None:
            for item in self.dataset:
                try:
                    self.queue.put(item, timeout=1)
                except queue.Full:
                    continue
        self.queue.put(None)

    def start_workers(self):
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        for _ in range(self.num_workers):
            self.executor.submit(self.worker)

    def stop_workers(self):
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def reset(self):
        self.stop_workers()
        self.start_workers()

    def __del__(self):
        self.stop_workers()


class voh(nn.Module):
    # -----------
    # Setup
    # -----------
    @classmethod
    def create(cls, name, conf=None):
        """Create a new model"""
        return (
            voh()
            .set_name(name)
            .set_conf(conf or dmap())
            .set_seed()
            .set_model()
            .set_optim()
            .set_iter()
            .set_loss()
            .finalize()
        )

    @classmethod
    def load(cls, name, conf=None, strict=True):
        """Load the pre-trained"""
        t = torch.load(which_model(name), map_location="cpu")
        return (
            voh()
            .set_name(t.get("name"))
            .set_conf(t["conf"])
            .set_conf(conf or dmap(), kind=default.META, warn=False)
            .set_seed()
            .set_model(t["model"], strict=strict)
            .set_optim(t.get("optim"))
            .set_iter(t.get("it"))
            .set_loss(t.get("loss"))
            .finalize()
        )

    def set_name(self, name):
        self.name = name
        return self

    def set_conf(self, conf, kind=None, warn=True):
        ref = self.conf if hasattr(self, "conf") else def_conf()
        self.conf = ref | uniq_conf(conf, def_conf(kind=kind), warn=warn)
        return self

    def set_seed(self, seed=None):
        seed = seed or randint(1 << 31)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return self

    def set_model(self, model=None, strict=True):
        guard(
            self.conf.size_out_enc == self.conf.size_in_dec,
            f"The output size({self.conf.size_out_enc}) of"
            " the encoder does not match the input size"
            f"({self.conf.size_in_dec}) of the decoder.",
        )
        self.encoder = Encoder(self.conf)
        self.decoder = Decoder(self.conf)
        if model:
            self.load_state_dict(model, strict=strict)
        return self

    def set_iter(self, it=None):
        self.it = it or 0
        return self

    def set_loss(self, loss=None):
        self.loss = float("inf") if loss is None else loss
        return self

    def set_optim(self, optim=None):
        o = self.conf.optim
        if not o:
            self.optim = None
            return self
        self.optim = dict(
            sgd=torch.optim.SGD(
                self.parameters(),
                lr=self.conf.lr,
                weight_decay=self.conf.weight_decay,
                momentum=self.conf.momentum,
            ),
            adamw=torch.optim.AdamW(
                self.parameters(),
                lr=self.conf.lr,
                betas=self.conf.betas,
                weight_decay=self.conf.weight_decay,
            ),
            adam=torch.optim.Adam(
                self.parameters(),
                lr=self.conf.lr,
                betas=self.conf.betas,
            ),
        ).get(o) or error(f"No such optim supported: {o}")
        if optim:
            self.optim.load_state_dict(optim)
        return self

    def into(self, device=None, dtype=None):
        """Set the data type and device for all tensors based on availability"""
        device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        dtype = dtype or (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        return self.to(device=device, dtype=dtype)

    def optimize(self):
        return torch.compile(self, backend="eager")

    def finalize(self):
        torch._dynamo.config.suppress_errors = True
        torch._logging.set_logs(dynamo=logging.ERROR)
        torch._dynamo.eval_frame.OptimizedModule.__repr__ = lambda x: ""
        return self.into().optimize()

    # -----------
    # Model
    # -----------
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def numel(self):
        return sum(p.numel() for p in self.parameters())

    def show(self):
        dumper(
            model=self.name,
            parameters=f"{self.numel:_}",
            in_enc=self.conf.size_in_enc,
            hidden_enc=self.conf.size_hidden_enc,
            out_enc=self.conf.size_out_enc,
            in_dec=self.conf.size_in_dec,
            attention_dec=self.conf.size_attn_pool,
            out_dec=f"{self.conf.size_out_dec}  (embedding size)",
            kernels=str(self.conf.size_kernel_blocks),
            blocks_B=len(self.conf.size_kernel_blocks),
            repeats_R=self.conf.num_repeat_blocks,
        )
        if exists(path_model(self.name)):
            dumper(
                path=which_model(self.name),
                size=size_model(self.name),
            )

    def info(self, kind=None):
        if not kind:
            dumper(
                encoder=self.encoder,
                decoder=self.decoder,
            )
        dumper(
            **kind_conf(self.conf, kind=kind),
            loss=self.loss,
            it=(
                f"{self.it}  "
                f"({100 * self.it / (self.conf.epochs * self.conf.steps):.2f}"
                "% complete)"
            ),
        )

    def __repr__(self):
        return ""

    def forward(self, x):
        x = x.to(device=self.device)
        mask = create_mask(x)
        return cf_(
            f_(self.decoder, mask),
            f_(self.encoder, mask),
        )(x)

    @torch.no_grad()
    def fb(self, f):
        """Load a given wav file in forms of log Mel-filterbank energies"""
        return (
            filterbank(f, n_mels=self.conf.num_mel_filters, sr=self.conf.samplerate)
            .unsqueeze(0)
            .to(device=self.device)
        )

    @torch.no_grad()
    def embed(self, f):
        self.eval()
        return self(self.fb(f))

    @torch.no_grad()
    def cosim(self, x, y):
        return F.cosine_similarity(self.embed(x), self.embed(y)).item()

    # -----------
    # Training
    # -----------
    def get_trained(self):
        dl = self.dl()  # daatloader
        for _ in tracker(
            range(self.conf.steps * self.conf.epochs),
            "training",
            start=self.it,
        ):
            if dl.dataset is None:
                dl.train()

            print("Loop")
            anchor, positive, negative = next(dl)

            self.train()
            self.update_lr(self.optim)
            self.optim.zero_grad(set_to_none=True)
            loss = tripletloss(
                self(anchor),
                self(positive),
                self(negative),
                margin=self.conf.margin_loss,
            )
            loss.backward()
            self.optim.step()
            self.log(loss)
            self.validate(vloader)
            self.it += 1

    @torch.no_grad()
    def validate(self, vloader):
        if not self.it_interval(self.conf.int_val):
            return
        self.eval()
        loss = 0
        for _ in tracker(range(self.conf.size_val), "validation"):
            anchor, positive, negative = next(vloader)
            loss += tripletloss(
                self(anchor),
                self(positive),
                self(negative),
                margin=self.conf.margin_loss,
            )
        loss /= self.conf.size_val
        if loss < self.loss:
            self.save(snap=f"-{loss:.2f}-{self.it:06d}")
        self.loss = min(self.loss, loss)
        dumper(val_loss=(f"{loss:.4f} ({self.loss:.4f} best so far)"))

    def update_lr(self, optim, force=False):
        if not force and not self.it_interval(self.conf.int_sched_lr):
            return
        self.lr = (
            sched_lr(
                self.it,
                lr=self.conf.lr,
                lr_min=self.conf.lr_min,
                steps=self.conf.steps,
                warmup=int(self.conf.steps * self.conf.ratio_warmup),
            )
            if self.conf.int_sched_lr
            else self.conf.lr
        )
        for param_group in optim.param_groups:
            param_group["lr"] = self.lr

    def dl(self):
        guard(
            self.conf.size_in_enc == self.conf.num_mel_filters,
            f"The input size({self.conf.size_in_enc}) of"
            " the encoder does not match the number of "
            f"Mel-filterbanks({self.conf.num_mel_filters}).",
        )
        guard(
            self.conf.ds_train and exists(self.conf.ds_train),
            f"No such training set 'ds_train', {self.conf.ds_train}",
        )
        guard(
            self.conf.ds_val and exists(self.conf.ds_val),
            f"No such validation set 'ds_val', {self.conf.ds_val}",
        )
        shared = dict(
            n_mels=self.conf.num_mel_filters,
            sr=self.conf.samplerate,
            size_batch=self.conf.size_batch,
        )
        return _dataloader(
            datasets=(
                _dataset(self.conf.ds_train, **shared),  # training set
                _dataset(self.conf.ds_val, **shared),  # validation set
            ),
            size_buffer=500,
            num_workers=self.conf.num_workers or os.cpu_count(),
            collate_fn=collate_fn,
            shuffle=False,
        )

    def log(self, loss):
        setattr(self, "_loss", getattr(self, "_loss", 0) + loss)
        if not self.it_interval(self.conf.size_val):
            return
        self._loss /= self.conf.size_val
        dumper(lr=f"{self.lr:.6f}", avg_loss=f"{self._loss:.4f}")
        self._loss = 0

    def save(self, name=None, snap=None):
        """Create a checkpoint"""
        name = name or self.name
        if null(name):
            error("The model name is not specified.")
        path = path_model(name)
        mkdir(dirname(path))
        torch.save(
            dict(
                name=name,
                it=self.it,
                loss=self.loss,
                optim=self.optim.state_dict() if self.optim else None,
                conf=dict(self.conf),
                model=self.state_dict(),
            ),
            normpath(path),
        )
        if snap:
            d = f"{path}.snap"
            mkdir(d)
            shell(f"cp -f {path} {d}/{basename(path)}{snap}")
        return path

    def it_interval(self, val):
        return self.it and self.it % val == 0
