import logging
import queue
import random
import sys
import time
from multiprocessing import Event, Manager, Process, set_start_method

import torch
from foc import *
from ouch import *
from torch import nn
from torch.multiprocessing import Event, Process, Queue
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .model import *
from .utils import *

set_start_method("fork", force=True)


class _dataset(Dataset):

    def __init__(self, path, n_mels=80, sr=16000, size=None):
        super().__init__()
        self.n_mels = n_mels
        self.sr = sr
        self.db = read_json(path)
        self.size = size or len(flatl(self.db.values()))

    def __len__(self):
        return self.size

    def __getitem__(self, _):
        return tuple(self.fetch())

    def fetch(self):
        return map(
            filterbank(n_mels=self.n_mels, sr=self.sr),
            triplet(self.db),
        )


class _dataloader:
    def __init__(self, dl, size_buffer=100, **kwargs):
        self.dl = DataLoader(dl, **kwargs)
        self.buffer = Manager().Queue(maxsize=size_buffer)
        self.size_buffer = size_buffer
        self.stop = Event()
        self.process = None

    def start_process(self):
        if self.process is None or not self.process.is_alive():
            self.process = Process(target=self.generate)
            self.process.start()

    def generate(self):
        while not self.stop.is_set():
            for batch in self.dl:
                if self.stop.is_set() or self.buffer.full():
                    break
                self.buffer.put(batch)
            if self.buffer.full():
                break

    def repopulate(self):
        self.start_process()
        while not self.buffer.full():
            if self.stop.is_set():
                break
            time.sleep(0.1)

    def __iter__(self):
        self.start_process()
        return self

    def __next__(self):
        if self.buffer.empty() and self.stop.is_set():
            raise StopIteration
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            self.repopulate()
            return self.__next__()

    def __enter__(self):
        self.repopulate()
        return self

    def __exit__(self, *args):
        self.stop.set()
        if self.process:
            self.process.join()


def collate_fn(batch):
    anchor, positive, negative = zip(*batch)
    return pad_(anchor), pad_(positive), pad_(negative)


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
            .finalize()
        )

    @classmethod
    def load(cls, name, strict=True):
        """Load the pre-trained"""
        t = torch.load(which_model(name), map_location="cpu")
        return (
            voh()
            .set_name(t.get("name", ""))
            .set_conf(t["conf"])
            .set_seed()
            .set_model()
            .load_model(t["model"], strict=strict)
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

    def set_model(self):
        guard(
            self.conf.size_out_enc == self.conf.size_in_dec,
            f"The output size({self.conf.size_out_enc}) of"
            " the encoder does not match the input size"
            f"({self.conf.size_in_dec}) of the decoder.",
        )
        self.encoder = Encoder(self.conf)
        self.decoder = Decoder(self.conf)
        return self

    def load_model(self, model, strict=True):
        self.load_state_dict(model, strict=strict)
        return self

    def into(self, device=None, dtype=None, **kwargs):
        """set a default dtype and device based on availability"""
        device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        dtype = dtype or (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        return self.to(device=device, dtype=dtype, **kwargs)

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
        dumper(**kind_conf(self.conf, kind=kind))

    def __repr__(self):
        return ""

    def forward(self, x):
        mask = create_mask(x).to(device=self.device)
        return cf_(
            f_(self.decoder, mask),
            f_(self.encoder, mask),
        )(x)

    @torch.no_grad()
    def fb(self, f):
        """Load a given wav file in forms of log Mel-filterbank energies"""
        return (
            filterbank(
                f,
                n_mels=self.conf.num_mel_filters,
                sr=self.conf.samplerate,
            )
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
        optim = self.get_optim()
        tloader, vloader = self.dl()
        for it in tracker(range(self.conf.steps), "training"):
            anchor, positive, negative = next(tloader)
            self.train()
            lr = self.update_lr(optim, it)
            optim.zero_grad(set_to_none=True)
            loss = tripletloss(
                self(anchor.to(self.device)),
                self(positive.to(self.device)),
                self(negative.to(self.device)),
                margin=self.conf.margin_loss,
            )
            loss.backward()
            optim.step()
            self.log(loss, it, lr)
            self.validate(vloader, it)

    def get_optim(self):
        o = self.conf.optim
        return dict(
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

    def update_lr(self, optim, it):
        if self.conf.decay:
            lr = sched_lr(
                it,
                lr=self.conf.lr,
                lr_min=self.conf.lr_min,
                steps=self.conf.steps,
                warmup=int(self.conf.steps * self.conf.ratio_warmup),
            )
            for param_group in optim.param_groups:
                param_group["lr"] = lr
            return lr
        else:
            return self.conf.lr

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
        ),
        shared = dict(
            batch_size=self.conf.size_batch,
            num_workers=self.conf.num_workers or os.cpu_count(),
            collate_fn=collate_fn,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=self.conf.prefetch,
            multiprocessing_context="fork",
        )
        return (
            _dataloader(
                _dataset(  # training dataset
                    self.conf.ds_train,
                    n_mels=self.conf.num_mel_filters,
                    sr=self.conf.samplerate,
                ),
                size_buffer=self.conf.period_val,
                **shared,
            ),
            _dataloader(
                _dataset(  # validation dataset
                    self.conf.ds_val,
                    n_mels=self.conf.num_mel_filters,
                    sr=self.conf.samplerate,
                    size=sys.maxsize,
                ),
                size_buffer=self.conf.size_val,
                **shared,
            ),
        )

    def log(self, loss, it, lr):
        self.conf.avg_loss += loss
        if not it or it % self.conf.size_val != 0:
            return
        self.conf.avg_loss /= self.conf.size_val
        dumper(lr=f"{lr:.6f}", avg_loss=f"{self.conf.avg_loss:.4f}")
        self.conf.avg_loss = 0

    @torch.no_grad()
    def validate(self, vloader, it):
        if not it or it % self.conf.period_val != 0:
            return
        self.eval()
        loss = 0
        for _ in tracker(range(self.conf.size_val), "validation"):
            anchor, positive, negative = next(vloader)
            loss += tripletloss(
                self(anchor.to(self.device)),
                self(positive.to(self.device)),
                self(negative.to(self.device)),
            )
        loss /= self.conf.size_val
        if loss < self.conf.min_loss:
            self.save(snap=f"-{loss:.2f}-{it:06d}")
        self.conf.min_loss = min(self.conf.min_loss, loss)
        dumper(val_loss=(f"{loss:.4f} ({self.conf.min_loss:.4f} best so far)"))
        return loss

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
