import logging
import random
import sys

import torch
from foc import *
from ouch import *
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .model import *
from .utils import *


def vohconf(conf=None, **kwds):
    o = dmap(
        # main/data
        model="o/pilot",
        ds_train=None,
        ds_validate=None,
        num_mel_filters=80,
        samplerate=16000,
        # --------------------------------------
        # trainer
        seed=42,
        decay=True,
        dropout=0.1,
        margin_loss=0.5,
        lr=1e-4,
        lr_min=1e-5,
        ratio_warmup=0.01,
        size_batch=16,
        size_validate=20,
        num_workers=None,
        it_val=100,
        it_log=10,
        steps=None,
        # --------------------------------------
        # model architecture
        size_in_enc=None,
        size_hidden_enc=1024,
        size_out_enc=3072,
        size_kernel_prolog=3,
        size_kernel_epilog=1,
        size_kernel_blocks=(7, 11, 15),
        num_repeat_blocks=2,
        ratio_reduction=8,  # SE-block reduction ratio
        size_in_dec=None,
        size_attn_pool=128,
        size_out_dec=192,  # embedding size
    )
    o = o | uniq_conf(conf, o) | uniq_conf(dict(**kwds), o)
    o.size_in_enc = o.num_mel_filters
    o.size_in_dec = o.size_out_enc
    return o


class vohDataset(Dataset):

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
        return map(filterbank(n_mels=self.n_mels, sr=self.sr), triplet(self.db))


def collate_fn(batch):
    anchor, positive, negative = zip(*batch)
    return pad_(anchor), pad_(positive), pad_(negative)


class voh(nn.Module):

    # -----------
    # Setup
    # -----------
    @classmethod
    def new(cls, conf=None, **kwds):
        """Create a new model"""
        o = voh()
        o.initialize()
        o.set_conf(conf, **kwds)
        o.set_seed()
        o.set_model()
        o.set_meta()
        return o.into().optimize()

    @classmethod
    def load(cls, model, strict=True):
        """Load the pre-trained"""
        guard(exists(model, "f"), f"Not found model: {model}")
        o = voh()
        o.initialize()
        t = torch.load(model, map_location="cpu")
        o.set_conf(t["conf"])
        o.set_seed()
        o.set_model()
        o.load_model(t["model"], strict=strict)
        o.set_meta(t["meta"])
        return o.into().optimize()

    def initialize(self):
        torch.mps.empty_cache()
        torch.mps.set_per_process_memory_fraction(0.8)
        torch._dynamo.config.suppress_errors = True
        torch._logging.set_logs(dynamo=logging.ERROR)

    def set_conf(self, conf=None, **kwds):
        self.conf = vohconf(conf, **kwds)

    def set_seed(self, seed=None):
        seed = seed or self.conf.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def set_model(self):
        self.encoder = Encoder(self.conf)
        self.decoder = Decoder(self.conf)

    def load_model(self, model, strict=True):
        self.load_state_dict(model, strict=strict)

    def set_meta(self, meta=None):
        o = dmap(
            avg_loss=0,
            min_loss=float("inf"),
        )
        self.meta = o | uniq_conf(meta, o)

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

    def info(self):
        dumper(encoder=self.encoder, decoder=self.decoder)
        dumper(num_parameters=f"{self.numel:_}", **self.conf)

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
        optim = torch.optim.Adam(self.parameters(), lr=self.conf.lr)
        tloader, vloader = self.dl()
        for it, (anchor, positive, negative) in tracker(
            enumerate(tloader),
            "training",
            total=self.conf.steps,
        ):
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
        conf = dict(
            batch_size=self.conf.size_batch,
            num_workers=self.conf.num_workers or os.cpu_count(),
            collate_fn=collate_fn,
            shuffle=False,
            persistent_workers=True,
            multiprocessing_context="fork",
        )
        self.ds_train = vohDataset(
            self.conf.ds_train,
            n_mels=self.conf.num_mel_filters,
            sr=self.conf.samplerate,
        )
        self.ds_validate = vohDataset(
            self.conf.ds_validate,
            n_mels=self.conf.num_mel_filters,
            sr=self.conf.samplerate,
            size=sys.maxsize,
        )
        self.conf.steps = self.ds_train.size // self.conf.size_batch
        return (
            DataLoader(self.ds_train, **conf),
            DataLoader(self.ds_validate, **conf),
        )

    def log(self, loss, it, lr):
        self.meta.avg_loss += loss
        if not it or it % self.conf.it_log != 0:
            return
        self.meta.avg_loss /= self.conf.it_log
        dumper(lr=f"{lr:.6f}", avg_loss=f"{self.meta.avg_loss:.4f}")
        self.meta.avg_loss = 0

    @torch.no_grad()
    def validate(self, vloader, it):
        if not it or it % self.conf.it_val != 0:
            return
        self.eval()
        loss = 0
        for anchor, positive, negative in tracker(
            take(self.conf.size_validate, vloader),
            "validation",
            total=self.conf.size_validate,
        ):
            loss += tripletloss(
                self(anchor.to(self.device)),
                self(positive.to(self.device)),
                self(negative.to(self.device)),
            )
        loss /= self.conf.size_validate
        if loss < self.meta.min_loss:
            self.save(filepath=self.conf.model, snap=f"-{loss:.2f}-{it:06d}")
        self.meta.min_loss = min(self.meta.min_loss, loss)
        dumper(val_loss=(f"{loss:.4f} ({self.meta.min_loss:.4f} best so far)"))
        return loss

    def save(self, filepath=None, snap=None):
        """Create a checkpoint"""
        filepath = filepath or self.conf.model
        d = dirname(filepath)
        d and mkdir(d)
        torch.save(
            dict(
                conf=dict(self.conf),
                meta=dict(self.meta),
                model=self.state_dict(),
            ),
            normpath(filepath),
        )
        if snap:
            d = f"{filepath}.snap"
            mkdir(d)
            shell(f"cp -f {filepath} {d}/{basename(filepath)}{snap}")
