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


def collate_fn(batch):
    anchor, positive, negative = zip(*batch)
    return pad_(anchor), pad_(positive), pad_(negative)


def defmodel(conf="model.conf"):
    return read_conf(f"{dirname(__file__)}/../conf/{conf}")


def deftrain(conf="train.conf"):
    return read_conf(f"{dirname(__file__)}/../conf/{conf}")


class voh(nn.Module):

    # -----------
    # Setup
    # -----------
    @classmethod
    def create(cls, conf=None):
        """Create a new model"""
        return voh().set_conf(conf).set_seed().set_model().finalize()

    @classmethod
    def load(cls, model, strict=True):
        """Load the pre-trained"""
        t = torch.load(which_model(model), map_location="cpu")
        return (
            voh()
            .set_conf(t["conf"])
            .set_seed()
            .set_model()
            .load_model(t["model"], strict=strict)
            .finalize()
        )

    def set_conf(self, conf, kind=None):
        self.conf = def_conf() | uniq_conf(conf, def_conf(kind=kind))
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
            model=self.conf.model,
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
        if exists(path_model(self.conf.model)):
            dumper(
                path=which_model(self.conf.model),
                size=size_model(self.conf.model),
            )

    def info(self):
        dumper(
            encoder=self.encoder,
            decoder=self.decoder,
            **self.conf,
        )

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
        conf = dict(
            batch_size=self.conf.size_batch,
            num_workers=self.conf.num_workers or os.cpu_count(),
            collate_fn=collate_fn,
            shuffle=False,
            prefetch_factor=1,
            persistent_workers=True,
            multiprocessing_context="fork",
        )
        self.ds_train = _dataset(
            self.conf.ds_train,
            n_mels=self.conf.num_mel_filters,
            sr=self.conf.samplerate,
        )
        self.ds_val = _dataset(
            self.conf.ds_val,
            n_mels=self.conf.num_mel_filters,
            sr=self.conf.samplerate,
            size=sys.maxsize,
        )
        self.conf.steps = (
            self.conf.steps or (self.ds_train.size // self.conf.size_batch)
        ) * self.conf.epochs
        return (
            DataLoader(self.ds_train, **conf),
            DataLoader(self.ds_val, **conf),
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
        for anchor, positive, negative in tracker(
            take(self.conf.size_val, vloader),
            "validation",
            total=self.conf.size_val,
        ):
            loss += tripletloss(
                self(anchor.to(self.device)),
                self(positive.to(self.device)),
                self(negative.to(self.device)),
            )
        loss /= self.conf.size_val
        if loss < self.conf.min_loss:
            self.save(model=self.conf.model, snap=f"-{loss:.2f}-{it:06d}")
        self.conf.min_loss = min(self.conf.min_loss, loss)
        dumper(val_loss=(f"{loss:.4f} ({self.conf.min_loss:.4f} best so far)"))
        return loss

    def save(self, model=None, snap=None):
        """Create a checkpoint"""
        path = path_model(model or self.conf.model)
        mkdir(dirname(path))
        torch.save(
            dict(
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
