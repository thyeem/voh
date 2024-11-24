import logging
import multiprocessing as mp
import random

import torch
from foc import *
from ouch import *
from torch.nn import functional as F

from .dl import _dataloader, _dataset
from .model import *
from .utils import *

mp.set_start_method("fork", force=True)


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
            .set_iter()
            .set_optim()
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
            .set_iter(t.get("it"))
            .set_optim(t.get("optim"))
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
        self.update_lr(force=True)
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
        if hasattr(self.optim, "state"):
            for state in self.optim.state.values():  # update optimize's state
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
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

    @property
    def summary(self):
        return (
            dmap(
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
            | dmap(
                path=which_model(self.name),
                size=size_model(self.name),
            )
            if exists(path_model(self.name))
            else dmap()
        )

    def show(self):
        dumper(**self.summary)

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
        tl, vl = self.dl()  # dataloader: (training, validation)
        g = self.conf.steps * self.conf.epochs  # global steps
        with tl, vl:
            vl.pause()
            for _ in tracker(range(g), "training", start=self.it, total=g):
                anchor, positive, negative = next(tl)
                self.train()
                self.update_lr()
                self.optim.zero_grad(set_to_none=True)
                loss = tripletloss(
                    self(anchor),
                    self(positive),
                    self(negative),
                    margin=self.conf.margin_loss,
                )
                loss.backward()
                self.optim.step()
                self.validate(tl, vl)
                self.log(loss)
                self.it += 1

    @torch.no_grad()
    def validate(self, tl, vl):
        if not self.it_interval(self.conf.int_val):
            return
        self.eval()
        tl.pause()
        vl.resume()
        loss = 0
        for _ in tracker(range(self.conf.size_val), "validation"):
            anchor, positive, negative = next(vl)
            loss += tripletloss(
                self(anchor),
                self(positive),
                self(negative),
                margin=self.conf.margin_loss,
            )
        vl.pause()
        tl.resume()
        loss /= self.conf.size_val
        if loss < self.loss:
            self.save(snap=f"-{loss:.2f}-{self.it:06d}")
        self.loss = min(self.loss, loss)

    def update_lr(self, force=False):
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
        for param_group in self.optim.param_groups:
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
        kwds = dict(  # shared keywords for dataset
            n_mels=self.conf.num_mel_filters,
            sr=self.conf.samplerate,
            size_batch=self.conf.size_batch,
        )
        kwdl = dict(  # shared keywords for dataloader
            size_queue=self.conf.size_batch,
            num_workers=self.conf.num_workers,
        )
        return (
            _dataloader(_dataset(self.conf.ds_train, **kwds), **kwdl),
            _dataloader(_dataset(self.conf.ds_val, **kwds), **kwdl),
        )

    def log(self, loss):
        setattr(self, "_loss", getattr(self, "_loss", 0) + loss)
        if not self.it_interval(self.conf.size_val):
            return
        self._loss /= self.conf.size_val
        record = [
            f"{self.it:06d}",
            f"{self.lr:.6f}",
            f"{self._loss:.4f} ({None})",
            f"{self.loss:.4f} ({None})",
        ]
        header = ["Step", "lr", "Avg Loss (EMA)", "Loss (EMA)"]
        # nohead = True if self.it_interval(10 * self.conf.size_val) else None
        print(tabulate(header=header, fn=" " * 10 + _, missing="?")([record]))
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
