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
            .set_stat()
            .finalize()
        )

    @classmethod
    def load(cls, name, conf=None, strict=True, debug=False):
        """Load the pre-trained"""
        path = name if debug else which_model(name)
        t = torch.load(path, map_location="cpu")
        return (
            voh()
            .set_name(t.get("name"))
            .set_conf(t["conf"])
            .set_conf(conf or dmap(), kind=default.META, warn=False)
            .set_seed()
            .set_model(t["model"], strict=strict)
            .set_iter(t.get("it"))
            .set_optim(t.get("optim"))
            .set_stat(t.get("stat"))
            .finalize()
        )

    def set_name(self, name):
        self.name = name
        return self

    def set_conf(self, conf, kind=None, warn=True):
        ref = self.conf if hasattr(self, "conf") else def_conf()
        self.conf = ref | uniq_conf(conf, def_conf(kind=kind), warn=warn)
        self.guards()
        return self

    def set_seed(self, seed=None):
        seed = seed or randint(1 << 31)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return self

    def set_model(self, model=None, strict=True):
        self.encoder = Encoder(self.conf)
        self.decoder = Decoder(self.conf)
        if model:
            self.load_state_dict(model, strict=strict)
        return self

    def set_iter(self, it=None):
        self.it = 0 if self.conf.reset else (it or 0)
        return self

    def set_stat(self, stat=None):
        if self.conf.reset or stat is None:
            self.stat = dmap(
                loss=float("inf"),
                vloss=float("inf"),
                minloss=float("inf"),
                alpha=0.2,
            )
        else:
            self.stat = dmap(stat)
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
        if not self.conf.reset and optim:
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
        torch._dynamo.eval_frame.OptimizedModule.__repr__ = self.__repr__
        return self.into().optimize()

    def guards(self, train=False):
        guard(
            self.conf.size_out_enc == self.conf.size_in_dec,
            f"error, output size({self.conf.size_out_enc}) of"
            " the encoder does not match the input size"
            f"({self.conf.size_in_dec}) of the decoder.",
        )
        guard(
            self.conf.num_mining <= self.conf.size_batch,
            f"error, number of hard negatives ({self.conf.num_mining}) "
            f"must be less than batch size ({self.conf.size_batch})",
        )
        train and guard(
            self.conf.size_in_enc == self.conf.num_mel_filters,
            f"error, input size({self.conf.size_in_enc}) of"
            " the encoder does not match the number of "
            f"Mel-filterbanks({self.conf.num_mel_filters}).",
        )
        train and guard(
            self.conf.ds_train and exists(self.conf.ds_train),
            f"error, not found value/set for 'ds_train': {self.conf.ds_train}",
        )
        train and guard(
            self.conf.ds_val and exists(self.conf.ds_val),
            f"error, not found value/set for 'ds_val': {self.conf.ds_val}",
        )

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
        return dmap(
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
        ) | (
            dmap(
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
        o = dmap(**kind_conf(self.conf, kind=kind)) | dmap(
            loss=f"{self.stat.loss:.4f}",
            val_loss=f"{self.stat.vloss:.4f}",
            it=(
                f"{self.it}  "
                f"({100 * self.it / (self.conf.epochs * self.conf.steps):.2f}"
                "% complete)"
            ),
        )
        dumper(**o)

    def __repr__(self):
        return self.name if hasattr(self, "name") else ""

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
        dl = self.dl()  # dataloader of (training + validation) set
        dl.train()
        g = self.conf.steps * self.conf.epochs  # global steps
        self.update_lr()
        self._loss = self._vloss = 0
        with dl:
            for _ in tracker(range(g), "training", start=self.it, total=g):
                self.train()
                if self.on_interval(self.conf.int_sched_lr):
                    self.update_lr()

                # load triplet -> model-pass -> normalize
                anchor, positive, negative = map(
                    cf_(f_(F.normalize, dim=-1), self),
                    next(dl),
                )
                # loss = triplet-loss + contrastive-loss
                loss = triplet_contrastive_loss(
                    anchor,
                    positive,
                    negative,
                    margin_min=self.conf.margin_min,
                    margin_max=self.conf.margin_max,
                    alpha=self.conf.alpha,
                    tau=self.conf.tau,
                    num_mining=self.conf.num_mining,
                    prob_mining=self.conf.prob_mining,
                )
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                self.optim.step()

                self._loss += loss
                if self.on_interval(self.conf.int_val) and not self.on_warmup():
                    self.validate(dl)
                if self.on_interval(self.conf.size_val):
                    self.log()
                self.it += 1
        self.checkpoint()

    @torch.no_grad()
    def validate(self, dl):
        self.eval()
        dl.eval()
        self._vloss = 0
        for _ in tracker(range(self.conf.size_val), "validation"):
            anchor, positive, negative = map(
                cf_(f_(F.normalize, dim=-1), self),
                next(dl),
            )
            self._vloss += triplet_contrastive_loss(
                anchor,
                positive,
                negative,
                margin_min=self.conf.margin_min,
                margin_max=self.conf.margin_max,
                alpha=self.conf.alpha,
                tau=self.conf.tau,
            ).item()
        self._vloss /= self.conf.size_val
        self.stat.vloss = ema(alpha=self.stat.alpha)(self.stat.vloss, self._vloss)
        if self.stat.vloss < self.stat.minloss:
            self.stat.minloss = self.stat.vloss
            self.checkpoint()
        dl.train()

    def update_lr(self):
        self._lr = (
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
            param_group["lr"] = self._lr

    def dl(self):
        self.guards(train=True)
        shared = dict(  # shared keywords for dataset
            n_mels=self.conf.num_mel_filters,
            sr=self.conf.samplerate,
            size_batch=self.conf.size_batch,
        )
        return _dataloader(
            _dataset(  # training dataset
                self.conf.ds_train,
                p=self.conf.prob_aug,
                num_aug=self.conf.num_aug,
                **shared,
            ),
            _dataset(  # validation dataset
                self.conf.ds_val,
                p=None,
                **shared,
            ),
            num_workers=self.conf.num_workers,
        )

    def log(self):
        loss = (self._loss / self.conf.size_val).item()
        self.stat.loss = ema(alpha=self.stat.alpha)(self.stat.loss, loss)
        record = [
            f"{self.it:06d}",
            f"{self._lr:.8f}",
            f"{loss:.4f}",
            f"{self.stat.loss:.4f}",
            f"{self._vloss:.4f}",
            f"{self.stat.vloss:.4f} ({self.stat.minloss:.4f})",
        ]
        header = ["Step", "lr", "Loss", "Loss(EMA)", "vLoss", "vLoss(EMA)"]
        print(tabulate([record], header=header, fn=" " * 10 + _))
        self._loss = 0

    def save(self, name=None, ckpt=None):
        name = name or self.name
        if null(name):
            error("The model name is not specified.")
        path = path_model(name)
        mkdir(dirname(path))
        torch.save(
            dict(
                name=name,
                it=self.it,
                stat=dict(self.stat),
                optim=self.optim.state_dict() if self.optim else None,
                conf=dict(self.conf) | dict(reset=False),
                model=self.state_dict(),
            ),
            normpath(path),
        )
        if ckpt:
            d = f"{path}.ckpt"
            mkdir(d)
            shell(f"cp -f {path} {d}/{basename(path)}{ckpt}")
        return path

    def checkpoint(self, retain=24):
        self.save(
            ckpt=f"-v{self.stat.vloss:.3f}-t{self.stat.loss:.3f}-{self.it:06d}",
        )
        path = f"{path_model(self.name)}.ckpt"
        for f in shell(f"find {path} -type f | sort -V")[retain:]:
            os.remove(f)

    def on_interval(self, val):
        return self.it and self.it % val == 0

    def on_warmup(self):
        return self.it <= int(self.conf.steps * self.conf.ratio_warmup)
