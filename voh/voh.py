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
        t = torch.load(path, map_location="cpu", weights_only=False)
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
        self.seed = seed or randint(1 << 31)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        return self

    def set_model(self, model=None, strict=True):
        self.encoder = Encoder(self.conf)
        self.decoder = Decoder(self.conf)
        if model:
            self.load_state_dict(model, strict=strict)
        return self

    def set_iter(self, it=0):
        self.it = 0 if self.conf.reset else it
        return self

    def set_stat(self, stat=None):
        if self.conf.reset or stat is None:
            self.stat = dmap(
                loss=dmap(t=float("inf"), v=float("inf")),
                minloss=float("inf"),
                alpha=0.1,
            )
        else:
            self.stat = dmap(stat)
        self.dq = dmap(
            loss=dmap(
                t=dataq(self.conf.size_val),
                v=dataq(self.conf.size_val),
                triplet=dataq(self.conf.size_val),
                ce=dataq(self.conf.size_val),
                dist=dataq(self.conf.size_val),
                penalty=dataq(self.conf.size_val),
            ),
            pos=dmap(
                t=dataq(self.conf.int_val * self.conf.size_batch),
                v=dataq(self.conf.size_val * self.conf.size_batch),
            ),
            neg=dmap(
                t=dataq(self.conf.int_val * self.conf.size_batch**2),
                v=dataq(self.conf.size_val * self.conf.size_batch),
            ),
            mine=dmap(
                pos=dataq(self.conf.int_val * self.conf.size_batch),
                neg=dataq(self.conf.int_val * self.conf.size_batch),
                size=dataq(self.conf.int_val),
                diff=dataq(self.conf.int_val * self.conf.size_batch**2),
            ),
        )
        self.ema = ema(alpha=self.stat.alpha)
        return self

    def set_logger(self):
        self.logger = None

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
    def ipad(self):
        return -1e10

    @property
    def summary(self):
        return dmap(
            model=self.name,
            parameters=f"{self.numel:_}",
            in_enc=self.conf.size_in_enc,
            hidden_enc=self.conf.size_hidden_enc,
            out_enc=self.conf.size_out_enc,
            in_dec=self.conf.size_in_dec,
            out_dec=f"{self.conf.size_out_dec}  (embedding size)",
            kernels=str(self.conf.size_kernels),
            dilations=str(self.conf.size_dilations),
            blocks_B=max(0, len(self.conf.size_kernels) - 2),
            repeats_R=self.conf.num_repeats,
            attention_dec=self.conf.size_attn_pool,
            num_heads=self.conf.num_heads,
            ratio_reduction=self.conf.ratio_reduction,
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
            loss=f"{self.stat.loss.t:.4f}",
            val_loss=f"{self.stat.loss.v:.4f}",
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
        mask = create_mask(x, ipad=self.ipad)
        return cf_(
            f_(self.decoder, mask),
            f_(self.encoder, mask),
        )(x)

    @torch.no_grad()
    def read(self, ysr):
        """Load a given waveform in forms of log Mel-filterbank energies tensor."""
        return (
            filterbank(
                ysr,
                n_mels=self.conf.num_mel_filters,
                sr=self.conf.samplerate,
                max_frames=self.conf.max_frames,
            )
            .unsqueeze(0)
            .to(device=self.device)
        )

    @torch.no_grad()
    def embed(self, f):
        self.eval()
        return cf_(self, self.read, readwav)(f)

    @torch.no_grad()
    def cosim(self, a, b):
        return F.cosine_similarity(self.embed(a), self.embed(b)).item()

    def verify(self, a, b, threshold=0.7):
        return self.cosim(a, b) >= threshold

    # -----------
    # Training
    # -----------
    @property
    def tset(self):
        return lazy(  # bulder of training dataset
            _dataset,
            self.conf.ds_train,
            n_mels=self.conf.num_mel_filters,
            sr=self.conf.samplerate,
            max_frames=self.conf.max_frames,
            size_batch=self.conf.size_batch,
            ipad=self.ipad,
        )

    @property
    def vset(self):
        return lazy(  # builder of validation dataset
            _dataset,
            self.conf.ds_val,
            n_mels=self.conf.num_mel_filters,
            sr=self.conf.samplerate,
            max_frames=self.conf.max_frames,
            size_batch=self.conf.size_batch,
            ipad=self.ipad,
            augment=False,
        )

    @property
    def dl(self):
        self.guards(train=True)
        return lazy(  # builder of training/validation dataset loader
            _dataloader,
            self.tset,
            self.vset,
            num_workers=self.conf.num_workers,
        )

    def get_trained(self):
        dl = self.dl()
        dl.train()
        g = self.conf.steps * self.conf.epochs  # global steps
        with dl:
            for _ in tracker(range(g), "training", start=self.it):
                triplet, (i, j) = self.mine(dl)
                self.update_lr(sched=True)
                self.train()
                anchor, positive, negative = map(self, triplet)
                loss = self.get_loss(anchor[i], positive[i], negative[j])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                if self.on_interval(self.conf.acc_steps):
                    self.optim.step()
                    self.optim.zero_grad(set_to_none=True)
                self.dq.loss.t.update(loss.item())
                self.validate(dl, sched=True)
                self.log(sched=True)
                self.perf(sched=True)
                self.it += 1

    def get_loss(self, anchor, positive, negative):
        ap = F.cosine_similarity(anchor, positive, dim=-1)
        an = F.cosine_similarity(anchor, negative, dim=-1)

        # triplet loss
        triplet = self.conf.rho * F.relu(an - ap + self.conf.margin).mean()

        # cross-entropy loss
        ce = self.conf.kappa * F.cross_entropy(
            torch.stack([ap, an], dim=-1),
            torch.zeros(an.size(0), dtype=torch.long, device=anchor.device),
        )

        # positive-embedding distance
        dist = self.conf.lam * (anchor - positive).norm(dim=-1).mean()

        # negative-similarity penalty
        penalty = torch.exp(self.conf.nu * an).mean()

        self.dq.loss.triplet.update(triplet.item())
        self.dq.loss.ce.update(ce.item())
        self.dq.loss.dist.update(dist.item())
        self.dq.loss.penalty.update(penalty.item())
        return triplet + ce + dist + penalty

    @torch.no_grad
    def mine(self, dl):
        """Find the indices of challenging negatives based on distribution."""
        self.eval()
        while True:
            triplet = next(dl)
            anchor, positive, negative = map(self, triplet)
            ap = F.cosine_similarity(anchor, positive, dim=-1).unsqueeze(1)
            an = F.cosine_similarity(
                anchor.unsqueeze(1),
                negative.unsqueeze(0),
                dim=-1,
            )
            diff = ap - an
            cut = dataq(an.view(-1).tolist()).quantile(1 - self.conf.hard_ratio)
            q = ((an > cut) & (diff < self.conf.margin)).nonzero()
            if not len(q):
                continue
            self.dq.mine.diff.update(diff.tolist())
            self.dq.mine.size.update(len(q))
            self.dq.mine.neg.update(an[q].tolist())
            self.dq.mine.pos.update(ap[q[:, 0]].tolist())
            self.dq.neg.t.update(an.tolist())
            self.dq.pos.t.update(ap.tolist())
            return triplet, q.unbind(dim=1)

    @torch.no_grad()
    def validate(self, dl, sched=False):
        if sched and not self.on_interval(self.conf.int_val):
            return
        self.eval()
        dl.eval()
        for _ in tracker(range(self.conf.size_val), "validation"):
            anchor, positive, negative = map(self, next(dl))
            loss = self.get_loss(anchor, positive, negative).item()
            ap = F.cosine_similarity(anchor, positive, dim=-1).tolist()
            an = F.cosine_similarity(anchor, negative, dim=-1).tolist()
            self.dq.loss.v.update(loss)
            self.dq.pos.v.update(ap)
            self.dq.neg.v.update(an)
        self.stat.loss.v = self.ema(self.stat.loss.v, self.dq.loss.v.median)
        if self.stat.loss.v < self.stat.minloss:
            self.stat.minloss = self.stat.loss.v
            self.checkpoint()
        dl.train()

    def update_lr(self, sched=False):
        if sched and not self.on_interval(self.conf.int_sched_lr):
            return
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

    def log(self, sched=False):
        if sched and not self.on_interval(self.conf.size_val):
            return
        self.stat.loss.t = self.ema(self.stat.loss.t, self.dq.loss.t.median)
        log = [
            (
                [
                    "STEP",
                    "LR",
                    "d-SIMILARITY",
                    "POSITIVES",
                    "NEGATIVES",
                    "LOSS(EMA)",
                ]
                if self.on_interval(5 * self.conf.size_val)
                else []
            ),
            [
                f"{self.it:06d}",
                f"{self._lr:.8f}",
                f"{self.dq.mine.diff.median:7.4f}/{self.dq.mine.diff.mad:.4f}",
                f"{self.dq.pos.t.median:7.4f}/{self.dq.pos.t.mad:.4f}",
                f"{self.dq.neg.t.median:7.4f}/{self.dq.neg.t.mad:.4f}",
                f"{self.dq.loss.t.median:.4f}({self.stat.loss.t:.4f})",
            ],
            [
                "",
                "{:.1f}/{:.1f}/{:.1f}".format(*self.dq.mine.size.quartile),
                "{:7.4f}/{:.4f}".format(*_[0, -1](self.dq.mine.diff.quartile)),
                f"{self.dq.mine.pos.median:7.4f}/{self.dq.mine.pos.mad:.4f}",
                f"{self.dq.mine.neg.median:7.4f}/{self.dq.mine.neg.mad:.4f}",
                f"{self.dq.loss.triplet.median:.2f}/"
                f"{self.dq.loss.ce.median:.2f}/"
                f"{self.dq.loss.dist.median:.2f}/"
                f"{self.dq.loss.penalty.median:.2f}",
            ],
            [
                "",
                f">= {self.stat.minloss:.4f}",
                f"{self.dq.mine.diff.min:7.4f}/{self.dq.mine.diff.max:.4f}",
                f"{self.dq.pos.v.median:7.4f}/{self.dq.pos.v.mad:.4f}",
                f"{self.dq.neg.v.median:7.4f}/{self.dq.neg.v.mad:.4f}",
                f"{self.dq.loss.v.median:.4f}({self.stat.loss.v:.4f})",
            ],
        ]
        print(tabulate(filterl(bool, log), nohead=True), "\n")

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
                stat=deepdict(self.stat),
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
            ckpt=f"-v{self.stat.loss.v:.3f}-t{self.stat.loss.t:.3f}-{self.it:06d}",
        )
        path = f"{path_model(self.name)}.ckpt"
        for f in shell(f"find {path} -type f | sort -V")[retain:]:
            os.remove(f)

    def on_interval(self, val):
        return self.it and self.it % val == 0

    @torch.no_grad()
    def perf(self, size=None, sched=False):
        if sched and not self.on_interval(self.conf.int_perf):
            return
        if size or not hasattr(self, "__perf__"):
            db = read_json(self.conf.ds_val)
            size = size or self.conf.size_perf
            self.__perf__ = (
                randpair(db, size=size, mono=False),
                randpair(db, size=size, mono=True),
            )
        perf(
            self.load(self.name),  # use the best-so-far
            self.__perf__,
            out=f"/tmp/{self.name}-{self.seed}",
        )
