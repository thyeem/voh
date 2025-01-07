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
                loss=dmap(t=float("inf"), v=float("inf")),
                minloss=float("inf"),
                alpha=0.1,
            )
        else:
            self.stat = dmap(stat)
        self.dq = dmap(
            pos=dmap(
                t=dataQ(self.conf.size_val * self.conf.size_batch),
                v=dataQ(self.conf.size_val * self.conf.size_batch),
            ),
            neg=dmap(
                t=dataQ(self.conf.size_val * self.conf.size_batch**2),
                v=dataQ(self.conf.size_val * self.conf.size_batch**2),
            ),
            mine=dmap(
                loss=dataQ(self.conf.size_mineq),
                th=dataQ(self.conf.size_mineq),
            ),
            loss=dmap(t=dataQ(self.conf.size_val), v=dataQ(self.conf.size_val)),
        )
        self.dq.mine.th.update(1 - self.conf.ratio_hard)
        self.ema = ema(alpha=self.stat.alpha)
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

    def forward(self, x, mask=None):
        x = x.to(device=self.device)
        if mask is None:
            mask = create_mask(x)
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
                max_frames=self.conf.max_frames,
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
    @torch.no_grad()
    def prepare(self, dl):
        dl.train()
        if not self.dq.mine.loss.data:
            self.train()
            size = self.dq.mine.loss.data.maxlen // (self.conf.size_batch**2)
            for _ in tracker(range(size), "preparing"):
                anchor, positive, negative = self.next(dl)
                self.update_stat(anchor, positive, negative)

    def next(self, dl):
        return map(
            cf_(f_(F.normalize, dim=-1), self),
            next(dl),
        )

    def get_trained(self):
        dl = self.dl()  # dataloader of (training + validation) set
        g = self.conf.steps * self.conf.epochs  # global steps
        with dl:
            self.prepare(dl)
            for _ in tracker(range(g), "training", start=self.it, total=g):
                self.train()
                self.update_lr(sched=True)
                self.validate(dl, sched=True)
                anchor, positive, negative = self.next(dl)
                self.update_stat(anchor, positive, negative)
                loss = self.get_loss(anchor, positive, negative).mean()
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                self.optim.step()
                self.dq.loss.t.update(loss.item())
                self.log(sched=True)
                self.it += 1
        self.checkpoint()

    def get_loss(self, anchor, positive, negative, mining=False):
        if self.training and not mining:  # hard mining in training mode
            i, j = self.mine(anchor, positive, negative)
            anchor = anchor[i]
            positive = positive[i]
            negative = negative[j]
        if mining:  # creating similarity matrix when mining
            ap = F.cosine_similarity(anchor, positive, dim=-1).unsqueeze(1)
            an = F.cosine_similarity(
                anchor.unsqueeze(1),
                negative.unsqueeze(0),
                dim=-1,
            )
        else:
            ap = F.cosine_similarity(anchor, positive, dim=-1)
            an = F.cosine_similarity(anchor, negative, dim=-1)
        return F.relu(self.conf.margin + an - ap) + (1 - torch.acos(an) / np.pi)

    @torch.no_grad()
    def mine(self, anchor, positive, negative):
        """Find the indices of the most challenging negatives."""
        loss = self.get_loss(anchor, positive, negative, mining=True)
        threshold = 1 - self.conf.ratio_hard
        q = (loss > self.dq.mine.loss.percentile(100 * threshold)).nonzero()
        while not len(q):
            if threshold <= 0.5:  # ensure non-zero mask
                q = (loss == torch.max(loss)).nonzero()
                break
            threshold -= 0.01
            q = (loss > self.dq.mine.loss.percentile(100 * threshold)).nonzero()
        self.dq.mine.th.update(threshold)
        self.dq.mine.loss.update(loss.tolist())
        return q[:, 0], q[:, 1]

    @torch.no_grad()
    def update_stat(self, anchor, positive, negative):
        ap = F.cosine_similarity(anchor, positive, dim=-1).tolist()
        an = F.cosine_similarity(
            anchor.unsqueeze(1), negative.unsqueeze(0), dim=-1
        ).tolist()
        if self.training:
            self.dq.pos.t.update(ap)
            self.dq.neg.t.update(an)
        else:
            self.dq.pos.v.update(ap)
            self.dq.neg.v.update(an)

    @torch.no_grad()
    def validate(self, dl, sched=False):
        if sched and (
            not self.on_interval(self.conf.int_val)
            or (self.it <= int(self.conf.steps * self.conf.ratio_warmup))
        ):  # turn off validation on warmup stage
            return
        self.eval()
        dl.eval()
        for _ in tracker(range(self.conf.size_val), "validation"):
            anchor, positive, negative = self.next(dl)
            self.update_stat(anchor, positive, negative)
            self.dq.loss.v.update(
                self.get_loss(anchor, positive, negative).mean().item()
            )
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

    def dl(self):
        self.guards(train=True)
        shared = dict(  # shared keywords for dataset
            n_mels=self.conf.num_mel_filters,
            sr=self.conf.samplerate,
            max_frames=self.conf.max_frames,
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

    def log(self, sched=False):
        if sched and not self.on_interval(self.conf.size_val):
            return
        self.stat.loss.t = self.ema(self.stat.loss.t, self.dq.loss.t.median)
        cutval = self.dq.mine.loss.percentile(100 * self.dq.mine.th.median)
        log = [
            (
                [
                    "Step",
                    "LR",
                    "TH(Value)",
                    "P Median/MAD",
                    "N Median/MAD",
                    "Loss(EMA) >= MIN",
                ]
                if self.on_interval(5 * self.conf.size_val)
                else []
            ),
            [
                f"{self.it:06d}",
                f"{self._lr:.8f}",
                f"{self.dq.mine.th.median:.2f}({cutval:.4f})",
                f"{self.dq.pos.t.median:.4f}/{self.dq.pos.t.mad:.4f}",
                f"{self.dq.neg.t.median:.4f}/{self.dq.neg.t.mad:.4f}",
                f"{self.dq.loss.t.median:.4f}({self.stat.loss.t:.4f})",
            ],
            [
                "val",
                "-",
                "-",
                f"{self.dq.pos.v.median:.4f}/{self.dq.pos.v.mad:.4f}",
                f"{self.dq.neg.v.median:.4f}/{self.dq.neg.v.mad:.4f}",
                f"{self.dq.loss.v.median:.4f}({self.stat.loss.v:.4f})"
                f" >= {self.stat.minloss:.4f}",
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
