import logging
import multiprocessing as mp
import random
from bisect import bisect
from collections import Counter

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
            ),
            pos=dmap(
                t=dataq(1000 * self.conf.size_batch),
                v=dataq(self.conf.size_val * self.conf.size_batch),
            ),
            neg=dmap(
                t=dataq(1000 * self.conf.size_batch, 1.0),
                v=dataq(self.conf.size_val * self.conf.size_batch),
            ),
            mine=dmap(
                size=dataq(1000),
                diff=dataq(1000 * self.conf.size_batch**2),
            ),
        )
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

    def readf(self, f):
        return cf_(self.read, readwav)(f)

    @torch.no_grad()
    def embed(self, x, file=True):
        self.eval()
        return cf_(self, (self.readf if file else self.read))(x)

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
                self.validate(dl, sched=True)
                self.dist(sched=True)
                triplet = next(dl)
                self.eval()
                with torch.no_grad():
                    anchor, positive, negative = map(self, triplet)
                    i, j = self.mine(anchor, positive, negative)
                self.update_lr(sched=True)
                self.train()
                anchor, positive, negative = map(self, triplet)
                loss = self.get_loss(anchor[i], positive[i], negative[j])
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
                self.optim.step()
                self.dq.loss.t.update(loss.item())
                self.log(sched=True)
                self.it += 1

    def get_loss(self, anchor, positive, negative):
        ap = F.cosine_similarity(anchor, positive, dim=-1)
        an = F.cosine_similarity(anchor, negative, dim=-1)
        loss = F.relu(an - ap + self.conf.margin)
        return (
            self.conf.scale * loss.mean()  # triplet loss
            + torch.exp(an).mean()  # penalize negatives
            + torch.exp(-ap).mean()  # penalize positives
        )

    def mine(self, anchor, positive, negative):
        """Find the indices of challenging negatives based on distribution."""
        ap = (
            F.cosine_similarity(anchor, positive, dim=-1)
            .unsqueeze(1)
            .expand(-1, len(positive))
        )
        an = F.cosine_similarity(
            anchor.unsqueeze(1),
            negative.unsqueeze(0),
            dim=-1,
        )
        hard = (
            an > self.dq.neg.t.percentile(100 * (1 - self.conf.hard_ratio))
        ).nonzero()
        semi = (
            (an > ap) & (an < ap + self.conf.semi_ratio * self.conf.margin)
        ).nonzero()
        q = (
            (an == torch.max(an)).nonzero()
            if not len(hard) and not len(semi)
            else torch.cat([hard, semi], dim=0)
        )
        self.dq.mine.diff.update((ap - an).tolist())
        self.dq.mine.size.update(len(q))
        self.dq.pos.t.update(ap.tolist())
        self.dq.neg.t.update(an.tolist())
        return q[:, 0], q[:, 1]

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
                    "LOSS(EMA) >= MIN",
                ]
                if self.on_interval(5 * self.conf.size_val)
                else []
            ),
            [
                f"{self.it:06d}",
                f"{self._lr:.8f}",
                f"{self.dq.mine.diff.median:7.4f}/{self.dq.mine.diff.mad:.4f}",
                f"{self.dq.pos.t.median:.4f}/{self.dq.pos.t.mad:.4f}",
                f"{self.dq.neg.t.median:.4f}/{self.dq.neg.t.mad:.4f}",
                f"{self.dq.loss.t.median:.4f}({self.stat.loss.t:.4f})",
            ],
            [
                "",
                "/".join(f"{x:.1f}" for x in self.dq.mine.size.quartile),
                f"{self.dq.mine.diff.min:7.4f}/{self.dq.mine.diff.max:.4f}",
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

    def dist_pairs(self, size):
        def waveform_pairs(mono):
            return [mapl(readwav, p) for p in randpair(db, mono=mono, size=size)]

        if not hasattr(self, "nfix") or not hasattr(self, "pfix"):
            db = read_json(self.conf.ds_val)
            self.nfix = waveform_pairs(False)
            self.pfix = waveform_pairs(True)
        return self.nfix, self.pfix

    @torch.no_grad()
    def dist(self, data=None, size=42, sched=False):
        def t(pairs, desc, mono=True):
            cosims = [
                F.cosine_similarity(*map(f_(self.embed, file=False), pair)).item()
                for pair in tracker(pairs, desc)
            ]
            bins = [0.6, 0.7, 0.8, 0.9, 1.01]
            hist = Counter(bisect(bins, cosim) for cosim in cosims)
            pdf = [hist.get(i, 0) / len(pairs) for i in range(len(bins))]
            return dmap(
                pdf=pdf,
                cdf=(scanl1 if mono else scanr1)(op.add, pdf),
                median=np.median(cosims),
                mad=np.median(np.abs(np.array(cosims) - np.median(cosims))),
            )

        if sched and not self.on_interval(self.conf.int_dist):
            return
        nfix, pfix = data if data else self.dist_pairs(size)
        n, p = t(nfix, "n-distrib", False), t(pfix, "p-distrib", True)
        print(
            tabulate(
                [
                    [f"{x:.4f}" for x in n.pdf] + [f"{n.median:.4f}"],
                    [f"{x:.4f}" for x in n.cdf] + [f"{n.mad:.4f}"],
                    ["<0.6", "<0.7", "<0.8", "<0.9", "<1.0", "med/mad"],
                    [f"{x:.4f}" for x in p.pdf] + [f"{p.median:.4f}"],
                    [f"{x:.4f}" for x in p.cdf] + [f"{p.mad:.4f}"],
                ],
                style="grid",
                nohead=True,
            ),
        )
