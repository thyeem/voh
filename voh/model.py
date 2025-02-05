import torch
import torch.nn as nn
from foc import *
from ouch import *
from torch.nn import functional as F

from .utils import *


class Encoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.ln = LayerNorm(conf.size_in_enc)
        self.prolog = Context(
            conf.size_in_enc,
            conf.size_hidden_enc,
            conf.size_kernels[0],
            dilation=conf.size_dilations[0],
            reduction=conf.ratio_reduction,
            end=True,
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    conf.size_hidden_enc,
                    conf.size_hidden_enc,
                    conf.num_repeats,  # repeated block R-times
                    size_kernel,
                    dilation=size_dilation,
                    reduction=conf.ratio_reduction,
                    dropout=conf.dropout,
                )
                for size_kernel, size_dilation in zipl(
                    conf.size_kernels, conf.size_dilations
                )[1:-1]
            ]
        )
        self.epilog = Context(
            conf.size_hidden_enc,
            conf.size_out_enc,
            conf.size_kernels[-1],
            dilation=conf.size_dilations[-1],
            reduction=conf.ratio_reduction,
            end=True,
        )

    def forward(self, mask, x):
        return cf_(
            f_(self.epilog, mask),  # (B, C'out, T)
            cf_(*[f_(b, mask) for b in rev(self.blocks)]),  # (B, C'hidden, T)
            f_(self.prolog, mask),  # (B, C'hidden, T)
            self.ln,  # (B, C'in, T)
        )(x)


class Block(nn.Module):
    """Core block used in Encoder"""

    def __init__(
        self,
        size_in,
        size_out,
        num_repeats,
        size_kernel,
        dilation=1,
        reduction=8,
        dropout=0.1,
    ):
        super().__init__()
        self.repeats = nn.ModuleList(
            [
                RepeatR(
                    size_in,
                    size_out,
                    size_kernel,
                    dilation=dilation,
                    dropout=dropout,
                )
                for _ in range(num_repeats)
            ]
        )
        self.context = Context(
            size_in,
            size_out,
            size_kernel,
            dilation=dilation,
            reduction=reduction,
            end=False,
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, mask, x):
        return cf_(
            self.dropout,
            self.relu,
            f_(self.context, mask),  # SE-context
            cf__(*[f_(r, mask) for r in self.repeats]),  # Repeat Rx
        )(x)


class RepeatR(nn.Module):
    """Unit block repeated R-times in Block module"""

    def __init__(
        self,
        size_in,
        size_out,
        size_kernel,
        dilation=1,
        dropout=0.1,
    ):
        super().__init__()
        self.tsc = TimeSeparable(
            size_in,
            size_out,
            size_kernel,
            dilation=dilation,
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, mask, x):
        return cf_(
            self.dropout,
            self.relu,
            _ + x,  # skip connection
            f_(self.tsc, mask),
        )(x)


class Context(nn.Module):
    """SE-Context module"""

    def __init__(
        self,
        size_in,
        size_out,
        size_kernel,
        dilation=1,
        reduction=8,
        end=False,
    ):
        super().__init__()
        self.tsc = TimeSeparable(
            size_in,
            size_out,
            size_kernel,
            dilation=dilation,
        )
        self.se = SqueezeExcite(size_out, reduction=reduction)
        self.relu = nn.ReLU(inplace=True) if end else None
        self.end = end  # flag for [pro|epil]log block

    def forward(self, mask, x):
        return cf_(
            self.relu if self.end else id,
            id if self.end else _ + x,  # skip connection when not in end block
            self.se,
            f_(self.tsc, mask),
        )(x)


class TimeSeparable(nn.Module):
    """Time-Channel Separable Conv (1d-Depthwise Separable Conv)"""

    def __init__(
        self,
        size_in,
        size_out,
        size_kernel,
        dilation=1,
    ):
        super().__init__()
        self.ln = LayerNorm(size_in)
        self.depthwise = MaskedConv1d(
            size_in,
            size_in,
            size_kernel,
            dilation=dilation,
            groups=size_in,
            bias=False,
        )
        self.pointwise = MaskedConv1d(
            size_in,
            size_out,
            1,
            dilation=dilation,
            bias=False,
        )

    def forward(self, mask, x):
        return cf_(
            # time-separable-conv :: 1d-depthwise-conv -> pointwise-conv
            f_(self.pointwise, mask),
            f_(self.depthwise, mask),
            self.ln,
        )(x)


class SqueezeExcite(nn.Module):
    "Squeeze-and-Excitation block"

    def __init__(self, channels, reduction=8):
        super().__init__()
        guard(
            channels % reduction == 0,
            f"Error, SE({channels}) must be divisible by {reduction}",
        )
        self.ln = LayerNorm(channels)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        return cf_(
            _ * x,  # (B, C, T)
            torch.sigmoid,  # (B, C, 1)
            ob(_.transpose)(1, -1),  # (B, C, 1)
            self.fc2,  # (B, 1, C)
            self.relu,  # (B, 1, C // reduction)
            self.fc1,  # (B, 1, C // reduction)
            ob(_.transpose)(1, -1),  # (B, 1, C)
            ob(_.mean)(dim=-1, keepdim=True),  # (B, C, 1)
            self.ln,
        )(x)


class MaskedConv1d(nn.Module):
    def __init__(
        self,
        size_in,
        size_out,
        size_kernel,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            size_in,
            size_out,
            size_kernel,
            stride=stride,
            padding=(dilation * (size_kernel - 1)) // 2,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, mask, x):
        return self.conv(x) * mask


class LayerNorm(nn.LayerNorm):
    def __init__(self, channels):
        super().__init__(normalized_shape=channels)

    def forward(self, x):
        return cf_(
            ob(_.transpose)(1, -1),  # (B, C, T)
            super().forward,  # nn.LayerNorm along C
            ob(_.transpose)(1, -1),  # (B, T, C)
            # supposed to be (B, C, T)
        )(x)


class Decoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.ln = LayerNorm(conf.size_in_dec)
        self.pool = AttentivePool(
            conf.size_in_dec,
            conf.size_attn_pool,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
        )
        self.emb = Embedding(
            conf.size_in_dec * 2 * conf.num_heads,
            conf.size_out_dec,
        )

    def forward(self, mask, x):
        return cf_(
            self.emb,  # (B, E)
            f_(self.pool, mask),  # (B, 2C, 1)
            self.ln,  # (B, C, T)
        )(x)


class AttentivePool(nn.Module):
    """Attention pooling layer with multi-head attention approach"""

    def __init__(self, size_in, size_attn_pool, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Conv1d(size_in, size_attn_pool, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.attn = nn.Conv1d(size_attn_pool, num_heads, 1)

    def forward(self, mask, x):
        B, C, T = x.shape
        return cf_(
            ob(_.view)(B, -1),  # (B, H * 2C)
            f_(torch.cat, dim=-1),  # (B, H, 2C)
            f_(wtd_mu_sigma, x.unsqueeze(1)),  # ((B, H, C), (B, H, C))
            ob(_.unsqueeze)(2),  # (B, H, 1, T)
            f_(F.softmax, dim=-1),  # (B, H, T)
            ob(_.masked_fill)(mask == 0, float("-inf")),
            self.attn,  # (B, C'attn, T) -> (B, H, T)
            self.dropout,
            self.tanh,
            self.proj,  # (B, C, T) -> (B, C'attn, T)
        )(x)


class Embedding(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.fc = nn.Linear(size_in, size_out, bias=True)

    def forward(self, x):
        return self.fc(x)  # (B, 2C) -> (B, E)
