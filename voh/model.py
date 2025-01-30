import torch
import torch.nn as nn
from foc import *
from ouch import *
from torch.nn import functional as F

from .utils import *


def pad_conv(size_kernel, dilation=1):
    return (dilation * (size_kernel - 1)) // 2


class Encoder(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.ln = LayerNorm(conf.size_in_enc)
        self.prolog = Context(
            conf.size_in_enc,
            conf.size_hidden_enc,
            conf.size_kernel_prolog,
            reduction=conf.ratio_reduction,
            residual=False,
            end=True,
        )
        # num-of-blocks(B) = 1(prolog) + num-of-size_kernel_blocks + 1(epilog)
        self.blocks = nn.ModuleList(
            [
                Block(
                    conf.size_hidden_enc,
                    conf.size_hidden_enc,
                    conf.num_repeat_blocks,  # repeated block R-times
                    size_kernel,
                    reduction=conf.ratio_reduction,
                    dropout=conf.dropout,
                )
                for size_kernel in conf.size_kernel_blocks
            ]
        )
        self.epilog = Context(
            conf.size_hidden_enc,
            conf.size_out_enc,
            conf.size_kernel_epilog,
            reduction=conf.ratio_reduction,
            residual=False,
            end=True,
        )

    def forward(self, mask, x):
        return cf_(
            f_(self.epilog, mask),  # (B, C'out, T)
            cf__(*[f_(b, mask) for b in self.blocks]),  # (B, C'hidden, T)
            f_(self.prolog, mask),  # (B, C'hidden, T)
            self.ln,  # (B, C'in, T)
        )(x)


class Block(nn.Module):
    """Core block used in Encoder"""

    def __init__(
        self,
        size_in,
        size_out,
        num_repeat,
        size_kernel,
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
                )
                for _ in range(num_repeat)
            ]
        )
        self.context = Context(
            size_in,
            size_out,
            size_kernel,
            reduction=reduction,
            residual=True,
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

    def __init__(self, size_in, size_out, size_kernel, padding=None, dropout=0.1):
        super().__init__()
        if padding is None:
            padding = pad_conv(size_kernel)

        self.tsc = TimeSeparable(size_in, size_out, size_kernel, padding=padding)
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
        padding=None,
        reduction=8,
        residual=False,
        end=False,
    ):
        super().__init__()
        if padding is None:
            padding = pad_conv(size_kernel)

        self.tsc = TimeSeparable(size_in, size_out, size_kernel, padding=padding)
        self.se = SqueezeExcite(size_out, reduction=reduction)
        self.relu = nn.ReLU(inplace=True) if end else None
        self.res = residual

    def forward(self, mask, x):
        return cf_(
            self.relu if self.relu else id,
            _ + x if self.res else id,  # skip connection
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
        padding=None,
    ):
        super().__init__()
        if padding is None:
            padding = pad_conv(size_kernel)

        self.ln = LayerNorm(size_in)
        self.depthwise = MaskedConv1d(
            size_in,
            size_in,
            size_kernel,
            padding=padding,
            groups=size_in,
            bias=False,
        )
        self.pointwise = MaskedConv1d(
            size_in,
            size_out,
            1,
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
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        return cf_(
            _ * x,  # (B, C, C)
            torch.sigmoid,  # (B, C, 1)
            ob(_.transpose)(1, -1),  # (B, C, 1)
            self.fc2,  # (B, 1, C)
            self.relu,
            self.fc1,  # (B, 1, C // reduction)
            ob(_.transpose)(1, -1),  # (B, 1, C)
            ob(_.mean)(dim=-1, keepdim=True),  # (B, C, 1)
        )(x)


class MaskedConv1d(nn.Module):
    def __init__(
        self,
        size_in,
        size_out,
        size_kernel,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            size_in,
            size_out,
            size_kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, mask, x):
        return self.conv(x) * mask


class LayerNorm(nn.LayerNorm):
    def __init__(self, channels, elementwise_affine=False):
        super().__init__(
            normalized_shape=channels,
            elementwise_affine=elementwise_affine,
        )

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
        self.pool = AttentivePool(conf.size_in_dec, conf.size_attn_pool)
        self.emb = Embedding(conf.size_in_dec * 2, conf.size_out_dec)
        self.ln = nn.LayerNorm(conf.size_out_dec, elementwise_affine=False)

    def forward(self, mask, x):
        return cf_(
            self.ln,
            self.emb,  # (B, E)
            f_(self.pool, mask),  # (B, 2C, 1)
        )(x)


class AttentivePool(nn.Module):
    """Attention pooling layer with mask and statistical pooling"""

    def __init__(self, size_in, size_attn_pool):
        super().__init__()
        self.tdnn = TDNN(size_in * 3, size_attn_pool, 1)
        self.tanh = nn.Tanh()
        self.conv = nn.Conv1d(size_attn_pool, size_in, 1)

    def forward(self, mask, x):
        B, C, T = x.size()  # dim(x) = (B, C, T)
        norm_mask = mask / torch.sum(mask, dim=2, keepdim=True)  # (B, 1, T)

        # stats from encoder-output
        mean, std = wtd_mu_sigma(x, norm_mask)  # (B, C, 1) each
        # enriched-concatenated input: (B, 3C, T)
        y = torch.cat((x, mean.expand(-1, -1, T), std.expand(-1, -1, T)), dim=1)

        # attention value
        alpha = cf_(
            F.softmax,
            ob(_.masked_fill)(mask == 0, float("-inf")),
            self.conv,
            self.tanh,
            self.tdnn,
        )(y)
        # stats from attention
        mu, sigma = wtd_mu_sigma(x, alpha)  # (B, C, 1) each
        return torch.cat((mu, sigma), dim=1)  # (B, 2C, 1)


class Embedding(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.ln = LayerNorm(size_in)
        self.conv = nn.Conv1d(size_in, size_out, 1)

    def forward(self, x):
        return cf_(
            ob(_.squeeze)(dim=-1),
            self.conv,
            self.ln,
        )(x)


class TDNN(nn.Module):
    """1D-TDNN (Time-Delay Neural Network)"""

    def __init__(
        self,
        size_in,
        size_out,
        size_kernel=1,
        dilation=1,
        stride=1,
        padding=None,
    ):
        super().__init__()
        if padding is None:
            padding = pad_conv(size_kernel, dilation)
        self.ln = LayerNorm(size_in)
        self.conv = nn.Conv1d(
            size_in,
            size_out,
            size_kernel,
            dilation=dilation,
            stride=stride,
            padding=padding,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return cf_(
            self.relu,
            self.conv,
            self.ln,
        )(x)
