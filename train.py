from foc import *
from ouch import *

from voh import *

conf = dmap(
    # main/data
    model="o/pilot",
    ds_train="data-train.db",
    ds_val="data-val.db",
    num_mel_filters=80,
    samplerate=16000,
    # --------------------------------------
    # trainer
    seed=randint(1 << 31),
    decay=True,
    dropout=0.1,
    margin_loss=0.7,
    lr=5e-5,
    lr_min=1e-6,
    ratio_warmup=0.01,
    size_batch=8,
    size_val=20,
    num_workers=4,
    it_val=100,
    it_log=20,
    steps=None,
    # --------------------------------------
    # model architecture
    size_in_enc=None,
    size_hidden_enc=256,
    size_out_enc=512,
    size_kernel_prolog=3,
    size_kernel_epilog=1,
    size_kernel_blocks=(5, 7, 9, 11),
    num_repeat_blocks=2,
    ratio_reduction=8,
    size_in_dec=None,
    size_attn_pool=32,
    size_out_dec=128,
)

resume = True

if resume:
    md = voh.load(conf.model)
else:
    md = voh.new(conf)
    if exists(md.conf.model):
        prompt(
            f"{md.conf.model} exists. Are you sure to proceed?",
            fail=lazy(error, f"Aborted, {md.conf.model}."),
        )

md.info()
md.get_trained()
