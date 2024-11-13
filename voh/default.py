from ouch import *

CORE = 1  # model-fundamental conf
META = 2  # model-training conf

conf = dmap(
    # -----------------------
    # model configuration
    # -----------------------
    model=(CORE, "default"),
    size_in_enc=(CORE, 80),
    size_hidden_enc=(CORE, 256),
    size_out_enc=(CORE, 512),
    size_kernel_prolog=(CORE, 3),
    size_kernel_epilog=(CORE, 1),
    size_kernel_blocks=(CORE, (5, 7, 9, 11)),
    num_repeat_blocks=(CORE, 2),
    ratio_reduction=(CORE, 8),
    size_in_dec=(CORE, 512),
    size_attn_pool=(CORE, 32),
    size_out_dec=(CORE, 128),
    dropout=(CORE, 0.1),
    # -----------------------
    # dataset/training
    # -----------------------
    num_mel_filters=(META, 80),
    samplerate=(META, 16000),
    ds_train=(META, None),
    ds_val=(META, None),
    seed=(META, None),
    decay=(META, True),
    margin_loss=(META, 0.7),
    lr=(META, 5e-5),
    lr_min=(META, 1e-7),
    ratio_warmup=(META, 0.01),
    steps=(META, None),
    epochs=(META, None),
    num_workers=(META, 8),
    size_batch=(META, 8),
    size_val=(META, 20),
    period_val=(META, 100),
    avg_loss=(META, 0),
    min_loss=(META, 99999),
)

modelpath = f"{dirname(__file__)}/../o"
