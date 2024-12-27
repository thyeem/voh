from ouch import *

CORE = 1  # model-fundamental conf
META = 2  # model-training conf

conf = dmap(
    # -----------------------
    # model configuration
    # -----------------------
    size_in_enc=(CORE, 80),
    size_hidden_enc=(CORE, 512),
    size_out_enc=(CORE, 1024),
    size_kernel_prolog=(CORE, 3),
    size_kernel_epilog=(CORE, 1),
    size_kernel_blocks=(CORE, (5, 9, 13, 17)),
    num_repeat_blocks=(CORE, 2),
    ratio_reduction=(CORE, 8),
    size_in_dec=(CORE, 1024),
    size_attn_pool=(CORE, 128),
    size_out_dec=(CORE, 128),
    dropout=(CORE, 0.1),
    # -----------------------
    # dataset/training
    # -----------------------
    num_mel_filters=(META, 80),
    samplerate=(META, 16000),
    max_frames=(META, 400),
    ds_train=(META, None),
    ds_val=(META, None),
    reset=(META, False),
    seed=(META, None),
    alpha=(META, 0.5),
    tau=(META, 1.0),
    margin_min=(META, 0.2),
    margin_max=(META, 0.3),
    weight_decay=(META, 1e-4),
    momentum=(META, 0.9),
    betas=(META, (0.9, 0.999)),
    ratio_warmup=(META, 0.01),
    optim=(META, None),
    lr=(META, 3e-4),
    lr_min=(META, 1e-6),
    epochs=(META, 1),
    steps=(META, 10000),
    int_sched_lr=(META, 20),  # interval of updating lr, None for no decay
    size_batch=(META, 8),
    size_val=(META, 20),  # also interval of logging
    int_val=(META, 100),  # interval of validation
    num_workers=(META, 2),
    num_aug=(META, 3),
    prob_aug=(META, 0.3),
    neg_mining=(META, 0.1),  # number of hard-negative mining
)

modelpath = f"{dirname(__file__)}/../o"
