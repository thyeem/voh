from ouch import *

CORE = 1  # model-fundamental conf
META = 2  # model-training conf

conf = dmap(
    # -----------------------
    # model configuration
    # -----------------------
    size_in_enc=(CORE, 80),
    size_hidden_enc=(CORE, 256),
    size_out_enc=(CORE, 512),
    size_kernels=(CORE, (3, 5, 9, 13, 1)),
    size_dilations=(CORE, (1, 1, 1, 1, 1)),
    num_repeats=(CORE, 2),
    ratio_reduction=(CORE, 8),
    size_in_dec=(CORE, 512),
    size_attn_pool=(CORE, 64),
    num_heads=(CORE, 4),
    size_out_dec=(CORE, 128),
    dropout=(CORE, 0.1),
    # -----------------------
    # dataset/training
    # -----------------------
    num_mel_filters=(META, 80),  # number of Mel-filterbanks
    samplerate=(META, 16000),  # waveform sample rate
    max_frames=(META, 400),  # maximum time frames (trim)
    ds_train=(META, None),  # filepath of training dataset
    ds_val=(META, None),  # path of validation dataset
    reset=(META, False),  # reset meta data when retraining
    seed=(META, None),  # random seed
    margin=(META, 0.3),  # size of additive angular margin
    scale=(META, 10),  # loss scaling factor
    lam=(META, 0.1),  # coefficient for distance
    nu=(META, 1.2),  # coefficient for negative-sim penalty
    hard_ratio=(META, 0.1),  # ratio of hard negative samples
    weight_decay=(META, 1e-4),  # optimizer's weight decay
    momentum=(META, 0.9),  # optimizer's momentum
    betas=(META, (0.9, 0.999)),  # optimizer's betas
    ratio_warmup=(META, 0.01),  # ratio of warmup to total step
    optim=(META, None),  # kind of optimizer: {sgd, adam, adamw}
    lr=(META, 3e-4),  # learning rate
    lr_min=(META, 1e-6),  # mininum of learning rate
    epochs=(META, 1),  # number of epochs
    acc_steps=(META, 4),  # number of accumulated steps per update
    steps=(META, 10000),  # number of steps per epoch
    int_sched_lr=(META, 20),  # interval of updating lr, None for no decay
    size_batch=(META, 8),  # number of samples per iteration
    size_val=(META, 20),  # size of validation / interval of logging
    int_val=(META, 100),  # interval of validation
    num_workers=(META, 2),  # number of dataloader workers
    size_perf=(META, 4000),  # size of simple-perf test
    int_perf=(META, 4000),  # interval of simple-perf test
)

modelpath = f"{dirname(__file__)}/../o"
