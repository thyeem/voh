import hashlib
import json
import math
import os
import re
import wave
from functools import lru_cache, partial

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from foc import *
from ouch import *
from pytubefix import YouTube
from scipy.signal import butter, fftconvolve, sosfilt
from torch.nn import functional as F

from . import default


# ----------------------
# Fundamentals
# ----------------------
def wtd_percentile(x, weights, ps):
    x, weights = zip(*sort(zip(x, weights)))
    cum_weights = np.cumsum(weights)
    cum_weights = cum_weights / cum_weights[-1]
    if isinstance(ps, list):
        return [np.interp(p, cum_weights, x) for p in ps]
    else:
        return np.interp(ps, cum_weights, x)


def wtd_median(x, weights):
    return wtd_percentile(x, weights, 0.5)


def wtd_mad(x, weights):
    median = wtd_median(x, weights)
    return wtd_median(np.abs(x - median), weights)


def hmean(x):  # harmonic mean
    return np.mean([1 / e if e != 0 else np.inf for e in x]) ** (-1)


@fx
def ema(prev, new, alpha=0.1):
    if prev in (float("inf"), float("-inf")):
        return new
    return alpha * new + (1 - alpha) * prev


@fx
def filterbank(
    f,
    sr=16000,
    n_fft=512,
    hop_length=160,
    n_mels=80,
    fmin=0,
    fmax=None,
    from_ysr=False,
):
    """Generate log of Mel-filterbank energies from a given wavfile.
    -----------------------------
    When default values are used
    -----------------------------
    duration of each FFT window:
    512(n_fft, FFT_window_size) / 16000(sr, sample_rate) = 32 ms

    time shift between frames:
    160(hop_length) / 16000 (sr) = 10 ms
    frames overlap = 32 - 10 = 22 ms

    num_frames = 1 + [SAMPLES(t*sr) - WINDOW_SIZE(n_fft)] / hop_length
     1-sec wav = 1 + [1*16000 - 512] / 160 = 98 (frames/sec)
    """
    y, orig_sr = f if from_ysr else readwav(f)
    if sr != orig_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    y = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax or sr // 2,
    )  # (C, T) = (n_mels, num_frames)
    return cf_(
        torch.log,
        torch.Tensor.float,
        torch.from_numpy,
        _ + 1e-10,
    )(y)


def tripletloss(anchor, positive, negative, margin=1.0):
    return F.relu(
        F.pairwise_distance(anchor, positive)
        - F.pairwise_distance(anchor, negative)
        + margin
    ).mean()


@torch.no_grad()
def wtd_mu_sigma(x, alpha, dim=2, eps=1e-10):
    """Compute mean and standard deviation of input weighted by alpha"""
    mean = torch.sum(alpha * x, dim=dim, keepdim=True)  # (B, C, 1)
    var = torch.sum(alpha * (x - mean).pow(2), dim=dim)
    std = torch.sqrt(var.clamp(min=eps)).unsqueeze(dim)  # (B, C, 1)
    return mean, std


@torch.no_grad()
def create_mask(x, max_len=None, pad=0):
    """Creates a mask based on a given input: dim of (B, 1, T)"""
    lengths = torch.any(x != pad, dim=1).sum(dim=1)
    B = lengths.size(0)
    T = torch.max(lengths).item() if max_len is None else max_len
    return (
        torch.arange(T, device=x.device).expand(B, T) < lengths.unsqueeze(1)
    ).unsqueeze(1)


def pad_(o, ipad=0):
    """Pad along the last ``dim`` so that it's the same dimenstion.
    Assueming that ``o`` has dimensions of ``(B, C, T)``.
    """
    L = max(x.size(-1) for x in o)
    return torch.stack(
        [F.pad(x, (0, L - x.size(1)), value=ipad).squeeze() for x in o],
    )


@fx
def sched_lr(it, lr=1e-3, lr_min=1e-5, steps=10000, warmup=1000):
    """Learning rate scheduler (cosine-annealing with warmup)"""
    it %= steps
    if it < warmup:
        return lr * it / warmup
    if it > steps:
        return lr_min
    decay_ratio = (it - warmup) / (steps - warmup)
    guard(0 <= decay_ratio <= 1, f"Error, invalid decay ratio: {decay_ratio}")
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * decay_ratio))


@torch.no_grad()
def cosim(model, x, y):
    model.eval()
    return F.cosine_similarity(
        model.get_embedding(x),
        model.get_embedding(y),
    ).item()


# ----------------------
# Helper/Util
# ----------------------
@fx
def dumper(**kwargs):
    print(neatly(dmap(**kwargs), _cols=20, _sort=False))


def which_model(name, dir=default.modelpath):
    path = path_model(name, dir=dir)
    guard(exists(path), f"Error, model '{name}' not found")
    return path


def path_model(name, dir=default.modelpath):
    return f"{dir}/{base58e(name.encode())}"


def size_model(name):
    path = which_model(name)
    return du_hs(path)


def list_models(dir=default.modelpath):
    data = [
        (
            base58d(basename(f)).decode(),
            du_hs(f),
            timeago(
                timestamp() - timestamp(os.path.getmtime(f), to_utc=True),
            ),
        )
        for f in ls(dir, f=True)
    ]
    print(tabulate(data, header=mapl(str.upper, ("name", "size", "modified"))))


def def_conf(kind=None):
    def cond(x):
        return fst(x) == kind if kind else True

    return dmap({k: snd(v) for k, v in default.conf.items() if cond(v)})


def uniq_conf(x, o, warn=True):
    conf = dmap()
    for k in x:
        if k in o:
            conf[k] = o[k] if x[k] is None else x[k]
        else:
            if warn:
                print(f"Warning, ignored invalid key: '{k}'.")
    return conf


def kind_conf(x, kind):
    return dmap({k: x[k] for k in [o for o in x if o in def_conf(kind=kind)]})


def read_json(f):
    return json.loads(reader(f).read())


@fx
def write_json(f, o, v=False):
    writer(f).write(json.dumps(o, ensure_ascii=False))
    v and print(f"Wrote {f}")


def read_jsonl(f):
    return [json.loads(x) for x in reader(f).read().splitlines()]


@fx
def write_jsonl(f, l, v=False):
    fh = writer(f)
    for e in l:
        fh.write(f"{json.dumps(e)}\n")
    v and print(f"Wrote {f}")


@fx
def index_data(path, out="data.db", split=None):
    """Index the dataset of audio files (*.wav).
    If a training set ratio 'split' is provided,
    split the dataset into a training set and a validation set.
    """
    db = {}
    for d in tracker(ls(path), f"indexing {path}"):
        wavs = ls(d, grep=r"\.wav$")
        if wavs:
            db[basename(d)] = wavs
    if split is not None:
        guard(0 < split < 1, f"Invalid split ratio: {split}")
        keys = shuffle(list(db))
        i = int(len(keys) * split)
        o = out.split(".")[0]
        write_json(f"{o}-train.db", {k: db[k] for k in keys[:i]}, v=True)
        write_json(f"{o}-val.db", {k: db[k] for k in keys[i:]}, v=True)
    write_json(out, db, v=True)


@fx
def segment_data(
    path,
    dur=4,
    min_dur=2.5,
    keep_orig=False,
    eps=0.1,
    workers=None,
):
    """Segment audio files in the given path into shorter clips."""
    f = partial(
        segment_wav,
        dur=dur,
        min_dur=min_dur,
        keep_orig=keep_orig,
        eps=eps,
    )
    for speaker in tracker(ls(path), "segmenting wavs"):
        wavs = ls(speaker, grep=r"\.wav$")
        parmap(f, wavs, workers=workers)


def segment_wav(wav, dur=4, min_dur=2.5, keep_orig=False, eps=0.1):
    """Segment a given wav file into shorter clips."""
    dur_wav = get_duration(wav)
    if not dur_wav or dur_wav < min_dur:
        shell(f"rm -f {wav}")
        return
    if dur_wav <= dur + eps:
        return
    label = basename(stripext(wav))
    split_wav(wav, label=label, dur=dur, min_dur=min_dur, v=False)
    if not keep_orig:
        shell(f"rm -f {wav}")


@fx
def reduce_data(path, cutoff=300, workers=None):
    """Reduce the number of wav files for each speaker."""
    for speaker in tracker(ls(path), "reduction"):
        wavs = ls(speaker, grep=r".wav$")
        if len(wavs) > cutoff:
            wavs = shuffle(wavs)[cutoff:]
            parmap(os.remove, wavs, workers=workers)


def create_mmap(f="data.db", out="mmap", sr=16000):
    """Parallel execution of the ``_create_mmap`` function"""
    db = read_json(f)
    speakers = db.keys()
    parmap(_create_mmap, speakers, repeat(db), repeat(out), repeat(sr))


def _create_mmap(speaker, db, out, sr):
    """Create memory-map files using a given"""
    print(f"creating memmap for {speaker}")
    d = f"{out}/{speaker}"
    if exists(d) and len(db[speaker]) == len(ls(d, grep=r".mmap$")):
        return
    mkdir(d)
    for wav in db[speaker]:
        fm = re.sub(r"\.wav$", ".mmap", basename(wav))
        write_memmap(f"{d}/{fm}", readwav(wav, mmap=False), sr=sr)


def write_memmap(f, ysr, sr=16000):
    y, sr = resample(ysr, sr=sr)
    sr = np.int32(sr)
    fp = np.memmap(f, dtype="float32", mode="w+", shape=(len(y) + 1,))
    fp[0] = sr.view("float32")
    fp[1:] = y
    fp.flush()


def read_memmap(f):
    guard(exists(f, "f"), f"Error, not found memmap file: {f}")
    fp = np.memmap(f, dtype="float32", mode="r")
    sr = fp[0].view("int32").item()
    y = fp[1:]
    return y, sr


def speaker_id(f, maxlen=24):
    """Extract speaker ID from the directory name of a given audio file.
    Naming rule:
        [ID]_[Speaker Code]_[time or meta]*
    """
    d = basename(dirname(f))[:maxlen]
    d = d == "." and f or d
    if d == "tmp":  # when untagged tmp wav
        return "unlabeled"
    else:
        return capture(r"^[^_]+_[^_]+", d) or d


def archive(f, d="arch/"):
    paths = f.split("/")
    if len(paths) < 2 or paths[1] != "tmp":  # archive tmpfile only
        return f
    d = f"{normpath(d, abs=True)}/{speaker_id(f)}"
    mkdir(d)
    t = f"{d}/{basename(f)}"
    if not exists(t):
        shell(f"cp -f {f} {t}")
    return t


def play(x, rev=False):
    y, sr = x if isinstance(x, tuple) else readwav(x)
    y = y[::-1] if rev else y
    sd.play(y, sr, blocking=True)
    return x


def readwav(f):
    return sf.read(f)


@fx
def savewav(f, ysr):
    y, sr = ysr
    sf.write(f, y, int(sr))
    return f


@fx
def randwav(db, key=None, size=1):
    """Get a random speaker wav file from the given indexed data
    if 'key' is given, returns a wav-file whose label contains 'key'
    """

    def jar(x):
        speakers = list(x)
        q = key or choice(speakers)
        err = f_(error, f"No matching key found: {q}")
        return grep(rf"{q}")(speakers) or err()

    def wav(x):
        return cf_(f_(normpath, abs=True), choice, db.get, choice, jar)(x)

    return wav(db) if size == 1 else [wav(db) for _ in range(size)]


@fx
def augwav(f, augmentor=None, out=None, wav=True):
    augmentor = augmentor or demo_aug(v=True)
    return cf_(
        (
            savewav(out or tmpfile(suffix=".wav", dir=f"/tmp/{speaker_id(f)}"))
            if wav
            else id
        ),
        augmentor,
        readwav,
    )(f)


@fx
def randpair(db, mono=False, sync=False, key=None, size=1):
    """Get random pair(s) of voices from the given indexed data.
    If ``mono=True``, returns positive pair(s). Otherwise negative pair(s).
    If ``sync=True``, returns a pair of different times. Affects ``mono`` only.
    If ``key``, returns a pair whose label contains ``key := str | (str, str)``
    """

    def jar(x):
        def err(k):
            error(f"No matching key found: {k}")

        speakers = list(x)
        p, *q = flatl(key or speaker_id(choice(speakers)))
        if mono:
            guard(not q, f"Too many keys provided: {q}")
            return grep(rf"{p}")(speakers) or err(p)
        else:  # duo-speakers
            if q:  # with two keys
                guard(len(q) == 1, f"Too many keys provided: {q[1:]}")
                return (
                    grep(rf"{p}")(x) or err(p),
                    grep(rf"{fst(q)}")(x) or err(fst(q)),
                )
            else:
                return (
                    grep(rf"{p}")(x) or err(p),
                    grep(rf"^(?!.*{p}).*$")(x) or err(f"not '{p}'"),
                )

    def mono_speaker(x):
        if sync:
            return cf_(choice(size=2), x.get, choice, jar)(x)
        else:  # mono-async speaker
            return cf_(map(cf_(choice, x.get)), choice(size=2), jar)(x)

    def duo_speaker(x):
        return cf_(map(cf_(choice, x.get)), bimap(choice, choice), jar)(x)

    def pair(x):
        return cf_(
            tuple,
            mapl(f_(normpath, abs=True)),
            mono_speaker if mono else duo_speaker,
        )(x)

    return pair(db) if size == 1 else [pair(db) for _ in range(size)]


def to_pairs(x):
    """Ensure the given data is in pair form"""
    if isinstance(x, str):
        return mapl(
            cf_(
                filter(capture(r".wav$")),
                str.split,
            )
        )(reader(x).read().splitlines())
    else:
        return [x] if isinstance(x, tuple) else list(x)


def triplet(db, hardset=False):
    """Get a triplet for Triplet Loss approach"""
    if hardset:
        anchor, positive, negative = db[randint(len(db))]
    else:
        anchor, positive = randpair(db, mono=True, sync=False, size=1)
        _, negative = randpair(db, mono=False, key=speaker_id(anchor))
    return anchor, positive, negative


def mining_hardset(md, db, out="hardset.db", threshold=0.6, size=None):
    """Mine the most challenging examples for models to distinguish"""
    size = size or len(flatl(db.values())) // 10
    o = set()
    gathered = set()
    while True:
        a, b = randpair(db, mono=False)
        if _pair_id(a, b) in gathered:
            continue
        gathered.add(_pair_id(a, b))
        cosim = md.cosim(a, b)
        if cosim > threshold:
            o.add((a, randwav(db, key=speaker_id(a)), b))
            o.add((b, randwav(db, key=speaker_id(b)), a))
            print(f"{len(o):06d}  {speaker_id(a)}  {speaker_id(b)}  {cosim:.4f}")
        if len(o) >= size:
            break
    write_json(out, list(o))


def tasting(md, db, mono=False, size=10):
    v = []
    for o in randpair(db, mono=mono, size=size):
        cosim = md.cosim(*o)
        print(f"{cosim:.4f}")
        v.append(cosim)
    print(f"mean: {np.mean(v):.4f}")


def get_pre_trained():
    import nemo.collections.asr as nemo_asr
    from nemo.utils import logging

    logging.setLevel(logging.ERROR)
    return nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large"
    )


# ----------------------
# Audio perturber
# ----------------------
_ffmpeg = "ffmpeg -y -loglevel error"


def rfnum(x):
    """returns fixed scalar value or `rand(low, high)`"""
    return rand(*x) if isinstance(x, tuple) and len(x) == 2 else x


@safe
def get_duration(f):
    size = os.path.getsize(f)
    size_header = 44
    wav = wave.Wave_read(f)
    return (size - size_header) / (
        wav.getframerate() * wav.getnchannels() * wav.getsampwidth()
    )


@fx
def gaussian_noise(ysr, snr=10, v=False, MIN=3, MAX=30):
    """add Gaussian random noise to a given audio"""
    y, sr = ysr  # (waveform, sample rate)
    snr = max(MIN, min(rfnum(snr), MAX))  # clip the SNR value between (MIN, MAX)
    y_power = np.mean(y**2)
    r = 20 ** (snr / 10)
    noise_power = y_power / r
    noise = np.random.normal(0, np.sqrt(noise_power), y.shape)
    ysr = y + noise, sr
    if v:
        dumper(gaussian_noise=f"snr={snr:.1f} (signal/noise, dB)")
    return ysr


@fx
def sfx_noise(ysr, sfx=None, snr=5, v=False):
    y, sr = ysr
    snr = rfnum(snr)
    b, _ = resample(sfx or randsfx(), sr=sr)
    if len(y) > len(b):
        b = np.tile(b, len(y) // len(b) + 1)
    b = b[: len(y)]
    r = 10 ** (snr / 10)
    ysr = librosa.util.normalize(y * r + b), sr
    if v:
        dumper(sfx_noise=f"snr={snr:.1f} (signal/SFX, dB)")
    return ysr


@fx
def time_stretch(ysr, rate=1.25, v=False):
    """apply time stretching to a given audio"""
    y, sr = ysr
    rate = rfnum(rate)
    ysr = librosa.effects.time_stretch(y, rate=rate), sr
    if v:
        dumper(time_stretch=f"rate={rate:.1f} ({rate:.1f}x speed)")
    return ysr


@fx
def pitch_shift(ysr, semitone=7.0, v=False):
    """shift pitch of a given audio"""
    y, sr = ysr
    semitone = rfnum(semitone)
    ysr = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitone), sr
    if v:
        dumper(pitch_shift=f"semitone={semitone:.2f} ({2 ** (semitone/12):.2f}x Hz)")
    return ysr


@fx
def time_shift(ysr, sec=0.5, v=False):
    """apply time shift to a given audio"""
    y, sr = ysr
    sec = rfnum(sec)
    ysr = np.roll(y, int(sec * sr)), sr
    if v:
        dumper(time_shift=f"sec={sec:.2f} (second)")
    return ysr


@fx
def clip_distortion(ysr, threshold=0.5, v=False):
    """apply clipping distortion to a given audio"""
    y, sr = ysr
    threshold = rfnum(threshold)
    ysr = np.clip(y, -threshold, threshold), sr
    if v:
        dumper(clip_distortion=f"threshold={threshold:.2f} (clipped)")
    return ysr


@fx
def resample(ysr, sr=16000, v=False):
    """resample a given audio"""
    y, orig_sr = ysr
    sr = rfnum(sr)
    if sr != orig_sr:
        ysr = librosa.resample(y, orig_sr=orig_sr, target_sr=sr), sr
        if v:
            dumper(resample=f"sr={sr:.1f} (sample rate)")
    return ysr


@fx
def normalize(ysr, v=False):
    """normalize a given audio"""
    y, sr = ysr
    ysr = librosa.util.normalize(y), sr
    if v:
        rms = 20 * np.log10(
            np.sqrt(np.mean(fst(ysr) ** 2)) / np.sqrt(np.mean(y**2)),
        )
        peak = 20 * np.log10(np.max(np.abs(fst(ysr))) / np.max(np.abs(y)))
        dumper(normalize=f"(peak {peak:.1f} dB, rms {rms:.1f} dB)")
    return ysr


@fx
def gain(ysr, db=6, v=False):
    """adjust the gain of a given audio"""
    y, sr = ysr
    db = rfnum(db)
    ysr = y * (10 ** (db / 10.0)), sr
    if v:
        dumper(gain=f"db={db:.1f}")
    return ysr


@fx
def equalize(ysr, low=1.2, mid=1.5, high=1.3, v=False):
    y, sr = ysr
    low, mid, high = rfnum(low), rfnum(mid), rfnum(high)
    stft = librosa.stft(y)
    stft_db = librosa.amplitude_to_db(np.abs(stft))
    freq = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2)
    freq = freq[: stft.shape[0]]
    low_freq = freq < 200
    mid_freq = (freq >= 200) & (freq <= 2000)
    high_freq = freq > 2000
    stft_db[low_freq] *= low
    stft_db[mid_freq] *= mid
    stft_db[high_freq] *= high
    ysr = (
        librosa.istft(
            librosa.db_to_amplitude(stft_db) * np.exp(1j * np.angle(stft)),
        ),
        sr,
    )
    if v:
        dumper(equalize=f"low={low:.1f}, mid={mid:.1f}, high={high:.1f}")
    return ysr


@fx
def reverb(ysr, delay=0.3, decay=0.3, v=False):
    """add an echo effect to a given audio"""
    y, sr = ysr
    delay, decay = rfnum(delay), rfnum(decay)
    delay_samples = int(sr * delay)
    echo = np.zeros(len(y) + delay_samples)
    echo[: len(y)] = y
    echo[delay_samples:] += decay * y
    ysr = echo[: len(y)], sr
    if v:
        dumper(reverb=f"delay={delay:.2f}, decay={decay:.2f}")
    return ysr


def room_impulse_response(duration=1.0, sr=44100, decay=7.0, v=False):
    """generate a synthetic room impulse response with exponential decay"""
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    y = np.random.randn(len(t)) * np.exp(-decay * t)
    return y / np.max(np.abs(y))


@fx
def room_simulator(ysr, decay=7.0, v=False):
    y, sr = ysr
    decay = rfnum(decay)
    ysr = (
        librosa.util.normalize(
            fftconvolve(
                y,
                room_impulse_response(sr=sr, decay=decay),
                mode="full",
            )[: len(y)]
        ),
        sr,
    )
    if v:
        dumper(room_simulator=f"decay={decay:.1f}")
    return ysr


@fx
def bandpass(ysr, low=200, high=3000, v=False):
    """apply air absorption effect on a given audio"""
    y, sr = ysr
    low, high = rfnum(low), rfnum(high)
    sos = butter(10, [low, high], btype="band", fs=sr, output="sos")
    ysr = sosfilt(sos, y), sr
    if v:
        dumper(bandpass=f"low={low:.1f}, high={high:.1f} (Hz)")
    return ysr


@fx
def trim(ysr, threshold=20, v=False):
    """trim leading and trailing silence from a given audio"""
    y, sr = ysr
    threshold = rfnum(threshold)
    ysr = fst(librosa.effects.trim(y, top_db=threshold)), sr
    if v:
        dumper(trim=f"threshold={threshold} (dB)")
    return ysr


@fx
def time_inverse(ysr, v=False):
    """reverse a given audio in time domain"""
    y, sr = ysr
    ysr = y[::-1], sr
    if v:
        dumper(time_inverse=f"({len(y) / sr:.2f} second)")
    return ysr


def demo_aug(p=1, v=False):
    g = lazy(
        shuffle,
        [
            gaussian_noise(snr=(10, 20), v=v),
            sfx_noise(snr=(5, 10), v=v),
            time_shift(sec=(-1.5, 1.5), v=v),
            time_stretch(rate=(0.8, 1.2), v=v),
            pitch_shift(semitone=(-2, 2), v=v),
            clip_distortion(threshold=(0.1, 0.5), v=v),
            gain(db=(-6, 6), v=v),
            normalize(v=v),
            equalize(low=(0.9, 1.1), mid=(0.9, 1.1), high=(0.9, 1.1), v=v),
            reverb(delay=(0.15), decay=(0.3), v=v),
            resample(sr=(8000, 44100), v=v),
            room_simulator(decay=(20, 40), v=v),
            bandpass(low=(100, 400), high=(1500, 4000), v=v),
        ],
    )
    return cf_(*map(probify(p=p), force(g)))


def perturb(p=0.3, num_aug=3, v=False):
    augmetors = [
        gain(db=(-3, 3), v=v),
        gaussian_noise(snr=(5, 10), v=v),
        sfx_noise(snr=(5, 10), v=v),
        equalize(low=(0.5, 1.5), mid=(0.5, 1.5), high=(0.5, 1.5), v=v),
        time_stretch(rate=(0.8, 1.2), v=v),
        pitch_shift(semitone=(-2.0, 2.0), v=v),
        bandpass(low=(100, 400), high=(1500, 4000), v=v),
        reverb(delay=(0.1), decay=(0.3), v=v),
        room_simulator(decay=(10, 20), v=v),
    ]
    return probify(p=p)(cf_(*choice(augmetors, size=num_aug)))


def randsfx():
    @lru_cache
    def sfx():
        return ls("sfx")

    return readwav(choice(sfx()))


def add_to_sfx(src, out="sfx", ss=None, to=None):
    """Given wav sources are added to the SFX directory"""
    ss = f"-ss {ss}" if ss else ""
    to = f"-to {to}" if to else ""
    files = ls(src) if isinstance(src, str) else src
    mkdir(out)
    for f in files:
        wav = (
            f"{out}/"
            f"{hashlib.sha256(reader(f, mode='rb').read()).hexdigest()[:16]}"
            ".wav"
        )
        if not exists(wav):
            dumper(converting=f"{f}  ->  {wav}")
            shell(f"{_ffmpeg} -i {f} {ss} {to} -ac 1 {wav}")
    return out


# ----------------------
# Youtube pipeline
# ----------------------
FACTOR_MCMC = 10
FACTOR_MAD = 3
MIN_SAMPLE = 40
BURN_IN = 0.4
NUM_REF = 5
COL = 24


def _log(left, right):
    print(left.rjust(COL), right)


def realwav(
    YOUTUBE,
    dir="real",
    period=-1,  # silence-remove stop period
    duration=0.6,  # silence-remove stop duration
    threshold="-30dB",  # silence-remove stop threshold
    dur=4,  # wav-file clip duration in second
    model=None,  # if model is set, perform quarantine step
    limit=0.7,  # quarantine baseline limit
):
    for label, url, ss, to in map(
        str.split,
        filter(
            cf_(not_, ob(_.startswith)("#")),
            reader(YOUTUBE).read().splitlines(),
        ),
    ):
        wavtube(
            url,
            label,
            ss=ss,
            to=to,
            period=period,
            duration=duration,
            threshold=threshold,
            dir=dir,
            dur=dur,
            model=model,
            limit=limit,
        )
    index_data(dir, out="real.db")


def wavtube(
    url,
    label,
    ss=None,
    to=None,
    period=-1,
    duration=0.6,
    threshold="-30dB",
    dir=".",
    dur=4,
    model=None,
    limit=0.7,
):
    path = normpath(f"{dir}/{label}", abs=True)
    if exists(path):
        _log("skipping", f"{path}")
        return
    mkdir(path)
    return cf_(
        quarantine(path, model, limit=limit, dir=f"{dir}/quar") if model else id,
        split_wav(label=label, dir=f"{dir}/{label}", dur=dur),
        trim_wav(period=period, duration=duration, threshold=threshold),
        convert_to_wav(ss=ss, to=to),
        download_video,
    )(url)


def download_video(url, out=None, v=True):
    mp4 = out or tmpfile(suffix=".mp4")
    v and _log("downloading", f"{mp4} from {url}")
    YouTube(url).streams.get_lowest_resolution().download(filename=mp4)
    return mp4


@fx
def convert_to_wav(src, ss=None, to=None, out=None, v=True):
    ss = f"-ss {ss}" if ss else ""
    to = f"-to {to}" if to else ""
    wav = out or tmpfile(suffix=".wav")
    v and _log("converting", f"{src}  ->  {wav}")
    if shell(f"{_ffmpeg} -i {src} {ss} {to} -ac 1 {wav}"):
        error(f"failed to convert file to wav: {src}")
    return wav


@fx
def trim_wav(wav, period=-1, duration=0.6, threshold="-30dB", out=None, v=True):
    trimmed = out or tmpfile(suffix=".wav")
    v and _log("trimming", f"{wav}  ->  {trimmed}")
    if shell(
        f"{_ffmpeg} -i {wav}"
        f" -af silenceremove=stop_periods={period}"
        f":stop_duration={duration}"
        f":stop_threshold={threshold}"
        f" {trimmed}"
    ):
        error(f"failed to trim wav: {wav}")
    return trimmed


@fx
def split_wav(wav, label=None, dir=None, dur=4, min_dur=2.5, v=True):
    """Split wav file into multiple segments of specified duration."""
    label = label or randbytes(4).hex()
    path = dir or dirname(wav)
    mkdir(path)
    output = f"{normpath(path)}/{label}_%04d.wav"
    v and _log("splitting", f"{wav}  ->  {output}")
    if shell(f"{_ffmpeg} -i {wav} -f segment -segment_time {dur} {output}"):
        error(f"failed to split wav: {wav}")

    o = ls(path, grep=rf"{label}_.+\.wav$")
    if o:
        o_dur = get_duration(o[-1])
        if not o_dur:
            shell(f"rm -f {unwords(o)}")
        if o_dur and o_dur < min_dur:
            shell(f"rm -f {o[-1]}")
        return path
    else:
        error(f"something wrong in splitting: {wav}")


@fx
def quarantine(path, model, limit=0.7, dir=".", v=True):
    """Find outliers in terms of cosim between two audios and remove them"""

    model = model.to("mps" if torch.backends.mps.is_available() else "cpu")
    # sampling
    jar = [normpath(f, abs=True) for f in ls(path, grep=r"\.wav")]
    v and _log("quarantining", f"{path}\t{len(jar)} file(s)")
    size = max(len(jar), MIN_SAMPLE) * FACTOR_MCMC
    counts, cosims, cache = mcmc_sample(model, jar, limit, size)

    # automated thresholding
    tol = tolerance(list(cosims.values()), list(counts.values()))

    # challenge/kill
    refs = _toptier(cosims, NUM_REF)  # top-tier references
    doom = challenge(model, refs, jar, tol, cache)
    doom_files(dir, doom)
    v and _log("removed", f"{len(doom)} file(s)")
    return path


def _pair_id(a, b):
    return tuple(sort((a, b)))


def _toptier(d, tier):
    return mapl(fst, sort(d.items(), key=lambda x: x[1], reverse=True)[:tier])


def mcmc_sample(model, jar, limit, size):
    cache = {}
    counts = {k: 0 for k in jar}
    cosims = {k: [0] for k in jar}
    warmup = int(size * BURN_IN)
    a, b = choice(jar, size=2)
    corr = cosim(model, a, b)
    cache[_pair_id(a, b)] = corr
    for i in tracker(range(size + warmup), "MCMC sampling".rjust(COL)):
        new_a, new_b = choice(jar, size=2)
        k = _pair_id(new_a, new_b)
        if k in cache:
            new_corr = cache[k]
        else:
            new_corr = cosim(model, new_a, new_b)
            cache[k] = new_corr
        if new_corr > limit or rand(0, 1) < np.exp(
            (new_corr - corr) / (1 - corr),
        ):
            a, b, corr = new_a, new_b, new_corr
        if i > warmup and corr > limit:
            counts[a] += 1
            counts[b] += 1
            cosims[a].append(corr)
            cosims[b].append(corr)
    cosims = {k: np.mean(v) for k, v in cosims.items()}
    return counts, cosims, cache


def challenge(model, refs, targets, tol, cache):
    doom = []
    for t in tracker(targets, "challenge".rjust(COL)):
        corrs = []
        for r in refs:
            k = _pair_id(r, t)
            corrs.append(cache[k] if k in cache else cosim(model, r, t))
            score = hmean(corrs)
        if score < tol:
            doom.append(t)
            _log("-", f"\t{score:.4f}  {basename(t)}")
    return doom


def tolerance(d, o):
    q1, median, q3 = wtd_percentile(d, o, [0.25, 0.5, 0.75])
    _log("median", f"\t{median:.4f}")
    _log("Q1", f"\t{q1:.4f}")
    _log("Q3", f"\t{q3:.4f}")
    mad = wtd_mad(d, o)
    _log("MAD", f"\t{mad:.4f}")
    tol = median - FACTOR_MAD * mad
    _log("tolerance", f"\t{tol:.4f} ({FACTOR_MAD}-MAD)")
    return tol


def doom_files(dir, doom):
    if doom:
        for f in doom:
            d = normpath(f"{dir}/{basename(dirname(f))}", abs=True)
            mkdir(d)
            shell(f"mv -f {f} {d}/{basename(f)}")


@fx
def ytclip(url, ss, to, out=None):
    return cf_(
        trim_wav(period=-1, duration=0.6, threshold="-30dB", out=out),
        convert_to_wav(ss=f"00:{ss}", to=f"00:{to}"),
        download_video,
    )(url)
