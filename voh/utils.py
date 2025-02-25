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


def norm_ppf(q, mean=0, std=1):
    def inv_erf(x):
        a = 8 * (np.pi - 3) / (3 * np.pi * (4 - np.pi))
        y = np.log(1 - x**2)
        z = 2 / (np.pi * a) + y / 2
        return np.sign(x) * np.sqrt(np.sqrt(z**2 - y / a) - z)

    return mean + std * np.sqrt(2) * inv_erf(2 * q - 1)


@fx
def ema(prev, new, alpha=0.1):
    if prev in (float("inf"), float("-inf")):
        return new
    return alpha * new + (1 - alpha) * prev


@fx
def filterbank(
    ysr,
    sr=16000,
    n_fft=512,
    hop_length=160,
    n_mels=80,
    fmin=0,
    fmax=None,
    max_frames=None,
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
    y, orig_sr = ysr
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
        lambda x: x[:, :max_frames] if max_frames else id,
        torch.log,
        torch.Tensor.float,
        torch.from_numpy,
        _ + 1e-10,
    )(y)


def wtd_mu_sigma(x, alpha, dim=-1, eps=1e-10):
    """Compute mean and standard deviation of input weighted by alpha."""
    mean = torch.sum(x * alpha, dim=dim)
    std = torch.sqrt(
        torch.clamp(
            torch.sum(x**2 * alpha, dim=dim) - mean**2,
            min=eps,
        )
    )
    return mean, std


@torch.no_grad()
def create_mask(x, max_frames=None, ipad=0):
    """Creates a mask based on a given input: dim of (B, 1, T)"""
    num_frames = torch.any(x != ipad, dim=1).sum(dim=-1)  # (B,)
    T = torch.max(num_frames).item() if max_frames is None else max_frames
    num_frames = torch.clamp(num_frames, max=T)
    B = num_frames.size(0)
    # For each batch element, indices < num_frames are True
    return (  # (B, T) -> (B, 1, T)
        torch.arange(T, device=x.device).expand(B, T) < num_frames.unsqueeze(1)
    ).unsqueeze(1)


def pad_(ts, max_frames=None, ipad=0):
    """Pad along the last ``dim`` so that they all have the same dimension.
    Assume that ``ts`` has a dimension of ``(B, C, T)``,
    where B for batches, C for channels, T for time frames.
    """
    L = max(t.size(-1) for t in ts) if max_frames is None else max_frames
    padded = []
    for t in ts:
        T = t.size(-1)
        if T <= L:
            o = torch.full((t.size(0), L), ipad, dtype=t.dtype, device=t.device)
            o[..., :T] = t
        else:
            o = t[..., :L].clone()
        padded.append(o)
    return torch.stack(padded, dim=0)


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


# ----------------------
# Helper/Util
# ----------------------
@fx
def dumper(**kwargs):
    print(neatly(dmap(**kwargs), margin=20, sort=False))


def which_model(name, dir=default.modelpath):
    path = path_model(name, dir=dir)
    guard(exists(path), f"Error, model '{name}' not found")
    return path


def path_model(name, dir=default.modelpath):
    nick = name.encode().decode("ascii", errors="ignore")
    return f"{dir}/{nick}-{base58e(name.encode())}"


def size_model(name):
    path = which_model(name)
    return du_hs(path)


def list_models(dir=default.modelpath):
    def model_name(f):
        o = f.split("-")
        if len(o) > 1:
            return base58d(o[-1]).decode()
        else:
            error(f"Error, found invalid model path: {f}")

    data = [
        (
            model_name(f),
            du_hs(f),
            timestamp() - timestamp(os.path.getmtime(f), to_utc=True),
        )
        for f in ls(dir, f=True)
    ]
    print(
        tabulate(
            [mapl(str.upper, ("name", "size", "modified"))]
            + [(n, s, timeago(t)) for n, s, t in sort(data, key=nth(3))],
            nohead=True,
        ),
    )


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
def segment_data(path, dur=4, min_dur=1.5, keep_orig=False, workers=None):
    """Segment audio files in the given path into shorter clips."""
    f = partial(
        segment_wav,
        dur=dur,
        min_dur=min_dur,
        keep_orig=keep_orig,
    )
    for speaker in tracker(ls(path), "segmenting wavs"):
        wavs = ls(speaker, grep=r"\.wav$")
        parmap(f, wavs, workers=workers)


def segment_wav(wav, dur=4, min_dur=1.5, keep_orig=False):
    """Segment a given wav file into shorter clips."""
    dur_wav = get_duration(wav)
    if not dur_wav or dur_wav < min_dur:
        shell(f"rm -f {wav}")
        return
    if dur_wav <= dur:
        return
    label = basename(stripext(wav))
    split_wav(wav, label=label, dur=dur, min_dur=min_dur, v=False)
    if not keep_orig:
        shell(f"rm -f {wav}")


@fx
def reduce_data(path, cutoff=300, workers=None, sr=None):
    """Reduce the number of wav files for each speaker."""
    for speaker in tracker(ls(path), "reduction"):
        wavs = ls(speaker, grep=r".wav$")
        if len(wavs) > cutoff:
            wavs = shuffle(wavs)[cutoff:]
            parmap(os.remove, wavs, workers=workers)
        if sr:
            wavs = ls(speaker, grep=r".wav$")
            for wav in tracker(wavs, "resample"):
                ysr = readwav(wav)
                if sr != snd(ysr):
                    savewav(wav, resample(ysr, sr=sr))


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
    return cf_(
        fst if size == 1 else id,
        (
            cf_(
                mapl(f_(normpath, abs=True)),
                choice(size=size),
                db.get,
                choice,
                guard_(bool, f"No matching key found: {key}"),
                grep(rf"{key}"),
            )
            if key
            else cf_(
                mapl(cf_(f_(normpath, abs=True), choice, db.get)),
                choice(size=size),
            )
        ),
        list,
    )(db)


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
    """Ensure the given data is in pair form."""
    if isinstance(x, str):
        return mapl(
            cf_(
                filter(capture(r".wav$")),
                str.split,
            )
        )(reader(x).read().splitlines())
    else:
        return [x] if isinstance(x, tuple) else list(x)


def triplet(db, size=1):
    """Get a triplet of (anchor, positive, negative) pairs."""

    def nodup(db, x):
        tol = 0
        while True:
            o = randwav(db, key=speaker_id(x))
            if o != x:
                return o
            if tol > 5:
                break
            tol += 1
        return x

    r = randwav(db, size=size * 2)
    anchor, negative = r[:size], r[size:]
    positive = [nodup(db, a) for a in anchor]
    return anchor, positive, negative


@torch.no_grad
def perf(model, data, bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], out=None):
    def t(pairs, desc, mono):
        cosims = [
            F.cosine_similarity(*map(model.embed, pair)).item()
            for pair in tracker(pairs, desc)
        ]
        pdf = fst(
            np.histogram(cosims, bins=cons(float("-inf"), bins)),
        ) / len(pairs)
        return dmap(
            pdf=pdf,
            cdf=(scanl1 if mono else scanr1)(op.add, pdf),
            median=np.median(cosims),
            mad=np.median(np.abs(np.array(cosims) - np.median(cosims))),
        )

    out = out or f"/tmp/{randbytes(4).hex()}"
    # data = ([nagative-pair], [positive-pair])
    n, p = t(fst(data), "n-distrib", False), t(snd(data), "p-distrib", True)
    tbl = tabulate(
        [
            [f"{x:.4f}" for x in n.pdf] + [f"{n.median:.4f}"],
            [f"{x:.4f}" for x in n.cdf] + [f"{n.mad:.4f}"],
            mapl(cf_("<" + _, str), bins) + ["med/mad"],
            [f"{x:.4f}" for x in p.pdf] + [f"{p.median:.4f}"],
            [f"{x:.4f}" for x in p.cdf] + [f"{p.mad:.4f}"],
        ],
        style="grid",
        nohead=True,
    )
    print(tbl)
    writer(out, "a").write(f"{tbl}\n\n")


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
        dumper(gaussian_noise=f"snr={snr:.2f} (signal/noise, dB)")
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
        dumper(time_stretch=f"rate={rate:.2f} ({rate:.2f}x speed)")
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


@fx
def perturb(ysr, n=None, v=False):
    augmetors = [
        gaussian_noise(snr=(5, 10), v=v),
        sfx_noise(snr=(5, 10), v=v),
        time_shift(sec=(-1, 1), v=v),
        time_stretch(rate=(0.8, 1.2), v=v),
        pitch_shift(semitone=(-2.0, 2.0), v=v),
        clip_distortion(threshold=(0.1, 0.5), v=v),
        gain(db=(-3, 3), v=v),
        equalize(low=(0.5, 1.5), mid=(0.5, 1.5), high=(0.5, 1.5), v=v),
        reverb(delay=(0.1), decay=(0.3), v=v),
        room_simulator(decay=(10, 20), v=v),
        bandpass(low=(100, 400), high=(1500, 4000), v=v),
    ]
    return cf_(*choice(augmetors, size=n or len(augmetors)))(ysr)


@fx
def perturb_tiny(ysr, v=False):
    """Augmentor the same as TitaNet-L's augmentor"""
    return cf_(
        *shuffle(
            [
                probify(p=0.5)(gaussian_noise(snr=(0, 15), v=v)),
                probify(p=0.2)(time_stretch(rate=(0.95, 1.05), v=v)),
            ]
        )
    )(ysr)


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
    corr = model.cosim(a, b)
    cache[_pair_id(a, b)] = corr
    for i in tracker(range(size + warmup), "MCMC sampling".rjust(COL)):
        new_a, new_b = choice(jar, size=2)
        k = _pair_id(new_a, new_b)
        if k in cache:
            new_corr = cache[k]
        else:
            new_corr = model.cosim(new_a, new_b)
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
            corrs.append(cache[k] if k in cache else model.cosim(r, t))
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
