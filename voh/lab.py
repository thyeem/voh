import math
import re
from bisect import bisect
from collections import Counter

import torch
from foc import *
from ouch import *
from sklearn.metrics import auc

from .utils import *
from .voh import *


def TPR(x):
    """True Positive Rate(TPR): 'recall' or 'sensitivity', 1-FNR"""
    tp, fn, *_ = x
    return safe_div(tp, tp + fn)


def FNR(x):
    """False Negative Rate(FNR): miss-rate or 'Type-II error, 1-TPR"""
    tp, fn, *_ = x
    return safe_div(fn, tp + fn)


def FPR(x):
    """False Positive Rate(FPR): false-alarm or 'Type-I error, 1-TNR"""
    *_, fp, tn = x
    return safe_div(fp, fp + tn)


def TNR(x):
    """True Negative Rate(TNR): 'specificity', 'selectivity', 1-FPR"""
    *_, fp, tn = x
    return safe_div(tn, fp + tn)


def PPV(x):
    """Positive Predictive Value(PPV): 'precision', 1-FDR"""
    tp, _, fp, _ = x
    return safe_div(tp, tp + fp)


def FDR(x):
    """False Discovery Rate(FDR), 1-PPV"""
    tp, _, fp, _ = x
    return safe_div(fp, tp + fp)


def FOR(x):
    """False Omission Rate(FOR), 1-NPV"""
    _, fn, _, tn = x
    return safe_div(fn, fn + tn)


def NPV(x):
    """Negative Predictive Value(NPV), 1-FOR"""
    _, fn, _, tn = x
    return safe_div(tn, fn + tn)


def ACC(x):
    """accuracy: proportion of correct predictions among
    the total number of cases
    """
    tp, fn, fp, tn = x
    return safe_div(tp + tn, tp + tn + fp + fn)


def BA(x):
    """balanced accuracy"""
    return (TPR(x) + TNR(x)) / 2


def F1(x):
    """F1-score = 2 * PPV * TPR / (PPV + TPR)
    Harmonic mean(HM) of 'precision(PPV)' and 'recall(TPR)'.
    """
    tp, fn, fp, tn = x
    return safe_div(2 * tp, 2 * tp + fp + fn)


def FM(x):
    """Fowlkes-Mallows index(FM) = sqrt(PPV * TPR)
    Geometric mean(GM) of 'precision(PPV)' and 'recall(TPR)'.
    """
    return math.sqrt(PPV(x) * TPR(x))


def CSI(x):
    """Critical Success Index(CSI) = TP / (TP + FP + FN)
    Also called 'Threat Score(TS)' or 'Jaccard index'
    """
    tp, fn, fp, _ = x
    return safe_div(tp, tp + fp + fn)


def MCC(x):
    """Matthews Correlation Coefficient(MCC) measures association for
    two binary variables 'actual' (P,N), 'predicted' (PP, PN).

     P = TP + FN
     N = FP + TN
    PP = TP + FP
    PN = FN + TN
    MCC = sqrt(PPV * TPR * TNR * NPV) - sqrt(FDR * FNR * FPR * FOR)
    """
    tp, fn, fp, tn = x
    return safe_div(
        tp * tn - fp * fn,
        math.sqrt((tp + fp) * (tp + fn) * (fp + tn) * (fn + tn)),
    )


def safe_div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0


@fx
def tt(x, model, threshold=0.7, expected=None, sound=False):
    errlog = tmpfile(suffix=".err") if expected is not None else None
    if errlog:
        out = writer(errlog)
        print(errlog)
    error = 0
    for i, (a, b) in enumerate(to_pairs(x), start=1):
        if not exists(a):
            print(f"skip: {a}")
            continue
        if not exists(b):
            print(f"skip: {b}")
            continue
        if sound:
            (a, b) | mapl(play)
        o = model.verify(a, b, threshold=threshold, v=True)
        if o != expected:
            error += 1
            if out:
                out.write(f"{model.cosim(a,b):.4f}  {archive(a)}  {archive(b)}\n")
                out.flush()
    if errlog:
        print(errlog)
    print(f"\nError Rate(%)={(error/i)*100:.2f}, total={i}")


@fx
def ttd(x, target, ref, threshold=0.7, expected=None, sound=False):
    errlog = tmpfile(suffix=".err") if expected is not None else None
    if errlog:
        out = writer(errlog)
        print(errlog)
    for a, b in to_pairs(x):
        if not exists(a):
            print(f"skip: {a}")
            continue
        if not exists(b):
            print(f"skip: {b}")
            continue
        if sound:
            (a, b) | mapl(play)
        r = ref.verify(a, b, threshold=threshold, v=True)
        t = target.verify(a, b, threshold=threshold, v=True)
        if errlog and (t != expected or r != expected):
            out.write(f"{target.cosim(a,b):.4f}  {archive(a)}  {archive(b)}\n")
            out.write(f"{ref.cosim(a,b):.4f}  {archive(a)}  {archive(b)}\n")
            out.write("\n")
            out.flush()
    if errlog:
        print(errlog)


def plot_wav(f):
    y, sr = readwav(f)
    time = np.linspace(0, len(y) / sr, num=len(y))
    plt.ion()
    plt.figure(figsize=(10, 4))
    plt.plot(time, y, linewidth=0.2)
    plt.title("Waveform of the audio")
    plt.xlabel("time (s)")
    plt.ylabel("amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_freqspec(f, max_freq=16000):
    y, sr = readwav(f)
    spec = np.fft.fft(y)
    freq = np.fft.fftfreq(len(spec), 1 / sr)
    idx = np.where((freq >= 0) & (freq <= max_freq))
    plt.ion()
    plt.figure(figsize=(10, 4))
    plt.plot(freq[idx], np.abs(spec[idx]), linewidth=0.2)
    plt.title("Frequency spectrum")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("magnitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def spectrogram(f, dB=False):
    y, sr = readwav(f)
    y = np.abs(librosa.stft(y)) ** 2
    if dB:
        y = librosa.amplitude_to_db(y, ref=np.max)
    return y, sr


def plot_spectrogram(f, n_mels=128):
    s, sr = spectrogram(f, dB=True)
    plt.ion()
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(s, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("spectrogram")
    plt.tight_layout()
    plt.show()


def melspectrogram(f, n_mels=128, dB=False):
    y, sr = readwav(f)
    y = librosa.feature.melspectrogram(y=y, n_mels=n_mels)
    if dB:
        y = librosa.power_to_db(y, ref=np.max)
    return y, sr


def plot_melspectrogram(f, n_mels=128):
    m, sr = melspectrogram(f, n_mels=n_mels, dB=True)
    plt.ion()
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(m, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-frequency spectrogram")
    plt.tight_layout()
    plt.show()


def eval_error_rate(
    model,
    db,
    mono=False,
    sync=False,
    threshold=0.7,
    out=None,
    p=None,
    augmentor=None,
):
    """Evaluate model's Type I/II errors based on monte-carlo method.
    mono == False -> test false-positive or FP (Type I error)
    mono == True  -> test false-negative or FN (Type II error)
    """

    aug = probify(p=p)(augwav(augmentor=augmentor)) if augmentor and p else id

    def loader():
        while True:
            a, b = randpair(db, mono=mono, sync=sync)
            yield a, aug(b)

    if out:
        out = writer(f"{out}.{'frr' if mono else 'far'}")
    i, error = 1, 0
    for a, b in loader():
        cosim = model.cosim(a, b)
        res = True if cosim > threshold else False
        onError = res if not mono else not res
        if onError:
            error += 1
            if out:
                out.write(f"{archive(a)}  {archive(b)}\n")
                out.flush()
        if i % 10 == 0:
            print(f"Error Rate(%)={(error/i)*100:.2f}, total={i}")
        i += 1


def eval_bin_metrics(
    model,
    db,
    prefix="bcm/test",
    size=500,
    step=0.001,
    pairs=None,
    augmentor=None,
    p=None,
):
    """Generates binary classification metrics including confusion matrix and
    its derived metrics (FPR, TPR, PPV, ACC, F1, MCC..)
    """

    def header(o):
        o.write(f"{'-' * 124}\n")
        o.write(
            f"{'THRES':^10}"
            f"{'P':^8}"
            f"{'N':^8}"
            f"{'TP':^8}"
            f"{'FN':^8}"
            f"{'FP':^8}"
            f"{'TN':^8}"
            f"{'TPR':^8}"
            f"{'FPR':^8}"
            f"{'PPV':^8}"
            f"{'NPV':^8}"
            f"{'ACC':^8}"
            f"{'F1':^8}"
            f"{'CSI':^8}"
            f"{'MCC':^8}"
            "\n"
        )
        o.write(f"{'-' * 124}\n")
        o.flush()

    def metric(o, x, threshold):
        tp, fn, fp, tn = x  # unpack conf-mat elems
        o.write(
            f"{threshold:^10.4f}"
            f"{tp+fn:^8d}"
            f"{fp+tn:^8d}"
            f"{tp:^8d}"
            f"{fn:^8d}"
            f"{fp:^8d}"
            f"{tn:^8d}"
            f"{TPR(x):^8.4f}"
            f"{FPR(x):^8.4f}"
            f"{PPV(x):^8.4f}"
            f"{NPV(x):^8.4f}"
            f"{ACC(x):^8.4f}"
            f"{F1(x):^8.4f}"
            f"{CSI(x):^8.4f}"
            f"{MCC(x):^8.4f}"
            "\n"
        )

    def cache(d, pairs):
        total = len(pairs)
        fmt = f">{len(str(total))}d"
        for i, (a, b) in enumerate(pairs, 1):
            d[(a, b)] = model.cosim(a, b)
            print(
                f"[{i:{fmt}} of {total:{fmt}}]"
                f"{d.get((a,b)):^16.4f}"
                f"{speaker_id(a):^18}"
                f"{speaker_id(b):^18}"
            )

    def fails(o, d, pairs, threshold):
        o.write(f"\nthreshold {threshold:.4f}:\n")
        for a, b in uniq(pairs):
            o.write(f"{d.get((a,b)):.4f}  {archive(a)}  {archive(b)}\n")

    aug = probify(p=p)(augwav(augmentor=augmentor)) if augmentor and p else id
    mkdir(dirname(prefix))
    out = f"{prefix}-{size:06d}"
    if pairs is None:
        P = pairs_pos(size, db=db, augmentor=augmentor, p=p)
        N = pairs_neg(size, db=db, augmentor=augmentor, p=p)
    else:
        P, N = pairs
    far = writer(f"{out}.far")  # collects FP
    frr = writer(f"{out}.frr")  # collects FN
    bcm = writer(f"{out}")
    d = {}
    header(bcm)
    cache(d, P)
    cache(d, N)

    for threshold in np.arange(0.0, 1.0, step):
        print(f"\rprocessing {threshold:.4f}", end="", flush=True)
        tp = tn = 0
        fp = []
        fn = []
        for a, b in N:
            if d.get((a, b)) >= threshold:  # detect false-positive
                fp.append((a, b))
            else:
                tn += 1
        for a, b in P:
            if d.get((a, b)) >= threshold:  # detect false-negative
                tp += 1
            else:
                fn.append((a, b))
        x = (tp, len(fn), len(fp), tn)  # confusion matrix
        metric(bcm, x, threshold)
        if 0.60 < threshold < 0.75:
            fails(far, d, fp, threshold)
            fails(frr, d, fn, threshold)


def pairs_pos(size, db, sync=False, key=None, augmentor=None, p=None):
    aug = probify(p=p)(augwav(augmentor=augmentor)) if augmentor and p else id
    return [
        bimap(
            id,
            aug,
            randpair(db, mono=1, sync=sync, key=key),
        )
        for _ in range(size)
    ]


def pairs_neg(size, db, key=None, augmentor=None, p=None):
    aug = probify(p=p)(augwav(augmentor=augmentor)) if augmentor and p else id
    return [
        bimap(
            id,
            aug,
            randpair(db, mono=0, key=key),
        )
        for _ in range(size)
    ]


def readbcm(f):
    lines = reader(f).read().splitlines()
    hdrs = re.split(r"\s+", lines[1].strip())
    cols = {hdr: [] for hdr in hdrs}
    for line in lines[3:]:
        vals = re.split(r"\s+", line.strip())
        for k, v in zip(hdrs, vals):
            cols[k].append(
                float(v) if re.match(r"^-?\d+\.?\d*$", v) else int(v),
            )
    return cols


def plot_curves(fs, xr=[0.0, 1.0], yr=[0.0, 1.02]):
    plt.ion()
    plt.figure()
    colors = [
        ("blue", "green"),
        ("red", "orange"),
        ("purple", "cyan"),
    ]
    for (ci, cj), f in zip(colors, flat(fs)):
        cols = readbcm(f)
        fpr, tpr, ppv = cols["FPR"], cols["TPR"], cols["PPV"]
        roc_auc = auc(fpr, tpr)  # Receiver Operating Characteristic curve
        prc_auc = auc(tpr, ppv)  # Precision-Recall curve
        plt.plot(
            fpr,
            tpr,
            color=ci,
            lw=1.5,
            label=f"{basename(f)}: ROC (area={roc_auc:.4f})",
        )
        plt.plot(
            tpr,
            ppv,
            color=cj,
            lw=1.5,
            label=f"{basename(f)}: PRC (AUC={prc_auc:.4f})",
        )
        plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
        plt.xlim(xr)
        plt.ylim(yr)
        plt.xlabel("False Positive Rate / Recall")
        plt.ylabel("True Positive Rate / Precision")
        plt.title("ROC and PRC Curves")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


@torch.no_grad()
def tasting(model, pairs):
    md = voh.load(model)
    md.eval()
    cosims = []
    for a, b in pairs:
        cosim = md.cosim(a, b)
        print(
            f"{cosim:.4f} {speaker_id(a):>16} {speaker_id(b):>16}",
            end="\r",
            flush=True,
        )
        cosims.append(cosim)
    bins = [0.6, 0.7, 0.8, 0.9, 1.01]
    hist = Counter(bisect(bins, cosim) for cosim in cosims)
    pdf = [hist.get(i, 0) / len(pairs) for i in range(len(bins))]
    cdf = scanl1(op.add, pdf)
    median = np.median(cosims)
    mad = np.median(np.abs(np.array(cosims) - median))
    data = [
        [f"{x:.4f}" for x in pdf] + [f"{median:.4f}"],
        [f"{x:.4f}" for x in cdf] + [f"{mad:.4f}"],
    ]
    header = ["<0.6", "<0.7", "<0.8", "<0.9", "<1.0", "Median/MAD"]
    print(tabulate(data, header=header))
