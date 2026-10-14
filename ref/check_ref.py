"""Compare float vs integer reference: top-1 agreement and logit correlation."""
import sys, numpy as np
from floatmodel import FloatModel, random_model, load
from intmodel import quantize, IntModel


def calibrate(cfg, w, seqs):
    fm = FloatModel(cfg, w); fm.stats = {}
    for s in seqs:
        fm.reset()
        for t in s: fm.step(t)
    return fm.stats


def compare(cfg, w, seqs_cal, seqs_test):
    qp = quantize(cfg, w, calibrate(cfg, w, seqs_cal))
    fm, im = FloatModel(cfg, w), IntModel(qp)
    agree = n = 0; cors = []
    for s in seqs_test:
        fm.reset(); im.reset()
        for t in s:
            lf = fm.step(t); ti, li = im.step(t)
            agree += int(np.argmax(lf) == ti); n += 1
            cors.append(np.corrcoef(lf, li.astype(float))[0, 1])
    return agree / n, float(np.mean(cors)), qp


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfg, w, vocab = load(sys.argv[1])
    else:
        cfg, w = random_model()
    r = np.random.default_rng(1)
    cal = [r.integers(6, cfg["V"], 40) for _ in range(20)]
    test = [r.integers(6, cfg["V"], 40) for _ in range(5)]
    a, c, _ = compare(cfg, w, cal, test)
    print(f"top1 agreement {a:.3f}  mean logit corr {c:.4f}")
