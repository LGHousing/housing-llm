"""Float vs integer model on real questions: teacher-forced top-1 agreement + both greedy replies."""
import sys, re, json, numpy as np
from floatmodel import FloatModel, load
from intmodel import quantize, IntModel

EOS = 5


def questions(run):
    import os
    if os.environ.get("EVAL_FILE"):
        return [l.strip() for l in open(os.environ["EVAL_FILE"]) if l.strip()]
    return [l[3:].strip() for l in open(f"{run}/samples.txt") if l.startswith("Q: ")]


def encode(q, w2i):
    return [2, 3] + [w2i.get(x, 1) for x in q.split()] + [4]


def greedy(m, prompt, n=30):
    m.reset(); out = []
    for t in prompt[:-1]:
        m.step(t)
    t = prompt[-1]
    for _ in range(n):
        r = m.step(t)
        t = int(np.argmax(r)) if not isinstance(r, tuple) else r[0]
        if t == EOS: break
        out.append(t)
    return out


def calib_stats(cfg, w, prompts):
    fm = FloatModel(cfg, w); fm.stats = {}
    for p in prompts:
        greedy(fm, p)
    return fm.stats


if __name__ == "__main__":
    run = sys.argv[1]
    cfg, w, vocab = load(run)
    w2i = {x: i for i, x in enumerate(vocab)}
    qs = questions(run)
    prompts = [encode(q, w2i) for q in qs]
    cal = [encode(q, w2i) for q in open("../calib.txt").read().splitlines() if q.strip()]
    qp = quantize(cfg, w, calib_stats(cfg, w, cal))
    fm, im = FloatModel(cfg, w), IntModel(qp)
    agree = n = same = 0
    for q, p in zip(qs, prompts):
        fr = greedy(fm, p)
        ir = greedy(im, p)
        same += fr == ir
        # teacher-forced agreement along the float reply
        fm.reset(); im.reset()
        seq = p + fr
        for i, t in enumerate(seq[:-1]):
            lf = fm.step(t); ti, _ = im.step(t)
            if i >= len(p) - 1:
                agree += int(np.argmax(lf) == ti); n += 1
        if not __import__("os").environ.get("QUIET"): print(f"Q: {q}\n  float: {' '.join(vocab[t] for t in fr)}\n  int:   {' '.join(vocab[t] for t in ir)}")
    print(f"teacher-forced top-1 agreement {agree}/{n} = {agree / n:.3f}; identical greedy replies {same}/{len(qs)}")
