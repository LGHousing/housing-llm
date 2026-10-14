"""Fake-quant ablation: which rounding site costs float/int agreement?"""
import sys, numpy as np
from floatmodel import FloatModel, load, rmsnorm
from eval_q import questions, encode, greedy, calib_stats

run = sys.argv[1]
cfg, w, vocab = load(run); w2i = {x: i for i, x in enumerate(vocab)}
cal = [encode(q, w2i) for q in open("../calib.txt").read().splitlines() if q.strip()]
st = calib_stats(cfg, w, cal)
A = {k: 127.0 / np.percentile(v, 99.9) for k, v in st.items()}

def qrow(W):
    s = np.abs(W).max(1, keepdims=True) / 127; return np.round(W / s) * s
def qa(x, a): return np.clip(np.round(x * a), -127, 127) / a

class FQ(FloatModel):
    def __init__(s, cfg, w, sites):
        ww = dict(w)
        if "W" in sites:
            for k in ww:
                if ww[k].ndim == 2 and k != "tok_emb": ww[k] = qrow(ww[k])
        if "E" in sites: ww["tok_emb"] = qrow(w["tok_emb"])
        super().__init__(cfg, ww); s.sites = sites
    def step(s, tok):
        w, hd, pos, S = s.w, s.hd, s.pos, s.sites
        h = w["tok_emb"][tok].astype(np.float64)
        for l in range(s.L):
            xn = rmsnorm(h, 1.0)
            if "x" in S: xn = qa(xn, A[f"l{l}.xa"])
            x = xn * w[f"l{l}.attn_norm"]
            q, k, v = w[f"l{l}.wq"] @ x, w[f"l{l}.wk"] @ x, w[f"l{l}.wv"] @ x
            if "q" in S: q = qa(q, A[f"l{l}.q"])
            if "k" in S: k = qa(k, A[f"l{l}.k"])
            if "v" in S: v = qa(v, A[f"l{l}.v"])
            s.k[l].append(k); s.v[l].append(v)
            K, V = np.array(s.k[l]), np.array(s.v[l]); o = np.zeros(s.d)
            for hh in range(s.H):
                kh = hh * s.KVH // s.H
                sc = K[:, kh*hd:(kh+1)*hd] @ q[hh*hd:(hh+1)*hd] / np.sqrt(hd) + s.slopes[hh] * (np.arange(pos+1) - pos)
                p = np.exp(sc - sc.max()); p /= p.sum()
                if "p" in S: p = np.floor(p * 255) / 255
                o[hh*hd:(hh+1)*hd] = p @ V[:, kh*hd:(kh+1)*hd]
            if "o" in S: o = qa(o, A[f"l{l}.o"])
            h = h + w[f"l{l}.wo"] @ o
            xn = rmsnorm(h, 1.0)
            if "x" in S: xn = qa(xn, A[f"l{l}.xm"])
            r = np.maximum(w[f"l{l}.w1"] @ (xn * w[f"l{l}.mlp_norm"]), 0)
            if "r" in S: r = qa(r, A[f"l{l}.r"])
            h = h + w[f"l{l}.w2"] @ r
        xn = rmsnorm(h, 1.0)
        if "x" in S: xn = qa(xn, A["xf"])
        s.pos += 1
        H = w["tok_emb"] * w["final_norm"]
        if "H" in S: H = np.round(H / (np.abs(H).max() / 127)) * (np.abs(H).max() / 127)
        return H @ xn

qs = questions(run); prompts = [encode(q, w2i) for q in qs]
fm = FloatModel(cfg, w)
refs = [p + greedy(fm, p) for p in prompts]
def agree(sites):
    m = FQ(cfg, w, sites); a = n = 0
    for p, seq in zip(prompts, refs):
        m.reset(); fm.reset()
        for i, t in enumerate(seq[:-1]):
            lm = m.step(t); lf = fm.step(t)
            if i >= len(p) - 1: a += int(np.argmax(lm) == np.argmax(lf)); n += 1
    return a / n
for s in ["W", "E", "H", "x", "q", "k", "v", "p", "o", "r", "WEHxqkvpor"]:
    print(s, f"{agree(set(s)):.3f}")
