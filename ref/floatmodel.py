"""Float reference forward pass for the TASK.md architecture contract.

Pre-norm decoder, RMSNorm (gain, eps 1e-5), causal attention with ALiBi,
n_kv_heads may be < n_heads (MQA/GQA), non-gated ReLU MLP, tied head.
"""
import json
import numpy as np


def load(run_dir):
    cfg = json.load(open(f"{run_dir}/config.json"))
    w = dict(np.load(f"{run_dir}/model.npz"))
    vocab = json.load(open(f"{run_dir}/vocab.json"))
    return cfg, w, vocab


def alibi_slopes(n_heads):
    return np.array([2.0 ** (-8.0 * (i + 1) / n_heads) for i in range(n_heads)])


def rmsnorm(x, g, eps=1e-5):
    return x / np.sqrt(np.mean(x * x) + eps) * g


class FloatModel:
    def __init__(self, cfg, w):
        self.cfg, self.w = cfg, w
        self.d, self.L, self.H = cfg["d"], cfg["L"], cfg["n_heads"]
        self.KVH = cfg.get("n_kv_heads", self.H)
        self.hd = self.d // self.H
        self.slopes = np.array(cfg.get("alibi_slopes") or alibi_slopes(self.H))
        self.stats = None
        self.reset()

    def rec(self, site, arr):
        if self.stats is not None:
            self.stats.setdefault(site, []).append(np.abs(arr).max())

    def reset(self):
        self.k = [[] for _ in range(self.L)]
        self.v = [[] for _ in range(self.L)]
        self.pos = 0

    def step(self, tok):
        w, hd, pos = self.w, self.hd, self.pos
        h = w["tok_emb"][tok].astype(np.float64)
        for l in range(self.L):
            self.rec("h", h)
            self.rec(f"l{l}.xa", rmsnorm(h, 1.0))
            x = rmsnorm(h, w[f"l{l}.attn_norm"])
            q = w[f"l{l}.wq"] @ x
            self.rec(f"l{l}.q", q)
            self.k[l].append(w[f"l{l}.wk"] @ x)
            self.v[l].append(w[f"l{l}.wv"] @ x)
            self.rec(f"l{l}.k", self.k[l][-1])
            self.rec(f"l{l}.v", self.v[l][-1])
            K, Vv = np.array(self.k[l]), np.array(self.v[l])
            o = np.zeros(self.d)
            for hh in range(self.H):
                kh = hh * self.KVH // self.H
                qs = q[hh * hd:(hh + 1) * hd]
                ks = K[:, kh * hd:(kh + 1) * hd]
                vs = Vv[:, kh * hd:(kh + 1) * hd]
                s = ks @ qs / np.sqrt(hd) + self.slopes[hh] * (np.arange(pos + 1) - pos)
                p = np.exp(s - s.max())
                p /= p.sum()
                o[hh * hd:(hh + 1) * hd] = p @ vs
            self.rec(f"l{l}.o", o)
            h = h + w[f"l{l}.wo"] @ o
            self.rec(f"l{l}.xm", rmsnorm(h, 1.0))
            x = rmsnorm(h, w[f"l{l}.mlp_norm"])
            r = np.maximum(w[f"l{l}.w1"] @ x, 0)
            self.rec(f"l{l}.r", r)
            h = h + w[f"l{l}.w2"] @ r
        self.rec("h", h)
        self.rec("xf", rmsnorm(h, 1.0))
        x = rmsnorm(h, w["final_norm"])
        self.pos += 1
        return w["tok_emb"] @ x


def random_model(V=512, d=96, L=2, H=4, KVH=1, f=384, ctx=64, seed=0):
    r = np.random.default_rng(seed)
    hd = d // H
    w = {"tok_emb": r.normal(0, 1, (V, d)) * 0.5, "final_norm": 1 + 0.1 * r.normal(size=d)}
    for l in range(L):
        w[f"l{l}.attn_norm"] = 1 + 0.1 * r.normal(size=d)
        w[f"l{l}.mlp_norm"] = 1 + 0.1 * r.normal(size=d)
        w[f"l{l}.wq"] = r.normal(0, d ** -0.5, (d, d))
        w[f"l{l}.wk"] = r.normal(0, d ** -0.5, (KVH * hd, d))
        w[f"l{l}.wv"] = r.normal(0, d ** -0.5, (KVH * hd, d))
        w[f"l{l}.wo"] = r.normal(0, d ** -0.5, (d, d)) * 0.5
        w[f"l{l}.w1"] = r.normal(0, d ** -0.5, (f, d))
        w[f"l{l}.w2"] = r.normal(0, f ** -0.5, (d, f)) * 0.5
    cfg = dict(V=V, d=d, L=L, n_heads=H, n_kv_heads=KVH, f=f, ctx=ctx)
    return cfg, {k: v.astype(np.float32) for k, v in w.items()}
