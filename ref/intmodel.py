"""Integer reference: the exact numeric spec the HTSL compiler must reproduce.

Only operations Housing has on 64-bit Longs: + - * (wrapping), / (truncate
toward zero), >> (arithmetic), << & | ^. All values below stay far inside
int64; the asserts in quantize() check the bounds that make this true.

Pipeline per token (see TASK.md ledger, "Integer design"):
  h        residual stream, Q16 (real * 2^16)
  norm     rms = isqrt(sum(h^2) / d + 1) by NEWTON_ITERS Newton steps from 2^24;
           xq = clamp(trunc(h * A_site / rms), -127, 127)      (gain folded into W)
  matmul   acc = Wq . xq  (int8 x int8, exact)
  requant  r = (acc * M + 2^(S-1)) >> S                        (round half up)
  attn     z = q8 . k8; zq = requant(z) in Q8 nats; + B_h*(s - pos);
           y = exp_q30(zq - max_s zq)  (max(1+x/64,0)^64, Q30);
           p8 = y * 255 / sum(y); out = sum_s p8 * v8
  head     logit = Hq . xq (one scale for the whole head), argmax, first max wins
"""
import numpy as np

S = 24            # requant shift
NEWTON_ITERS = 20
Q = 16            # residual fixed point
EXP_SQUARINGS = int(__import__("os").environ.get("EXP_SQ", "6"))
PROB_MAX = int(__import__("os").environ.get("PROB_MAX", "255"))   # attention probs are ints 0..PROB_MAX


def tdiv(a, b):
    """Java long division: truncate toward zero. b > 0 everywhere we use it."""
    a = np.asarray(a, dtype=np.int64)
    q = np.abs(a) // b
    return np.where(a < 0, -q, q).astype(np.int64)


def requant(acc, M):
    return (np.asarray(acc, np.int64) * np.asarray(M, np.int64) + (1 << (S - 1))) >> S


def isqrt_newton(m):
    y = np.int64(1 << 24)
    for _ in range(NEWTON_ITERS):
        y = (y + m // y) >> 1
    return y


def exp_q30(e):
    """exp(e/256) in Q30 for e <= 0 via max(1 + x/64, 0)^64."""
    y = (np.int64(1) << 30) + (np.asarray(e, np.int64) << (30 - 8 - EXP_SQUARINGS))
    y = np.maximum(y, 0)
    for _ in range(EXP_SQUARINGS):
        y = (y * y) >> 30
    return y


def quant_rows(W):
    """int8 per output row. Returns (Wq int64 [out,in], row scale float [out])."""
    s = np.abs(W).max(axis=1) / 127.0
    s[s == 0] = 1.0
    return np.round(W / s[:, None]).astype(np.int64), s


def act_scale(stats, site, pct=float(__import__("os").environ.get("ACT_PCT", "99.9"))):
    return 127.0 / np.percentile(stats[site], pct)


def quantize(cfg, w, stats):
    d, L, H, f, V = cfg["d"], cfg["L"], cfg["n_heads"], cfg["f"], cfg["V"]
    KVH = cfg.get("n_kv_heads", H)
    hd = d // H
    slopes = np.array(cfg.get("alibi_slopes") or [2.0 ** (-8.0 * (i + 1) / H) for i in range(H)])
    qp = dict(d=d, L=L, H=H, KVH=KVH, hd=hd, f=f, V=V, ctx=cfg["ctx"])
    M = lambda real: np.round(np.asarray(real) * (1 << S)).astype(np.int64)

    Eq, sE = quant_rows(w["tok_emb"].astype(np.float64))
    qp["Eq"], qp["ME"] = Eq, np.round(sE * (1 << Q)).astype(np.int64)
    qp["B"] = np.round(slopes * 256).astype(np.int64)

    for l in range(L):
        Aa = act_scale(stats, f"l{l}.xa")
        Am = act_scale(stats, f"l{l}.xm")
        Aq, Ak, Av = (act_scale(stats, f"l{l}.{s}") for s in "qkv")
        Ao, Ar = act_scale(stats, f"l{l}.o"), act_scale(stats, f"l{l}.r")
        g1, g2 = w[f"l{l}.attn_norm"], w[f"l{l}.mlp_norm"]
        for name, W, Ain, Aout in (("wq", w[f"l{l}.wq"] * g1, Aa, Aq),
                                   ("wk", w[f"l{l}.wk"] * g1, Aa, Ak),
                                   ("wv", w[f"l{l}.wv"] * g1, Aa, Av),
                                   ("w1", w[f"l{l}.w1"] * g2, Am, Ar)):
            Wq, s = quant_rows(W.astype(np.float64))
            qp[f"l{l}.{name}"], qp[f"l{l}.{name}.M"] = Wq, M(s / Ain * Aout)
        for name, W, Ain in (("wo", w[f"l{l}.wo"], Ao), ("w2", w[f"l{l}.w2"], Ar)):
            Wq, s = quant_rows(W.astype(np.float64))
            qp[f"l{l}.{name}"], qp[f"l{l}.{name}.M"] = Wq, M(s / Ain * (1 << Q))
        qp[f"l{l}.Aa"], qp[f"l{l}.Am"] = int(round(Aa)), int(round(Am))
        qp[f"l{l}.Ms"] = M(256.0 / (Aq * Ak * np.sqrt(hd)))      # z -> Q8 nats
        qp[f"l{l}.Mo"] = M(Ao / (PROB_MAX * Av))                    # out -> o8
    Wh = (w["tok_emb"] * w["final_norm"]).astype(np.float64)
    qp["head"] = np.round(Wh / (np.abs(Wh).max() / 127.0)).astype(np.int64)   # one scale: compare raw ints
    qp["Af"] = int(round(act_scale(stats, "xf")))

    # Bounds that keep every op exact in int64.
    hmax = max(stats["h"]) * (1 << Q) * 4
    assert hmax ** 2 * d < 2 ** 62, "residual too large for sum of squares"
    for k, v in qp.items():
        if k.endswith(".M"):
            assert np.abs(v).max() < 2 ** 38, k
    return qp


def norm_q(h, A, d):
    ss = np.sum(h * h)
    rms = isqrt_newton(ss // d + 1)
    return np.clip((tdiv(h * (2 * A), rms) + 1) >> 1, -127, 127)     # round, not truncate


class IntModel:
    def __init__(self, qp):
        self.qp = qp
        self.reset()

    def reset(self):
        L = self.qp["L"]
        self.K = [[] for _ in range(L)]
        self.Vc = [[] for _ in range(L)]
        self.pos = 0

    def step(self, tok):
        p, pos = self.qp, self.pos
        d, H, KVH, hd = p["d"], p["H"], p["KVH"], p["hd"]
        h = p["Eq"][tok] * p["ME"][tok]
        for l in range(p["L"]):
            x = norm_q(h, p[f"l{l}.Aa"], d)
            q8 = np.clip(requant(p[f"l{l}.wq"] @ x, p[f"l{l}.wq.M"]), -127, 127)
            k8 = np.clip(requant(p[f"l{l}.wk"] @ x, p[f"l{l}.wk.M"]), -127, 127)
            v8 = np.clip(requant(p[f"l{l}.wv"] @ x, p[f"l{l}.wv.M"]), -127, 127)
            self.K[l].append(k8)
            self.Vc[l].append(v8)
            K, Vv = np.array(self.K[l]), np.array(self.Vc[l])
            o8 = np.zeros(d, np.int64)
            rel = np.arange(pos + 1) - pos
            for hh in range(H):
                kh = hh * KVH // H
                z = K[:, kh * hd:(kh + 1) * hd] @ q8[hh * hd:(hh + 1) * hd]
                zq = requant(z, p[f"l{l}.Ms"]) + p["B"][hh] * rel
                y = exp_q30(zq - zq.max())
                p8 = (y * PROB_MAX) // y.sum()
                out = p8 @ Vv[:, kh * hd:(kh + 1) * hd]
                o8[hh * hd:(hh + 1) * hd] = np.clip(requant(out, p[f"l{l}.Mo"]), -127, 127)
            h = h + requant(p[f"l{l}.wo"] @ o8, p[f"l{l}.wo.M"])
            x = norm_q(h, p[f"l{l}.Am"], d)
            r8 = np.clip(requant(np.maximum(p[f"l{l}.w1"] @ x, 0), p[f"l{l}.w1.M"]), 0, 127)
            h = h + requant(p[f"l{l}.w2"] @ r8, p[f"l{l}.w2.M"])
        self.h = h
        x = norm_q(h, p["Af"], d)
        logits = p["head"] @ x
        self.pos += 1
        return int(np.argmax(logits)), logits
