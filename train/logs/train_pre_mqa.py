"""Train a tiny contract-conformant chat transformer on TinyChat (MPS) and export it.

Architecture (TASK.md contract): pre-norm decoder, RMSNorm gain (eps 1e-5), MHA no bias,
ALiBi, non-gated ReLU MLP, tied embedding/head, word-level vocab, ctx 64.
Attention scores are scaled by 1/sqrt(head_dim). No embedding scaling.
Loss is on reply tokens only (everything after <bot>, including <eos>).
"""
import argparse, json, math, os, pickle, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import SPECIALS, PAD, UNK, BOS, USER, BOT, EOS, words, prompt_ids

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = 48  # training sequence length (all kept examples <= 48 tokens; contract ctx is 64)

QUESTIONS = [
    "do you like cats ?", "what is your name ?", "how are you ?", "what do you like to eat ?",
    "do you want to play ?", "what color is the sky ?", "can you help me ?", "where do you live ?",
    "what is your favorite toy ?", "are you happy ?", "hello !", "do you like music ?",
    "what makes you sad ?", "do you have a friend ?", "is it raining today ?", "do you like school ?",
    "what is your favorite food ?", "can you tell me a story ?", "i feel sad today .", "good night !",
]


def budget(V, d, L, f):
    lhs = L * (4 * d * d + 2 * d * f) + V * d
    rhs = (300 - 35 - math.ceil(V / 25)) * 1200
    return lhs, rhs


def alibi_slopes(n):
    # standard ALiBi slopes (power-of-two n): 2^(-8/n * (i+1))
    assert n & (n - 1) == 0
    return [2 ** (-8.0 / n * (i + 1)) for i in range(n)]


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.g


class Block(nn.Module):
    def __init__(self, d, h, f):
        super().__init__()
        self.h = h
        self.attn_norm, self.mlp_norm = RMSNorm(d), RMSNorm(d)
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.w1, self.w2 = nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

    def forward(self, x, bias):
        B, t, d = x.shape
        hd = d // self.h
        y = self.attn_norm(x)
        q, k, v = (w(y).view(B, t, self.h, hd).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        att = (q @ k.transpose(-1, -2)) / math.sqrt(hd) + bias[:, :t, :t]
        x = x + self.wo((att.softmax(-1) @ v).transpose(1, 2).reshape(B, t, d))
        return x + self.w2(F.relu(self.w1(self.mlp_norm(x))))


class Model(nn.Module):
    def __init__(self, V, d, L, h, f, ctx=64):
        super().__init__()
        self.tok_emb = nn.Embedding(V, d)
        self.layers = nn.ModuleList(Block(d, h, f) for _ in range(L))
        self.final_norm = RMSNorm(d)
        sl = torch.tensor(alibi_slopes(h))
        i = torch.arange(ctx)
        rel = (i[None, :] - i[:, None]).float()  # j - i  (<= 0 for allowed keys)
        bias = sl[:, None, None] * rel[None]
        bias = bias.masked_fill(rel[None] > 0, float("-inf"))
        self.register_buffer("bias", bias, persistent=False)
        for n, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)
                if n.endswith("wo.weight") or n.endswith("w2.weight"):
                    p.data.mul_(1 / math.sqrt(2 * L))

    def forward(self, idx):
        x = self.tok_emb(idx)
        for b in self.layers:
            x = b(x, self.bias)
        return self.final_norm(x) @ self.tok_emb.weight.T


def build_vocab(counts, V, pin):
    mc = [w for w, _ in counts.most_common()]
    n = V - len(SPECIALS)
    pinned = [w for w in pin if w not in mc[:n]]
    base = [w for w in mc if w not in set(pinned)][: n - len(pinned)]
    return SPECIALS + base + pinned


def build_data(convs, w2i, max_user_unk, seed=0):
    """One example per (user, bot) pair (a pair is ~36 tokens, two never fit 64 reliably)."""
    rng = random.Random(seed)
    order = list(range(len(convs)))
    rng.shuffle(order)
    n_val = len(convs) // 50
    out = {}
    for name, ids in (("val", order[:n_val]), ("train", order[n_val:])):
        rows, kept, seen = [], 0, 0
        for ci in ids:
            for u, b in convs[ci]:
                seen += 1
                ue = [w2i.get(w, UNK) for w in u]
                be = [w2i.get(w, UNK) for w in b]
                if UNK in be or ue.count(UNK) > max_user_unk:
                    continue
                seq = [BOS, USER] + ue + [BOT] + be + [EOS]
                if len(seq) > T + 1:
                    continue
                rows.append(seq); kept += 1
        X = np.zeros((len(rows), T + 1), dtype=np.int16)
        M = np.zeros((len(rows), T + 1), dtype=np.bool_)
        for r, s in enumerate(rows):
            X[r, : len(s)] = s
            p = s.index(BOT)
            M[r, p + 1 : len(s)] = True  # predict reply tokens and <eos>
        out[name] = (X, M)
        print(f"{name}: kept {kept}/{seen} pairs ({kept/seen:.3f})", flush=True)
    return out


@torch.no_grad()
def evaluate(model, X, M, dev, bs=2048):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i : i + bs].astype(np.int64)).to(dev)
        m = torch.from_numpy(M[i : i + bs]).to(dev)
        lg = model(x[:, :-1])
        l = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none")
        mm = m[:, 1:].reshape(-1)
        tot += l[mm].sum().item(); n += mm.sum().item()
    model.train()
    return tot / n


@torch.no_grad()
def greedy(model, ids, dev, max_new=40):
    model.eval()
    ids = list(ids); out = []
    for _ in range(max_new):
        x = torch.tensor([ids[-64:]], device=dev)
        nxt = int(model(x)[0, -1].argmax())
        if nxt == EOS:
            break
        out.append(nxt); ids.append(nxt)
    model.train()
    return out


def np_forward(P, cfg, ids):
    """Pure numpy reference forward from the exported npz (sanity check of the export)."""
    d, h, L = cfg["d"], cfg["n_heads"], cfg["L"]
    hd = d // h
    sl = np.array(cfg["alibi_slopes"], dtype=np.float32)
    rms = lambda x, g: x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * g
    x = P["tok_emb"][ids]
    t = len(ids)
    rel = np.arange(t)[None, :] - np.arange(t)[:, None]
    bias = np.where(rel[None] > 0, -np.inf, sl[:, None, None] * rel[None])
    for i in range(L):
        y = rms(x, P[f"l{i}.attn_norm"])
        q, k, v = ((y @ P[f"l{i}.{w}"].T).reshape(t, h, hd).transpose(1, 0, 2) for w in ("wq", "wk", "wv"))
        a = q @ k.transpose(0, 2, 1) / np.sqrt(hd) + bias
        a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        x = x + (a @ v).transpose(1, 0, 2).reshape(t, d) @ P[f"l{i}.wo"].T
        y = rms(x, P[f"l{i}.mlp_norm"])
        x = x + np.maximum(y @ P[f"l{i}.w1"].T, 0) @ P[f"l{i}.w2"].T
    return rms(x, P["final_norm"]) @ P["tok_emb"].T


def export(model, cfg, vocab, outdir, dev):
    os.makedirs(outdir, exist_ok=True)
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    P = {"tok_emb": sd["tok_emb.weight"], "final_norm": sd["final_norm.g"]}
    for i in range(cfg["L"]):
        P[f"l{i}.attn_norm"] = sd[f"layers.{i}.attn_norm.g"]
        P[f"l{i}.mlp_norm"] = sd[f"layers.{i}.mlp_norm.g"]
        for w in ("wq", "wk", "wv", "wo", "w1", "w2"):
            P[f"l{i}.{w}"] = sd[f"layers.{i}.{w}.weight"]
    P = {k: v.astype(np.float32) for k, v in P.items()}
    np.savez(os.path.join(outdir, "model.npz"), **P)
    json.dump(vocab, open(os.path.join(outdir, "vocab.json"), "w"), indent=0)
    json.dump(cfg, open(os.path.join(outdir, "config.json"), "w"), indent=2)
    w2i = {w: i for i, w in enumerate(vocab)}
    lines, agree, total = [], 0, 0
    for q in QUESTIONS:
        p = prompt_ids(q, w2i)
        r = greedy(model, p, dev)
        # verify numpy reference greedy matches torch greedy
        ids = list(p); nr = []
        for _ in range(40):
            nxt = int(np_forward(P, cfg, ids[-64:])[-1].argmax())
            if nxt == EOS: break
            nr.append(nxt); ids.append(nxt)
        agree += int(nr == r); total += 1
        qtxt = " ".join(vocab[i] for i in p[2:-1])
        lines.append(f"Q: {q}\n   (as tokens: {qtxt})\nA: {' '.join(vocab[i] for i in r)}\n")
    open(os.path.join(outdir, "samples.txt"), "w").write("\n".join(lines))
    print(f"numpy-reference greedy == torch greedy on {agree}/{total} prompts", flush=True)
    return agree, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--V", type=int, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--f", type=int, required=True)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=60000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max_user_unk", type=int, default=2)
    args = ap.parse_args()

    lhs, rhs = budget(args.V, args.d, args.L, args.f)
    assert lhs <= rhs, f"budget violated {lhs} > {rhs}"
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = pickle.load(open(os.path.join(ROOT, "data", "tinychat_pairs.pkl"), "rb"))
    pin = sorted({w for q in QUESTIONS for w in words(q)})
    vocab = build_vocab(d["counts"], args.V, pin)
    w2i = {w: i for i, w in enumerate(vocab)}
    data = build_data(d["convs"], w2i, args.max_user_unk)
    del d
    Xtr, Mtr = data["train"]; Xva, Mva = data["val"]
    Xva, Mva = Xva[:50000], Mva[:50000]

    model = Model(args.V, args.d, args.L, args.heads, args.f).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"params {nparams}  budget lhs {lhs} <= {rhs}  dev {dev}", flush=True)
    decay = [p for p in model.parameters() if p.dim() == 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": nodecay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    warm = 500
    sched = lambda s: min(1, s / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / args.max_steps))))

    outdir = os.path.join(ROOT, "runs", args.name)
    os.makedirs(outdir, exist_ok=True)
    best, best_state, bad, t0, hist = 1e9, None, 0, time.time(), []
    Xtr_t = torch.from_numpy(Xtr.astype(np.int64)); Mtr_t = torch.from_numpy(Mtr)
    N = len(Xtr)
    step = 0
    while step < args.max_steps:
        perm = torch.randperm(N)
        for i in range(0, N - args.bs + 1, args.bs):
            idx = perm[i : i + args.bs]
            x = Xtr_t[idx].to(dev); m = Mtr_t[idx].to(dev)
            for g in opt.param_groups: g["lr"] = args.lr * sched(step)
            lg = model(x[:, :-1])
            l = F.cross_entropy(lg.reshape(-1, lg.size(-1)), x[:, 1:].reshape(-1), reduction="none")
            mm = m[:, 1:].reshape(-1).float()
            loss = (l * mm).sum() / mm.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.eval_every == 0 or step == args.max_steps:
                vl = evaluate(model, Xva, Mva, dev)
                el = time.time() - t0
                hist.append((step, round(loss.item(), 4), round(vl, 4), round(el)))
                print(f"step {step} train {loss.item():.4f} val {vl:.4f} lr {args.lr*sched(step):.2e} t {el:.0f}s", flush=True)
                if vl < best - 0.002:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                if bad >= args.patience:
                    print("plateau, stopping", flush=True)
                    step = args.max_steps
            if step >= args.max_steps:
                break
    model.load_state_dict(best_state)
    train_time = time.time() - t0
    cfg = dict(name=args.name, V=args.V, d=args.d, L=args.L, n_heads=args.heads, f=args.f, ctx=64,
               alibi_slopes=alibi_slopes(args.heads), rmsnorm_eps=1e-5, attn_scale="1/sqrt(head_dim)",
               val_loss=round(best, 4), val_loss_def="mean CE over reply tokens (+<eos>), nats",
               param_count=nparams, budget_lhs=lhs, budget_rhs=rhs, train_time_s=round(train_time),
               train_pairs=int(N), steps=hist[-1][0], batch_size=args.bs, lr=args.lr, weight_decay=args.wd,
               max_user_unk=args.max_user_unk, history=hist, specials=SPECIALS)
    export(model, cfg, vocab, outdir, dev)
    print(f"DONE {args.name} best val {best:.4f} time {train_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
