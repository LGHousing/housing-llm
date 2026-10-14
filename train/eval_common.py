"""Score every exported run on one shared val set: held-out pairs whose user and bot words
are in-vocab for ALL runs. Uses the numpy reference forward (export path), reply-token CE."""
import glob, json, os, pickle, random, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from train import np_forward
from tokenizer import BOS, USER, BOT, EOS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
runs = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "config.json")))
models = []
for p in runs:
    d = os.path.dirname(p)
    cfg = json.load(open(p)); vocab = json.load(open(os.path.join(d, "vocab.json")))
    P = dict(np.load(os.path.join(d, "model.npz")))
    models.append((cfg, {w: i for i, w in enumerate(vocab)}, P))
common = set.intersection(*(set(m[1]) for m in models))

convs = pickle.load(open(os.path.join(ROOT, "data", "tinychat_pairs.pkl"), "rb"))["convs"]
order = list(range(len(convs))); random.Random(0).shuffle(order)
val_ids = order[: len(convs) // 50]  # same held-out conversations as train.py
pairs = [(u, b) for ci in val_ids for u, b in convs[ci] if all(w in common for w in u + b)][:N]
print(f"common words {len(common)}; shared val pairs {len(pairs)}")

res = {}
for cfg, w2i, P in models:
    tot = n = top1 = 0
    for u, b in pairs:
        seq = [BOS, USER] + [w2i[w] for w in u] + [BOT] + [w2i[w] for w in b] + [EOS]
        lg = np_forward(P, cfg, seq[:-1])
        s = len(u) + 2  # position of <bot>: predicts first reply token
        z = lg[s:]; tgt = np.array(seq[s + 1:])
        z = z - z.max(-1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        tot -= lp[np.arange(len(tgt)), tgt].sum(); n += len(tgt)
        top1 += (z.argmax(-1) == tgt).sum()
    res[cfg["name"]] = dict(ce=round(tot / n, 4), top1=round(top1 / n, 4), tokens=int(n))
    print(cfg["name"], res[cfg["name"]], flush=True)
json.dump(dict(pairs=len(pairs), results=res), open(os.path.join(ROOT, "train", "eval_common.json"), "w"), indent=2)
