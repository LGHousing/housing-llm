import pickle, numpy as np, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pickle.load(open(os.path.join(ROOT, "data", "tinychat_pairs.pkl"), "rb"))
cnt, convs = d["counts"], d["convs"]
tot = sum(cnt.values())
mc = [w for w, _ in cnt.most_common()]
plen = np.array([len(u) + len(b) + 4 for c in convs for u, b in c])
print("pair len (bos user .. bot .. eos) mean %.1f p50 %d p90 %d p99 %d, frac<=64 %.4f" % (plen.mean(), *np.percentile(plen, [50, 90, 99]), (plen <= 64).mean()))
import random; random.seed(0); sample = random.sample(convs, 50000)
for V in (512, 768, 1024, 1536, 2048):
    vs = set(mc[:V - 6])
    cov = sum(cnt[w] for w in vs) / tot
    full = np.mean([all(w in vs for w in u + b) for c in sample for u, b in c])
    botfull = np.mean([all(w in vs for w in b) for c in sample for u, b in c])
    print(f"V={V}: token coverage {cov:.4f}  pairs fully in-vocab {full:.4f}  bot turns fully in-vocab {botfull:.4f}")
print("top 60:", mc[:60])
