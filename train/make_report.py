"""Build train/REPORT.md from data stats and runs/*/ artifacts."""
import glob, json, os, pickle, random, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from train import budget

d = pickle.load(open(os.path.join(ROOT, "data", "tinychat_pairs.pkl"), "rb"))
cnt, convs = d["counts"], d["convs"]
tot = sum(cnt.values())
mc = [w for w, _ in cnt.most_common()]
random.seed(0)
sample = random.sample(convs, 50000)
plen = np.array([len(u) + len(b) + 4 for c in convs for u, b in c])

out = ["# housing-llm training report (helper A)", ""]
out += ["## Data", "",
        f"- Source: HF `starhopp3r/TinyChat`, 1,000,000 rows, format `[INST] user [/INST] bot [INST] ...`.",
        f"- Parsed {len(convs):,} conversations ({1_000_000 - len(convs)} rows had no pair), {len(plen):,} (user, bot) pairs, {tot:,} word tokens, {len(cnt):,} distinct words.",
        f"- Encoded pair length `<bos> <user> .. <bot> .. <eos>`: mean {plen.mean():.1f}, p50 {np.percentile(plen,50):.0f}, p90 {np.percentile(plen,90):.0f}, p99 {np.percentile(plen,99):.0f}. Two pairs (~72) do not fit ctx 64, so every example is one turn: `<bos> <user> u <bot> b <eos>`; each pair of a multi-turn conversation is its own example.",
        "- Split: 2% of conversations held out (by conversation, seed 0) for val; val loss uses the first 50k kept val pairs.",
        "- Filter: keep a pair only if the bot turn is 100% in-vocab (so the model never learns to emit `<unk>`) and the user turn has <= 4 `<unk>`. Examples are <= 49 tokens.",
        "- Loss: cross-entropy on reply tokens only (tokens after `<bot>`, including `<eos>`).",
        "- Tokenizer (`train/tokenizer.py`): lowercase; `. , ? ! '` separate tokens; `; :` -> `,`; `- / em-dash` split words; curly quotes -> `'`; every other character dropped. Vocab = specials + most frequent words, with the 20 sample-question words pinned in (replacing the rarest words; cats, name, toy, live, eat, color, etc. were below the cutoff at V=512).",
        ""]
out += ["## Vocab coverage (pure top-frequency vocab)", "",
        "| V | word-token coverage | pairs fully in-vocab | bot turns fully in-vocab |", "|---|---|---|---|"]
for V in (512, 768, 1024):
    vs = set(mc[: V - 6])
    cov = sum(cnt[w] for w in vs) / tot
    full = np.mean([all(w in vs for w in u + b) for c in sample for u, b in c])
    bf = np.mean([all(w in vs for w in b) for c in sample for u, b in c])
    out.append(f"| {V} | {cov:.3f} | {full:.3f} | {bf:.3f} |")
out += ["", "Coverage of the actual (pinned) vocab per run is in the runs table.", ""]

runs = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "config.json")))
rows = []
for p in runs:
    c = json.load(open(p))
    vocab = json.load(open(os.path.join(os.path.dirname(p), "vocab.json")))
    vs = set(vocab[6:])
    cov = sum(cnt[w] for w in vs) / tot
    rows.append((c, cov))
out += ["## Runs", "",
        "Budget: `L*(2*d*d + 2*d*kv_dim + 2*d*f) + V*d <= (300 - 35 - ceil(V/25)) * 1200`, where kv_dim = d for MHA (this is the TASK.md formula) and head_dim for MQA.", "",
        "| run | V | d | L | f | heads | kv heads | params | budget lhs / rhs | coverage | train pairs | steps | val loss (nats/reply token) | train time |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for c, cov in rows:
    out.append(f"| {c['name']} | {c['V']} | {c['d']} | {c['L']} | {c['f']} | {c['n_heads']} | {c.get('n_kv_heads', c['n_heads'])} | {c['param_count']:,} | {c['budget_lhs']:,} / {c['budget_rhs']:,} | {cov:.3f} | {c['train_pairs']:,} | {c['steps']:,} | {c['val_loss']:.4f} | {c['train_time_s']/60:.1f} min |")
out += ["", "All runs: AdamW (0.9, 0.95), lr 3e-3, 500-step warmup, cosine to 0.1x over 60k steps, weight decay 0.1 on matrices only, grad clip 1.0, batch 512, fp32 on MPS. Eval every 1000 steps; stop after `patience` evals without a 0.002 improvement (5 for MHA runs, 60 = full schedule for the MQA run). No MHA run triggered early stopping; all four ran 60k steps. The exported checkpoint is the last eval that improved best val by >= 0.002 (so the final export can sit ~0.001 above the lowest logged val).",
        "Val losses are not strictly comparable across V: each V filters a different (larger-V = harder, more varied) subset of replies, and the softmax runs over a different vocab size.",
        "Export check: `np_forward` in train.py (pure numpy, reads only model.npz + config.json) reproduces torch greedy replies on 20/20 sample prompts for every run.", ""]
ec = os.path.join(ROOT, "train", "eval_common.json")
if os.path.exists(ec):
    E = json.load(open(ec))
    out += ["## Shared-set comparison (fair across V)", "",
            f"`train/eval_common.py`: {E['pairs']} held-out val pairs whose user and reply words are in every run's vocab (effectively the V=512 vocab), scored with the numpy reference forward on the exported npz. Reply-token CE and greedy top-1 accuracy (teacher forced).", "",
            "| run | CE (nats) | top-1 | reply tokens |", "|---|---|---|---|"]
    for k, v in sorted(E["results"].items(), key=lambda kv: kv[1]["ce"]):
        out.append(f"| {k} | {v['ce']:.4f} | {v['top1']:.3f} | {v['tokens']:,} |")
    out += ["", "Caveat: this set only contains words from the 512-word vocab, so the wider coverage of V=768/1024 earns no credit here.", ""]
rec = os.path.join(ROOT, "train", "RECOMMENDATION.md")
if os.path.exists(rec):
    out += [open(rec).read().strip(), ""]
out += ["## Samples (greedy, stop at `<eos>` or 40 tokens)", ""]
for c, _ in rows:
    out += [f"### {c['name']}", "", "```", open(os.path.join(ROOT, "runs", c["name"], "samples.txt")).read().strip(), "```", ""]
open(os.path.join(ROOT, "train", "REPORT.md"), "w").write("\n".join(out) + "\n")
print("wrote REPORT.md with", len(rows), "runs")
