"""Parse TinyChat into (user_words, bot_words) pair lists and word counts."""
import os, re, pickle, collections, sys
sys.path.insert(0, os.path.dirname(__file__))
from tokenizer import words
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "hf_cache"))

ds = load_dataset("starhopp3r/TinyChat")["train"]
split = re.compile(r"\[INST\]|\[/INST\]")
convs, bad = [], 0
cnt = collections.Counter()
for row in ds:
    t = row["text"]
    # expect "[INST] u [/INST] b [INST] u [/INST] b ..."
    parts = re.split(r"(\[INST\]|\[/INST\])", t)
    pairs, cur_u, mode = [], None, None
    for p in parts:
        p = p.strip()
        if p == "[INST]": mode = "u"; continue
        if p == "[/INST]": mode = "b"; continue
        if not p: continue
        w = words(p)
        if mode == "u": cur_u = w
        elif mode == "b" and cur_u is not None:
            pairs.append((cur_u, w)); cur_u = None
    if not pairs: bad += 1; continue
    for u, b in pairs: cnt.update(u); cnt.update(b)
    convs.append(pairs)
print("convs", len(convs), "bad", bad, "pairs", sum(map(len, convs)))
print("distinct words", len(cnt), "total words", sum(cnt.values()))
with open(os.path.join(ROOT, "data", "tinychat_pairs.pkl"), "wb") as f:
    pickle.dump({"convs": convs, "counts": cnt}, f)
