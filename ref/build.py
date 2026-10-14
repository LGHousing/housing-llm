"""Quantize a run (or random model), compile to HTSL, write expected outputs from IntModel."""
import argparse, json, numpy as np
from floatmodel import FloatModel, random_model, load
from intmodel import quantize, IntModel
from check_ref import calibrate
from compile import compile_model, layout, stats, write, vocab_functions

ap = argparse.ArgumentParser()
ap.add_argument("--run"); ap.add_argument("--out", required=True)
ap.add_argument("--ctx", type=int, default=24); ap.add_argument("--gen", type=int, default=8)
ap.add_argument("--prompt", default="")
ap.add_argument("--prompts", default="")   # file, one question per line (multi-conversation run)
ap.add_argument("--random", default="")   # e.g. "V=512,d=96,L=2,H=4,KVH=1,f=384"
a = ap.parse_args()

if a.run:
    cfg, w, vocab = load(a.run)
else:
    kw = {k: int(v) for k, v in (x.split("=") for x in a.random.split(","))} if a.random else {}
    cfg, w = random_model(**kw); vocab = [f"w{i}" for i in range(cfg["V"])]
r = np.random.default_rng(1)
w2i = {x: i for i, x in enumerate(vocab)}
cal = [r.integers(6, cfg["V"], 30) for _ in range(30)]
if a.prompt:
    prompt = [2, 3] + [w2i.get(x, 1) for x in a.prompt.split()] + [4]
else:
    prompt = [2, 3] + list(r.integers(6, cfg["V"], 5)) + [4]
if a.run:
    from eval_q import questions, encode, calib_stats
    cal = [encode(q, w2i) for q in open("../calib.txt").read().splitlines() if q.strip()]
    qp = quantize(cfg, w, calib_stats(cfg, w, cal))
else:
    qp = quantize(cfg, w, calibrate(cfg, w, cal))

def conversation(prompt):
    im = IntModel(qp); fm = FloatModel(cfg, w)
    steps, tok, agree = [], prompt[0], 0
    for i in range(len(prompt) - 1 + a.gen):
        if i < len(prompt):
            tok = prompt[i]
        bi, _ = im.step(tok)
        agree += int(np.argmax(fm.step(tok)) == bi)
        steps.append(dict(tok=int(tok), prompt=i < len(prompt), bi=bi, h=[str(int(v)) for v in im.h]))
        tok = bi
        if i >= len(prompt) - 1 and bi == 5:
            break
    assert len(steps) <= a.ctx, (len(steps), a.ctx)
    return steps, agree


prompts = [prompt]
if a.prompts:
    prompts = [[2, 3] + [w2i.get(x, 1) for x in q.split()] + [4] for q in open(a.prompts).read().splitlines() if q.strip()]
convs, agree_all, n_all = [], 0, 0
for p in prompts:
    steps, agree = conversation(p)
    convs.append(steps); agree_all += agree; n_all += len(steps)
    print("Q:", " ".join(vocab[t] for t in p[2:-1]), "| int reply:", " ".join(vocab[s["bi"]] for s in steps[len(p) - 1:]))
print(f"float/int top1 agreement over all steps: {agree_all}/{n_all}")
steps = convs[0]

e = compile_model(qp, a.ctx)
fns, H = layout(e, vocab_functions(vocab))
st = stats(e, fns, H)
write(fns, a.out)
json.dump(dict(convs=convs, steps=steps, d=qp["d"], stats=st, teams=H.teams()), open(f"{a.out}/expected.json", "w"))
print(json.dumps(st))
json.dump(vocab, open(f"{a.out}/vocab.json", "w"))
