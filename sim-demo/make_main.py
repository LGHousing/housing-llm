"""Write htsw:main for the housing-llm sim demo.  Usage: python3 make_main.py "do you like cats ?"
Words must be in vocab.json (lowercase, punctuation as separate words). Run with: htsw run import.json
"""
import json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
vocab = json.load(open(os.path.join(here, "vocab.json")))
w2i = {w: i for i, w in enumerate(vocab)}
q = sys.argv[1] if len(sys.argv) > 1 else "do you like cats ?"
words = q.lower().split()
missing = [w for w in words if w not in w2i]
if missing:
    sys.exit(f"not in vocab: {missing} (512 words, see vocab.json)")
ids = [2, 3] + [w2i[w] for w in words] + [4]
GEN = 48 - len(ids) + 1

lines = ['chat "Q: ' + " ".join(words).replace('"', "'") + '"', 'function "Init"', "var done = 0 false", "var gen = 0 false"]
for i, t in enumerate(ids):
    lines += ["if () {", f"    var tok = {t} false"]
    if i == len(ids) - 1:
        lines.append("    var gen = 1 false")   # Out prints only while generating
    lines += ['    function "Run0"', '    function "Run1"', "}"]
gen_fns = []
for g in range(0, GEN, 25):
    name = f"Gen_{g // 25}"
    body = []
    for _ in range(min(25, GEN - g)):
        body += ["if (var done == 0) {", '    function "Run0"', '    function "Run1"', "}"]
    open(os.path.join(here, "fns", f"{name}.htsl"), "w").write("\n".join(body) + "\n")
    gen_fns.append(name)
lines += [f'function "{n}"' for n in gen_fns]
open(os.path.join(here, "fns", "htsw_main.htsl"), "w").write("\n".join(lines) + "\n")

imp = json.load(open(os.path.join(here, "import.json")))
names = {f["name"] for f in imp["functions"]}
for n in ["htsw:main"] + gen_fns:
    if n not in names:
        imp["functions"].append({"name": n, "actions": "fns/htsw_main.htsl" if n == "htsw:main" else f"fns/{n}.htsl"})
json.dump(imp, open(os.path.join(here, "import.json"), "w"), indent=1)
print("prompt ids:", " ".join(map(str, ids)))
