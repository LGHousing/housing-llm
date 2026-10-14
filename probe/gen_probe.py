"""Live Housing throughput probe for housing-llm (Elijah runs it; nothing drives the client).

/lbench  : one burst at the current level (10/20/30/40 blocks = 6,250..25,000 Long ops),
           checks the result, then steps the level up.
/lloop   : toggles a 5-tick repeating run at the current level; turning it off prints
           runs, seconds, effective ticks/s and actions/s, plus how many runs were wrong.
Each block = 25 containers x (8 x [t = W; t *= x; acc += t] + x += 1) = 625 actions,
the same op mix as the compiled model.
"""
import json, os, random, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Desktop/HTSW/ChiknImportFolder/imports/llm-probe")
NBLK, PER_CONT, CONTS = 40, 8, 25
M = (1 << 64) - 1
w64 = lambda x: ((x & M) ^ (1 << 63)) - (1 << 63)
random.seed(7)
W = [[random.randint(-2**40, 2**40) for _ in range(PER_CONT)] for _ in range(CONTS)]

os.makedirs(f"{OUT}/fns", exist_ok=True)
block = []
for c in range(CONTS):
    block.append("if () {")
    for g in range(PER_CONT):
        block += [f"    var t = {W[c][g]} false", "    var t *= %var.player/x 0% false", "    var acc += %var.player/t 0% false"]
    block += ["    var x += 1 false", "}"]
block = "\n".join(block) + "\n"
for b in range(NBLK):
    open(f"{OUT}/fns/LB{b + 1:02d}.htsl", "w").write(block)

# expected acc after 10*lv blocks, starting acc=0, x=1
exp, acc, x = {}, 0, 1
for b in range(NBLK):
    for c in range(CONTS):
        for g in range(PER_CONT):
            acc = w64(acc + w64(W[c][g] * x))
        x += 1
    if (b + 1) % 10 == 0:
        exp[(b + 1) // 10] = acc

bench = ["var acc = 0 false", "var x = 1 false"]
for lv in range(1, 5):
    bench.append(f"if (var lv >= {lv}) {{")
    bench += [f'    function "LB{b + 1:02d}"' for b in range((lv - 1) * 10, lv * 10)]
    bench.append("}")
bench.append("var ok = 0 false")
for lv in range(1, 5):
    bench.append(f"if (var lv == {lv}, var acc == {exp[lv]}) {{")
    bench.append("    var ok = 1 false")
    bench.append("}")
bench += ["var na = %var.player/lv 0% false", "var na *= 6250 false",
          "if (var ok == 0) {", "    var bad += 1 false", "}"]
open(f"{OUT}/fns/LBench.htsl", "w").write("\n".join(bench) + "\n")

burst = ["var lv += 0 false", "if (var lv == 0) {", "    var lv = 1 false", "}",
         'function "LBench"',
         "if (var ok == 1) {",
         '    chat "&aLLM bench OK: %var.player/na% ops in one trigger (level %var.player/lv%)"',
         "} else {",
         '    chat "&cLLM bench WRONG: actions were skipped (level %var.player/lv%)"',
         "}",
         "var lv += 1 false",
         "if (var lv > 4) {", "    var lv = 1 false", "}",
         'chat "&7next /lbench level: %var.player/lv% (x6250 ops)"']
open(f"{OUT}/fns/LBurst.htsl", "w").write("\n".join(burst) + "\n")

loop = ["if (var on == 1) {", '    function "LBench"', "    var runs += 1 false", "}"]
open(f"{OUT}/fns/LLoop.htsl", "w").write("\n".join(loop) + "\n")

toggle = ["var lv += 0 false", "var on += 0 false", "if (var lv == 0) {", "    var lv = 1 false", "}",
          "if (var on == 1) {",
          "    var on = 0 false",
          "    var el = %date.unix% false",
          "    var el -= %var.player/st 0% false",
          "    var tps = %var.player/runs 0% false",
          "    var tps *= 5 false",
          "    var tps /= %var.player/el 0% false",
          "    var aps = %var.player/runs 0% false",
          "    var aps *= %var.player/na 0% false",
          "    var aps /= %var.player/el 0% false",
          '    chat "&eLLM loop: %var.player/runs% runs in %var.player/el%s, wrong runs %var.player/bad%"',
          '    chat "&eeffective ticks/s %var.player/tps% (20 = no lag), ops/s %var.player/aps%"',
          "} else {",
          "    var on = 1 false", "    var runs = 0 false", "    var bad = 0 false",
          "    var st = %date.unix% false",
          '    chat "&eLLM loop ON at level %var.player/lv% (%var.player/lv%x6250 ops every 5 ticks). /lloop again to stop."',
          "}"]
open(f"{OUT}/fns/LToggle.htsl", "w").write("\n".join(toggle) + "\n")

imp = {"functions": [{"name": f"LB{b + 1:02d}", "actions": f"fns/LB{b + 1:02d}.htsl"} for b in range(NBLK)]
       + [{"name": n, "actions": f"fns/{n}.htsl"} for n in ("LBench", "LBurst", "LToggle")]
       + [{"name": "LLoop", "actions": "fns/LLoop.htsl", "repeatTicks": 5}],
       "commands": [{"name": "lbench", "actions": "fns/LBurstCmd.htsl", "mode": "Self", "requiredPriority": 0, "listed": True},
                    {"name": "lloop", "actions": "fns/LToggleCmd.htsl", "mode": "Self", "requiredPriority": 0, "listed": True}]}
open(f"{OUT}/fns/LBurstCmd.htsl", "w").write('function "LBurst"\n')
open(f"{OUT}/fns/LToggleCmd.htsl", "w").write('function "LToggle"\n')
json.dump(imp, open(f"{OUT}/import.json", "w"), indent=1)
json.dump({str(k): str(v) for k, v in exp.items()}, open(f"{OUT}/expected.json", "w"))
print("wrote", OUT, "expected", exp)
