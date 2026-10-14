"""Compile the integer model (intmodel.quantize output) to an HTSW import.

Every number the model needs is a literal inside a Change Variable action.
Program per token = Run0 (ph=0) then, >= 5 ticks later, Run1 (ph=1).
Inputs: var tok (token id), var pos (position, auto-incremented).
Output: var bi (argmax token) -> copied into tok for the next step.

Layout rules (Housing limits):
  - <= 25 Change Variable per container; phase functions have 25 conditionals
    `if (var ph == 0) {..} else {..}`: if-branches run in Run0, else in Run1.
  - special functions (embedding lookup, KV-cache writes) use real conditionals
    on tok / pos / pb and are called once per token.
  - <= 10 Trigger Function per container (dispatchers).
  - <= 300 functions total.
"""
import json, math, os, sys
import numpy as np
from intmodel import S, Q, NEWTON_ITERS, EXP_SQUARINGS, PROB_MAX

LANE = 21
LANE_MAX = (1 << (LANE - 1)) - 1
HALF = 1 << (S - 1)
MAXCV = 25
MAXCOND = 25
MAXCALL = 10
MAXFN = 300


PLAYER_MAX = 300      # Longs per holder: 2,560 B / 8 B = 320, keep margin
COND_VARS = ("ph", "ps", "tk", "pb")


class Holders:
    """Assign every var to player, global or a team (T1, T2, ...), <= PLAYER_MAX each."""
    def __init__(self, names_in_order):
        scalars = set(COND_VARS) | {"t", "m", "u", "c", "s2", "ss", "ry", "mx", "sm", "es", "best", "bi",
                                    "tok", "pos", "lsh", "g0", "g1", "g2"}
        hot = [n for n in names_in_order if n in scalars or (n[0] == "e" and n[1:].isdigit())]
        rest = [n for n in names_in_order if n not in set(hot)]
        self.of = {}
        for n in hot:
            self.of[n] = ("player",)
        assert len(hot) <= PLAYER_MAX, len(hot)
        free = PLAYER_MAX - len(hot)
        for n in rest[:free]:
            self.of[n] = ("player",)
        rest = rest[free:]
        # global holds the vocab strings (dynamic-read output), so no Longs there
        for i in range(0, len(rest), PLAYER_MAX):
            for n in rest[i:i + PLAYER_MAX]:
                self.of[n] = ("team", f"T{i // PLAYER_MAX + 1}")

    def teams(self):
        return sorted({h[1] for h in self.of.values() if h[0] == "team"})

    def read(self, n):
        h = self.of[n]
        if h[0] == "player":
            v = f"%var.player/{n} 0%"      # " 0": unset reads as "" otherwise
        elif h[0] == "global":
            v = f"%var.global/{n} 0%"
        else:
            v = f"%var.team/{n} {h[1]}%"   # no fallback allowed: Init zeroes team vars
        assert len(v) <= 32, v
        return v

    def render(self, act):
        var, op, val = act
        if isinstance(val, str):
            val = self.read(val)
        h = self.of[var]
        head = {"player": f"var {var}", "global": f"globalvar {var}"}.get(h[0]) or f"teamvar {var} {h[1]}"
        return f"{head} {op} {val} false"


def P(name):
    return name


class Emitter:
    def __init__(self):
        self.stream = []          # ("a", line) | ("call", fn)
        self.specials = {}        # fn name -> list of (cond, [lines])

    def a(self, var, op, val):
        if isinstance(val, str) and val.lstrip("-").isdigit():
            val = int(val)
        self.stream.append(("a", (var, op, val)))

    def call(self, fn):
        self.stream.append(("call", fn))

    def special(self, base, conds):
        """conds: list of (condition text, [lines]); split into functions of 25 conditionals."""
        names = []
        for i in range(0, len(conds), MAXCOND):
            name = f"{base}_{i // MAXCOND}"
            self.specials[name] = conds[i:i + MAXCOND]
            names.append(name)
        for n in names:
            self.call(n)

    # ---- building blocks -------------------------------------------------

    def clamp(self, T, lo, hi):
        if lo == 0:
            self.a("m", "=", T); self.a("m", ">>=", 63); self.a("m", "^=", -1); self.a(T, "&=", "m")
        else:
            self.a("t", "=", T); self.a("t", "-=", lo); self.a("m", "=", "t"); self.a("m", ">>=", 63)
            self.a("t", "&=", "m"); self.a(T, "-=", "t")
        self.a("t", "=", hi); self.a("t", "-=", T); self.a("m", "=", "t"); self.a("m", ">>=", 63)
        self.a("t", "&=", "m"); self.a(T, "+=", "t")

    def requant(self, T, M):
        self.a(T, "*=", int(M)); self.a(T, "+=", HALF); self.a(T, ">>=", S)

    def extract(self, targets):
        """Accumulator c holds len(targets) signed 21-bit lanes; write them to targets."""
        n = len(targets)
        for lane, T in enumerate(targets):
            if lane == n - 1:
                self.a(T, "=", "c")
                break
            self.a(T, "=", "c"); self.a(T, "<<=", 64 - LANE); self.a(T, ">>=", 64 - LANE)
            self.a("c", "-=", T); self.a("c", ">>=", LANE)

    def matmul(self, W, xs, handler, tag):
        """W int [out,in], xs input var names (|x| <= 127). handler(row, var) consumes each output."""
        out, K = W.shape
        for b0 in range(0, out, 3):
            rows = list(range(b0, min(b0 + 3, out)))
            # chunk K so every lane's worst case stays inside 21 bits
            chunks, start, bound = [], 0, np.zeros(len(rows))
            for j in range(K):
                nb = bound + 127 * np.abs(W[rows, j])
                if nb.max() > LANE_MAX:
                    chunks.append((start, j)); start, nb = j, 127 * np.abs(W[rows, j])
                bound = nb
            chunks.append((start, K))
            for ci, (j0, j1) in enumerate(chunks):
                first = True
                for j in range(j0, j1):
                    packed = sum(int(W[r, j]) << (LANE * l) for l, r in enumerate(rows))
                    if packed == 0:
                        continue
                    if first:
                        self.a("c", "=", packed); self.a("c", "*=", xs[j]); first = False
                    else:
                        self.a("t", "=", packed); self.a("t", "*=", xs[j]); self.a("c", "+=", "t")
                if first:
                    self.a("c", "=", 0)
                tg = [f"{tag}{r}" if ci == 0 else f"g{l}" for l, r in enumerate(rows)]
                self.extract(tg)
                if ci > 0:
                    for l, r in enumerate(rows):
                        self.a(f"{tag}{r}", "+=", f"g{l}")
            for r in rows:
                handler(r, f"{tag}{r}")

    def norm(self, d, A, out="x"):
        self.a("ss", "=", "h0"); self.a("ss", "*=", "h0")
        for i in range(1, d):
            self.a("t", "=", f"h{i}"); self.a("t", "*=", f"h{i}"); self.a("ss", "+=", "t")
        self.a("ss", "/=", d); self.a("ss", "+=", 1); self.a("ry", "=", 1 << 24)
        for _ in range(NEWTON_ITERS):
            self.a("t", "=", "ss"); self.a("t", "/=", "ry"); self.a("ry", "+=", "t"); self.a("ry", ">>=", 1)
        for i in range(d):
            self.a(f"{out}{i}", "=", f"h{i}"); self.a(f"{out}{i}", "*=", 2 * int(A)); self.a(f"{out}{i}", "/=", "ry")
            self.a(f"{out}{i}", "+=", 1); self.a(f"{out}{i}", ">>=", 1)
            self.clamp(f"{out}{i}", -127, 127)


def compile_model(qp, ctx):
    d, L, H, KVH, hd, f, V = (qp[k] for k in ("d", "L", "H", "KVH", "hd", "f", "V"))
    KVd = KVH * hd
    assert hd * 127 * 127 <= LANE_MAX and KVd % 3 == 0 and hd % 3 == 0
    nblk = math.ceil(ctx / 3)
    e = Emitter()

    # ---- embedding lookup (special: conditional on tok) ---------------------
    ne = math.ceil(d / 8)
    conds = []
    for tk in range(V):
        row = [int(v) & 0xFF for v in qp["Eq"][tk]] + [0] * (ne * 8 - d)
        lines = []
        for g in range(ne):
            u = sum(row[g * 8 + k] << (8 * k) for k in range(8))
            if u >= 1 << 63:
                u -= 1 << 64
            lines.append((f"e{g}", "=", u))
        lines.append(("es", "=", int(qp["ME"][tk])))
        conds.append((f"var tok == {tk}", lines))
    e.special("Emb", conds)
    # ---- per-token prelude: KV block / lane, causal masks -------------------
    # copies that are always set: Housing comparisons on an unset var fail
    e.a("ps", "=", "pos"); e.a("tk", "=", "tok")
    e.a("pb", "=", "pos"); e.a("pb", "/=", 3)
    e.a("lsh", "=", "pos"); e.a("t", "=", "pb"); e.a("t", "*=", 3); e.a("lsh", "-=", "t"); e.a("lsh", "*=", LANE)
    for s in range(ctx):
        e.a(f"mk{s}", "=", "pos"); e.a(f"mk{s}", "-=", s); e.a(f"mk{s}", ">>=", 63); e.a(f"mk{s}", "&=", -(1 << 40))

    for i in range(d):
        g, k = divmod(i, 8)
        e.a(f"h{i}", "=", f"e{g}")
        if 56 - 8 * k:
            e.a(f"h{i}", "<<=", 56 - 8 * k)
        e.a(f"h{i}", ">>=", 56); e.a(f"h{i}", "*=", "es")

    def to_int8(M, lo=-127):
        def h(r, var):
            e.requant(var, M[r]); e.clamp(var, lo, 127)
        return h

    def to_resid(M):
        def h(r, var):
            e.requant(var, M[r]); e.a(f"h{r}", "+=", var)
        return h

    xs = [f"x{i}" for i in range(d)]
    for l in range(L):
        # ---- attention -------------------------------------------------------
        e.norm(d, qp[f"l{l}.Aa"])
        e.matmul(qp[f"l{l}.wq"], xs, to_int8(qp[f"l{l}.wq.M"]), "q")
        e.matmul(qp[f"l{l}.wk"], xs, to_int8(qp[f"l{l}.wk.M"]), "k")
        e.matmul(qp[f"l{l}.wv"], xs, to_int8(qp[f"l{l}.wv.M"]), "v")
        for i in range(KVd):
            e.a(f"k{i}", "<<=", "lsh")
        for j in range(KVd // 3):
            e.a(f"vp{j}", "=", f"v{3 * j}")
            e.a("t", "=", f"v{3 * j + 1}"); e.a("t", "<<=", LANE); e.a(f"vp{j}", "+=", "t")
            e.a("t", "=", f"v{3 * j + 2}"); e.a("t", "<<=", 2 * LANE); e.a(f"vp{j}", "+=", "t")
        e.special(f"KW{l}", [(f"var pb == {b}", [(f"K{l}_{i}_{b}", "+=", f"k{i}") for i in range(i0, min(i0 + MAXCV, KVd))])
                             for b in range(nblk) for i0 in range(0, KVd, MAXCV)])
        e.special(f"VW{l}", [(f"var ps == {s}", [(f"V{l}_{s}_{j}", "=", f"vp{j}") for j in range(j0, min(j0 + MAXCV, KVd // 3))])
                             for s in range(ctx) for j0 in range(0, KVd // 3, MAXCV)])
        Ms = int(qp[f"l{l}.Ms"])
        for hh in range(H):
            kh = hh * KVH // H
            for b in range(nblk):
                for i in range(hd):
                    kv, qv = f"K{l}_{kh * hd + i}_{b}", f"q{hh * hd + i}"
                    if i == 0:
                        e.a("c", "=", kv); e.a("c", "*=", qv)
                    else:
                        e.a("t", "=", kv); e.a("t", "*=", qv); e.a("c", "+=", "t")
                e.extract([f"z{s}" for s in range(3 * b, min(3 * b + 3, 3 * nblk))])
            e.a("mx", "=", -(1 << 62))
            B = int(qp["B"][hh])
            for s in range(ctx):
                z = f"z{s}"
                e.a(z, "*=", Ms); e.a(z, "+=", HALF + (B * s << S)); e.a(z, ">>=", S); e.a(z, "+=", f"mk{s}")
                e.a("t", "=", z); e.a("t", "-=", "mx"); e.a("m", "=", "t"); e.a("m", ">>=", 63)
                e.a("m", "^=", -1); e.a("t", "&=", "m"); e.a("mx", "+=", "t")
            for s in range(ctx):
                z = f"z{s}"
                e.a(z, "-=", "mx"); e.a(z, "<<=", 30 - 8 - EXP_SQUARINGS); e.a(z, "+=", 1 << 30)
                e.a("m", "=", z); e.a("m", ">>=", 63); e.a("m", "^=", -1); e.a(z, "&=", "m")
                for _ in range(EXP_SQUARINGS):
                    e.a(z, "*=", z); e.a(z, ">>=", 30)
                e.a("sm", "=" if s == 0 else "+=", z)
            for s in range(ctx):
                e.a(f"z{s}", "*=", PROB_MAX); e.a(f"z{s}", "/=", "sm")
            for j in range(hd // 3):
                for s in range(ctx):
                    vv = f"V{l}_{s}_{kh * hd // 3 + j}"
                    if s == 0:
                        e.a("c", "=", vv); e.a("c", "*=", "z0")
                    else:
                        e.a("t", "=", vv); e.a("t", "*=", f"z{s}"); e.a("c", "+=", "t")
                tg = [f"o{hh * hd + 3 * j + l2}" for l2 in range(3)]
                e.extract(tg)
                for T in tg:
                    e.requant(T, qp[f"l{l}.Mo"]); e.clamp(T, -127, 127)
        e.matmul(qp[f"l{l}.wo"], [f"o{i}" for i in range(d)], to_resid(qp[f"l{l}.wo.M"]), "n")
        # ---- MLP ---------------------------------------------------------------
        e.norm(d, qp[f"l{l}.Am"])
        e.matmul(qp[f"l{l}.w1"], xs, to_int8(qp[f"l{l}.w1.M"], lo=0), "r")
        e.matmul(qp[f"l{l}.w2"], [f"r{j}" for j in range(f)], to_resid(qp[f"l{l}.w2.M"]), "n")

    # ---- head + argmax (first max wins) --------------------------------------
    e.norm(d, qp["Af"])
    e.a("best", "=", -(1 << 62)); e.a("bi", "=", 0)

    def argmax(r, var):
        e.a("t", "=", var); e.a("t", "-=", "best"); e.a("u", "=", "t"); e.a("u", "-=", 1)
        e.a("u", ">>=", 63); e.a("u", "^=", -1); e.a("t", "&=", "u"); e.a("best", "+=", "t")
        e.a("s2", "=", r); e.a("s2", "-=", "bi"); e.a("s2", "&=", "u"); e.a("bi", "+=", "s2")
    e.matmul(qp["head"], xs, argmax, "n")
    e.a("tok", "=", "bi"); e.a("pos", "+=", 1)
    e.call("Out")
    return e


VOCAB_GLOBAL_BYTES = 2400


def vocab_functions(vocab):
    """Vocab table (strings) + Out: print word `bi` with a computed-key read, no per-word conditionals.
    Words [6, K) live in global, [K, V) in team V (one holder can't hold all 2.7 KB)."""
    K, used = 6, 0
    while K < len(vocab) and used + len(vocab[K]) <= VOCAB_GLOBAL_BYTES:
        used += len(vocab[K]); K += 1
    q = lambda w: w.replace('"', "'")
    acts = [(f'globalvar w{i} = "{q(vocab[i])}" false' if i < K else f'teamvar w{i} V = "{q(vocab[i])}" false')
            for i in range(6, len(vocab))]
    lines = []
    for j in range(0, len(acts), MAXCV):
        lines.append("if () {"); lines += ["    " + x for x in acts[j:j + MAXCV]]; lines.append("}")
    assert len(lines) // (MAXCV + 2) + 1 <= MAXCOND
    out = ["if (var gen == 1, var bi > 5, var bi < %d) {" % K,
           '    var p = "%var.global/w" false',
           '    var q = "%var.player/p%%var.player/bi%%" false',   # -> %var.global/w123%
           "    var r = %var.player/q% false",                     # second pass reads it
           '    chat "%var.player/r%"',
           "}",
           "if (var gen == 1, var bi >= %d) {" % K,
           '    var p = "%var.team/w" false',
           '    var q = "%var.player/p%%var.player/bi%" false',    # -> %var.team/w400
           '    var q = "%var.player/q% V%" false',                 # -> %var.team/w400 V%
           "    var r = %var.player/q% false",
           '    chat "%var.player/r%"',
           "}",
           "if (var bi == 5) {",
           "    var done = 1 false",
           "}"]
    return {"Vocab": "\n".join(lines) + "\n", "Out": "\n".join(out) + "\n"}


def var_order(e):
    seen = dict.fromkeys(("ph", "pos", "tok"))
    acts = [x for k, x in e.stream if k == "a"] + [a for c in e.specials.values() for _, b in c for a in b]
    for var, _, val in acts:
        seen.setdefault(var, None)
        if isinstance(val, str):
            seen.setdefault(val, None)
    return list(seen)


def calls_fn(base, names):
    """Dispatcher functions calling `names` in order (<= 10 calls per container)."""
    out, part = {}, 0
    while True:
        fn = base if part == 0 else f"{base}_{part}"
        take = names[:MAXCOND * MAXCALL + MAXCALL - 1]
        names = names[len(take):]
        body, top = take[:MAXCOND * MAXCALL], take[MAXCOND * MAXCALL:]
        lines = []
        for i in range(0, len(body), MAXCALL):
            lines.append("if () {")
            lines += [f'    function "{x}"' for x in body[i:i + MAXCALL]]
            lines.append("}")
        lines += [f'function "{x}"' for x in top]
        if names:
            lines.append(f'function "{base}_{part + 1}"')
        out[fn] = lines
        part += 1
        if not names:
            return out


def layout(e, extra=None):
    """Pack the stream into phase functions + dispatchers. Returns ({fn: htsl}, Holders)."""
    H = Holders(var_order(e))
    segs, cur = [], []            # segs: list of ("chunks", [[lines]]) | ("call", fn)
    for kind, x in e.stream:
        if kind == "a":
            cur.append(x)
        else:
            if cur:
                segs.append(("chunks", [cur[i:i + MAXCV] for i in range(0, len(cur), MAXCV)])); cur = []
            segs.append(("call", x))
    if cur:
        segs.append(("chunks", [cur[i:i + MAXCV] for i in range(0, len(cur), MAXCV)]))
    # flatten to items: chunk | call
    items = []
    for kind, x in segs:
        if kind == "call":
            items.append(("call", x))
        else:
            items.extend(("chunk", c) for c in x)

    def assign(sub):
        """Greedy: chunks fill 25 slots per function; a call closes the open function."""
        fns, calls, open_ = [], [], False
        for kind, x in sub:
            if kind == "call":
                calls.append(("special", x)); open_ = False
            else:
                if not open_ or len(fns[-1]) == MAXCOND:
                    fns.append([]); calls.append(("phase", len(fns) - 1)); open_ = True
                fns[-1].append(x)
        return fns, calls

    # functions used by phase 0 grows with split, phase 1 shrinks: binary search the crossover
    lo, hi = 1, len(items) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if len(assign(items[:mid])[0]) < len(assign(items[mid:])[0]):
            lo = mid + 1
        else:
            hi = mid
    best = None
    for split in range(max(1, lo - 30), min(len(items), lo + 30)):
        f0, c0 = assign(items[:split]); f1, c1 = assign(items[split:])
        n = max(len(f0), len(f1))
        if best is None or n < best[0]:
            best = (n, f0, c0, f1, c1)
    n, f0, c0, f1, c1 = best

    out = {}
    for k in range(n):
        a = f0[k] if k < len(f0) else []
        b = f1[k] if k < len(f1) else []
        lines = []
        for slot in range(max(len(a), len(b))):
            lines.append("if (var ph == 0) {")
            lines += ["    " + H.render(x) for x in (a[slot] if slot < len(a) else [])]
            lines.append("} else {")
            lines += ["    " + H.render(x) for x in (b[slot] if slot < len(b) else [])]
            lines.append("}")
        out[f"F{k:03d}"] = "\n".join(lines) + "\n"
    for name, conds in e.specials.items():
        lines = []
        for cond, body in conds:
            lines.append(f"if ({cond}) {{")
            lines += ["    " + H.render(x) for x in body]
            lines.append("}")
        out[name] = "\n".join(lines) + "\n"
    for p, calls in ((0, c0), (1, c1)):
        names = [f"F{x:03d}" if kind == "phase" else x for kind, x in calls]
        for fn, lines in calls_fn(f"Run{p}", names).items():
            if fn == f"Run{p}":
                lines = [H.render(("ph", "=", p))] + lines
            out[fn] = "\n".join(lines) + "\n"
    # Init: start a conversation. Zero pos, the KV cache, and every team var (team reads have no fallback).
    # K lanes accumulate with += and V/K are read before written for future positions;
    # every other var is written before it is read within a token.
    zero = [n for n in H.of if n == "pos" or (n[0] in "KV" and "_" in n)]
    acts = [H.render((n, "=", 0)) for n in zero]
    per_fn = MAXCOND * MAXCV
    init_fns = []
    for i in range(0, len(acts), per_fn):
        name = f"Init_{i // per_fn}"
        lines = []
        for j in range(i, min(i + per_fn, len(acts)), MAXCV):
            lines.append("if () {"); lines += ["    " + x for x in acts[j:j + MAXCV]]; lines.append("}")
        out[name] = "\n".join(lines) + "\n"
        init_fns.append(name)
    for name, body in (extra or {}).items():
        out[name] = body
    if extra and "Vocab" in extra:
        init_fns.append("Vocab")
    for fn, lines in calls_fn("Init", init_fns).items():
        out[fn] = "\n".join(lines) + "\n"
    return out, H


def stats(e, fns, H):
    n_actions = sum(1 for k, _ in e.stream if k == "a") + sum(len(b) for c in e.specials.values() for _, b in c)
    exec_actions = sum(1 for k, _ in e.stream if k == "a")
    vars_ = set(var_order(e))
    per = {}
    for n, h in H.of.items():
        per[h[-1]] = per.get(h[-1], 0) + 1
    assert max(per.values()) * 8 <= 2560, per
    if not os.environ.get("ALLOW_OVER_300"):
        assert len(fns) <= MAXFN, len(fns)
    low = {}
    for v in vars_:
        assert low.setdefault(v.lower(), v) == v, f"case-insensitive var clash: {v} / {low[v.lower()]}"
    return dict(functions=len(fns), stored_actions=n_actions, executed_actions_per_token=exec_actions,
                variables=len(vars_), storage_bytes=8 * len(vars_), holders=per)


def write(fns, outdir):
    os.makedirs(f"{outdir}/fns", exist_ok=True)
    for name, body in fns.items():
        open(f"{outdir}/fns/{name}.htsl", "w").write(body)
    imp = {"functions": [{"name": n, "actions": f"fns/{n}.htsl"} for n in fns]}
    json.dump(imp, open(f"{outdir}/import.json", "w"), indent=1)
