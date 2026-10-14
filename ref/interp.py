"""Execute an Emitter stream in program order with Housing Long semantics (debug aid)."""
import re

M64 = (1 << 64) - 1


def w64(x):
    x &= M64
    return x - (1 << 64) if x >> 63 else x


class Interp:
    def __init__(self, e):
        self.e, self.v = e, {}

    def val(self, s):
        m = re.fullmatch(r"%var\.player/(\S+) 0%", s)
        return self.v.get(m.group(1), 0) if m else int(s)

    def act(self, act):
        var, op, val = act
        a = self.v.get(var, 0)
        b = self.v.get(val, 0) if isinstance(val, str) else val
        if op == "=": r = b
        elif op == "+=": r = a + b
        elif op == "-=": r = a - b
        elif op == "*=": r = a * b
        elif op == "/=": r = a if b == 0 else (abs(a) // abs(b)) * (1 if (a >= 0) == (b >= 0) else -1)
        elif op == "<<=": r = a << (b & 63)
        elif op == ">>=": r = a >> (b & 63)
        elif op == "&=": r = a & b
        elif op == "|=": r = a | b
        elif op == "^=": r = a ^ b
        else: raise ValueError(op)
        self.v[var] = w64(r)

    def cond(self, c):
        m = re.fullmatch(r"var (\S+) == (-?\d+)", c)
        return self.v.get(m.group(1), 0) == int(m.group(2))

    def run_token(self, hook=None):
        for kind, x in self.e.stream:
            if kind == "a":
                self.act(x)
            elif x in self.e.specials:
                for c, body in self.e.specials[x]:
                    if self.cond(c):
                        for line in body: self.act(line)
            if hook: hook(kind, x, self.v)
