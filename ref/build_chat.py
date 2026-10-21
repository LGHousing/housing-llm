"""Generate the click-to-type UI and batched model dispatchers.

    python3 ref/build_chat.py [--house house]

Vocabulary order is frequency order with pinned evaluation words at the end
(train/train.py:build_vocab). Keep that order inside each prefix menu.
Owns house/chat, UI manifest entries, and Run0 / Run1 / Run0_1 dispatchers.
"""
import argparse
import hashlib
import string
import json
from pathlib import Path
import re
import shutil

K = 7
BITS = 9
CONTEXT = 48
MAX_INPUT = 16
ROOT = "LLM Keypad"
FUNCTIONS = ("Feed", "Reply", "New Chat", "Chat Next", "Chat Load", "Chat Keys", "Chat Show", "Chat Pick")


def build(house):
    vocab = json.loads((house / "vocab.json").read_text())
    assert vocab[:6] == ["<pad>", "<unk>", "<bos>", "<user>", "<bot>", "<eos>"]
    assert len(set(vocab)) == len(vocab)
    # Keep room for the reply within the model context.
    max_input = MAX_INPUT
    assert len(vocab) <= 1 << BITS
    words = [i for i, w in enumerate(vocab) if re.fullmatch("[a-z]+", w)]
    prefixes = {}

    def visit(prefix, tokens):
        choices = tokens[:K]
        # Keep exact short words reachable even outside the seven most common.
        exact = next((t for t in tokens if vocab[t] == prefix), None)
        if exact is not None and exact not in choices:
            choices = sorted(choices[:K - 1] + [exact])
        prefixes[prefix] = choices
        if len(prefix) < 2 or len(tokens) > K:
            letters = sorted({vocab[t][len(prefix)] for t in tokens if len(vocab[t]) > len(prefix)})
            for letter in letters:
                child = prefix + letter
                visit(child, [t for t in tokens if vocab[t].startswith(child)])

    visit("", words)
    depth = max(map(len, prefixes))
    assert depth <= 12  # Numeric base-27 prefix must fit a signed Long.
    # V's strings + packed lookup + current prefix and seven suggestion names.
    v_bytes = sum(len(w.encode()) for w in vocab[460:])
    v_bytes += 8 * len(prefixes) + K * max(map(len, vocab)) + depth
    assert v_bytes <= 2560, f"Team V lookup needs {v_bytes} bytes"

    def code(prefix):
        n = 0
        for c in prefix:
            n = n * 27 + ord(c) - ord('a') + 1
        return n

    packed = {code(p): sum(t << (BITS * i) for i, t in enumerate(ts)) for p, ts in prefixes.items()}
    version = int(hashlib.sha256(json.dumps([vocab, packed]).encode()).hexdigest()[:7], 16)
    routes = {}
    for prefix, tokens in prefixes.items():
        for i, token in enumerate(tokens):
            routes.setdefault(str(token), dict(prefix=prefix, suggestion=i))
    assert all(str(t) in routes for t in words), "A word is hidden behind more frequent prefix matches"
    manifest = json.loads((house / "import.json").read_text())
    for kind in ("functions", "menus", "commands", "events"):
        manifest[kind] = [x for x in manifest.get(kind, [])
                          if not x.get("actions", "").startswith("chat/")
                          and not (kind == "menus" and x["name"].startswith("LLM "))]
    assert not (set(FUNCTIONS) & {x["name"] for x in manifest["functions"]})
    assert not any(x["name"] == "llm" for x in manifest["commands"])
    assert len(manifest["functions"]) + len(FUNCTIONS) <= 300
    out = house / "chat"
    if out.exists():
        shutil.rmtree(out)

    def write(name, text):
        file = out / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text.strip() + "\n")
        return "chat/" + name

    def function(name, body, repeat=None):
        entry = dict(name=name, actions=write(f"fns/{name}.htsl", body))
        if repeat is not None:
            entry["repeatTicks"] = repeat
        manifest["functions"].append(entry)

    # Preserve the compiler's exact call order, changing only its dispatchers.
    passes_file = house / "model-passes.json"
    if not (house / "fns/Run0.htsl").read_text().startswith("// Batched"):
        def calls(name):
            result = []
            for child in re.findall(r'function "([^"]+)"', (house / f"fns/{name}.htsl").read_text()):
                result.extend(calls(child) if child.startswith("Run0_") else [child])
            return result
        passes_file.write_text(json.dumps({name: calls(name) for name in ("Run0", "Run1")}, indent=2) + "\n")
    passes = json.loads(passes_file.read_text())
    overflow = []
    for phase, name in enumerate(("Run0", "Run1")):
        sequence = passes[name]
        chunks = [sequence[i:i + 10] for i in range(0, len(sequence), 10)]
        lines = ["// Batched by ref/build_chat.py; call once per scheduler tick.", f"var ph = {phase} false"]
        for i, chunk in enumerate(chunks):
            block = [f"if (var ui_part == {i} 0" + (f", var ph == {phase}" if i >= 25 else "") + ") {"]
            block += [f'    function "{fn}"' for fn in chunk]
            block += [f"    var ui_part = {i + 1 if i + 1 < len(chunks) else 0} false",
                      f"    var ui_work = {phase + 1 if i + 1 < len(chunks) else phase + 2} false", "    exit", "}"]
            (lines if i < 25 else overflow).extend(block)
        lines += ['function "Run0_1"']
        (house / f"fns/{name}.htsl").write_text("\n".join(lines) + "\n")
    (house / "fns/Run0_1.htsl").write_text("// Batched overflow for both model passes.\n" + "\n".join(overflow) + "\n")

    owner_guard = '''if (!globalvar ui_owner == "%player.name%" "") {
    chat "[LLM] Click New Chat to start. Only one player can chat at a time."
    exit
}'''
    ready_guard = owner_guard + '''
if (!var ui_phase == 0 5) {
    chat "[LLM] Please wait for the current word or reply."
    exit
}'''
    release = '''globalvar ui_owner = "" false
var ui_phase = 5 false
var ui_work = 0 false'''
    release_indented = release.replace("\n", "\n    ")
    function("Feed", f'''{owner_guard}
if (!var ui_work == 0 0) {{
    exit
}}
if or (var ui_phase < 1 0, var ui_phase > 4 0) {{
    exit
}}
if (var pos >= {CONTEXT}) {{
    chat "[LLM] Context full. Click New Chat."
    {release_indented}
    exit
}}
var ui_part = 0 false
var ui_work = 1 false
''')
    # No pauses or queued work: restart/cancel can replace any failed phase.
    function("New Chat", '''globalvar ui_owner = "%player.name%" false
var ui_phase = 6 false
var ui_work = 0 false
var ui_part = 0 false
var gen = 0 false
var done = 0 false
var ui_n = 0 false
chat "[LLM] Starting a new question..."
''')
    function("Reply", f'''{ready_guard}
if (var pos <= 2) {{
    chat "[LLM] Choose some words first."
    exit
}}
var gen = 1 false
var done = 0 false
var tok = 4 false
var ui_phase = 4 false
chat "[LLM] Reply:"
function "Feed"
''')
    # The only scheduler. Each invocation does one small, synchronous batch.
    # -1 is left behind if a batch aborts before its success marker.
    function("Chat Next", f'''if (!globalvar ui_owner == "%player.name%" "") {{
    exit
}}
if or (var ui_phase == 0 5, var ui_phase == 5 5) {{
    exit
}}
if (var ui_work == -1) {{
    chat "[LLM] Calculation stopped (phase %var.player/ph%, batch %var.player/ui_part%). Click New Chat."
    {release_indented}
    displayMenu "{ROOT}"
    exit
}}
if (var ui_phase == 6) {{
    var ui_work = -1 false
    function "Chat Load"
    exit
}}
if (var ui_work == 1) {{
    var ui_work = -1 false
    function "Run0"
    exit
}}
if (var ui_work == 2) {{
    var ui_work = -1 false
    function "Run1"
    exit
}}
if (!var ui_work == 3) {{
    exit
}}
var ui_work = 0 false
if (var ui_phase == 1) {{
    var tok = 3 false
    var ui_phase = 2 false
    function "Feed"
    exit
}}
if (var ui_phase == 2) {{
    var done = 0 false
    var ui_phase = 0 false
    chat "[LLM] Choose words, then Send."
    var ui_key = 0 false
    function "Chat Keys"
    exit
}}
if (var ui_phase == 3) {{
    var done = 0 false
    var ui_phase = 0 false
    var ui_n = 0 false
    var ui_key = 0 false
    function "Chat Keys"
    exit
}}
if (!var ui_phase == 4) {{
    exit
}}
if (!var done == 1, var pos >= {CONTEXT}) {{
    chat "[LLM] Reply stopped at the context limit."
}}
if or (var done == 1, var pos >= {CONTEXT}) {{
    {release_indented}
    chat "[LLM] Finished. Use New Chat for another question."
    exit
}}
function "Feed"
''', repeat=5)
    new_chat = write("actions/new.htsl", 'function "New Chat"')
    send = write("actions/send.htsl", '''function "Reply"
if (var ui_phase == 4) {
    closeMenu
}''')
    end = write("actions/end.htsl", f'''
{release}
chat "[LLM] Chat released. Anyone can start a new question."
closeMenu''')
    back = write("actions/root.htsl", f'displayMenu "{ROOT}"')
    manifest["commands"].append(dict(name="llm", mode="Self", requiredPriority=0,
        listed=True, actions=back))
    if not manifest["events"]:
        del manifest["events"]

    def item(key, label, material="paper", damage=0):
        tag = '{display:{Name:' + json.dumps(label, ensure_ascii=False) + '}'
        return write(f"items/{key}.snbt", '{id:"minecraft:' + material
            + f'",Count:1b,Damage:{damage}s,tag:' + tag + '}}')

    def slot(index, key, label, actions, material="paper", damage=0):
        return dict(slot=index, nbt=item(key, label, material, damage), actions=actions)

    # All seven 9-bit token IDs fit in one nonnegative 63-bit Long.
    # Dynamic numeric placeholders have commas; L casts undo that formatting.
    loads = [f'teamvar "k{n:,}" V = {value} false' for n, value in packed.items()]
    loads += [f'globalvar ui_c{i} = "{c}" false' for i, c in enumerate(string.ascii_lowercase, 1)]
    batches = [loads[i:i + 20] for i in range(0, len(loads), 20)]
    count = len(batches)
    loader = [f"if (var ui_part == 0, globalvar ui_rev == {version} 0) {{",
              f"    var ui_part = {count} false", "    var ui_work = 0 false", "    exit", "}"]
    for i, batch in enumerate(batches):
        loader += [f"if (var ui_part == {i}) {{"] + ["    " + line for line in batch]
        if i + 1 == count:
            loader += [f"    globalvar ui_rev = {version} false"]
        loader += [f"    var ui_part = {i + 1} false", "    var ui_work = 0 false", "    exit", "}"]
    for i, fn in enumerate(("Init_0", "Init_1", "Init_2", "Vocab"), count):
        loader += [f"if (var ui_part == {i}) {{", f'    function "{fn}"',
                   f"    var ui_part = {i + 1} false", "    var ui_work = 0 false", "    exit", "}"]
    loader += ["var ui_phase = 1 false", "var tok = 2 false", "var ui_work = 0 false", 'function "Feed"']
    function("Chat Load", "\n".join(loader))

    keys = f'''{ready_guard}
if (var ui_key > 0, var ui_n >= {27 ** (depth - 1)}) {{
    chat "[LLM] Choose a suggestion, Backspace, or Clear letters."
    exit
}}
if (var ui_key > 0) {{
    var ui_n *= 27 false
    var ui_n += %var.player/ui_key% false
}}
if (var ui_key == -1) {{
    var ui_n /= 27 false
}}
if (var ui_key == -2) {{
    var ui_n = 0 false
}}
var p = "%var.team/k" false
var q = "%var.player/p%%var.player/ui_n%" false
var q = "%var.player/q% V%L" false
var ui_pack = %var.player/q% false
if (var ui_pack == "L") {{
    var ui_pack = 0 false
}}
teamvar ui_p V = "" false
var t = %var.player/ui_n% false
'''
    for _ in range(depth):
        keys += '''if () {
    var m = %var.player/t% false
    var t /= 27 false
    var u = %var.player/t% false
    var u *= 27 false
    var m -= %var.player/u% false
}
if (var m > 0) {
    var p = "%var.global/ui_c" false
    var q = "%var.player/p%%var.player/m%%" false
    var r = %var.player/q% false
    teamvar ui_p V = "%var.player/r%%var.team/ui_p V%" false
}
'''
    keys += 'function "Chat Show"'
    function("Chat Keys", keys)

    def lookup(target):
        return f'''if (var bi < 460, var bi > 0) {{
    var p = "%var.global/w" false
    var q = "%var.player/p%%var.player/bi%%" false
    {target} = %var.player/q% false
}}
if (var bi >= 460) {{
    var p = "%var.team/w" false
    var q = "%var.player/p%%var.player/bi%" false
    var q = "%var.player/q% V%" false
    {target} = %var.player/q% false
}}
'''

    show = ready_guard + "\n"
    for i in range(K):
        show += f'''var bi = %var.player/ui_pack% false
var bi >>= {BITS * i} false
var bi &= {(1 << BITS) - 1} false
if (var bi == 0) {{
    teamvar ui_s{i} V = "-" false
}}
''' + lookup(f"teamvar ui_s{i} V")
    show += f'displayMenu "{ROOT}"'
    function("Chat Show", show)

    pick = f'''{ready_guard}
if (var pos >= {max_input + 2}) {{
    chat "[LLM] Question full. Click Send or New Chat."
    exit
}}
if (var ui_key >= 0) {{
    var tok = %var.player/ui_pack% false
    var ui_key *= {BITS} false
    var tok >>= %var.player/ui_key% false
    var tok &= {(1 << BITS) - 1} false
}}
if (var tok == 0) {{
    chat "[LLM] No word in that slot. Try Backspace or Clear letters."
    exit
}}
var bi = %var.player/tok% false
''' + lookup("var r") + '''chat "[LLM] Added: %var.player/r%"
var ui_phase = 3 false
function "Feed"
'''
    function("Chat Pick", pick)

    slots = []
    # Busy clicks must not consume Chat Keys' cooldown just before the scheduler
    # uses it to refresh the menu after finishing a token.
    key_guard = 'if (!var ui_phase == 0 5) {\n    exit\n}\n'
    # Callan: never generate custom player heads / SkullOwner texture NBT.
    # Only Housing's built-in skulls are supported; keep letters as named wool.
    for i, letter in enumerate(string.ascii_lowercase):
        action = write(f"actions/key-{letter}.htsl", key_guard + f'var ui_key = {i+1} false\nfunction "Chat Keys"')
        slots.append(slot(i, "key-" + letter, "§f§l" + letter.upper(), action,
                          material="wool", damage=3))
    for i, value, label, key in [(26, -1, "§eBackspace", "backspace"),
                                  (27, -2, "§eClear letters: §f%var.team/ui_p V%", "clear")]:
        action = write(f"actions/{key}.htsl", key_guard + f'var ui_key = {value} false\nfunction "Chat Keys"')
        slots.append(slot(i, key, label, action, material="arrow"))
    slots += [slot(30, "new", "§b§lNew Chat", new_chat, material="diamond"),
              slot(32, "send", "§a§lSend", send, material="emerald"),
              slot(35, "end", "§c§lEnd Chat", end, material="barrier")]
    punctuation = {}
    for i, word in enumerate(("?", ".", ",", "!", "'"), 38):
        token = vocab.index(word)
        # Guard before setting tok: a punctuation click during Run0/Run1 must
        # never change the model input while Chat Pick rejects the busy click.
        action = write(f"actions/token-{token}.htsl", ready_guard + f'\nvar ui_key = -1 false\nvar tok = {token} false\nfunction "Chat Pick"')
        slots.append(slot(i, f"token-{token}", "§f§l" + word, action))
        punctuation[word] = i
    for i in range(K):
        action = write(f"actions/pick-{i}.htsl", f'var ui_key = {i} false\nfunction "Chat Pick"')
        slots.append(slot(46 + i, f"suggestion-{i}", f"§a%var.team/ui_s{i} V%", action, material="book"))
    manifest["menus"].append(dict(name=ROOT, size=6, slots=slots))
    (house / "import.json").write_text(json.dumps(manifest, indent=1) + "\n")
    write("routes.json", json.dumps(dict(root=ROOT, maxInput=max_input, context=CONTEXT,
        routes=routes, new=30, send=32, end=35, backspace=26, clear=27,
        punctuation=punctuation, suggestionStart=46, prefixDepth=depth,
        prefixes=prefixes, version=version, teamVBytes=v_bytes), indent=2))
    print(f'{len(manifest["functions"])} functions, {len(manifest["menus"])} menu; '
          f'{len(prefixes)} prefix entries, {depth} letters max; team V <= {v_bytes} bytes')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house", type=Path, default=Path(__file__).resolve().parents[1] / "house")
    build(parser.parse_args().house)
