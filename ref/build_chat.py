"""Generate the click-to-type UI without changing any compiled model functions.

    python3 ref/build_chat.py [--house house]

Vocabulary order is frequency order with pinned evaluation words at the end
(train/train.py:build_vocab). Keep that order inside each prefix menu.
Only house/chat and the UI entries in import.json are owned by this script.
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
    assert not any(x["event"] in ("Player Join", "Player Quit") for x in manifest["events"]), \
        "Merge the existing Join/Quit actions instead of replacing them."
    assert len(manifest["functions"]) + len(FUNCTIONS) <= 300
    out = house / "chat"
    if out.exists():
        shutil.rmtree(out)

    def write(name, text):
        file = out / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text.strip() + "\n")
        return "chat/" + name

    def function(name, body):
        manifest["functions"].append(dict(name=name, actions=write(f"fns/{name}.htsl", body)))

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
    function("Feed", f'''// One token takes 10 ticks: five between passes, five before reuse.
{owner_guard}
if (var ui_work == 1 0) {{
    exit
}}
if or (var ui_phase < 1 0, var ui_phase > 4 0) {{
    exit
}}
if (var pos >= {CONTEXT}) {{
    chat "[LLM] Context full. Click New Chat for another question."
    {release_indented}
    exit
}}
var ui_work = 1 false
function "Run0"
pause 5
{owner_guard}
function "Run1"
pause 5
{owner_guard}
var ui_work = 0 false
function "Chat Next"
''')
    function("New Chat", '''if (var ui_work == 1 0) {
    chat "[LLM] Please wait a moment before starting a new question."
    exit
}
if (!globalvar ui_owner == "" "", !globalvar ui_owner == "%player.name%" "") {
    chat "[LLM] Someone else is chatting. Try again when they finish."
    exit
}
if (globalvar ui_owner == "%player.name%" "", var ui_phase > 0 0, var ui_phase < 5 0) {
    chat "[LLM] Please wait for the current word or reply."
    exit
}
globalvar ui_owner = "%player.name%" false
var ui_phase = 1 false
var ui_work = 0 false
var gen = 0 false
var done = 0 false
var ui_n = 0 false
function "Chat Load"
function "Init"
var tok = 2 false
chat "[LLM] Starting a new question..."
function "Feed"
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
    # A delayed call chain, bounded by pos, rather than a synchronous loop.
    # Functions return immediately when a nested function pauses, so all
    # sequencing belongs here, after Feed's second five-tick pause.
    function("Chat Next", f'''{owner_guard}
if (var ui_phase == 1) {{
    var tok = 3 false
    var ui_phase = 2 false
    function "Feed"
    exit
}}
if (var ui_phase == 2) {{
    var done = 0 false
    var ui_phase = 0 false
    chat "[LLM] Choose up to {max_input} words or punctuation marks, then Send."
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
    chat "[LLM] Finished. Use /llm and New Chat for another question."
    exit
}}
function "Feed"
''')
    new_chat = write("actions/new.htsl", 'function "New Chat"')
    send = write("actions/send.htsl", '''function "Reply"
if (var ui_phase == 4) {
    closeMenu
}''')
    end = write("actions/end.htsl", f'''{ready_guard}
{release}
chat "[LLM] Chat released. Anyone can start a new question."
closeMenu''')
    back = write("actions/root.htsl", f'displayMenu "{ROOT}"')
    manifest["commands"].append(dict(name="llm", mode="Self", requiredPriority=0,
        listed=True, actions=back))
    manifest["events"].append(dict(event="Player Quit", actions=write("actions/quit.htsl", f'''
if (globalvar ui_owner == "%player.name%" "") {{
    {release_indented}
}}''')))
    # Wait out any paused work from a previous connection before permitting a
    # fresh claim under the same player name. Do not rely on disconnect silently
    # cancelling delayed actions (the harness, for example, does not do that).
    manifest["events"].append(dict(event="Player Join", actions=write("actions/join.htsl", '''
var ui_work = 1 false
pause 10
var ui_work = 0 false
var ui_phase = 5 false''')))

    def item(key, label, lore=(), material="paper"):
        # JSON string quoting also produces valid SNBT string literals.
        return write(f"items/{key}.snbt", '{id:"minecraft:' + material + '",Count:1b,tag:{display:{Name:'
            + json.dumps(label) + ',Lore:' + json.dumps(list(lore)) + '}}}')

    def slot(index, key, label, actions, lore=(), material="paper"):
        return dict(slot=index, nbt=item(key, label, lore, material), actions=actions)

    # All seven 9-bit token IDs fit in one nonnegative 63-bit Long.
    # Dynamic numeric placeholders have commas; L casts undo that formatting.
    loads = [f'teamvar "k{n:,}" V = {value} false' for n, value in packed.items()]
    loads += [f'globalvar ui_c{i} = "{c}" false' for i, c in enumerate(string.ascii_lowercase, 1)]
    loader = [f"if (globalvar ui_rev == {version} 0) {{", "    exit", "}"]
    for offset in range(0, len(loads), 25):
        loader += ["if () {"] + ["    " + line for line in loads[offset:offset + 25]] + ["}"]
    loader.append(f"globalvar ui_rev = {version} false")
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
    show += 'chat "[LLM] Letters: %var.team/ui_p V%"\n'
    show += 'chat "[LLM] ' + ' | '.join(f'{i+1}: %var.team/ui_s{i} V%' for i in range(K)) + '"\n'
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
    for i, letter in enumerate(string.ascii_lowercase):
        action = write(f"actions/key-{letter}.htsl", f'var ui_key = {i+1} false\nfunction "Chat Keys"')
        slots.append(slot(i, "key-" + letter, letter.upper(), action, ["Add a letter to narrow the suggestions."]))
    for i, value, label, key in [(27, -1, "Backspace", "backspace"), (28, -2, "Clear letters", "clear")]:
        action = write(f"actions/{key}.htsl", f'var ui_key = {value} false\nfunction "Chat Keys"')
        slots.append(slot(i, key, label, action, ["Changes the letters, not words already added."], "arrow"))
    slots += [slot(30, "new", "New Chat", new_chat, material="book"),
              slot(31, "send", "Send", send, material="emerald"),
              slot(32, "end", "End Chat", end, material="barrier")]
    punctuation = {}
    for i, word in enumerate(("?", ".", ",", "!", "'", "<unk>"), 36):
        token = vocab.index(word)
        # Guard before setting tok: a punctuation click during Run0/Run1 must
        # never change the model input while Chat Pick rejects the busy click.
        action = write(f"actions/token-{token}.htsl", ready_guard + f'\nvar ui_key = -1 false\nvar tok = {token} false\nfunction "Chat Pick"')
        slots.append(slot(i, f"token-{token}", word, action))
        punctuation[word] = i
    for i in range(K):
        action = write(f"actions/pick-{i}.htsl", f'var ui_key = {i} false\nfunction "Chat Pick"')
        slots.append(slot(45 + i, f"suggestion-{i}", f"{i+1}: %var.team/ui_s{i} V%", action,
            [f"Suggestion {i+1}. The numbered list is also in chat."]))
    manifest["menus"].append(dict(name=ROOT, size=6, slots=slots))
    (house / "import.json").write_text(json.dumps(manifest, indent=1) + "\n")
    write("routes.json", json.dumps(dict(root=ROOT, maxInput=max_input, context=CONTEXT,
        routes=routes, new=30, send=31, end=32, backspace=27, clear=28,
        punctuation=punctuation, suggestionStart=45, prefixDepth=depth,
        prefixes=prefixes, version=version, teamVBytes=v_bytes), indent=2))
    print(f'{len(manifest["functions"])} functions, {len(manifest["menus"])} menu; '
          f'{len(prefixes)} prefix entries, {depth} letters max; team V <= {v_bytes} bytes')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house", type=Path, default=Path(__file__).resolve().parents[1] / "house")
    build(parser.parse_args().house)
