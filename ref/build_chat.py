"""Generate the click-to-type UI without changing any compiled model functions.

    python3 ref/build_chat.py [--house house]

Vocabulary order is frequency order with pinned evaluation words at the end
(train/train.py:build_vocab). Keep that order inside each prefix menu.
Only house/chat and the UI entries in import.json are owned by this script.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

CAP = 40
CONTEXT = 48
MAX_INPUT = 16
ROOT = "LLM Words"
SYMBOLS = "LLM Symbols"
FUNCTIONS = ("Feed", "Reply", "New Chat", "Chat Next")


def build(house):
    vocab = json.loads((house / "vocab.json").read_text())
    assert vocab[:6] == ["<pad>", "<unk>", "<bos>", "<user>", "<bot>", "<eos>"]
    assert len(set(vocab)) == len(vocab)
    # Leave room for the prefix in the single-message sentence preview.
    max_input = min(MAX_INPUT, 240 // (max(map(len, vocab)) + 1))
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
teamvar ui_q V = "" false
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
    displayMenu "{ROOT}"
    exit
}}
if (var ui_phase == 3) {{
    var done = 0 false
    var ui_phase = 0 false
    chat "[LLM] You:%var.team/ui_q V%"
    displayMenu "{ROOT}"
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
    symbols = write("actions/symbols.htsl", f'displayMenu "{SYMBOLS}"')
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

    def menu(name, slots):
        assert len({s["slot"] for s in slots}) == len(slots)
        assert all(0 <= s["slot"] < 54 for s in slots)
        manifest["menus"].append(dict(name=name, size=6, slots=slots))

    def word_slot(index, token):
        word = vocab[token]
        actions = write(f"words/{token}.htsl", f'''{ready_guard}
if (var pos >= {max_input + 2}) {{
    chat "[LLM] Question full. Click Send or New Chat."
    displayMenu "{ROOT}"
    exit
}}
var tok = {token} false
var ui_phase = 3 false
teamvar ui_q V = "%var.team/ui_q V% {word}" false
function "Feed"''')
        return slot(index, f"word-{token}", word, actions, ["Click to add this word."])

    routes = {}
    def prefix_menu(prefix, tokens, trail):
        name = "LLM " + prefix
        link = write(f"actions/prefix-{prefix}.htsl", f'displayMenu "{name}"')
        slots = []
        if len(tokens) <= CAP:
            for i, token in enumerate(tokens):
                slots.append(word_slot(i, token))
                routes[str(token)] = trail + [i]
        else:
            # Exact-prefix words (e.g. 's') remain selectable at this level.
            exact = [t for t in tokens if vocab[t] == prefix]
            for token in exact:
                i = len(slots)
                slots.append(word_slot(i, token))
                routes[str(token)] = trail + [i]
            letters = sorted({vocab[t][len(prefix)] for t in tokens if len(vocab[t]) > len(prefix)})
            for letter in letters:
                child = prefix + letter
                i = len(slots)
                action = prefix_menu(child, [t for t in tokens if vocab[t].startswith(child)], trail + [i])
                slots.append(slot(i, "prefix-" + child, child.upper(), action, ["Choose a word starting with " + child]))
        slots.append(slot(45, "back", "Back to letters", back, material="arrow"))
        menu(name, slots)
        return link

    words = [i for i, w in enumerate(vocab) if re.fullmatch("[a-z]+", w)]
    root_slots = []
    for i, letter in enumerate(sorted({vocab[t][0] for t in words})):
        link = prefix_menu(letter, [t for t in words if vocab[t].startswith(letter)], [i])
        root_slots.append(slot(i, "letter-" + letter, letter.upper(), link))
    for i, symbol in enumerate(("?", ".", ","), 36):
        root_slots.append(word_slot(i, vocab.index(symbol)))
        routes[str(vocab.index(symbol))] = [i]
    root_slots += [
        slot(40, "symbols", "Symbols and special tokens", symbols),
        slot(45, "new", "New Chat", new_chat, ["Start a fresh question."], "book"),
        slot(49, "send", "Send", send, ["The answer prints automatically in chat."], "emerald"),
        slot(53, "end", "End Chat", end, ["Release the model without sending."], "barrier"),
    ]
    menu(ROOT, root_slots)
    symbol_slots = []
    for token, word in enumerate(vocab):
        if re.fullmatch("[a-z]+", word):
            continue
        i = len(symbol_slots)
        if token in (0, 2, 3, 4, 5):
            # The reserved markers must not be injected into the question.
            explanation = {0: "Padding is internal and is not part of a question.",
                2: "New Chat adds <bos> automatically.", 3: "New Chat adds <user> automatically.",
                4: "Send adds <bot> automatically.", 5: "The reply ends automatically at <eos>."}[token]
            action = write(f"actions/special-{token}.htsl", f'chat "[LLM] {explanation}"')
            symbol_slots.append(slot(i, f"special-{token}", word + " (automatic)", action, [explanation]))
        else:
            symbol_slots.append(word_slot(i, token))
            routes.setdefault(str(token), [40, i])
    assert len(symbol_slots) <= CAP, "Add paging for the non-word tokens in this vocabulary."
    symbol_slots.append(slot(45, "back", "Back to letters", back, material="arrow"))
    menu(SYMBOLS, symbol_slots)
    assert len(manifest["menus"]) <= 50, "Menu budget exceeded; use larger pages or fewer prefixes."
    (house / "import.json").write_text(json.dumps(manifest, indent=1) + "\n")
    write("routes.json", json.dumps(dict(root=ROOT, maxInput=max_input, context=CONTEXT,
        routes=routes, new=45, send=49, end=53), indent=2))
    print(f'{len(manifest["functions"])} functions, {len(manifest["menus"])} menus; '
          f'{len(words)} words, {max_input}-token question limit')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house", type=Path, default=Path(__file__).resolve().parents[1] / "house")
    build(parser.parse_args().house)
