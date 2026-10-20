// Exercise real imported menu actions and the complete model with cooldowns.
// /home/opc/heavy-lock.sh node sim/check-chat.mjs [house]
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { Simulation } from "../build/tools/simulator.mjs";

const dir = path.resolve(process.argv[2] ?? "house");
const ui = JSON.parse(fs.readFileSync(path.join(dir, "chat/routes.json")));
const vocab = JSON.parse(fs.readFileSync(path.join(dir, "vocab.json")));
const manifest = JSON.parse(fs.readFileSync(path.join(dir, "import.json")));
const sim = new Simulation({ importPath: path.join(dir, "import.json"), seed: 1, autoFireJoin: false });
assert.deepEqual(sim.diagnostics.filter(d => d.level === "error" || d.level === "bug"), []);
for (const name of [...Array.from({ length: 10 }, (_, i) => `T${i + 1}`), "V"]) sim.createTeam({ name });
const alice = sim.addPlayer({ name: "Alice" });
const bob = sim.addPlayer({ name: "Bob" });
const get = (p, k) => p.state.vars.get(k)?.toString() ?? "";
const owner = () => sim.vars.global.get("ui_owner")?.toString() ?? "";
const added = p => p.chatMessages().filter(s => s.startsWith("[LLM] Added: "));
const passes = [];
// Keep only pass boundaries: logging 300K arithmetic actions/token is wasteful.
sim.transcript.append = e => {
    if (e.action.type === "FUNCTION" && ["Run0", "Run1"].includes(e.action.function)) {
        const p = e.player === "Alice" ? alice : bob;
        passes.push({ tick: e.tick, player: e.player, fn: e.action.function, tok: Number(get(p, "tok")) });
    }
};
const advance = n => { for (let i = 0; i < n; i++) sim.advance(1); };
function open(p) {
    const command = sim.importables.find(x => x.type === "COMMAND" && x.name === "llm");
    assert(command);
    sim.withPlayer(p.state, () => sim.runtime.runActions(command.actions));
    assert.equal(p.openMenu, ui.root);
}
function clickWord(p, word) {
    open(p);
    if (word in ui.punctuation) {
        p.clickMenuSlot(ui.punctuation[word]);
        return;
    }
    p.clickMenuSlot(ui.clear);
    advance(4);
    const route = ui.routes[vocab.indexOf(word)];
    for (const letter of route.prefix) {
        p.clickMenuSlot(letter.charCodeAt(0) - 97);
        advance(4);
    }
    p.clickMenuSlot(ui.suggestionStart + route.suggestion);
}
function waitReady(p, max = 25) {
    for (let i = 0; i < max && get(p, "ui_phase") !== "0"; i++) advance(1);
    assert.equal(get(p, "ui_phase"), "0", p.chatMessages().join("\n"));
    assert.equal(p.openMenu, ui.root);
}

assert.equal(manifest.functions.length, 300);
assert.equal(manifest.menus.length, 1);
assert.equal(manifest.menus[0].name, "LLM Keypad");
assert.equal(manifest.menus[0].slots.length, 44);
for (let i = 0; i < 26; i++) assert(manifest.menus[0].slots.some(s => s.slot === i));
const teamValue = name => sim.vars.team("V").get(name)?.value.toString() ?? "";
function assertSuggestions(prefix) {
    const ids = ui.prefixes[prefix] ?? [];
    for (let i = 0; i < 7; i++) assert.equal(teamValue(`ui_s${i}`), ids[i] ? vocab[ids[i]] : "-", `${prefix}, slot ${i}`);
    assert.equal(teamValue("ui_p"), prefix);
    assert(alice.chatMessages().at(-1).includes(`1: ${ids[0] ? vocab[ids[0]] : "-"}`));
}

open(alice);
alice.clickMenuSlot(ui.send); // Must require New Chat.
assert.equal(passes.length, 0);
alice.clickMenuSlot(ui.new);
assert.equal(owner(), "Alice");
open(bob);
bob.clickMenuSlot(ui.new);
assert.equal(owner(), "Alice");
bob.clickMenuSlot(7); // Another player's letter/pick must not touch model state.
bob.clickMenuSlot(ui.suggestionStart);
assert.equal(passes.length, 1);
waitReady(alice);
assert.equal(get(alice, "pos"), "2");
assert.deepEqual(passes.filter(p => p.fn === "Run0").map(p => p.tok), [2, 3]);
alice.clickMenuSlot(ui.send); // Empty question must not start a reply.
assert.equal(get(alice, "ui_phase"), "0");
advance(5); // Reply itself has a four-tick cooldown, even on the empty path.

// The loader runs once; all prefixes use actual letter actions and dynamic reads.
// Validate packed Longs with BigInt, never lossy JavaScript numbers.
for (const [prefix, ids] of Object.entries(ui.prefixes)) {
    let code = 0;
    for (const letter of prefix) code = code * 27 + letter.charCodeAt(0) - 96;
    const packed = ids.reduce((value, id, i) => value | (BigInt(id) << BigInt(9 * i)), 0n);
    assert.equal(teamValue(`k${code.toLocaleString("en-US")}`), String(packed));
    alice.clickMenuSlot(ui.clear); advance(4);
    for (const letter of prefix) {
        alice.clickMenuSlot(letter.charCodeAt(0) - 97); advance(4);
    }
    assertSuggestions(prefix);
}
for (const [token, route] of Object.entries(ui.routes)) assert.equal(ui.prefixes[route.prefix][route.suggestion], Number(token));
assert.equal(Object.keys(ui.routes).length, 501);
assert.equal(teamValue("k0"), String(ui.prefixes[""].reduce((n, id, i) => n | BigInt(id) << BigInt(9 * i), 0n)));
const beforeReload = teamValue("k0");
alice.runFunction("Chat Load");
assert.equal(teamValue("k0"), beforeReload);
alice.clickMenuSlot(ui.clear); advance(4);
alice.clickMenuSlot(22); advance(4); // w
alice.clickMenuSlot(0); advance(4); // wa
assertSuggestions("wa");
alice.clickMenuSlot(ui.backspace); advance(4);
assertSuggestions("w");
alice.clickMenuSlot(ui.clear); advance(4);
alice.clickMenuSlot(25); advance(4); // z: no words, no stale suggestions.
assertSuggestions("z");
const emptyPasses = passes.length;
alice.clickMenuSlot(ui.suggestionStart); advance(4);
assert.equal(passes.length, emptyPasses);
assert.equal(added(alice).length, 0);
alice.clickMenuSlot(ui.backspace); advance(4);
assertSuggestions("");
console.log("PASS: all 228 prefix lookups and 501 words, suggestion word variables, numbered chat fallback, backspace and empty slots");

for (const word of ["do", "you", "like", "cats", "?"]) {
    clickWord(alice, word);
    const before = added(alice).length;
    const tokBefore = get(alice, "tok");
    alice.clickMenuSlot(7); // Letter and punctuation during Feed must be ignored.
    alice.clickMenuSlot(ui.punctuation["!"]);
    alice.clickMenuSlot(ui.suggestionStart);
    assert.equal(get(alice, "tok"), tokBefore);
    open(alice);
    alice.clickMenuSlot(ui.new);
    alice.clickMenuSlot(ui.send);
    assert.equal(added(alice).length, before);
    waitReady(alice);
}
assert.deepEqual(added(alice), ["do", "you", "like", "cats", "?"].map(w => "[LLM] Added: " + w));
assert.equal(get(alice, "pos"), "7");

alice.setVar("done", 1); // Ignore an EOS prediction made while feeding the question.
const start = alice.chatMessages().length;
alice.clickMenuSlot(ui.send);
assert.equal(get(alice, "done"), "0");
assert.equal(get(alice, "ui_phase"), "4");
assert.equal(alice.openMenu, undefined);
for (let t = 0; t < 500 && get(alice, "ui_phase") !== "5"; t++) advance(1);
const reply = alice.chatMessages().slice(start).filter(x => !x.startsWith("[LLM]")).join(" ");
assert.equal(reply, "i enjoy them , they make me feel happy .");
assert.equal(get(alice, "done"), "1");
assert.equal(owner(), "");
assert.deepEqual(passes.filter(p => p.fn === "Run0").slice(0, 8).map(p => p.tok), [2, 3, 21, 18, 32, 502, 25, 4]);
for (let i = 0; i < passes.length; i += 2) {
    assert.equal(passes[i].fn, "Run0");
    assert.equal(passes[i + 1].fn, "Run1");
    // The current harness ticks newly appended schedulers in the same advance
    // iteration, so nested pause 5 can measure as four ticks. Assert the actual
    // Housing cooldown here and the declared five-tick waits below.
    assert(passes[i + 1].tick - passes[i].tick >= 4);
    if (i + 2 < passes.length) assert(passes[i + 2].tick - passes[i + 1].tick >= 4);
}
assert.deepEqual(sim.findFunctionImportable("Feed").actions.filter(a => a.type === "PAUSE").map(a => a.ticks), [5, 5]);
console.log(`PASS: README example via keypad clicks + Feed/Reply: ${reply}`);
console.log("PASS: startup markers, empty Send, duplicate clicks, shared-player lock, pass timing");

// Boundary/control tests seed position near the limit to avoid extra model runs.
open(bob);
bob.clickMenuSlot(ui.new);
waitReady(bob);
bob.setVar("pos", ui.maxInput + 2);
const before = passes.length;
clickWord(bob, "hello");
assert.equal(passes.length, before);
assert.equal(added(bob).length, 0);
open(bob);
bob.clickMenuSlot(ui.end);
assert.equal(owner(), "");

open(alice);
alice.clickMenuSlot(ui.new);
waitReady(alice);
alice.setVar("pos", ui.context - 1);
// Force a non-EOS prediction to exercise the hard bound independently of weights.
const out = sim.findFunctionImportable("Out");
const originalOut = out.actions;
out.actions = [];
alice.clickMenuSlot(ui.send);
advance(10);
out.actions = originalOut;
assert.equal(get(alice, "pos"), String(ui.context));
assert.equal(get(alice, "done"), "0");
assert.equal(get(alice, "ui_phase"), "5");
assert.equal(owner(), "");
assert(alice.chatMessages().includes("[LLM] Reply stopped at the context limit."));
const stopped = passes.length;
advance(20);
assert.equal(passes.length, stopped);

open(alice);
alice.clickMenuSlot(ui.new);
assert.equal(owner(), "Alice");
alice.disconnect(); // Disconnect halfway through Feed; guards must block old work.
assert.equal(owner(), "");
alice.reconnect();
open(alice);
alice.clickMenuSlot(ui.new); // Cannot reclaim under the same name until old work drains.
assert.equal(owner(), "");
open(bob);
bob.clickMenuSlot(ui.new);
waitReady(bob);
assert.equal(owner(), "Bob");
assert.equal(get(bob, "pos"), "2");
open(bob);
bob.clickMenuSlot(ui.end);
assert.equal(owner(), "");
assert.deepEqual(sim.runtime.diagnostics, []);
console.log("PASS: input cap, End Chat, 48-token hard stop, disconnect during Feed and next owner");
