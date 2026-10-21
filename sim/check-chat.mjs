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
const get = (p, k) => p.state.vars.get(k)?.value.toString() ?? "";
const owner = () => sim.vars.global.get("ui_owner")?.toString() ?? "";
const added = p => p.chatMessages().filter(s => s.startsWith("[LLM] Added: "));
const passes = [];
const batches = [];
const modelPasses = JSON.parse(fs.readFileSync(path.join(dir, "model-passes.json")));
const modelFunctions = new Set(Object.values(modelPasses).flat());
const mathCalls = [];
let tickActions = 0, lastTick = -1, maxTickActions = 0;
// Count actual work without retaining millions of arithmetic transcript entries.
sim.transcript.append = e => {
    if (e.tick !== lastTick) { tickActions = 0; lastTick = e.tick; }
    maxTickActions = Math.max(maxTickActions, ++tickActions);
    if (e.action.type !== "FUNCTION") return;
    const fn = e.action.function;
    const p = e.player === "Alice" ? alice : bob;
    if (modelFunctions.has(fn)) mathCalls.push(fn);
    if (["Run0", "Run1"].includes(fn) && batches.at(-1)?.tick !== e.tick) {
        const batch = { tick: e.tick, player: e.player, fn, tok: Number(get(p, "tok")), part: Number(get(p, "ui_part")) };
        batches.push(batch);
        if (batch.part === 0) passes.push(batch);
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
function waitReady(p, max = 2000) {
    for (let i = 0; i < max && get(p, "ui_phase") !== "0"; i++) advance(1);
    assert.equal(get(p, "ui_phase"), "0", p.chatMessages().join("\n"));
    assert.equal(p.openMenu, ui.root);
}

assert.equal(manifest.functions.length, 300);
assert.equal(manifest.menus.length, 1);
assert.equal(manifest.menus[0].name, "LLM Keypad");
assert.equal(manifest.menus[0].slots.length, 43);
assert.equal(manifest.events, undefined); // Existing house events must stay untouched.
assert.equal(sim.findFunctionImportable("Chat Next").repeatTicks, 5);
for (let i = 0; i < 26; i++) {
    const slot = manifest.menus[0].slots.find(s => s.slot === i);
    const nbt = fs.readFileSync(path.join(dir, slot.nbt), "utf8");
    assert(nbt.includes('minecraft:wool') && !nbt.includes('SkullOwner') && nbt.includes('Damage:3s'));
    assert(!nbt.includes('Lore'));
}
assert(!("<unk>" in ui.punctuation));
const teamValue = name => sim.vars.team("V").get(name)?.value.toString() ?? "";
function assertSuggestions(prefix) {
    const ids = ui.prefixes[prefix] ?? [];
    for (let i = 0; i < 7; i++) assert.equal(teamValue(`ui_s${i}`), ids[i] ? vocab[ids[i]] : "-", `${prefix}, slot ${i}`);
    assert.equal(teamValue("ui_p"), prefix);
}

open(alice);
alice.clickMenuSlot(ui.send); // Must require New Chat.
assert.equal(passes.length, 0);
alice.clickMenuSlot(ui.new);
assert.equal(owner(), "Alice");
open(bob);
assert.equal(owner(), "Alice");
bob.clickMenuSlot(7); // Another player's letter/pick must not touch model state.
bob.clickMenuSlot(ui.suggestionStart);
assert.equal(passes.length, 0);
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
console.log("PASS: all 228 prefix lookups and 501 words, suggestion word variables, backspace and empty slots");

for (const word of ["do", "you", "like", "cats", "?"]) {
    clickWord(alice, word);
    const before = added(alice).length;
    const tokBefore = get(alice, "tok");
    alice.clickMenuSlot(7); // Letter and punctuation during Feed must be ignored.
    alice.clickMenuSlot(ui.punctuation["!"]);
    alice.clickMenuSlot(ui.suggestionStart);
    assert.equal(get(alice, "tok"), tokBefore);
    open(alice);
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
for (let t = 0; t < 10000 && get(alice, "ui_phase") !== "5"; t++) advance(1);
const reply = alice.chatMessages().slice(start).filter(x => !x.startsWith("[LLM]")).join(" ");
assert.equal(reply, "i enjoy them , they make me feel happy .");
assert.equal(get(alice, "done"), "1");
assert.equal(owner(), "");
assert.deepEqual(passes.filter(p => p.fn === "Run0").slice(0, 8).map(p => p.tok), [2, 3, 21, 18, 32, 502, 25, 4]);
for (let i = 0; i < passes.length; i += 2) {
    assert.equal(passes[i].fn, "Run0");
    assert.equal(passes[i + 1].fn, "Run1");
    assert(passes[i + 1].tick - passes[i].tick >= 5);
    if (i + 2 < passes.length) assert(passes[i + 2].tick - passes[i + 1].tick >= 5);
}
for (let i = 1; i < batches.length; i++) assert(batches[i].tick - batches[i - 1].tick >= 5);
const expectedCalls = passes.flatMap(p => modelPasses[p.fn]);
assert.deepEqual(mathCalls, expectedCalls); // Exact original compiler order, no skipped/repeated math.
assert(maxTickActions < 3500, `Batch performed ${maxTickActions} actions`);
console.log(`PASS: model call order, five-tick spacing, maximum ${maxTickActions} actions/tick`);
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
for (let t = 0; t < 1200 && get(alice, "ui_phase") !== "5"; t++) advance(1);
out.actions = originalOut;
assert.equal(get(alice, "pos"), String(ui.context));
assert.equal(get(alice, "done"), "0");
assert.equal(get(alice, "ui_phase"), "5");
assert.equal(owner(), "");
assert(alice.chatMessages().includes("[LLM] Reply stopped at the context limit."));
const stopped = passes.length;
advance(20);
assert.equal(passes.length, stopped);

// End/New must recover at every stage without Join/Quit actions or delayed work.
open(alice);
alice.clickMenuSlot(ui.new);
advance(5);
assert.equal(get(alice, "ui_phase"), "6");
alice.clickMenuSlot(ui.end); // Cancel loading.
assert.equal(owner(), "");
const loadPart = get(alice, "ui_part");
advance(20);
assert.equal(get(alice, "ui_part"), loadPart);
open(alice);
alice.clickMenuSlot(ui.new);
for (let t = 0; t < 300 && get(alice, "ui_work") !== "1"; t++) advance(1);
advance(10); // Partway through BOS first pass.
assert.equal(get(alice, "pos"), "0");
assert(Number(get(alice, "ui_part")) > 0);
alice.clickMenuSlot(ui.new); // Restart while computing; resets pass index immediately.
assert.equal(get(alice, "ui_part"), "0");
waitReady(alice);
assert.equal(get(alice, "pos"), "2");

// Model an aborted batch by returning before its success marker is written.
const run0 = sim.findFunctionImportable("Run0");
const originalRun0 = run0.actions;
run0.actions = [];
clickWord(alice, "hello");
advance(5);
assert.equal(get(alice, "ui_work"), "-1");
advance(5);
run0.actions = originalRun0;
assert.equal(owner(), "");
assert.equal(get(alice, "ui_phase"), "5");
assert(alice.chatMessages().some(s => s.includes("Calculation stopped")));
open(alice);
alice.clickMenuSlot(ui.new);
waitReady(alice);
assert.equal(get(alice, "pos"), "2");
clickWord(alice, "hello");
advance(10);
alice.disconnect();
open(bob);
bob.clickMenuSlot(ui.end); // Another player can release an absent/stuck owner.
assert.equal(owner(), "");
open(bob);
bob.clickMenuSlot(ui.new);
alice.reconnect();
waitReady(bob);
assert.equal(owner(), "Bob");
assert.equal(get(bob, "pos"), "2");
open(bob);
bob.clickMenuSlot(ui.end);
assert.equal(owner(), "");
const stoppedAtEnd = batches.length;
advance(20);
assert.equal(batches.length, stoppedAtEnd);
assert.deepEqual(sim.runtime.diagnostics, []);
console.log("PASS: input/context caps; End/New during load and calculation; aborted batch recovery; absent owner replacement without events");
