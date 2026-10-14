// Run a compiled housing-llm import in htsw-harness, token by token, and compare
// every step (argmax token + full residual vector) with the integer reference.
//   vite-node sim/run.mjs build/<name>
import fs from "node:fs";
import path from "node:path";
import { Simulation } from "/Users/elijah/Desktop/HTSW/htsw-simulator/dist/index.js";

const dir = path.resolve(process.argv[2]);
const exp = JSON.parse(fs.readFileSync(path.join(dir, "expected.json"), "utf8"));
const vocab = fs.existsSync(path.join(dir, "vocab.json")) ? JSON.parse(fs.readFileSync(path.join(dir, "vocab.json"), "utf8")) : null;
const word = (t) => (vocab ? vocab[t] : `w${t}`);

let t0 = performance.now();
const sim = new Simulation({ importPath: path.join(dir, "import.json"), seed: 1, autoFireJoin: false });
const errs = sim.diagnostics.filter((d) => d.level === "error" || d.level === "bug");
console.log(`load ${(performance.now() - t0).toFixed(0)} ms, ${sim.importables.length} functions, ${errs.length} errors`);
for (const d of errs.slice(0, 5)) console.log("  ", d.message);
if (errs.length) process.exit(1);

for (const t of exp.teams ?? []) sim.createTeam({ name: t });
const p = sim.addPlayer({ name: "Bot" });
const get = (k) => { const v = p.state.vars.get(k); return v ? v.value.toString() : "0"; };
let ok = true, ms = 0, ntok = 0, convOk = 0;
const convs = exp.convs ?? [exp.steps];
for (const steps of convs) {
    p.runFunction("Init"); sim.advance(5);
    let good = true, out = [], q = [];
    steps.forEach((s, i) => {
        if (s.prompt) { p.setVar("tok", s.tok); if (i > 1) q.push(word(s.tok)); }
        const genNow = !s.prompt || i === steps.findLastIndex((x) => x.prompt);
        p.setVar("gen", genNow ? 1 : 0);
        const t1 = performance.now();
        p.runFunction("Run0"); sim.advance(5);
        p.runFunction("Run1"); sim.advance(5);
        ms += performance.now() - t1; ntok++;
        sim.transcript.clear();   // it logs every action; 300K per token would exhaust memory
        const bi = Number(get("bi"));
        const hbad = s.h.filter((v, j) => get(`h${j}`) !== v).length;
        // dynamic-read output: Out must have looked up the word for bi into r
        if (genNow && s.bi > 5 && vocab && get("r") !== vocab[s.bi]) { good = false; console.log(`  step ${i}: printed "${get("r")}" want "${vocab[s.bi]}"`); }
        if (bi !== s.bi || hbad) { good = false; console.log(`  step ${i}: sim=${word(bi)} ref=${word(s.bi)} residual ${s.h.length - hbad}/${s.h.length}`); }
        if (!steps[i + 1]?.prompt && !s.prompt || i === steps.findLastIndex((x) => x.prompt)) out.push(word(bi));
    });
    q.pop();
    console.log(`Q: ${q.join(" ")}\nA: ${out.join(" ")}   [${steps.length} steps ${good ? "MATCH" : "MISMATCH"}]`);
    ok &&= good; convOk += good;
}
console.log(`${convOk}/${convs.length} conversations bit-exact`);
const exp_steps = { length: ntok };
console.log(`${(ms / ntok).toFixed(0)} ms per token in simulator`);
console.log(ok ? "RESULT: SIM == REFERENCE" : "RESULT: MISMATCH");
process.exit(ok ? 0 : 1);
