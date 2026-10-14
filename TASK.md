# housing-llm: a chat LLM running as Hypixel Housing code

## Elijah's words (verbatim, 2026-10-02)

> Using the HTSW simulator & all my documentation on Hypiel Housing HTSW try and figure out if its possilbe and the approach that would be possible

> So the computation actually functions? Can you just iteratively test with the HTSW simualtor ? the simulator allows you to do all the computation right? can you just iteratively work on this and see if its possible to do it, get it fully working in simulator. Assume some input varaibles, you dont need to build the player UX, juts prove hte computation is possible from some preset input you can just hardcode yourself, ideally a set of variables or something whatever is most computationally easy

> Can we get some like distilled one that can answer questions tho? Find a balance between speed & actually. being interactable

> We can train it on my maxxed out m5 right? its pretty powerfu land we can just do it right

> Ok go

Housing facts from Elijah (2026-10-02): function limit is 300; if and else branches each hold 25 Change Variable; string vars keep leading spaces; chat message holds ~256 chars; L/D suffix typecasting works.

## Goal and done checks

Train a tiny word-level chat model on the M5, compile it to HTSL, prove in the HTSW simulator that a hardcoded question produces the same reply as the Python integer reference.

1. Simulator reply == integer reference, token for token, on >= 10 prompts.
2. Integer reference top-1 == float model top-1 on >= 95% of positions.
3. Generated code passes `htsw check` (300 functions, per-container limits).

Must preserve: htsw-repo and htsw-simulator are not edited. All work lives in `~/Desktop/HTSW/housing-llm/`. Never touch live Housing or the Minecraft client.

## Architecture contract (shared by training, integer reference, compiler)

Chosen so every op is cheap in Housing integer math:

- Decoder-only transformer, pre-norm, residual.
- RMSNorm with learned gain (no bias), eps 1e-5.
- Multi-head attention, n_kv_heads == n_heads, no biases, causal.
- Positions: **ALiBi** (no RoPE, no learned position table). Standard ALiBi slopes for n_heads.
- MLP: **non-gated ReLU**, `W2 · relu(W1 · x)`, no biases.
- Tied input embedding and output head. No final bias.
- Greedy decoding (argmax) for evaluation.
- Tokenizer: word level, lowercase. Punctuation `. , ? ! '` are separate tokens. Specials: `<pad>=0 <unk>=1 <bos>=2 <user>=3 <bot>=4 <eos>=5`.
- Prompt format: `<bos> <user> do you like cats ? <bot> yes , i like cats . <eos>`.
- Context length 64.

### Housing compute budget (what the model must fit)

Per token cost in "MAC actions" ~ 1 action per weight multiply-add.
Weight capacity ~ 1,200 MACs per Housing function; 300 functions total.
Reserve ceil(V/25) functions for the embedding lookup and ~35 for attention/norm/softmax/argmax.

    L * (4*d*d + 2*d*f) + V*d  <=  (300 - 35 - ceil(V/25)) * 1200

Examples that fit: V=512,d=96,L=2,f=384 (~270K); V=768,d=80,L=3,f=256 (~260K).

## Exported model format

`runs/<name>/model.npz` with float32 arrays:
`tok_emb [V,d]`, per layer i: `l{i}.attn_norm [d]`, `l{i}.wq [d,d]`, `l{i}.wk [d,d]`, `l{i}.wv [d,d]`, `l{i}.wo [d,d]`, `l{i}.mlp_norm [d]`, `l{i}.w1 [f,d]`, `l{i}.w2 [d,f]`; `final_norm [d]`. Weight layout is `[out, in]` (PyTorch Linear).
`runs/<name>/config.json`: V, d, L, n_heads, f, ctx, alibi slopes, val loss, param count.
`runs/<name>/vocab.json`: id -> word list.
`runs/<name>/samples.txt`: greedy replies to 20 fixed questions.

## Ledger

- 2026-10-02: spec written. Helper A (training), helper B (simulator scale test) dispatched.
- 2026-10-02: storage math. KV cache is stored as 3 x 21-bit lanes per Long (8 B per 3 values). With MHA d=96 L=2 that is 1,024 B per context token, so ctx ~24 fits in ~16 holders x 2.5 KB. Multi-query attention (n_kv_heads=1) cuts it 4x (ctx 64 ~ 17 KB). Plan: if helper A's runs are MHA, ask for one MQA retrain after it finishes (never message it mid-run).
- 2026-10-02: Integer design (ref/intmodel.py is the spec): int8 weights per-row scale, RMSNorm gain folded into following matmul, normalized activations int8 via Q16 residual / Newton isqrt, matmuls as exact int dot products (SWAR lanes are exact when per-row bounds fit 21 bits; compiler asserts and chunk-flushes), exp by 10 squarings in Q30, attention probs as 0..255 ints, branchless argmax (first max wins).
- 2026-10-02: helper B (agent 4dfdfd3b) done, scaletest/REPORT.md. 302 fns / 360K Change Var: check 1.58 s, load 1.54 s, ~1.4M actions/s, 192/192 accumulators match BigInt. Rules for compiler: bare operands only (`var t *= %var.player/x%`; quoted operands silently skip in sim), no L suffix on literals, compiler must enforce 300 fns itself (check doesn't), dispatcher fan-out max 260 per fn. Run harness with vite-node.
- 2026-10-02: MILESTONE. Compiler (ref/compile.py) + runner (sim/run.mjs). Random-weight model at target size (V=512 d=96 L=2 H=4 KVH=1 f=384, ctx 24): 271 functions, 308,940 stored / 301,516 executed actions per token, 2,185 vars (17.5 KB), `htsw check` OK, simulator == integer reference on 13/13 steps (argmax + all 96 residual values), ~0.9 s/token in sim.
  Bugs found on the way: (1) unset var in a placeholder reads "" -> every read uses `%var.player/x 0%`; (2) comparisons on unset vars fail (harness models real Housing) -> conditions use always-set copies `ps`, `tk`; (3) var names are case-insensitive -> compiler asserts no clash. Simulator Long ops fuzzed 396/399 exact (misses are shift >= 64, never emitted).
- 2026-10-02: storage made Housing-legal: compiler assigns vars to player / global / teams T1..Tn (<= 300 Longs each), team reads have no fallback so `Init` zeroes team vars + KV cache + pos; every action carries explicit `false` (Automatic Unset off). Full random build: 275 fns, holders player/global/T1-T5 300 each + T6 86, check OK, sim == reference.
- 2026-10-02: live throughput probe written + sim-verified: ChiknImportFolder/imports/llm-probe (probe/gen_probe.py). 44 fns, 25,000 Long ops. /lbench = one burst at 6,250 / 12,500 / 18,750 / 25,000 ops with correctness check; /lloop = 5-tick sustained loop reporting effective ticks/s and ops/s. For Elijah to import into a test house; never driven by agents.
- 2026-10-02: FIRST REAL CHAT IN SIM. Trained v512_d96_L2_f384 (MHA, val 1.4128). Integer vs float on 20 questions: teacher-forced top-1 250/262 = 95.4%, identical replies 11/20 (ref/eval_q.py). Compiled at ctx 21: "do you like cats ?" -> "i enjoy them , they make me feel happy and cheerful ." , sim == reference 20/20 steps, check OK, 333K actions/token, 4,268 vars over 15 holders. BUT 306 functions (> 300) -> built with ALLOW_OVER_300=1. In-limits target = MQA run (training).
- 2026-10-02: DONE. Final = runs/v512_d96_L2_f384_mqa (MQA, val 1.4341) compiled at ctx 48 -> build/final: 290 functions, 326,953 stored / 318,761 executed actions per token, 3,002 vars over player/global/T1-T9 (<= 300 each), `htsw check` OK. Simulator == integer reference on 20/20 conversations (406 tokens, every residual value), 263 ms/token in sim (build/final/sim-run.log). Integer vs float: 95.9% teacher-forced top-1 on 100 TinyChat questions (eval100.txt), calibrated on separate calib.txt. Fixes on the way: norm rounding (was truncation), harness transcript cleared per token (OOM otherwise).
- 2026-10-02: dynamic-read output (Cacti's "ROM tech" point). Out = computed-key read %var.global/w<bi>% (ids < 460) / two-step %var.team/w<bi> V% (rest); Vocab sets 506 strings (2,716 B, split global + team V); global holds no Longs now (T10 added). House build 292 fns; sim demo 295 fns (Say_* removed). 20/20 conversations bit-exact, 224 generated words looked up correctly (5 via team V). Embedding stays as code conditionals: as vars it would be 512 x 12 Longs = 49 KB.
