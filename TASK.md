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

## Keypad redesign acceptance (2026-10-20)

Replace the 44-menu UI with one menu: 26 letters, prefix backspace/clear,
seven frequency-ordered suggestions, punctuation, New Chat/Send/End Chat.
Generate a compact prefix lookup loader from vocab.json; keep the 292 model
functions unchanged and reuse the timing/reply wrappers. The full import must
pass htsw check and keypad clicks for the README example must reproduce its
reply. Report UI action counts and obsolete menu names. Menu-name placeholder
rendering requires an in-game check; include numbered chat suggestions as the
fallback. Work stays on feat/chat-ui, pushed without merge or live import.

## Live failure fix acceptance (2026-10-20)

IMPORT reproduced the error with Run0 alone, after Init alone succeeded.
Split both model passes into small scheduled batches, preserve their exact math
call order, and make New Chat/End Chat cancel any phase without Join/Quit events.
Keep 300 functions / one menu; add A-Z letter heads, word-only suggestions,
remove <unk> and unnecessary item subtitles. Check full HTSL and README reply,
including mid-load/mid-pass cancellation and simulated abort recovery. Push the
same branch; IMPORT handles live smoke testing and existing event actions.

## Ledger

- 2026-10-21: IMPORT job 7b977600 applied 800cbe8's 19 functions; new names
  survive export. Sixteen match exactly, F161 has 17 unresolved division
  spellings, F193 line 749 and F195 line 523 have substituted assignment actions.
  Requested direct editor inspection before trusting bulk exports; targeted
  repair and re-export if confirmed, followed by audit coverage of all 300
  functions after LG's GO. Added strict read-only ref/audit_live.py pinned to an
  explicit commit, with hashes, missing coverage and division differences kept
  unresolved. Checked against the 19 actual exports: 16 exact, 1 division-only,
  2 other mismatches, 281 missing. No model/importable changes in this update.

- 2026-10-21: IMPORT confirmed Hypixel rejects n19 as a blocked stat key and
  retains Kills; HTSW missed the refusal. Alias n19 -> norm19 in compiler reads
  and writes. Screen all emitted keys with conservative digit substitution;
  also alias n16, n160..n169, n190..n199 as precautions (only n19 confirmed).
  Reject unexpected suspect names and alias collisions. Regenerated 19 functions
  with exact name substitutions only; every other executable artifact, including
  Chat Next and F028, is byte-identical. All 318,761 math actions match compiler
  rendering, holder allocation unchanged, full HTSW check and keypad simulator
  pass with the exact README reply. Document old T4/T8 key cleanup to preserve
  storage headroom. IMPORT handles verification by live export and smoke test;
  F028 division inspection remains pending. Correct the previous export count:
  29 writes became Kills; F161's other 17 differences are division spellings,
  not n19 substitutions. F161 is already included in the name repair.

- 2026-10-20: Remaining live exports confirm all six n19 functions have the
  same substitution (46 writes total); F043/r65 matches exactly. F028 has 16
  division spelling differences, all in phase one: `/=` becomes `…/=`.
  HTSW fde9950d defines no alternate divide mode
  or ellipsis operator. Requested actual Operation field inspection, plus a
  fresh n19 assignment exported before execution. No holder relocation yet;
  verify repaired functions by exporting again before the next fresh chat.

- 2026-10-20: Five-call live batching passed batch 18 but failed batch 19.
  IMPORT isolated F063, then exported it read-only: exactly six writes to T4
  n19 were imported as Kills; remaining source lines match. Static audit counts
  exactly 300 T4 variables, with n19 at allocation position 158, r65 at 300.
  Compiler packing target is 300 Longs / 2,400 bytes, not a verified name cap;
  live extras and import creation order remain unknown. F063 n20..n23 are named
  correctly. All n19 references: F016, F063, F113, F160, F161, F168. Requested
  remaining five read-only exports plus boundary samples F028 (n95), F043 (r65).
  No further math, batching, holder, or Chat Next changes. Arithmetic probes
  were checked locally before the export arrived, then set aside under ignored
  build/f063-arithmetic-probes; no probe import is needed for this comparison.

- 2026-10-20: The ten-call scheduler reached phase 0 batch 8 (F056-F065) live,
  then aborted and recovered correctly. Failed-state read: sm=2^30, mx=-148,
  z0=255, z47=0; the normalization divisor was nonzero. Halve batches to five
  calls, sharing phase-aware selectors within Housing's flat conditional limit.
  Only Run0 / Run1 / Run0_1 change in the import; leave Chat Next and its working
  clock untouched (HTSW currently cannot parse its live repeat-interval lore).
  Now 60+54 calculation batches per token, about 29 seconds/token at 20 TPS.
  Full htsw check and keypad simulator pass, including exact original call
  order, README reply and recovery checks. Maximum measured tick: 3,315 actions
  (previously 6,553). Generator reproduces its output; only the three dispatcher
  bodies differ in the import. The failed range is now batches 18 and 19.

- 2026-10-20: IMPORT job ffe236d6 failed to creative-spawn the custom skull A.
  Replace only the 26 alphabet item payloads with light-blue wool named A–Z;
  keep every slot, click action, function and variable unchanged. Remove the
  unused texture asset/UUID code and update the existing item assertion.
  HTSW's importer validates item ID/damage/count/name after creative injection;
  wool is an ordinary supported item. Use a menu-only retry for this fallback.

- 2026-10-20: IMPORT reproduced the live failure in standalone Run0; Init alone
  succeeds. Replaced only Run0 / Run0_1 / Run1 dispatchers with ten-function
  batches, driven by Chat Next every five ticks. Original 283/257 calls remain
  in exact order; all 289 other model functions are byte-identical. Lookup and
  initialization now load over separate ticks. Feed enqueues work; there are no
  paused continuations. New/End always replace or release the shared owner;
  a missing batch success marker releases it on the next tick. Removed our
  Join/Quit events entirely. Added oak A-Z skull textures, coloured controls,
  word-only suggestions; removed <unk> and lore. Placeholder names were verified
  live by IMPORT. Full htsw check and simulator pass: README reply unchanged,
  all prefix/word routes, original math call order, five-tick batch spacing,
  maximum 6,553 actions in a tested tick, cancellation/restart during loading
  and calculation, aborted-batch recovery and replacement of an absent owner.
  Generator is reproducible. Build remains 300 functions / one 43-button menu /
  one command, no events. About 803 UI actions and 764 dispatcher actions.
  Nominal processing is 14 seconds/token. Exact server ceiling and skull
  appearance need IMPORT's live check; no live imports performed here.

- 2026-10-20: Replaced the cancelled 44-menu import with one 44-button keypad.
  Build now has 300 functions: original 292 unchanged plus Feed, Reply, New Chat,
  Chat Next, Chat Load, Chat Keys, Chat Show and Chat Pick. The generated lookup
  has 228 packed Longs for seven suggestions per prefix, at most three letters;
  team V's new total is bounded at 2,241 bytes. Numeric dynamic keys include
  Housing's comma formatting, and packed reads use an L cast. Exact short words
  remain selectable. Numbered chat suggestions work independently of item-name
  placeholder rendering, which still needs an in-game check. Replaced the long
  sentence preview with accepted-word echoes to respect string substitution
  limits. Rough UI import size: 705 actions (557 function, 139 menu, 9 command /
  events), one menu, no model changes. Full htsw check and keypad-driven README
  reply pass, as do every prefix/word route, backspace, empty suggestions and
  existing ownership, timing, input/context bounds and reconnect checks. Used
  shared test locks; no tool sources or live house modified. Cleanup names are
  recorded verbatim in house/old-menu-names.txt.

- 2026-10-20: Callan-approved click-to-type UI on `feat/chat-ui`. `ref/build_chat.py`
  adds 4 functions (296 total), 44 menus, `/llm`, and Join/Quit guards to `house/`;
  all 292 original functions and their import entries are unchanged. Letter menus
  retain vocab order; S (61) and W (42) split into second-letter menus at cap 40.
  New Chat feeds BOS/USER; Feed uses two five-tick pauses; Send feeds BOT and
  repeats through EOS or position 48. One shared conversation, 16-token input
  cap, sentence preview in team V, End Chat/disconnect release, reconnect delay.
  Full `htsw check` passes. `sim/check-chat.mjs` clicks the README question and
  reproduces `i enjoy them , they make me feel happy .`; all 507 selectable-token
  routes, busy/ownership guards, empty Send, question/context bounds, and
  disconnect/rejoin checks pass. Tools are bundled locally from read-only HTSW
  and harness sources. Harness nested timers can advance one tick early; checks
  enforce its four-tick function cooldown and the declared five-tick waits.
  Live import/smoke test belongs to IMPORT; teams T1..T10 and V must exist.
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
