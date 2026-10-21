housing-llm: a ~260K-parameter chat transformer compiled to HTSW.
300 functions (292 model functions, including 3 batched dispatchers, + 8 chat
helpers), ONE menu, ONE command. No event handlers.

KEYPAD CHAT

Teams T1..T10 and V must already exist. Import house/import.json and retain the
house's own Join/Quit actions. This build neither supplies nor needs those events.
The previous 44 menus are listed in old-menu-names.txt; IMPORT already removed
them from guh. The remaining menu is "LLM Keypad".

Players: /llm -> New Chat. Wait for "Choose words, then Send." Click letters to
narrow the seven suggestions, then click a word on the bottom row to add it.
Accepted words appear in chat. Wait for processing before selecting another.
Backspace removes a letter; Clear letters clears the current prefix (shown on
that button). These do not undo accepted words. Send prints the full reply,
one word per line. There is no free-text box or multi-turn history.

The alphabet uses light-blue wool blocks named A–Z (letters show on hover).
New Chat is blue, Send green, End Chat red.
Suggestions show just the word. Five punctuation buttons offer ? . , ! and '.
There is no <unk> button or item lore. All 501 alphabet words are reachable with
at most three letters. Empty matches show dashes. Allow a quarter second between
letter clicks because shared functions have a four-tick cooldown.

Only one player can use the shared model at a time. New Chat ALWAYS starts over,
including during loading/calculation or after another player's failure. End Chat
ALWAYS releases it, including another player's abandoned question. Closing the
menu alone keeps the question; /llm reopens it. A disconnected owner does not
need to reconnect: anyone may use New Chat or End Chat. These controls deliberately
allow players to cancel one another; use the house's access rules if needed.

Questions allow 16 words/punctuation marks, leaving room in the 48-token context.
The conservative batch size takes about 29 seconds per model token at 20 ticks/s:
New Chat takes about one minute; replies take roughly 29 seconds per output word.
Server load can make this longer. This is intended for live validation;
the server's actual execution ceiling has not yet been measured.

LIVE FAILURE AND RECOVERY

IMPORT verified Init alone succeeds, but the old Run0 alone fails with Housing's
generic execution error. That locates the failure in the model pass, rather than
the keypad or lookup loader; it does not yet establish the precise server limit.
The first ten-function batching attempt failed at phase 0 batch 8 (F056-F065).
At failure sm=2^30, z0=255 and z47=0, so the F057 normalization divisor was
nonzero. With five-function batches, batch 18 (F056-F060) passed and batch 19
(F061-F065) failed. Individual calls isolated the failure to F063. A read-only
live export then showed six write targets changed from team n19 on T4 to Kills;
all other lines matched the source. This confirms an import mismatch, not its
cause or the server's execution limit. Further batch reduction is not justified
by that failure. Do not resume a failed partial batch: use New Chat to reset
after IMPORT finishes examining the preserved diagnostic state.

Source T4 uses exactly 300 Long variables: 60 KV-cache values (V0_36_6 through
V0_47_6 and V0_0_7 through V0_47_7), o18..o95 (78), n0..n95 (96), r0..r65 (66).
n19 is number 158 in compiler allocation order; r65 is number 300. This is not
necessarily live creation order. The compiler chooses 300 Longs / 2,400 bytes
against its assumed 2,560-byte budget; 300 is not a verified Housing name limit.
Existing extra live variables could consume that margin. F063's n20..n23 write
targets are correct, so the export does not show a blanket failure of later names.
All n19 references occur in F016, F063, F113, F160, F161 and F168. IMPORT is
checking the other five exports; F028 (n95) and F043 (r65) are boundary samples.
Keep the source calculation and holder allocation unchanged pending those checks.

The five-call batching update changed ONLY Run0, Run1 and Run0_1. Chat Next is byte-identical;
keep its installed clock. HTSW currently fails to read back Housing's
"Current: 5 Ticks (Less than 1 second)" lore, even though the clock runs live.

Run0 and Run1 now run at most five original calculation functions per invocation.
Chat Next repeats every five ticks and runs exactly one batch. There are 60 Run0
batches and 54 Run1 batches per token, followed by one completion tick.
The README simulator run measured at most 3,315 actions in one tick, down from
6,553 with ten-call batches. This is not a measured live server limit.
The three dispatchers share selectors for F000-F255 so five-call batches fit
Housing's flat conditional limits. Run0/Run1 set ph only when ui_part=0 at pass
entry; their internal cross-calls preserve ph. Run1 skips phase-zero embedding
and KV-write batches, and finishes with Out. No new variables are needed.
The exact 283/257 calculation calls and their order are preserved in
model-passes.json. Only Run0, Run0_1 and Run1 changed among the original 292
functions; the 289 initialization/math/output functions are unchanged.

Feed enqueues a token instead of running a long chain. Loading uses 13 small
lookup batches (20 assignments each), then Init_0, Init_1, Init_2 and Vocab on
separate ticks. A version match skips lookup loading. Each batch marks itself
in progress (ui_work=-1), then records success. If execution aborts before that
marker, the next scheduler tick releases the chat and reports the failing phase
and batch. New Chat and End Chat work even if that recovery tick never runs.
There are no paused continuations that could resume an abandoned calculation.

IMPORT: keep Chat Next repeating OFF until all changed functions are installed,
then set its repeat interval to 5 ticks. The manifest carries repeatTicks=5.
Before replacing an old build, clear global ui_owner to stop old work. Optional:
set global ui_rev=0 to exercise the full loader. No per-player reset is required:
New Chat resets ui_phase, ui_work, ui_part, gen, done and ui_n itself. End Chat
also clears the owner and busy state. Team V ph_t="world" from the placeholder
probe is unused and may be unset. Do not reset player variable 18469 or replace
guh's existing Join/Quit events. Do not clear the model/team storage wholesale.

HOUSING ITEM RULE (Callan): never generate custom player heads or
SkullOwner/textures NBT. Housing permits only its own built-in skulls in menus
and items. Keep the named wool alphabet; do not retry custom head textures.

IMPORT verified dynamic item-name placeholders refresh on menu reopen. Custom
skull spawning failed live on the A key: HTSW observed an empty slot after the
creative packet. The fallback uses minecraft:wool, Damage:3s, Count:1b and only
a display name, without texture/profile NBT. HTSW accepts wool and compares its
item ID, damage, count and name on read-back. Import only LLM Keypad to apply
this fallback; functions, actions, layout and variable state are unchanged.

GENERATED LOOKUP AND IMPORT SIZE

Regenerate: python3 ref/build_chat.py
The generator owns house/chat, its manifest entries, and the three dispatchers.
When run after a fresh model compile it captures that model's original call order;
subsequent regenerations use model-passes.json.

The lookup has 228 packed Longs: empty prefix, 24 first letters, 130 second-letter
prefixes and 73 third-letter prefixes. Each Long holds seven 9-bit vocabulary IDs.
Dynamic keys include Housing's thousands separators, e.g. k1,108; an L cast
recovers each packed Long. Suggestions follow vocabulary order (frequency order
with pinned evaluation words at the end). Exact short words replace the seventh
suggestion when necessary. Team V storage is bounded at 2,241 bytes for its vocab,
lookup, suggestion strings and current prefix. The UI reuses idle model scratch
variables. Accepted-word echoes avoid Housing's short substitution-string limit.

Expected total: 300 functions, 1 menu with 43 buttons, 1 command, no events.
The UI has about 803 stored actions: 622 helper actions (335 in Chat Load),
180 menu-click actions and 1 command action. The three model dispatchers contain
603 actions. These counts include conditionals but NOT both branches executed;
they are not fleet import work units. No additional function slots are needed.

VALIDATION

On this shared box, use the test-slot scripts; do not run during import prep:
  /home/opc/light-lock.sh node sim/build-tools.mjs
  /home/opc/light-lock.sh node build/tools/htsw.mjs check house/import.json
  /home/opc/heavy-lock.sh node sim/check-chat.mjs
Set HTSW_ROOT and HTSW_SIM_ROOT if the read-only tool checkouts are elsewhere.
The regression drives all prefixes via menu clicks, reproduces the example,
checks every original model call in order, measures batch actions and timing,
and tests input/context limits, cancellation, restart, missing success markers,
and an absent owner's replacement with no Join/Quit events. It does not prove
Housing's undocumented live limits or Minecraft item appearance.

MANUAL MODEL INTERFACE (do not use concurrently with the chat UI)

Init still initializes the model by itself. Run0 and Run1 now execute ONE BATCH,
not a full pass. For a manual token: set tok, ui_part=0, ui_work=1. Run Run0 every
five ticks until ui_work=2, then Run1 every five ticks until ui_work=3. Never run
both in the same tick. Set gen=1 before processing <bot>=4; generated words copy
their vocabulary ID into tok. Repeat complete passes until done=1 or pos=48.
Use the chat UI for automatic scheduling.

Example: do you like cats ? -> ids 2 3 21 18 32 502 25 4
Expected reply: i enjoy them , they make me feel happy .
