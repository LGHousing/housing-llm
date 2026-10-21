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
all other lines matched the source. IMPORT then confirmed the cause directly:
typing n19 in the Variable editor produces "The stat key provided contains
blocked words! (1)" and leaves the default Kills. This is a filtered name, not
evidence of a storage or execution limit. HTSW did not detect the refusal.

Source T4 uses exactly 300 Long variables: 60 KV-cache values (V0_36_6 through
V0_47_6 and V0_0_7 through V0_47_7), o18..o95 (78), n0..n95 (96), r0..r65 (66).
n19 is number 158 in compiler allocation order; r65 is number 300. This is not
necessarily live creation order. The compiler chooses 300 Longs / 2,400 bytes
against its assumed 2,560-byte budget; 300 is not a verified Housing name limit.
Existing extra live variables could consume that margin. F063's n20..n23 write
targets are correct, so the export does not show a blanket failure of later names.
All n19 references occur in F016, F063, F113, F160, F161 and F168. Live exports
confirm the same substitution in all six (29 changed write targets total).
An earlier count of 46 included 17 division-spelling differences in F161.
F043, which writes the last T4 variable r65, matches exactly. F028's n95 writes
match, but its 16 phase-one divisions export as "…/=" instead of "/=". HTSW
v1.3.3 (fde9950d) defines only Divide, printed as "/="; this is not a recognized
alternate division mode. IMPORT subsequently inspected F028's live action and
confirmed Divide / 96; the ellipsis is an export spelling quirk. F028 passing
phase zero alone had not tested those lines.
F161 has 17 similar division differences; it is already included in the repair.
NAME FIX: the compiler now emits norm19 instead of n19. Its name screen also
flags n190..n199, n16 and n160..n169; those become norm190..norm199, norm16 and
norm160..norm169 as a precaution. Only n19 is confirmed blocked. F062 used n16
successfully live. The heuristic recognizes common digit-to-letter substitutions;
it cannot reproduce Hypixel's private filter, so verify new names by live export.
Internal tensor names, holder assignment, math and call order are unchanged.
Both compiler reads and writes use the same aliases. Unexpected suspicious names
and alias collisions fail the build; the keypad generator screens HTSL keys too.

Import ONLY these 19 functions from the name-fix build:
F015 F016 F062 F063 F111 F112 F113 F158 F159 F160 F161 F167 F168
F193 F194 F195 F198 F199 F200
Then export them back and verify the exact keys before a fresh New Chat.
F028, Chat Next, the keypad and all other executable files are byte-identical.
No extra functions, actions or logical variables are added.

Before starting a fresh chat, stop old work by clearing global ui_owner. Remove
obsolete model keys if present: T4 n16/n19; T8 n160..n169 and n190..n199. Retained
old keys consume storage; keeping all 20 on T8 would consume the compiler's
entire 160-byte margin once the replacements are written. Delete existing keys
through the variable editor if their names cannot be entered in an unset action.
Do not clear whole teams, Kills, the house's event variables, or player 18469.
New Chat writes the replacement values before reading them; Init is unchanged.

POST-IMPORT AUDIT (800cbe8)

IMPORT job 7b977600 imported the 19 renamed functions under HTSW v1.3.4. Fresh
exports contain no Kills and retain the new names. Sixteen match byte-for-byte;
F161 has 17 division spelling differences. Two exports contained actual
math differences: F193 line 749 and F195 line 523 replace c += t with earlier
t = constant actions. Inspect the actual editors to distinguish bad writes
from bad exports: F193 conditional 15 THEN action 6; F195 conditional 10 ELSE
action 19 (all positions one-based). IMPORT confirmed both wrong actions in
the editor and repaired them to Player, c, Increment, %var.player/t 0%,
Automatic Unset off. Fresh full exports still need to confirm both repairs.

After LG's deploy GO, verify those actions directly, then repair just the two
confirmed mistakes through a fresh live scan and targeted import or editor
changes. Re-export to verify. Deleting/recreating entire functions is a fallback,
not the first repair; it repeats thousands of writes through the same importer.
Audit all 300 functions before relying on the model. Unchanged, already-verified
exports can be reused. Do not treat a successful import status as a full read-back.
IMPORT inspected F028 conditional 23 ELSE action 14 under HTSW v1.3.5:
Player, ss, Divide, 96. This confirms the …/= export spelling represents Divide
there. The same spelling-only difference can now be accepted explicitly by the
audit; all other bytes must still match. Raw export hashes and diffs are retained.

Strict local audit (does not access or modify the live house):
  python3 ref/audit_live.py --revision 800cbe8 --accept-division-export \
    --report build/live-audit.json \
    /home/opc/housing-llm-live/800 /path/to/newer/exports
Each folder contains <function name>.live.htsl; later folders override earlier
ones. Only reuse an earlier export if that function has not changed live since.
The report records source/export hashes, all missing functions and exact diffs.
Default behavior remains strict: division-only differences are unresolved and
fail. --accept-division-export accepts only …/= in an action's operation field,
reports division_accepted separately from exact, and records that choice in JSON.
Values, other operations and text are not normalized. Exit 0 requires all 300
functions to be exact or explicitly accepted division-only matches; exit 1 means
incomplete or unresolved. This checks files, not whether the exporter read correctly.

The five-call batching update changed ONLY Run0, Run1 and Run0_1. Keep Chat Next's
installed clock. IMPORT reports HTSW v1.3.3 fixes the previous repeat-interval
read-back problem; the name fix does not require a clock re-import.

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
model-passes.json. Relative to the original 292 functions, three dispatchers
are batched and 19 calculation functions have variable-name substitutions.
The remaining 270 functions are unchanged.

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
