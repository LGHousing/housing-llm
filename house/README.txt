housing-llm: a ~260K-parameter chat transformer compiled to HTSW.
300 functions (292 unchanged model functions + 8 chat helpers), ONE menu.

KEYPAD CHAT

Setup: teams T1..T10 and V must already exist; this import does not create them.
Import house/import.json, including /llm and Player Join / Player Quit actions.
Merge those events with any existing house actions rather than overwriting them.
The old 44 menu names are listed exactly, one per line, in old-menu-names.txt.
Delete those old menus before importing the replacement "LLM Keypad" menu.

IN-GAME CHECK STILL REQUIRED: do custom-menu ITEM NAMES resolve placeholders
again each time the menu opens? The suggestion items use names such as
"1: %var.team/ui_s0 V%". Set ui_s0 on team V to hello, open the menu, change it
and reopen. The simulator checks the word variables, not Minecraft rendering.
If names remain literal, the same keypad still works: suggestions are numbered
1..7 in chat and each item starts with its number. Static names "Suggestion 1"
through "Suggestion 7" are the cleanest presentation fallback if needed.

Players: /llm -> New Chat. Click letters to narrow the seven suggestions, then
click a suggestion on the bottom row to add that word. The current letters and
numbered suggestions are printed in chat every time they change. Each accepted
word is echoed; after it is processed, the letters reset. Use Backspace to
remove a letter or Clear letters to start a different prefix. These do not undo
words already fed to the model. Click Send to generate the complete reply,
one word per chat line. There is no free-text box.

The keypad includes all 26 letters, ?, ., comma, !, apostrophe and <unk>.
Wait for the word to finish processing before choosing the next one, and allow
suggestions to refresh between letter clicks (shared functions have a four-tick
cooldown). Up to three letters suffice to reach every word. A prefix with no
matches shows seven dashes; use Backspace or Clear letters to recover.

One player can chat at a time. End Chat releases an unfinished question;
finishing a reply or leaving the house also releases it. Closing the menu alone
keeps the question; /llm reopens it. An online player with an unfinished question
holds the model until they Send, End Chat, or leave. New Chat starts over.
There is no multi-turn conversation or undo for an already accepted word.

Questions allow 16 tokens, counting punctuation, leaving up to 29 calculation
steps for a reply inside the 48-token context. Feed waits five ticks between
Run0 and Run1 and another five before reuse. Reply stops at <eos> or the context
limit. After joining, wait half a second before New Chat so old delayed work
can drain.

GENERATED LOOKUP AND IMPORT SIZE

Regenerate: python3 ref/build_chat.py
The generator replaces house/chat and its entries in house/import.json.
All 292 compiled model functions remain unchanged. Chat Load runs automatically
on New Chat, loading the lookup once per generated version. There are 228
prefix entries: empty prefix, 24 first letters, 130 second-letter prefixes and
73 third-letter prefixes. Seven 9-bit vocabulary IDs fit into each 63-bit Long.
Names such as "k1,108" match Housing's comma-formatted numeric placeholders;
lookups use an L cast to recover the packed Long. Vocab word reads use the same
dynamic lookup as the model's output. Missing entries become empty suggestions.

Suggestions follow vocab order (frequency order, with pinned words at the end).
If an exact word would be omitted, it replaces suggestion seven; this keeps the
single-letter words "s" and "t" reachable. Third letters are generated only for
crowded prefixes. All 501 words, five punctuation tokens and <unk> are selectable.
The generator checks the team V byte budget: at most 2,241 bytes for its existing
vocab strings, lookup, seven suggestion strings and current letters. UI scratch
work reuses model scratch variables while the model is idle. A long accumulated
sentence preview is omitted because Housing string substitution is limited.

Expected import: 8 UI functions, 1 menu with 44 populated slots, 1 command and
2 event handlers. Approximately 705 stored UI actions including conditionals:
557 in functions (268 in Chat Load), 139 menu-click actions, 9 command/event
actions. The original 292 model functions need no changes. Compare the old
build's 7,246 menu-click actions alone. These counts are NOT HTSW work units;
use the fleet's plan to estimate actual import time. There are no spare function
slots at the house's 300-function limit.

VALIDATION

On this shared box, use the test-slot scripts; do not run during import prep:
  /home/opc/light-lock.sh node sim/build-tools.mjs
  /home/opc/light-lock.sh node build/tools/htsw.mjs check house/import.json
  /home/opc/heavy-lock.sh node sim/check-chat.mjs
Set HTSW_ROOT and HTSW_SIM_ROOT if the read-only tool checkouts are elsewhere.
The test drives actual keypad clicks and dynamic reads for every prefix, checks
all word routes, backspace, empty results, the README reply, busy/ownership
guards, context limit, and disconnect/rejoin. The current harness advances newly
added timers in the same tick, so nested pause 5 can appear as four ticks; the
test checks four-tick function cooldowns and Feed's declared five-tick waits.
No tool source is patched. Menu-name rendering still needs the in-game check.

MANUAL MODEL INTERFACE (do not use concurrently with the chat UI)

Setup: create teams T1..T10 and V in the house. Run "Init" once per conversation
(zeroes the KV cache, loads the 512-entry vocab table into global + team V:
506 ordinary word/punctuation tokens plus six special tokens).

Per word: var tok = <word id> (vocab.json index), run "Run0", wait >= 5 ticks, run "Run1".
Set var gen = 1 before the last prompt id (<bot> = 4): from then on Run1 prints each reply
word in chat (dynamic read of the vocab table, no per-word conditionals) and copies it into
tok, so just keep running Run0 / Run1. var done becomes 1 at <eos>. Context limit 48 tokens.

Example: do you like cats ?  ->  ids 2 3 21 18 32 502 25 4
Expected reply: i enjoy them , they make me feel happy .
