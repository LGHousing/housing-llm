housing-llm: a ~260K-parameter chat transformer compiled to HTSW.
296 functions (292 unchanged model functions + 4 chat helpers), 44 menus.

CLICK-TO-TYPE CHAT

Setup: teams T1..T10 and V must already exist; this import does not create them.
Import house/import.json, including menus, /llm, and Player Join / Player Quit
actions. Those events protect the shared model when someone leaves or rejoins;
merge them with any existing house event actions instead of overwriting those.

Players: type /llm, click New Chat, then choose a letter and a word. S and W
have second-letter menus. Each word returns to the letters after processing;
the sentence so far appears in chat. The root has ?, . and , buttons; Symbols
has !, apostrophe and <unk>, plus explanations of the automatic special tokens.
Click Send and the answer prints by itself, one word per chat line. Open /llm
and click New Chat for another question. There is no free-text input.

Only one player can chat at a time. End Chat releases an unfinished question;
finishing a reply or leaving the house also releases it. Closing the menu alone
keeps the question: /llm reopens it. A player who stays online with an unfinished
question holds the model until they Send, End Chat, or leave. New Chat clears
the question; there is no individual-word undo or multi-turn conversation.

Questions allow 16 tokens (words and punctuation both count). This keeps the
sentence preview within a chat message and leaves up to 29 calculation steps
for a reply inside the model's 48-token context. Feed waits five ticks between
Run0 and Run1 and another five before the next word. Reply stops at <eos> or
the context limit. Wait for processing to finish before clicking another word.
After joining, wait half a second before New Chat so old delayed work can drain.

Rebuild menus/helpers: python3 ref/build_chat.py
This uses vocab order (frequency order, with pinned words at the end).
Keep the compiled model and vocab in sync before regenerating the UI.

Validation from the repository root (all tool output stays in build/):
  nice -n 10 node sim/build-tools.mjs
  nice -n 10 node build/tools/htsw.mjs check house/import.json
  nice -n 10 node sim/check-chat.mjs
Set HTSW_ROOT and HTSW_SIM_ROOT if the read-only tool checkouts are elsewhere.
The test clicks actual menus and checks the README reply, token IDs, cooldowns,
all word routes, busy/ownership guards, context limit, and disconnect/rejoin.
The current harness advances newly added timers in the same tick, so nested
pause 5 can appear as four ticks; the test checks four-tick function cooldowns
and separately checks that Feed declares both five-tick pauses. No tool source
is patched. In-game timing and menu presentation still need the fleet's smoke
test after import.

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
