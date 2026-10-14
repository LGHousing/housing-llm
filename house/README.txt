housing-llm: a ~260K-parameter chat transformer compiled to HTSW (292 functions).

Setup: create teams T1..T10 and V in the house. Run "Init" once per conversation
(zeroes the KV cache, loads the 506-word vocab table into global + team V).

Per word: var tok = <word id> (vocab.json index), run "Run0", wait >= 5 ticks, run "Run1".
Set var gen = 1 before the last prompt id (<bot> = 4): from then on Run1 prints each reply
word in chat (dynamic read of the vocab table, no per-word conditionals) and copies it into
tok, so just keep running Run0 / Run1. var done becomes 1 at <eos>. Context limit 48 tokens.

Example: do you like cats ?  ->  ids 2 3 21 18 32 502 25 4
Expected reply: i enjoy them , they make me feel happy .
