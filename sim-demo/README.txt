housing-llm SIM DEMO: a ~260K-parameter chat transformer compiled to HTSW (295 functions).

Run it:
  htsw run import.json
Prints the question and the model's reply, one word per chat line (~5 s).

Ask something else (words must be in vocab.json: 512 lowercase words, punctuation separate):
  python3 make_main.py "do you like music ?"
  htsw run import.json

What's in here:
  Run0 / Run1   the model (~319K Change Variable per token)
  Out           prints the reply word with a dynamic read: %var.global/w<bi>% (team V for the tail)
  Init, Vocab   reset the KV cache, load the vocab table
  htsw:main     feeds the question in word by word, then generates until <eos> (sim only:
                it calls Run0/Run1 back to back, which the 4-tick cooldown forbids in a real house)

Expected:
  do you like cats ?   ->  i enjoy them , they make me feel happy .
  do you like music ?  ->  yes , i enjoy music very much , it makes me feel happy .
