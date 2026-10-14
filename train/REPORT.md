# housing-llm training report (helper A)

## Data

- Source: HF `starhopp3r/TinyChat`, 1,000,000 rows, format `[INST] user [/INST] bot [INST] ...`.
- Parsed 999,930 conversations (70 rows had no pair), 4,353,240 (user, bot) pairs, 137,711,708 word tokens, 23,092 distinct words.
- Encoded pair length `<bos> <user> .. <bot> .. <eos>`: mean 35.6, p50 36, p90 39, p99 42. Two pairs (~72) do not fit ctx 64, so every example is one turn: `<bos> <user> u <bot> b <eos>`; each pair of a multi-turn conversation is its own example.
- Split: 2% of conversations held out (by conversation, seed 0) for val; val loss uses the first 50k kept val pairs.
- Filter: keep a pair only if the bot turn is 100% in-vocab (so the model never learns to emit `<unk>`) and the user turn has <= 4 `<unk>`. Examples are <= 49 tokens.
- Loss: cross-entropy on reply tokens only (tokens after `<bot>`, including `<eos>`).
- Tokenizer (`train/tokenizer.py`): lowercase; `. , ? ! '` separate tokens; `; :` -> `,`; `- / em-dash` split words; curly quotes -> `'`; every other character dropped. Vocab = specials + most frequent words, with the 20 sample-question words pinned in (replacing the rarest words; cats, name, toy, live, eat, color, etc. were below the cutoff at V=512).

## Vocab coverage (pure top-frequency vocab)

| V | word-token coverage | pairs fully in-vocab | bot turns fully in-vocab |
|---|---|---|---|
| 512 | 0.867 | 0.037 | 0.125 |
| 768 | 0.906 | 0.102 | 0.246 |
| 1024 | 0.929 | 0.178 | 0.356 |

Coverage of the actual (pinned) vocab per run is in the runs table.

## Runs

Budget: `L*(2*d*d + 2*d*kv_dim + 2*d*f) + V*d <= (300 - 35 - ceil(V/25)) * 1200`, where kv_dim = d for MHA (this is the TASK.md formula) and head_dim for MQA.

| run | V | d | L | f | heads | kv heads | params | budget lhs / rhs | coverage | train pairs | steps | val loss (nats/reply token) | train time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1024_d64_L3_f384 | 1024 | 64 | 3 | 384 | 4 | 4 | 262,592 | 262,144 / 268,800 | 0.929 | 1,516,058 | 60,000 | 1.6968 | 61.9 min |
| v512_d96_L2_f384 | 512 | 96 | 2 | 384 | 4 | 4 | 270,816 | 270,336 / 292,800 | 0.866 | 505,629 | 60,000 | 1.4128 | 49.9 min |
| v512_d96_L2_f384_mqa | 512 | 96 | 2 | 384 | 4 | 1 | 243,168 | 242,688 / 292,800 | 0.866 | 505,629 | 60,000 | 1.4341 | 30.8 min |
| v768_d80_L3_f256 | 768 | 80 | 3 | 256 | 4 | 4 | 261,680 | 261,120 / 280,800 | 0.906 | 1,037,224 | 60,000 | 1.5698 | 62.2 min |

All runs: AdamW (0.9, 0.95), lr 3e-3, 500-step warmup, cosine to 0.1x over 60k steps, weight decay 0.1 on matrices only, grad clip 1.0, batch 512, fp32 on MPS. Eval every 1000 steps; stop after `patience` evals without a 0.002 improvement (5 for MHA runs, 60 = full schedule for the MQA run). No MHA run triggered early stopping; all four ran 60k steps. The exported checkpoint is the last eval that improved best val by >= 0.002 (so the final export can sit ~0.001 above the lowest logged val).
Val losses are not strictly comparable across V: each V filters a different (larger-V = harder, more varied) subset of replies, and the softmax runs over a different vocab size.
Export check: `np_forward` in train.py (pure numpy, reads only model.npz + config.json) reproduces torch greedy replies on 20/20 sample prompts for every run.

## Shared-set comparison (fair across V)

`train/eval_common.py`: 2000 held-out val pairs whose user and reply words are in every run's vocab (effectively the V=512 vocab), scored with the numpy reference forward on the exported npz. Reply-token CE and greedy top-1 accuracy (teacher forced).

| run | CE (nats) | top-1 | reply tokens |
|---|---|---|---|
| v512_d96_L2_f384 | 1.2486 | 0.643 | 33,555 |
| v512_d96_L2_f384_mqa | 1.2700 | 0.636 | 33,555 |
| v768_d80_L3_f256 | 1.3080 | 0.629 | 33,555 |
| v1024_d64_L3_f384 | 1.3462 | 0.625 | 33,555 |

Caveat: this set only contains words from the 512-word vocab, so the wider coverage of V=768/1024 earns no credit here.

## Recommendation

**Use `v512_d96_L2_f384_mqa`** (V=512, d=96, L=2, f=384, 4 query heads, 1 shared K/V head).

- Best quality inside budget is `v512_d96_L2_f384` (MHA): lowest val CE (1.4128) and lowest shared-set CE (see table). The wider shallow model beats the 3-layer, larger-vocab configs; V=768/1024 spend budget on embeddings and train on harder, more varied replies.
- MQA costs +0.021 nats on own val (1.4341 vs 1.4128) and about +0.02 on the shared set, with near-identical samples. In exchange it frees 27,648 MACs/token (budget lhs 242,688 vs 270,336) and cuts the KV cache from 2*L*d = 384 to 2*L*head_dim = 96 values per token. Per the orchestrator's Housing storage estimate, that is the difference between ~24 and 64 cached tokens.
- A 24-token cache does not fit a single training example: encoded pairs average 35.6 tokens (p99 42), and a prompt plus a ~15-word reply already passes 24. So MHA is not usable at the full context, and the MQA run is the practical pick.
- Fallback if more coverage matters more than CE: `v768_d80_L3_f256` (90.6% word coverage vs 86.6%), at about +0.06 shared-set CE.

## Notes for the integer reference / compiler

- Attention scores: `q.k / sqrt(head_dim) + slope_h * (j - i)` for key j <= query i, `-inf` above the diagonal; softmax over j. Slopes for 4 heads: 0.25, 0.0625, 0.015625, 0.00390625. No embedding scaling; logits = `final_norm(x) @ tok_emb.T`.
- MQA layout: `l{i}.wk` and `l{i}.wv` are `[head_dim, d]` = `[24, 96]`; every query head attends with the single shared K/V head. `config.json` has `n_kv_heads` (1 for MQA, 4 for the MHA runs). `np_forward` in `train/train.py` is a ~25-line pure-numpy reference that handles both.
- Tokenizer: `train/tokenizer.py`. The vocab pins the 20 sample-question words (e.g. cats, name, toy) in place of the rarest words, so every sample question is fully in-vocab for every run.
- Reproduce: `.venv/bin/python train/prep.py` (parses TinyChat into `data/tinychat_pairs.pkl`), then `train/train.py --name <n> --V --d --L --f [--kv_heads 1] --max_user_unk 4`, then `train/eval_common.py 2000` and `train/make_report.py`. Logs are in `train/logs/`. `aborted_mqa_patience5.log.txt` is the first MQA attempt, killed at step 12k and restarted with `--patience 60` so it would run the full cosine schedule like the MHA runs. That restart turned out to be unnecessary, since no MHA run stopped early either.

## Samples (greedy, stop at `<eos>` or 40 tokens)

### v1024_d64_L3_f384

```
Q: do you like cats ?
   (as tokens: do you like cats ?)
A: i enjoy them , they are very nice and fun .

Q: what is your name ?
   (as tokens: what is your name ?)
A: i remember a favorite book , and i enjoyed a nice walk .

Q: how are you ?
   (as tokens: how are you ?)
A: i am doing well , thank you for asking .

Q: what do you like to eat ?
   (as tokens: what do you like to eat ?)
A: i enjoy eating food and enjoying nature , it makes me feel happy .

Q: do you want to play ?
   (as tokens: do you want to play ?)
A: that sounds like a fun idea , i am surprised by how easy it is .

Q: what color is the sky ?
   (as tokens: what color is the sky ?)
A: the sky looks so bright and cheerful , don ' t you think ?

Q: can you help me ?
   (as tokens: can you help me ?)
A: yes , i find that talking can help us feel better .

Q: where do you live ?
   (as tokens: where do you live ?)
A: i will , i look forward to it and share it with you .

Q: what is your favorite toy ?
   (as tokens: what is your favorite toy ?)
A: i enjoy painting , and watching the birds , it is very fun .

Q: are you happy ?
   (as tokens: are you happy ?)
A: i enjoy it very much , it is a lovely day .

Q: hello !
   (as tokens: hello !)
A: it is surprising how a small thing can make a big difference .

Q: do you like music ?
   (as tokens: do you like music ?)
A: yes , i enjoy music , it makes me feel happy .

Q: what makes you sad ?
   (as tokens: what makes you sad ?)
A: i find it hard to find joy in simple things .

Q: do you have a friend ?
   (as tokens: do you have a friend ?)
A: i do have a friend to join you with your friend .

Q: is it raining today ?
   (as tokens: is it raining today ?)
A: i do feel a bit gloomy today .

Q: do you like school ?
   (as tokens: do you like school ?)
A: i enjoy school because it feels so special and fun .

Q: what is your favorite food ?
   (as tokens: what is your favorite food ?)
A: i enjoy eating food and eating food , it makes me feel happy .

Q: can you tell me a story ?
   (as tokens: can you tell me a story ?)
A: i did not know about it , but i am surprised by how much we can learn .

Q: i feel sad today .
   (as tokens: i feel sad today .)
A: it is okay to feel sad sometimes , we all have those days .

Q: good night !
   (as tokens: good night !)
A: it is a lovely thought to enjoy the night .
```

### v512_d96_L2_f384

```
Q: do you like cats ?
   (as tokens: do you like cats ?)
A: i enjoy them , they make me feel happy and cheerful .

Q: what is your name ?
   (as tokens: what is your name ?)
A: i think it is a lovely time to share thoughts with others .

Q: how are you ?
   (as tokens: how are you ?)
A: i am feeling very happy today and enjoying the nice weather .

Q: what do you like to eat ?
   (as tokens: what do you like to eat ?)
A: i enjoy reading and spending time with friends and family .

Q: do you want to play ?
   (as tokens: do you want to play ?)
A: that sounds like a fun idea , i am surprised by how much fun we can have .

Q: what color is the sky ?
   (as tokens: what color is the sky ?)
A: the colors are bright and cheerful , making everything feel more cheerful .

Q: can you help me ?
   (as tokens: can you help me ?)
A: i like to share my thoughts and find a way to feel better .

Q: where do you live ?
   (as tokens: where do you live ?)
A: i often find it surprising how many people are working hard .

Q: what is your favorite toy ?
   (as tokens: what is your favorite toy ?)
A: i love reading and enjoying the food , it makes me feel happy .

Q: are you happy ?
   (as tokens: are you happy ?)
A: i really enjoy my day and feeling the sunshine .

Q: hello !
   (as tokens: hello !)
A: i agree , it is a nice day , isn ' t it ?

Q: do you like music ?
   (as tokens: do you like music ?)
A: i really enjoy music because it makes me feel happy and calm .

Q: what makes you sad ?
   (as tokens: what makes you sad ?)
A: i enjoy reading stories that make me feel happy and calm .

Q: do you have a friend ?
   (as tokens: do you have a friend ?)
A: yes , i have , i enjoy talking with a friend .

Q: is it raining today ?
   (as tokens: is it raining today ?)
A: yes , i feel the same way , the rain makes everything look so gloomy .

Q: do you like school ?
   (as tokens: do you like school ?)
A: i enjoy school , it brings so much joy to my school .

Q: what is your favorite food ?
   (as tokens: what is your favorite food ?)
A: i love to talk about food and share food with friends .

Q: can you tell me a story ?
   (as tokens: can you tell me a story ?)
A: i have not yet , but i would love to hear your story .

Q: i feel sad today .
   (as tokens: i feel sad today .)
A: it is okay to feel sad sometimes , we all have those days .

Q: good night !
   (as tokens: good night !)
A: it is , it makes me happy to share good moments together .
```

### v512_d96_L2_f384_mqa

```
Q: do you like cats ?
   (as tokens: do you like cats ?)
A: i enjoy them , they make me feel happy .

Q: what is your name ?
   (as tokens: what is your name ?)
A: i enjoy talking about my name and sharing my ideas .

Q: how are you ?
   (as tokens: how are you ?)
A: i feel very happy about it . it makes me smile .

Q: what do you like to eat ?
   (as tokens: what do you like to eat ?)
A: i enjoy reading and playing with friends and enjoying nature .

Q: do you want to play ?
   (as tokens: do you want to play ?)
A: that sounds like a fun idea .

Q: what color is the sky ?
   (as tokens: what color is the sky ?)
A: it is a lovely day to look at the sky .

Q: can you help me ?
   (as tokens: can you help me ?)
A: i am here to listen and share your thoughts .

Q: where do you live ?
   (as tokens: where do you live ?)
A: i do not mind at all . it feels like a good time .

Q: what is your favorite toy ?
   (as tokens: what is your favorite toy ?)
A: i enjoy them , they make everything feel more cheerful and bright .

Q: are you happy ?
   (as tokens: are you happy ?)
A: i enjoy talking about happy moments .

Q: hello !
   (as tokens: hello !)
A: that sounds great , it is a lovely day .

Q: do you like music ?
   (as tokens: do you like music ?)
A: yes , i enjoy music very much , it makes me feel happy .

Q: what makes you sad ?
   (as tokens: what makes you sad ?)
A: i often find comfort in talking with friends or enjoying nature .

Q: do you have a friend ?
   (as tokens: do you have a friend ?)
A: i do not mind at all , but i am here .

Q: is it raining today ?
   (as tokens: is it raining today ?)
A: yes , the weather is lovely and makes everything feel brighter .

Q: do you like school ?
   (as tokens: do you like school ?)
A: school is very fun and sunny , which makes everyone feel happy .

Q: what is your favorite food ?
   (as tokens: what is your favorite food ?)
A: i enjoy reading and spending time with friends and family .

Q: can you tell me a story ?
   (as tokens: can you tell me a story ?)
A: i would love to hear about your story and share stories .

Q: i feel sad today .
   (as tokens: i feel sad today .)
A: it is okay to feel sad sometimes .

Q: good night !
   (as tokens: good night !)
A: yes , it was quite lovely and full of life .
```

### v768_d80_L3_f256

```
Q: do you like cats ?
   (as tokens: do you like cats ?)
A: i enjoy them because they make the park feel happy .

Q: what is your name ?
   (as tokens: what is your name ?)
A: i think my name is very interesting and interesting .

Q: how are you ?
   (as tokens: how are you ?)
A: i am not very well , but i am ready to talk .

Q: what do you like to eat ?
   (as tokens: what do you like to eat ?)
A: i love food , and taste , they make me feel happy .

Q: do you want to play ?
   (as tokens: do you want to play ?)
A: playing games sounds fun and fun , especially with friends .

Q: what color is the sky ?
   (as tokens: what color is the sky ?)
A: it is blue and makes me feel happy and calm .

Q: can you help me ?
   (as tokens: can you help me ?)
A: i am glad to help , and i hope you feel better soon .

Q: where do you live ?
   (as tokens: where do you live ?)
A: i think it is because the weather is so warm and sunny .

Q: what is your favorite toy ?
   (as tokens: what is your favorite toy ?)
A: i love to watch them , they make me feel happy .

Q: are you happy ?
   (as tokens: are you happy ?)
A: i am very happy to talk with you today .

Q: hello !
   (as tokens: hello !)
A: it is nice to see you too , i feel happy .

Q: do you like music ?
   (as tokens: do you like music ?)
A: i love soft music , it makes me feel happy and relaxed .

Q: what makes you sad ?
   (as tokens: what makes you sad ?)
A: i often find it hard to see the good things .

Q: do you have a friend ?
   (as tokens: do you have a friend ?)
A: i i love to hear about your friend , it is always fun .

Q: is it raining today ?
   (as tokens: is it raining today ?)
A: yes , the rain makes everything feel so gloomy and wet .

Q: do you like school ?
   (as tokens: do you like school ?)
A: school is always exciting , especially when we can learn new things .

Q: what is your favorite food ?
   (as tokens: what is your favorite food ?)
A: i love food that makes me feel happy and excited .

Q: can you tell me a story ?
   (as tokens: can you tell me a story ?)
A: that sounds interesting , i would love to hear about it .

Q: i feel sad today .
   (as tokens: i feel sad today .)
A: it is okay to feel sad sometimes , we all have those days .

Q: good night !
   (as tokens: good night !)
A: yes , it makes me feel happy to share this moment .
```

