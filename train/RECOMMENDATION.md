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
