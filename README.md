# housing-llm

A ~260K-parameter word-level chat transformer (MQA, ALiBi, ReLU MLP) compiled to
Hypixel Housing code. Trained on an M5, quantized to exact integer math, compiled
to HTSL, and proven bit-exact against the Python integer reference in the HTSW
simulator on 20/20 conversations (2026-10-02). Spec and ledger: `TASK.md`.

## What to import

| Folder | What | Functions |
|---|---|---|
| `house/` | The real-house build. Import this. | 292 |
| `sim-demo/` | Simulator demo: `htsw run import.json` prints a reply. Not for a house (its `htsw:main` calls Run0/Run1 back to back, which the 4-tick cooldown forbids live). | 295 |

Both pass `htsw check`.

`house/` setup, before or after import: create teams `T1`..`T10` and `V` in the
house (the import contains functions only). Usage is in `house/README.txt`:
run `Init` once per conversation, then per word set `tok`, run `Run0`, wait >= 5
ticks, run `Run1`.

Headless fleet import: point the bot at this repo with `path: house`.

## Source

| Path | What |
|---|---|
| `train/` | tokenizer, data prep, training (`train.py`), reports |
| `runs/<name>/` | trained models (`model.npz`, `config.json`, `vocab.json`, samples). Final: `v512_d96_L2_f384_mqa` |
| `ref/intmodel.py` | integer model (the spec the compiler matches) |
| `ref/compile.py`, `ref/build.py` | HTSL compiler and build driver |
| `sim/run.mjs` | HTSW simulator runner (checks sim == integer reference) |
| `probe/` | live throughput probe generator |
| `model/` | base tokenizer/model files used during training |

Rebuild the house:

    python3 ref/build.py --run runs/v512_d96_L2_f384_mqa --out build/<x> --ctx 48 --prompts prompts.txt
    vite-node sim/run.mjs build/<x>

Training data (`data/`, 2.3 GB) and the Python venv are not committed.
