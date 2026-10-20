# housing-llm

A ~260K-parameter word-level chat transformer (MQA, ALiBi, ReLU MLP) compiled to
Hypixel Housing code. Trained on an M5, quantized to exact integer math, compiled
to HTSL, and proven bit-exact against the Python integer reference in the HTSW
simulator on 20/20 conversations (2026-10-02). Spec and ledger: `TASK.md`.

## What to import

| Folder | What | Functions |
|---|---|---|
| `house/` | The real-house build with click-to-type chat. Import this. | 296 |
| `sim-demo/` | Simulator demo: `htsw run import.json` prints a reply. Not for a house (its `htsw:main` calls Run0/Run1 back to back, which the 4-tick cooldown forbids live). | 295 |

Both pass `htsw check`.

`house/` setup: create teams `T1`..`T10` and `V` before use. Players open `/llm`,
click **New Chat**, choose words from the letter menus, then click **Send**.
The sentence appears in chat as words are added; the answer prints automatically.
One player can chat at a time, with up to 16 input words/punctuation marks.
**End Chat** releases an unfinished question. See `house/README.txt` for setup,
event import notes, limits, and the manual model interface.

The UI adds four helpers, 44 menus, one command, and Join/Quit handlers; all 292
compiled model functions remain unchanged. Regenerate it with
`python3 ref/build_chat.py` after building a matching model/vocabulary.

Check the full build and reproduce the example through actual menu clicks:

    nice -n 10 node sim/build-tools.mjs
    nice -n 10 node build/tools/htsw.mjs check house/import.json
    nice -n 10 node sim/check-chat.mjs

The tool builder reads `/home/opc/htsw` and `/home/opc/htsw-simulator` without
editing them; override `HTSW_ROOT` / `HTSW_SIM_ROOT` for other installations.
Their dependencies must already be installed. Generated tool bundles stay in
the ignored `build/` directory.

Headless fleet import: point the bot at this repo with `path: house`.

## Source

| Path | What |
|---|---|
| `train/` | tokenizer, data prep, training (`train.py`), reports |
| `runs/<name>/` | trained models (`model.npz`, `config.json`, `vocab.json`, samples). Final: `v512_d96_L2_f384_mqa` |
| `ref/intmodel.py` | integer model (the spec the compiler matches) |
| `ref/compile.py`, `ref/build.py` | HTSL compiler and build driver |
| `ref/build_chat.py` | Generates the word menus, chat helpers, and `/llm` entry point |
| `sim/check-chat.mjs` | Menu-to-reply regression and shared-state/timing checks |
| `sim/run.mjs` | HTSW simulator runner (checks sim == integer reference) |
| `probe/` | live throughput probe generator |
| `model/` | base tokenizer/model files used during training |

Rebuild the house:

    python3 ref/build.py --run runs/v512_d96_L2_f384_mqa --out build/<x> --ctx 48 --prompts prompts.txt
    vite-node sim/run.mjs build/<x>

Training data (`data/`, 2.3 GB) and the Python venv are not committed.
