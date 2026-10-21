# housing-llm

A ~260K-parameter word-level chat transformer (MQA, ALiBi, ReLU MLP) compiled to
Hypixel Housing code. Trained on an M5, quantized to exact integer math, compiled
to HTSL, and proven bit-exact against the Python integer reference in the HTSW
simulator on 20/20 conversations (2026-10-02). Spec and ledger: `TASK.md`.

## What to import

| Folder | What | Functions |
|---|---|---|
| `house/` | The real-house build with click-to-type chat. Import this. | 300 |
| `sim-demo/` | Simulator demo: `htsw run import.json` prints a reply. Not for a house (its `htsw:main` calls Run0/Run1 back to back, which the 4-tick cooldown forbids live). | 295 |

Both pass `htsw check`.

`house/` setup: create teams `T1`..`T10` and `V` before use. Players open `/llm`,
click **New Chat**, click letters on the single keypad, select a suggested word,
then click **Send**. Accepted words appear in chat; the answer
prints automatically. Backspace and Clear letters change the current prefix.
One player can chat at a time, with up to 16 input words/punctuation marks.
**End Chat** releases an unfinished question. See `house/README.txt` for setup,
recovery and import notes, limits, and the manual model interface.

The UI adds eight helpers, one 43-button menu, and one command. It does not
replace the house's Join/Quit events. New Chat and End Chat can recover any busy
state, including an abandoned player's question. The model's three dispatchers
now split each token into 114 batches, spaced five ticks apart; its 289 other
functions are unchanged. This smaller-batch setting takes roughly
29 seconds per token. The old full Run0 pass failed live; the exact Housing limit
still needs measuring. The generated 228-entry lookup fits in team V storage.

Letter buttons use light-blue wool named A–Z; suggestion buttons show only the word.
IMPORT verified that variable item names refresh in-game. The newly split calculation
still needs a live smoke test. Custom skull spawning failed live, so the keypad
uses ordinary named blocks; letters appear in their hover names. Regenerate with
`python3 ref/build_chat.py` after building a matching model/vocabulary.

Check the full build and reproduce the example through actual menu clicks:

    /home/opc/light-lock.sh node sim/build-tools.mjs
    /home/opc/light-lock.sh node build/tools/htsw.mjs check house/import.json
    /home/opc/heavy-lock.sh node sim/check-chat.mjs

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
| `ref/build_chat.py` | Generates the keypad, packed prefix lookup, chat helpers, and `/llm` |
| `sim/check-chat.mjs` | Menu-to-reply regression and shared-state/timing checks |
| `sim/run.mjs` | HTSW simulator runner (checks sim == integer reference) |
| `probe/` | live throughput probe generator |
| `model/` | base tokenizer/model files used during training |

Rebuild the house:

    python3 ref/build.py --run runs/v512_d96_L2_f384_mqa --out build/<x> --ctx 48 --prompts prompts.txt
    vite-node sim/run.mjs build/<x>

Training data (`data/`, 2.3 GB) and the Python venv are not committed.
