"""Conservative stat-key screening; Hypixel's private filter still needs live checks."""
import re

# Confirmed live: entering n19 is refused and leaves the default key "Kills".
# Keep internal tensor names and holder allocation stable; rename only on output.
# Also avoid the same suspicious prefix at longer indices and with 6 read as g.
# Only n19 is confirmed blocked; these additional renames are precautionary.
ALIASES = {f"n{i}": f"norm{i}" for i in (16, 19, *range(160, 170), *range(190, 200))}
LEET = str.maketrans("0123456789", "oizeasgtbg")
SUSPECT = re.compile(r"n[il]g|fuck|shit|cunt|fag|dick|cock|porn|sex|bitch")


def check_names(names):
    for name in sorted(set(names)):
        folded = re.sub(r"[^a-z0-9]", "", name.lower()).translate(LEET)
        if SUSPECT.search(folded):
            raise ValueError(f"Possible Housing-filtered variable {name!r}; choose a neutral alias")


def housing_name(name):
    result = ALIASES.get(name, name)
    check_names([result])
    return result


def check_htsl_names(text):
    # Include writes, conditions and placeholder reads, including the static
    # part of computed names. Callers generating computed keys must also check
    # their concrete keys (the keypad loader writes all of its keys explicitly).
    names = re.findall(r"\b(?:var|globalvar|teamvar)\s+([A-Za-z0-9_,]+)", text)
    names += re.findall(r"%var\.(?:player|global|team)/([A-Za-z0-9_,]+)", text)
    check_names(names)
