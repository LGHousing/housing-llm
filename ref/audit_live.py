"""Compare read-only Housing exports with a fixed source commit.

Pass export folders oldest first; later folders override the same function.
Only reuse older exports if those functions have not changed in the house.
Exit 1 means the audit is incomplete or has an unaccepted difference.
"""
import argparse
from collections import Counter
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess


# Match only the operation field of an action, never text inside a value/chat.
DIVISION_EXPORT = re.compile(
    rb"^([ \t]*(?:(?:var|globalvar)[ \t]+\S+|teamvar[ \t]+\S+[ \t]+\S+)[ \t]+)"
    rb"\xe2\x80\xa6/=(?=[ \t])", re.MULTILINE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", type=Path, nargs="+")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--accept-division-export", action="store_true",
                        help="Accept only the confirmed …/= export spelling of Divide; retain raw diffs")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments])

    revision = git("rev-parse", "--verify", args.revision + "^{commit}").decode().strip()
    manifest = json.loads(git("show", f"{revision}:house/import.json"))
    for folder in args.exports:
        if not folder.is_dir():
            parser.error(f"Export folder does not exist: {folder}")
    records = []
    for function in manifest["functions"]:
        name = function["name"]
        candidates = [folder / f"{name}.live.htsl" for folder in args.exports]
        exported = next((p for p in reversed(candidates) if p.is_file()), None)
        source = git("show", f"{revision}:house/{function['actions']}")
        record = dict(name=name, source=function["actions"],
                      source_sha256=hashlib.sha256(source).hexdigest())
        if exported is None:
            record["status"] = "missing"
        else:
            live = exported.read_bytes()
            record.update(export=str(exported.resolve()),
                          export_sha256=hashlib.sha256(live).hexdigest())
            if live == source:
                record["status"] = "exact"
            elif DIVISION_EXPORT.sub(rb"\1/=", live) == source:
                record["status"] = "division_accepted" if args.accept_division_export else "division_unresolved"
            else:
                record["status"] = "mismatch"
            if record["status"] != "exact":
                record["diff"] = list(difflib.unified_diff(
                    source.decode().splitlines(keepends=True),
                    live.decode().splitlines(keepends=True),
                    fromfile=f"{revision}:{function['actions']}", tofile=str(exported)))
                print(f"{record['status']}: {name}")
        records.append(record)
    counts = dict(Counter(r["status"] for r in records))
    report = dict(revision=revision, accept_division_export=args.accept_division_export,
                  counts=counts, functions=records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{revision}: {counts}; report: {args.report}")
    return int(any(r["status"] not in ("exact", "division_accepted") for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
