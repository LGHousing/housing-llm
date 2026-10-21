"""Compare read-only Housing exports with a fixed source commit.

Pass export folders oldest first; later folders override the same function.
Only reuse older exports if those functions have not changed in the house.
Known missing-name note/action pairs are reported as unreadable, never passed.
Additional content differences remain mismatches even when unreadable pairs exist.
Mismatch describes the exported file, not proven damage in the live house.
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
UNREADABLE_NOTE = re.compile(
    rb"^[ \t]*/// HTSW export: this Change Variable has no variable name in Housing "
    rb"\(read back: (.*)\)\r?\n?$")
EMPTY_ACTION = re.compile(
    rb'^(?P<pad>[ \t]*)(?P<kind>var|globalvar|teamvar)[ \t]+""(?P<tail>[^\r\n]*)\r?\n?$')
SOURCE_ACTION = re.compile(
    rb'^[ \t]*(?:var|globalvar|teamvar)[ \t]+[^\s"]+[^\r\n]*\r?\n?$')


def stub_covers(original, stub):
    if not SOURCE_ACTION.fullmatch(original) or not original.startswith(stub['pad'] + stub['kind'] + b' '):
        return False
    return not (stub['kind'] == b'teamvar' and stub['tail'].strip()) or original.split()[2] == stub['tail'].strip()


def source_locations(lines):
    """Editor positions for this build's flat conditional/action containers."""
    locations = []
    conditional, action, top = 0, 0, 0
    branch = None
    for line in lines:
        text = line.strip()
        if text.startswith(b'if (') and text.endswith(b'{'):
            conditional += 1
            branch, action = 'IF', 0
            location = f'conditional {conditional} condition'
        elif text == b'} else {':
            branch, action = 'ELSE', 0
            location = f'conditional {conditional} ELSE header'
        elif text == b'}':
            branch = None
            location = f'conditional {conditional} end'
        elif not text or text.startswith(b'//'):
            location = 'comment/blank'
        elif branch:
            action += 1
            location = f'conditional {conditional} {branch} action {action}'
        else:
            top += 1
            location = f'top-level action {top}'
        locations.append(location)
    return locations


def compact_differences(expected, collapsed):
    """Align on unchanged lines; keep unknown actions separate from other diffs."""
    normalized = [DIVISION_EXPORT.sub(rb'\1/=', line) for line, _, _ in collapsed]
    locations = source_locations(expected)
    differences = []

    def add(kind, a, b, c, d):
        differences.append(dict(
            kind=kind, source_line=a + 1, source_end_line=b if b > a else None,
            location=locations[a] if a < len(locations) else 'after source end',
            end_location=locations[b - 1] if b > a else None,
            export_line=collapsed[c][2] if c < len(collapsed) else None,
            expected=[line.decode().rstrip() for line in expected[a:b]],
            exported=[line.decode().rstrip() for line, _, _ in collapsed[c:d]]))

    for kind, a, b, c, d in difflib.SequenceMatcher(None, expected, normalized, autojunk=False).get_opcodes():
        if kind == 'equal':
            continue
        if kind == 'replace' and b - a == d - c:
            for i, j in zip(range(a, b), range(c, d)):
                stub = collapsed[j][1]
                if stub and stub_covers(expected[i], stub):
                    continue
                add('replace', i, i + 1, j, j + 1)
        else:
            add(kind, a, b, c, d)
    return differences


def compare(source, live, accept_division_export=False):
    """A missing action can explain one source action, never a missing block/line."""
    lines = live.splitlines(keepends=True)
    collapsed, unreadable = [], []
    i = 0
    while i < len(lines):
        note = UNREADABLE_NOTE.fullmatch(lines[i])
        stub = EMPTY_ACTION.fullmatch(lines[i + 1]) if note and i + 1 < len(lines) else None
        if stub:
            try:
                payload = json.loads(note[1])
            except (ValueError, UnicodeDecodeError):
                payload = None
            tail = stub['tail'].strip()
            # Team stubs may retain a team name; no operation/value may follow.
            valid_tail = not tail or (stub['kind'] == b'teamvar' and
                                      re.fullmatch(rb'[A-Za-z0-9_]+', tail) is not None)
            if isinstance(payload, dict) and payload.get('type') == 'CHANGE_VAR' and not payload.get('key') and valid_tail:
                unreadable.append(i + 1)
                collapsed.append((lines[i + 1], stub, i + 2))
                i += 2
                continue
        collapsed.append((lines[i], None, i + 1))
        i += 1

    expected = source.splitlines(keepends=True)
    other_difference = len(expected) != len(collapsed)
    division_difference = False
    for original, (exported, stub, _) in zip(expected, collapsed):
        if stub:
            # Preserve known holder/indent information, and only cover an action.
            if not stub_covers(original, stub):
                other_difference = True
        elif exported != original:
            if DIVISION_EXPORT.sub(rb'\1/=', exported) == original:
                division_difference = True
            else:
                other_difference = True

    if other_difference:
        status = 'mismatch'
    elif unreadable:
        status = 'unreadable'
    elif division_difference:
        status = 'division_accepted' if accept_division_export else 'division_unresolved'
    else:
        status = 'exact'
    return dict(status=status, unreadable_actions=len(unreadable),
                unreadable_export_lines=unreadable,
                division_difference=division_difference,
                differences=compact_differences(expected, collapsed) if other_difference else [])


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
            record.update(compare(source, live, args.accept_division_export))
            if record["status"] != "exact":
                record["diff"] = list(difflib.unified_diff(
                    source.decode().splitlines(keepends=True),
                    live.decode().splitlines(keepends=True),
                    fromfile=f"{revision}:{function['actions']}", tofile=str(exported)))
                print(f"{record['status']}: {name} (unreadable actions: {record['unreadable_actions']}, "
                      f"other differences: {len(record['differences'])})")
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
