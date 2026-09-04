"""Regenerate ONLY the `smiles_blind` column, under the current BLIND_MAP.

Why this is not `prepare_datasets.py --force`. That rebuilds the whole file, and rebuilding the
whole file regenerates `smiles_random` -- which silently unpairs every randomised-SMILES result
already collected, the failure the README warns about and the reason the ESOL randomisations are
pinned by `random_override` in the registry. The blinding fix touches one column and this script
touches one column; `smiles`, `value`, `smiles_random` and every value transform are copied
through byte for byte and checked afterwards.

It also runs the strict blinder over every row first, so a benchmark with a character no
substitution table covers fails BEFORE anything is written rather than halfway through.

The rewrite is TEXTUAL, field by field, and that is not fastidiousness. Round-tripping the file
through pandas moved `value_blind` by one unit in the last place on 5 of AqSolDB's 1,006 rows --
`31.501057082452437` came back as `31.50105708245244`, a different double -- because `to_csv`
does not always emit the shortest round-tripping repr. The change is numerically nothing and
would never have been noticed; it is also a silent edit to a target column, which is the kind of
thing this project has already been bitten by twice. Reading every field as a string and putting
back the one that changed makes the frozen columns byte-identical by construction rather than by
inspection.

    python src/reblind_datasets.py --dry            # what would change, per benchmark
    python src/reblind_datasets.py                  # rewrite, keeping a .bak of each file
"""
import argparse, csv, glob, os, shutil, sys
from collections import Counter

import pandas as pd

csv.field_size_limit(1 << 24)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_datasets import BLIND_MAP_VERSION, BlindingLeak, blind_smiles

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCREEN = os.path.join(ROOT, "data", "screening")

# Everything that must survive untouched. `smiles_random` is the one that would cost real money
# if it moved, but a silently altered target is worse, so the whole schema is compared.
FROZEN = ["mol_id", "smiles", "value", "smiles_random",
          "value_affine", "value_nonmono", "value_sine", "value_blind"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="", help="default: every screening CSV")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    files = ([os.path.join(SCREEN, f"{k}.csv") for k in args.datasets.split(",")]
             if args.datasets else sorted(glob.glob(os.path.join(SCREEN, "*.csv"))))
    files = [f for f in files if not os.path.basename(f).startswith("_")]

    print(f"BLIND_MAP version {BLIND_MAP_VERSION}, strict\n")
    plan, leaks = [], 0
    for f in files:
        key = os.path.basename(f)[:-4]
        # Every field as a string, so nothing is parsed and nothing can be reformatted.
        with open(f, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], rows[1:]
        if "smiles_blind" not in header or "smiles" not in header:
            print(f"  {key:14s} no smiles_blind column -- skipped")
            continue
        ci, cb = header.index("smiles"), header.index("smiles_blind")
        new, bad, changed = [], Counter(), 0
        for r in body:
            try:
                b = blind_smiles(r[ci])
            except BlindingLeak as e:
                b = None
                bad[str(e).split("'")[1] if "'" in str(e) else "?"] += 1
            new.append(b)
            if b is not None and b != r[cb]:
                changed += 1
        if bad:
            leaks += 1
            print(f"  {key:14s} LEAK: {sum(bad.values())} row(s) contain characters with no "
                  f"substitution: {dict(bad)}")
            continue
        print(f"  {key:14s} {len(body):6d} rows, {changed:6d} change "
              f"({100 * changed / max(len(body), 1):5.1f}%)")
        plan.append((f, key, header, body, cb, new))

    if leaks:
        sys.exit(f"\n{leaks} benchmark(s) leak -- extend BLIND_MAP before rewriting anything")
    if args.dry:
        print("\nDRY RUN: nothing written.")
        return

    for f, key, header, body, cb, new in plan:
        bak = f + ".bak_blindv1"
        if not os.path.exists(bak):
            shutil.copy2(f, bak)
        for r, b in zip(body, new):
            r[cb] = b
        tmp = f + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            w.writerows(body)
        os.replace(tmp, f)
        # Read back and prove the frozen fields are BYTE-identical to the backup. A rewrite that
        # quietly reformatted a float or dropped a row would be invisible until an analysis
        # disagreed with a result file three weeks later.
        with open(bak, newline="", encoding="utf-8") as fh:
            oldrows = list(csv.reader(fh))
        with open(f, newline="", encoding="utf-8") as fh:
            newrows = list(csv.reader(fh))
        cols = [header.index(c) for c in FROZEN if c in header]
        if len(oldrows) != len(newrows):
            sys.exit(f"{key}: row count changed -- restore from {os.path.basename(bak)}")
        diff = [i for i, (a, b) in enumerate(zip(oldrows, newrows))
                if [a[j] for j in cols] != [b[j] for j in cols]]
        if diff:
            sys.exit(f"{key}: frozen fields changed on {len(diff)} row(s) "
                     f"-- restore from {os.path.basename(bak)}")
        print(f"  {key:14s} rewritten; {len(cols)} frozen column(s) byte-identical")

    print(f"\nDone. Backups at data/screening/*.csv.bak_blindv1")
    print("Re-run `python src/audit_l5_prompts.py` before buying anything.")


if __name__ == "__main__":
    main()
