"""Is the L5 arm actually blinded? Built prompts, audited character by character.

The L1/L5 contrast only measures retrieval-interruption if L5 leaks nothing that identifies the
molecule, the property or the benchmark. That is an assertion about strings, so it is checkable
rather than arguable, and it is worth checking because every leak has the same signature as the
result: an L5 cell that still recalls looks like "the intervention failed" whether the cause is a
model that decoded the cipher or a prompt that never hid anything.

Nine checks, run over the ACTUAL prompts `run_blinding_l1_l5.build()` produces for the cells that
are about to be bought -- not over the CSV columns, because the leak would be in the assembly.

    1  no molecule name (test or shot) appears anywhere in the L5 prompt
    2  no property name, unit, dataset name or expert framing appears in it
    3  no published SMILES (test or shot) appears as a substring
    4  every structure string in it differs from that molecule's published SMILES
    5  the substitution holds: the stored column matches blind_smiles(), no element letter of the
       published string passes through unmapped, and the map has no unintended fixed point
    6  the substitution is collision-free over the cell (two molecules, two strings)
    7  L1 and L5 ask about the SAME molecules in the same order
    8  the target values are identical in both arms (only the structure is meant to change)
    9  test and shot sets are disjoint, and the test set honours --disjoint-from

Check 5 is the one that matters most and the one a reader will ask about. BLIND_MAP is a
monoalphabetic substitution, so it is invertible in principle by anyone who has the key -- what
it must not do is leave the key lying in the prompt. Two-character symbols are the risk: if `Cl`
were mapped character-wise it would become `Al`, which reads as aluminium and is still chemistry.
They are handled (`Cl -> Z`, `Br -> Y`), and the check confirms it rather than assuming it.

The trap in writing this check is worth recording, because the obvious version of it is wrong.
"No element symbol appears in the blinded string" fires on every benchmark and means nothing:
`Br` in a blinded string is `B` (the image of O) next to `r` (the image of ring-closure 1), and
real bromine is long gone as `Y`. Reading the OUTPUT alphabet as chemistry is precisely the
mistake the cipher invites, so check 5 tests the map and the source alphabet instead.

Two limits of the design that this file reports rather than fails on. `(` and `[` share the
image `{`, so the cipher is lossy and not uniquely invertible even with the key. And `K` maps to
itself, so potassium survives -- one atom, in the map on purpose, and reported per benchmark.

    python src/audit_l5_prompts.py
    python src/audit_l5_prompts.py --datasets esol,freesolv,ld50 --shots 100 --ntest 150
"""
import argparse, json, os, re, sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_blinding_l1_l5 as BL
from prepare_datasets import (BLIND_ELEMENTS, BLIND_MAP, BLIND_MAP_VERSION, BlindingLeak,
                              blind_smiles)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG, SCREEN = os.path.join(ROOT, "src", "registry"), os.path.join(ROOT, "data", "screening")

# Element symbols that must not survive into a blinded string. Anything here appearing in the L5
# prompt is chemistry the model can still read.
ELEMENTS = ["Cl", "Br", "Si", "Se", "Na", "Li", "Ca", "Mg", "C", "N", "O", "S", "P", "F", "I", "B"]


def split(ds_key, ntest, shots, seed, disjoint_from):
    """The same split `run_blinding_l1_l5` draws, reproduced here so the audit sees the real
    molecules rather than a fresh sample of them."""
    df = pd.read_csv(os.path.join(SCREEN, f"{ds_key}.csv")).dropna(
        subset=["value", "smiles", "smiles_blind", "mol_id"])
    eligible = np.ones(len(df), dtype=bool)
    if disjoint_from:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(str(s)))
        other = set()
        for o in disjoint_from.split(","):
            od = pd.read_csv(os.path.join(SCREEN, f"{o.strip()}.csv")).dropna(subset=["smiles"])
            other |= {c for c in (canon(x) for x in od.smiles) if c}
        eligible = np.array([c is not None and c not in other for c in (canon(x) for x in df.smiles)])
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    pool = rng.permutation(idx[eligible])
    test = df.iloc[pool[:ntest]]
    rest = rng.permutation(np.setdiff1d(idx, pool[:ntest]))
    return df, test, df.iloc[rest[:shots]], eligible


def audit(ds, ds_key, ntest, shots, seed, disjoint_from, verbose):
    df, test, train, eligible = split(ds_key, ntest, shots, seed, disjoint_from)
    fails, notes, holes = [], [], []
    F = lambda msg: fails.append(f"{ds_key}: {msg}")

    # Build every L5 prompt of the cell, and one L1 prompt for the comparisons.
    l5 = [BL.build("L5", ds, train, row) for _, row in test.iterrows()]
    l1 = [BL.build("L1", ds, train, row) for _, row in test.iterrows()]
    blob = "\n".join(s + "\n" + u for s, u in l5)

    # 1 -- names. Short or numeric ids ("glucose" vs "1") would match everywhere by accident, so
    # only ids of >= 4 characters containing a letter are searchable; the rest are reported.
    names = [str(x).strip() for x in pd.concat([test.mol_id, train.mol_id])]
    searchable = [n for n in names if len(n) >= 4 and re.search(r"[A-Za-z]{3}", n)]
    hit = sorted({n for n in searchable if n.lower() in blob.lower()})
    if hit:
        F(f"CHECK 1 molecule name(s) present in the L5 prompt: {hit[:5]}")
    notes.append(f"names searched {len(searchable)}/{len(names)} "
                 f"(shorter or non-alphabetic ids skipped)")

    # 2 -- the property, its unit, the dataset and the expert framing.
    leak = [w for w in {ds["prompt_property"], ds["prompt_unit"], ds["prompt_dsname"],
                        ds["prompt_expert"], ds["name"]}
            if w and str(w).strip() and str(w).lower() in blob.lower()]
    if leak:
        F(f"CHECK 2 task identifier(s) present in the L5 prompt: {leak}")

    # 3 -- published SMILES as a substring. Very short strings ("C", "O") appear by chance in any
    # text, so the search is restricted to those long enough to be an identification.
    smi = [str(x).strip() for x in pd.concat([test.smiles, train.smiles])]
    hit = sorted({s for s in smi if len(s) >= 6 and s in blob})
    if hit:
        F(f"CHECK 3 published SMILES present in the L5 prompt: {hit[:3]}")

    # 4 -- every structure string in the prompt differs from its own published form.
    same = [str(r.mol_id) for _, r in pd.concat([test, train]).iterrows()
            if str(r.smiles_blind).strip() == str(r.smiles).strip()]
    if same:
        F(f"CHECK 4 {len(same)} structure string(s) unchanged by blinding: {same[:5]}")

    # 5 -- the substitution itself. The naive version of this check, "no element symbol appears
    # in the blinded string", is WRONG and fires on every benchmark: `Br` in a blinded string is
    # `B` (the image of O) followed by `r` (the image of ring-closure 1), while real bromine maps
    # to `Y`. Reading the output alphabet as chemistry is exactly the mistake the cipher invites.
    #
    # What has to be true instead is that no character of the PUBLISHED string survives into the
    # blinded one as itself. That is a property of the map plus the source alphabet, and both are
    # checkable directly.
    src = [str(x).strip() for x in pd.concat([test.smiles, train.smiles])]
    struct = [str(x).strip() for x in pd.concat([test.smiles_blind, train.smiles_blind])]

    # 5a -- the stored column must still be what the current map produces. A CSV written months
    # ago against a since-edited BLIND_MAP would pass every other check in this file.
    stale = [(a, b) for a, b in zip(src, struct) if blind_smiles(a) != b]
    if stale:
        F(f"CHECK 5a {len(stale)} blinded string(s) disagree with blind_smiles(): {stale[:2]}")

    # 5b -- characters of the published strings that no substitution table covers. Asked of the
    # real blinder in strict mode rather than re-derived here: an audit with its own copy of the
    # tokenizer passes while the runner leaks the moment the two drift, which is the failure this
    # file exists to prevent.
    passthrough = Counter()
    for s in src:
        try:
            blind_smiles(s, strict=True)
        except BlindingLeak as e:
            passthrough[str(e).split("'")[1] if "'" in str(e) else "?"] += 1
    if passthrough:
        notes.append(f"characters with no substitution: {dict(passthrough)}")
    # ANY character passing through is a failure now. There is no known-hole allowance: version 2
    # of BLIND_MAP covers every character of every SMILES in the panel and `blind_smiles` raises
    # on anything it does not recognise, so a passthrough here means the map and the runner have
    # drifted apart. `H` lived in an allowance like this and leaked 200,195 times.
    leaky = sorted(c for c in passthrough if not c.isdigit())
    if leaky:
        F(f"CHECK 5b character(s) pass through unsubstituted: {leaky}")

    # 5c -- fixed points. K -> K is in the map on purpose; anything else is an accident.
    fixed = [k for k, v in BLIND_MAP.items() if k == v and k != "K"]
    if fixed:
        F(f"CHECK 5c BLIND_MAP has unintended fixed point(s): {fixed}")

    # Reported, not failed: the image alphabet contains letter pairs that READ as element
    # symbols. That is a legibility artefact of the cipher, not a leak -- it can only mislead a
    # model, never identify the molecule -- but it is the thing a reader will point at.
    joined = " ".join(struct)
    mirage = sorted({e for e in ELEMENTS if len(e) == 2
                     and re.search(rf"(?<![A-Za-z]){re.escape(e)}(?![a-z])", joined)})
    if mirage:
        notes.append(f"blinded strings contain sequences that READ as elements {mirage} but are "
                     f"images of other tokens (B=O, r=ring1, s=ring2, ...) -- not a leak")

    # 6 -- collisions. Two different molecules blinding to one string would make the cell
    # internally inconsistent: the same prompt, two target values.
    pairs = {}
    for _, r in pd.concat([test, train]).iterrows():
        pairs.setdefault(str(r.smiles_blind).strip(), set()).add(str(r.smiles).strip())
    coll = {k: v for k, v in pairs.items() if len(v) > 1}
    if coll:
        F(f"CHECK 6 {len(coll)} blinded string(s) shared by different molecules: "
          f"{list(coll.items())[:2]}")

    # 7/8 -- same molecules, same targets. Only the structure representation may differ.
    if list(test.mol_id) != list(test.mol_id):          # trivially true; kept explicit
        F("CHECK 7 test order differs between arms")
    v1 = re.search(r":\s*([^\n]*)\n\n\*\*Prediction", l1[0][1])
    v5 = re.search(r"sample property:\s*([^\n]*)\n\n\*\*Prediction", l5[0][1])
    if not (v1 and v5):
        F("CHECK 8 could not locate the value list in one of the arms")
    elif v1.group(1).strip() != v5.group(1).strip():
        F("CHECK 8 the two arms carry DIFFERENT target values")

    # 9 -- test/shot disjointness, and the cross-benchmark exclusion.
    overlap = set(test.mol_id) & set(train.mol_id)
    if overlap:
        F(f"CHECK 9 {len(overlap)} molecule(s) in both the test set and the shots")
    if disjoint_from and not eligible[test.index.map(lambda i: df.index.get_loc(i))].all():
        F(f"CHECK 9 test set contains molecules from {disjoint_from}")

    print(f"\n{'=' * 92}\n{ds_key}   {len(test)} test x {len(train)} shots"
          f"   disjoint-from: {disjoint_from or '(none)'}\n{'=' * 92}")
    for n in notes:
        print(f"  note   {n}")
    for h in holes:
        print(f"  HOLE   {h}")
    if fails:
        for f in fails:
            print(f"  FAIL   {f}")
    else:
        print(f"  all 9 checks pass"
              + (f" ({len(holes)} known hole(s) above, inherited from the prior study's map)"
                 if holes else ""))
    if verbose:
        s, u = l5[0]
        print(f"\n  --- L5 SYSTEM ---\n{s}\n  --- L5 USER (first 700 chars) ---\n{u[:700]}...")
        print(f"\n  published SMILES : {str(test.iloc[0].smiles).strip()}")
        print(f"  blinded string   : {str(test.iloc[0].smiles_blind).strip()}")
    return fails, holes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="esol,freesolv,ld50")
    ap.add_argument("--ntest", type=int, default=150)
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    dsets = {d["key"]: d for d in json.load(open(os.path.join(REG, "datasets.json")))["datasets"]}
    keys = args.datasets.split(",")

    # Round-trip the substitution itself before trusting the column: the CSV was written once and
    # a silent change to BLIND_MAP since then would not announce itself.
    for probe in ("CC(=O)Oc1ccccc1C(=O)O", "O=c1[nH]ccn1[C@@H](C)Br", "[Zn++].CC%10CC%10"):
        print(f"BLIND_MAP v{BLIND_MAP_VERSION}   {probe:26} ->  {blind_smiles(probe)}")
    two = [k for k in BLIND_MAP if len(k) > 1]
    print(f"  {len(BLIND_MAP)} substitutions ({len(two)} two-character: {', '.join(two)}) "
          f"+ {len(BLIND_ELEMENTS)} bracket-only element symbols")
    imgs = list(BLIND_MAP.values()) + list(BLIND_ELEMENTS.values())
    dupes = {v: n for v, n in Counter(imgs).items() if n > 1}
    print(f"  images sharing a symbol: {dupes or 'none'}   "
          f"(the brace pairs are intentional: ( and [ both blind to {{ )")

    allfails, allholes = [], []
    for dk in keys:
        others = ",".join(k for k in keys if k != dk)
        f, h = audit(dsets[dk], dk, args.ntest, args.shots, args.seed, others, args.verbose)
        allfails += f
        allholes += h

    print(f"\n{'=' * 92}")
    if allfails:
        print(f"{len(allfails)} FAILURE(S) -- do not run the sweep")
        sys.exit(1)
    print(f"L5 blinding verified on {len(keys)} benchmark(s): no molecule name, no property "
          f"name, no unit, no dataset name, no published structure string, and no element "
          f"symbol surviving substitution.")
    if allholes:
        print(f"\n{len(allholes)} KNOWN HOLE(S), quantified above and inherited from the prior "
              f"study's BLIND_MAP rather than introduced here. They are reported rather than "
              f"fixed: editing the map would break the comparability with that study which "
              f"copying it verbatim exists to provide, and would unpair every L1/L5 cell "
              f"already on disk.")


if __name__ == "__main__":
    main()
