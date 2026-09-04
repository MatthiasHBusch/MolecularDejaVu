"""
Level 1 vs Level 5, direct prediction only: does blocking structure recognition interrupt retrieval?

This is the comparison from the blinding study (Busch et al.), reproduced here on a benchmark the
audit finds heavily memorised, with the prompts taken from that study's `input_output_prompting`
and `io_sampleproperty_clear` entries rather than re-invented. The weighted-mean / pre-analysis
variants are deliberately not used: they add a reasoning scaffold that confounds the comparison.

    L1  chemist framing, IUPAC name + published SMILES, the property named, TRUE values
    L5  ML-regression framing, character-substituted structure string ONLY (no name, no property
        name), TRUE values

Only the STRUCTURE changes. The target scale is identical in both arms, so the two errors are
directly comparable -- which is what the label-transform axis cannot offer, because a model that
only partly recovers a relabelling is handicapped in a way that has nothing to do with recall.

One iteration per molecule: repeats measure sampling noise, and with a fixed budget the molecules
are worth more than the repeats.

    python src/run_blinding_l1_l5.py --dry
    python src/run_blinding_l1_l5.py --dataset esol --tag opus48 --ntest 100 --shots 100 --budget 8
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_reasoning_ladder as L

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG, SCREEN = os.path.join(ROOT, "src", "registry"), os.path.join(ROOT, "data", "screening")
OUT = os.path.join(ROOT, "results", "blinding")

# Verbatim role framings from the blinding study (Delaney_Prompts.jl).
SYS_L1 = """**Your Role**
You are a professional chemist with expert knowledge in physical chemistry and {prop} prediction.
You are tasked with predicting the {prop} ({unit}) of organic molecules based on their IUPAC names and SMILES strings.

**Problem Description**
You will be provided with:
1. A training dataset of molecules with their Names, SMILES, and known {prop} values.
2. A test molecule (Name and SMILES) for which you must predict the {prop}.

You have to use your knowledge and abilities to analyze the training datas molecules and the patterns and relationships between molecular properties and {prop} to make an accurate prediction."""

SYS_L5 = """**Your Role**
You are a professional machine learning model with expert knowledge in regression.
You are tasked with predicting a sample property based on a string based structure representation of the sample.

**Problem Description**
You will be provided with:
1. A training dataset of samples with their string based structure representation and known sample properties.
2. A test sample (string based structure representation) for which you must predict the sample property.

You have to use your knowledge and abilities to analyze the training datas samples and the patterns and relationships between sample properties to make an accurate prediction."""

USR_L1 = """**Training Data:**
- Names: {names}
- SMILES: {smiles}
- {Prop} ({unit}): {values}

**Prediction Task:**
Predict the {prop} for the following molecule:
- Name: {name}
- SMILES: {smi}

Provide your answer as a single numerical value with 3 significant figures at the end of your response."""

USR_L5 = """**Training Data:**
- Sample structure strings: {smiles}
- sample property: {values}

**Prediction Task:**
Predict the sample property for the following sample:
- Sample structure string: {smi}

Provide your answer as a single numerical value with 3 significant figures at the end of your response."""


def reasoning_tag(spec):
    """Filename component for the reasoning setting a cell was run at.

    The first sweep wrote `<ds>__<tag>__<level>.json`, which carries no record of the dose. That
    was survivable while every cell ran at its registry setting; it is not once the same cell is
    re-run at a controlled budget, because the existing file would either be skipped as complete
    or overwritten with a different experiment under the same name. Every new cell therefore
    carries its tag; the unsuffixed files already on disk stay as they are and the analysis reads
    them as the `legacy` arm.
    """
    s = str(spec)
    if s.startswith("max_tokens:"):
        return "t" + s.split(":", 1)[1]
    return {"none": "reg", "minimal": "reg"}.get(s, s)


def build(level, ds, train, row):
    j = lambda xs: ", ".join(str(x) for x in xs)
    prop, unit = ds["prompt_property"], ds["prompt_unit"]
    if level == "L1":
        return (SYS_L1.format(prop=prop, unit=unit),
                USR_L1.format(names=j(train.mol_id), smiles=j(train.smiles),
                              Prop=prop.capitalize(), prop=prop, unit=unit,
                              values=j(train.value), name=row.mol_id, smi=row.smiles))
    return (SYS_L5,
            USR_L5.format(smiles=j(train.smiles_blind), values=j(train.value),
                          smi=row.smiles_blind))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol")
    ap.add_argument("--tag", default="opus48")
    ap.add_argument("--ntest", type=int, default=100)
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--levels", default="L1,L5")
    ap.add_argument("--threads", type=int, default=10)
    # The completion cap has to cover the thinking budget AND the answer, and the budget does not
    # bind: measured here at a 1,024-token request, GPT-5.6 sol emits 3,845 thinking tokens on L5
    # and Kimi K3 3,009, because an unreadable structure string is what they deliberate hardest
    # about. The cap is therefore set from what the models DO, with room to spare, not from what
    # they were asked. A trace that eats the cap returns an empty answer, which scores as a miss
    # -- i.e. as a successful intervention -- so this failure has to be impossible rather than
    # unlikely. Nothing is charged for tokens that are not emitted, so the headroom is free.
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--budget", type=float, default=8.0)
    # Gemini 3.6 Flash needs this. Its registry setting is `minimal`, at which it emits zero
    # thinking tokens and does not retrieve at all -- the map calls it clean on all three
    # benchmarks. At a 128-token budget it reproduces 21% of LD50. Run at `minimal` the L1 arm
    # would measure nothing and the L1/L5 contrast would be empty. The comparison stays valid
    # because it is within-model: both arms use whatever setting is chosen here.
    ap.add_argument("--reasoning", default=None,
                    help="override the registry reasoning setting for this model")
    ap.add_argument("--seed", type=int, default=42)
    # ESOL, FreeSolv and LD50 share a lot of molecules -- 53% of FreeSolv is also in ESOL, and
    # 53% of ESOL is also in LD50. The labels are independent (solubility, hydration free
    # energy, toxicity) but the IDENTIFICATION step is shared, and that is precisely what L5
    # blocks. Drawing each benchmark's test set from the molecules the others do not contain
    # makes the arms independent; the in-context shots can come from anywhere, since they are
    # within-benchmark context and create no cross-benchmark dependence.
    ap.add_argument("--disjoint-from", default="",
                    help="comma-separated benchmarks whose molecules are excluded from the "
                         "TEST set (shots are unaffected)")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    ds = {d["key"]: d for d in json.load(open(os.path.join(REG, "datasets.json")))["datasets"]}[args.dataset]
    md = {m["tag"]: m for m in json.load(open(os.path.join(REG, "models.json")))["models"]}[args.tag]
    df = pd.read_csv(os.path.join(SCREEN, f"{args.dataset}.csv")).dropna(
        subset=["value", "smiles", "smiles_blind", "mol_id"])

    eligible = np.ones(len(df), dtype=bool)
    if args.disjoint_from:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(str(s)))
        other = set()
        for o in args.disjoint_from.split(","):
            od = pd.read_csv(os.path.join(SCREEN, f"{o.strip()}.csv")).dropna(subset=["smiles"])
            other |= {c for c in (canon(x) for x in od.smiles) if c}
        mine = [canon(x) for x in df.smiles]
        eligible = np.array([c is not None and c not in other for c in mine])
        print(f"  test molecules restricted to the {int(eligible.sum())} of {len(df)} not "
              f"present in {args.disjoint_from}")
        if eligible.sum() < args.ntest:
            sys.exit(f"only {int(eligible.sum())} eligible test molecules, need {args.ntest}")

    # One fixed split shared by both arms: the level effect must not be confounded with which
    # molecules were drawn.
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(df))
    test_pool = rng.permutation(idx[eligible])
    test = df.iloc[test_pool[:args.ntest]]
    # Shots come from whatever is left, eligible or not -- they are within-benchmark context.
    rest = rng.permutation(np.setdiff1d(idx, test_pool[:args.ntest]))
    train = df.iloc[rest[:args.shots]]
    print(f"{ds['name']} / {md['name']}: {len(test)} test, {len(train)} shots, "
          f"1 iteration, levels {args.levels}")

    if args.dry:
        for lv in args.levels.split(","):
            s, u = build(lv, ds, train.head(3), test.iloc[0])
            print(f"\n{'=' * 76}\n{lv}  (3 shots shown)\n{'=' * 76}")
            print(f"--- SYSTEM ---\n{s}\n--- USER ---\n{u}")
        print("\nDRY RUN: nothing queried.")
        return

    k = L.key()
    reasoning = args.reasoning or md["reasoning"]
    if args.reasoning:
        print(f"  reasoning setting overridden: {md['reasoning']} -> {reasoning}")
    os.makedirs(OUT, exist_ok=True)
    rtag = reasoning_tag(reasoning)
    for lv in args.levels.split(","):
        p = os.path.join(OUT, f"{args.dataset}__{args.tag}__{lv}__{rtag}.json")
        legacy = os.path.join(OUT, f"{args.dataset}__{args.tag}__{lv}.json")
        if os.path.exists(p):
            print(f"  [skip] {os.path.basename(p)} exists"); continue
        if os.path.exists(legacy):
            print(f"  [note] {os.path.basename(legacy)} exists at a different dose "
                  f"({json.load(open(legacy))['meta'].get('reasoning')}); "
                  f"writing {os.path.basename(p)} alongside it, nothing overwritten")
        t0, before = time.time(), L._spent
        jobs = [row for _, row in test.iterrows()]

        def one(row):
            s, u = build(lv, ds, train, row)
            for attempt in range(5):
                with L._lock:
                    if L._spent >= args.budget:
                        return None
                try:
                    r = L.call(k, md, reasoning, s, u, args.max_tokens)
                    with L._lock:
                        L._spent += r["cost"]
                    r["mol_id"] = str(row.mol_id).strip()
                    r["truth"] = float(row.value)
                    return r
                except Exception:
                    time.sleep(1.5 * (attempt + 1))
            return None

        calls = []
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            for i, r in enumerate(ex.map(one, jobs), 1):
                if r is not None:
                    calls.append(r)
                if i % 25 == 0:
                    print(f"       .. {lv} {i}/{len(jobs)}  ${L._spent - before:.3f}", flush=True)
        ok = [c for c in calls if c["value"] is not None]
        err = [abs(c["value"] - c["truth"]) for c in ok]
        json.dump(dict(meta=dict(dataset=args.dataset, tag=args.tag, level=lv,
                                 ntest=len(test), shots=len(train), seed=args.seed,
                                 reasoning=reasoning, arm=rtag, max_tokens=args.max_tokens,
                                 disjoint_from=args.disjoint_from,
                                 cost=L._spent - before,
                                 timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")),
                       calls=calls), open(p, "w"), indent=1)
        # Truncation is the one failure this experiment cannot absorb quietly: an answer eaten by
        # the completion cap scores as a miss, and a miss is what a clean cell looks like. Report
        # it per cell so a capped run cannot be read as an interrupted one.
        trunc = sum(1 for c in calls if c.get("finish") == "length")
        think = sorted(c.get("reasoning_tokens") or 0 for c in calls) or [0]
        print(f"  [ok] {lv}: {len(ok)}/{len(jobs)} parsed, medAE {np.median(err):.3f}, "
              f"median thinking {think[len(think) // 2]}/{args.max_tokens}, "
              f"{100 * trunc / max(len(calls), 1):.1f}% truncated, "
              f"${L._spent - before:.3f}, {time.time() - t0:.0f}s", flush=True)
    print(f"\nDONE. spent ${L._spent:.2f} of ${args.budget:.2f}")


if __name__ == "__main__":
    main()
