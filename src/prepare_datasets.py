"""
Dataset preparation for the expanded memorization screening.

Downloads / loads every dataset in src/registry/datasets.json and writes it to
data/screening/<key>.csv in one canonical schema, so that adding a benchmark to the study
is a data-only change (edit the registry, re-run this) rather than a code change.

Output columns
--------------
mol_id        stable key used as the JSON key in the result files. Guaranteed unique.
smiles        canonical SMILES exactly as published in the source file (this is what the
              models are shown in the main condition -- NOT re-canonicalised, because the
              published string is the one that appears in the corpus).
value         the ground-truth target.
smiles_random a random, valid, non-canonical SMILES for the SAME molecule (RDKit random
              atom ordering), verified to round-trip to an identical canonical structure.
              This is the chemistry-invariance control: chemistry is invariant to the
              rewriting, a lookup keyed to the published string is not.
smiles_blind  the prior study's deterministic character substitution (blinding levels 5/6):
              structure-preserving but chemically unrecognisable.
value_blind   the prior study's rank-preserving affine target transform (levels 2/4/6):
              inverted and rescaled to [0, 100].

Idempotent: an existing data/screening/<key>.csv is left alone unless --force is given,
so the randomised SMILES stay fixed across runs (they must, or the control is not paired).

Usage
-----
    python src/prepare_datasets.py                # prepare everything that is missing
    python src/prepare_datasets.py --only esol qm8
    python src/prepare_datasets.py --force        # regenerate (invalidates existing runs!)
"""
import argparse, io, json, os, sys, urllib.request
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG = os.path.join(ROOT, "src", "registry", "datasets.json")
OUT = os.path.join(ROOT, "data", "screening")

# Deterministic character substitution for the SMILES-blinding levels. Ring closures, branching
# and connectivity survive, chemical identity does not.
#
# VERSION 2, and the change is a leak fix rather than a preference. Version 1 was copied verbatim
# from the prior blinding study for comparability, and it had no entry for `H`, so hydrogen inside
# brackets passed through untouched: `[nH]` blinded to `{dH}` and `[C@@H]` to `{A**H}`. That is
# 200,195 occurrences over the panel and 4-11% of the molecules of the three benchmarks this study
# blinds, 32.7% of one 150-molecule test set. Worse, `I` maps to `H`, so an `H` in a version-1
# blinded string was ambiguous between iodine and a surviving hydrogen -- the leak and an image of
# the cipher wearing the same character.
#
# Three further source characters had no entry and passed through: `%` (the ring-closure marker
# for rings above 9, which also collides with `+`'s image), `0`, and every element symbol outside
# the organic subset -- 48 of them, all metals, all in AqSolDB. Version 2 covers all of them and
# `blind_smiles` now RAISES on anything it does not recognise, so the failure mode is a stack
# trace rather than a quiet leak. That is the actual fix; the added entries are its consequence.
#
# Cost of the change, stated because it is real: blinded strings are no longer character-identical
# to the prior study's, so the L5 arm is comparable to it by design and protocol rather than
# verbatim. The 26 L1/L5 cells already on disk were run under version 1 and are kept as the
# `legacy` arm; they are not re-scored against a version-2 column.
BLIND_MAP_VERSION = 2

BLIND_MAP = {
    "C": "A", "O": "B", "N": "D", "S": "E", "P": "F", "F": "G", "I": "H", "B": "J", "K": "K",
    "c": "a", "o": "b", "n": "d", "s": "e", "p": "f",
    "Cl": "Z", "Br": "Y", "Si": "X", "Se": "W", "Na": "V", "Li": "U", "Ca": "T", "Mg": "S",
    "1": "r", "2": "s", "3": "t", "4": "u", "5": "v", "6": "w", "7": "x", "8": "y", "9": "z",
    "=": "~", "#": "^", "(": "{", ")": "}", "[": "{", "]": "}", "@": "*", "+": "%", "-": "_",
    "/": "|", "\\": "!", ".": ",",
    # --- version 2 ---
    "H": "Q",    # was absent, so hydrogen survived. `Q` is not an element symbol and is not an
                 # image of anything else, so `H` in the output now means iodine and only iodine.
    "0": "q",    # was absent. Completes the digit run: 0-9 -> q,r,s,t,u,v,w,x,y,z.
    "%": "&",    # ring closures above 9 (`%10`). Was absent AND `+` maps to `%`, so a surviving
                 # `%` was indistinguishable from a blinded charge.
}

# Element symbols that occur ONLY inside brackets. They are matched bracket-locally rather than
# globally, which the organic-subset symbols above cannot be: `Cl` is chlorine wherever it
# appears, but `Sn` outside brackets is a sulfur bonded to an aromatic nitrogen at least as often
# as it is tin, and blinding it as tin would destroy a bond that the design keeps.
#
# All 48 are in AqSolDB and none in the three benchmarks this study blinds, so this table exists
# to make the guard below true rather than to change any string in the experiment.
_BRACKET_ONLY = [
    "Ag", "Al", "As", "Au", "Ba", "Be", "Bi", "Cd", "Ce", "Co", "Cr", "Cs", "Cu", "Dy", "Fe",
    "Gd", "Ge", "Hf", "Hg", "In", "Ir", "La", "Lu", "Mn", "Mo", "Nb", "Nd", "Ni", "Pb", "Pd",
    "Pr", "Pt", "Re", "Rh", "Ru", "Sb", "Sm", "Sn", "Sr", "Ta", "Te", "Ti", "Tl", "V", "W",
    "Y", "Zn", "Zr", "se",
]
# `$` prefixes every one of these and is used for nothing else, so `$a` is one token and can
# never be read as `$` next to `a`. `$` does not occur in any SMILES of the panel.
BLIND_ELEMENTS = {e: f"${chr(97 + i)}" if i < 26 else f"${chr(65 + i - 26)}"
                  for i, e in enumerate(sorted(_BRACKET_ONLY))}

_MULTI = sorted([k for k in BLIND_MAP if len(k) > 1], key=len, reverse=True)
_MULTI_BR = sorted(BLIND_ELEMENTS, key=len, reverse=True)


class BlindingLeak(ValueError):
    """A character no substitution table covers. Raised rather than passed through, because a
    passed-through character is invisible in the output and identifies chemistry in the prompt."""


def blind_smiles(s: str, strict: bool = True) -> str:
    """Structure-preserving, chemistry-destroying substitution.

    Bracket contents are substituted with the full element table, everything else with the
    organic-subset table. `strict` is the point of the function: an unmapped character used to
    be emitted unchanged, which is how `H` survived 200,195 times without anything failing.
    """
    out, i, in_bracket = [], 0, False
    while i < len(s):
        if s[i] == "[":
            in_bracket = True
        elif s[i] == "]":
            in_bracket = False
        if in_bracket:
            for m in _MULTI_BR:
                if s.startswith(m, i):
                    out.append(BLIND_ELEMENTS[m]); i += len(m); break
            else:
                for m in _MULTI:
                    if s.startswith(m, i):
                        out.append(BLIND_MAP[m]); i += len(m); break
                else:
                    if s[i] not in BLIND_MAP:
                        if strict:
                            raise BlindingLeak(
                                f"no substitution for {s[i]!r} in {s!r} (bracket) -- it would "
                                f"pass through into the blinded string")
                        out.append(s[i])
                    else:
                        out.append(BLIND_MAP[s[i]])
                    i += 1
            continue
        for m in _MULTI:
            if s.startswith(m, i):
                out.append(BLIND_MAP[m]); i += len(m); break
        else:
            if s[i] not in BLIND_MAP:
                if strict:
                    raise BlindingLeak(
                        f"no substitution for {s[i]!r} in {s!r} -- it would pass through into "
                        f"the blinded string")
                out.append(s[i])
            else:
                out.append(BLIND_MAP[s[i]])
            i += 1
    return "".join(out)


def blind_values(v: np.ndarray) -> np.ndarray:
    """Rank-inverting affine transform onto [0, 100] (prior study, blinding levels 2/4/6).

    Linear, so the intrinsic difficulty of the regression is unchanged; inverted and
    rescaled, so neither the value nor the scale of the original benchmark is recognisable.

    Crucially this map is MONOTONIC, and that is the point of comparing it with the two
    below: a model that recalls a molecule's true value can recover a rank-preserving map
    from a handful of in-context anchors and push its recollection through it. Monotonic
    blinding therefore hides the values without disabling recall.
    """
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi == lo:
        return np.zeros_like(v)
    return (hi - v) * 100.0 / (hi - lo)


def nonmono_values(v: np.ndarray, k: int = 5, perm=(2, 4, 1, 0, 3)) -> np.ndarray:
    """Binned permutation onto [0, 100]: non-monotonic but invertible.

    Equal-count bins reordered by a fixed derangement, order preserved within each bin. The
    global map cannot be recovered by assuming monotonicity from a few anchors, so recall of
    the true value is useless -- but the map is information-preserving, so a genuine
    in-context learner can still fit it. That contrast is what separates the two.
    Definition taken unchanged from the prior blinding study for comparability.
    """
    y = np.asarray(v, float)
    edges = np.quantile(y, np.linspace(0, 1, k + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    out = np.empty_like(y)
    for b in range(k):
        mask = (y > edges[b]) & (y <= edges[b + 1])
        if not mask.any():
            continue
        yb = y[mask]
        u = ((yb - yb.min()) / (yb.max() - yb.min()) if yb.max() > yb.min()
             else np.full(yb.shape, 0.5))
        out[mask] = (perm[b] + u) / k * 100.0
    return out


def sine_values(v: np.ndarray, periods: int = 2) -> np.ndarray:
    """Sine over whole periods onto [0, 100]: continuous, non-monotonic, many-to-one.

    Unlike the binned permutation this has no boundary discontinuities, so local structural
    similarity is preserved everywhere; the cost is non-injectivity, which lowers the
    achievable ceiling near the turning points. Over whole periods Pearson(out, v) ~ 0, so
    recall of the true value carries no information about the target at all.
    """
    y = np.asarray(v, float)
    if y.max() == y.min():
        return np.zeros_like(y)
    u = (y - y.min()) / (y.max() - y.min())
    return (np.sin(2.0 * np.pi * periods * u) + 1.0) / 2.0 * 100.0


def randomize_smiles(smi: str, seed: int):
    """A random valid non-canonical SMILES for the same molecule.

    Returns (string, ok). ok is False when no distinct valid rewriting could be produced --
    the string is then the canonical form. Callers must count the failures: a control that
    silently degrades to "canonicalise everything" tests nothing, and the identity is
    invisible in the output file for any molecule whose published SMILES is already canonical.

    Verified by round-trip: the returned string must parse back to the identical canonical
    SMILES, otherwise the 'same molecule, different string' premise is broken.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "", False
    ref = Chem.MolToSmiles(mol)
    rng = np.random.default_rng(seed)
    for _ in range(30):
        # RenumberAtoms rejects numpy integers; passing them raises and would send every
        # molecule down the fallback path, turning the whole control into a no-op.
        order = [int(x) for x in rng.permutation(mol.GetNumAtoms())]
        try:
            rnd = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
        except Exception:
            continue
        if rnd == smi:
            continue
        back = Chem.MolFromSmiles(rnd)
        if back is not None and Chem.MolToSmiles(back) == ref:
            return rnd, True
    # Molecules with only one possible SMILES (e.g. "C", "O") legitimately land here.
    return ref, False


def fetch(src: dict) -> pd.DataFrame:
    if src["kind"] == "local":
        return pd.read_csv(os.path.normpath(os.path.join(ROOT, src["path"])), sep=src.get("sep", ","))
    req = urllib.request.Request(src["url"], headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=300).read()
    # Several sources are tab-separated with quoted fields containing commas; sep sniffing
    # (engine="python") chokes on those, so the separator is declared in the registry.
    return pd.read_csv(io.BytesIO(raw), sep=src.get("sep", ","))


def prepare(cfg: dict, force: bool) -> dict:
    key = cfg["key"]
    dst = os.path.join(OUT, f"{key}.csv")
    if os.path.exists(dst) and not force:
        n = len(pd.read_csv(dst))
        print(f"  {key:14s} exists ({n} rows) -- skipped")
        return {"key": key, "status": "skipped", "n": n}

    df = fetch(cfg["source"])
    src = cfg["source"]
    if src.get("filter_col"):
        df = df[df[src["filter_col"]] == src["filter_val"]]

    idc, smc, vc = cfg["id_col"], cfg["smiles_col"], cfg["value_col"]
    df = df[[c for c in dict.fromkeys([idc, smc, vc])]].copy()
    df.columns = ["mol_id", "smiles", "value"] if idc != smc else ["smiles", "value"]
    if idc == smc:
        df.insert(0, "mol_id", df["smiles"])

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df["mol_id"] = df["mol_id"].astype(str).str.strip()
    df = df.dropna(subset=["value", "smiles"])
    df = df[df["smiles"] != ""]

    # mol_id is the JSON key in the result files; duplicates would merge two molecules'
    # predictions into one list and silently corrupt every downstream statistic.
    dup = df["mol_id"].duplicated(keep="first")
    if dup.any():
        df.loc[dup, "mol_id"] = [f"{m}__{i}" for i, m in enumerate(df.loc[dup, "mol_id"])]
    df = df.drop_duplicates(subset=["mol_id"]).reset_index(drop=True)

    invalid = [i for i, s in enumerate(df["smiles"]) if Chem.MolFromSmiles(s) is None]
    if invalid:
        print(f"  {key:14s} dropping {len(invalid)} unparseable SMILES")
        df = df.drop(index=invalid).reset_index(drop=True)

    rnd = [randomize_smiles(s, seed=i) for i, s in enumerate(df["smiles"])]
    df["smiles_random"] = [r[0] for r in rnd]
    n_fail = sum(1 for r in rnd if not r[1])

    # Some randomised sets were generated before this pipeline existed and have already been
    # queried against the models. Reusing those exact strings keeps the old and new runs on
    # one control; regenerating them would silently unpair the comparison.
    ov = cfg.get("random_override")
    if ov:
        prev = pd.read_csv(os.path.normpath(os.path.join(ROOT, ov["path"])))
        lut = {str(k).strip(): str(v).strip()
               for k, v in zip(prev[ov["id_col"]], prev[ov["smiles_col"]])}
        hit = df["mol_id"].map(lut)
        n_ov = int(hit.notna().sum())
        df["smiles_random"] = hit.fillna(df["smiles_random"])
        n_fail = sum(1 for i, r in enumerate(rnd) if not r[1] and pd.isna(hit.iloc[i]))
        print(f"  {key:14s} reused {n_ov} previously-queried randomised SMILES from {ov['path']}")

    n_same = int((df["smiles_random"] == df["smiles"]).sum())
    df["smiles_blind"] = [blind_smiles(s) for s in df["smiles"]]
    v = df["value"].to_numpy()
    # Rounded to 4 significant figures. Full float precision (17.830045523520486) is itself a
    # tell that the number came out of a transform, and it inflates every in-context prompt by
    # ~12 characters per shot for no information.
    r4 = lambda a: np.array([float(f"{x:.4g}") for x in a])
    df["value_affine"] = r4(blind_values(v))    # monotonic: invertible from anchors
    df["value_nonmono"] = r4(nonmono_values(v)) # non-monotonic, invertible
    df["value_sine"] = r4(sine_values(v))       # non-monotonic, many-to-one
    df["value_blind"] = df["value_affine"]      # back-compat alias

    os.makedirs(OUT, exist_ok=True)
    df.to_csv(dst, index=False)
    pct = 100.0 * n_same / max(len(df), 1)
    print(f"  {key:14s} {len(df):6d} rows  randomised: {n_fail} no distinct rewriting, "
          f"{n_same} ({pct:.1f}%) identical to published string"
          f"  value range [{df['value'].min():.4g}, {df['value'].max():.4g}]")
    if pct > 5:
        print(f"  {'':14s} WARNING: randomisation control is weak for this dataset "
              f"({pct:.1f}% unchanged) -- molecules are too small to rewrite.")
    return {"key": key, "status": "written", "n": len(df),
            "random_failed": n_fail, "random_identical": n_same}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    reg = json.load(open(REG))["datasets"]
    if args.only:
        reg = [d for d in reg if d["key"] in args.only]
        missing = set(args.only) - {d["key"] for d in reg}
        if missing:
            sys.exit(f"unknown dataset key(s): {sorted(missing)}")

    print(f"Preparing {len(reg)} datasets -> {OUT}")
    report = []
    for cfg in reg:
        try:
            report.append(prepare(cfg, args.force))
        except Exception as e:
            print(f"  {cfg['key']:14s} FAILED: {type(e).__name__}: {e}")
            report.append({"key": cfg["key"], "status": "failed", "error": str(e)})

    ok = [r for r in report if r["status"] in ("written", "skipped")]
    print(f"\n{len(ok)}/{len(report)} datasets ready.")
    json.dump(report, open(os.path.join(OUT, "_prepare_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
