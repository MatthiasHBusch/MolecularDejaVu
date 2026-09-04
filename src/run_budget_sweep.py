"""
The screening matrix re-run at a CONTROLLED thinking budget, in disjoint blocks.

Why a token budget and not `effort: medium`. The ladder measured what each endpoint actually
emits when asked for `medium`: 196 tokens on the positive control, 3,408 on the post-cutoff set,
and anywhere between for the rest. A "medium sweep" is therefore thirty different budgets
wearing one label, which is precisely the confound Fig. 4 exists to avoid. Asking every model
for the same number of thinking tokens is both defensible and, unlike an effort label, a
BOUNDED cost.

Why truncation is not the problem it was. In the ladder, 133 of 18,048 calls stopped on
`length` -- all of them at `high`/`medium`, where thinking is unbounded and ran into the
completion cap. The worst cell (ld50/gpt55/high) put its 95th-percentile thinking at exactly
the 8,000-token cap and lost 17% of its answers. With an explicit `reasoning.max_tokens` the
provider stops thinking at the cap and then writes the answer, so the completion cap only has
to cover cap + answer. This runner sets `max_tokens = cap + --answer-slack`, records
`finish_reason` per call, retries a truncated empty answer once with double slack, and reports
the truncation rate per cell so a quiet cell cannot be mistaken for a clean one.

Blocks, and why disjointness is per CELL and not per dataset. A block used to be a slice of one
fixed permutation, `perm(dataset, SEED)[offset : offset+n]`, so that two blocks of the same cell
were disjoint by arithmetic. That only works if every block of a cell reads the same
permutation, and block 0 does not: it was bought in three launches under a per-process
`hash()` seed, leaving 2-3 different 250-molecule sets per benchmark (see `_dsseed`). Slice
arithmetic against a permutation no block actually used produced 20-40% overlap.

So a block is defined by SUBTRACTION rather than by offset, against a FROZEN TARGET -- the
molecule set every cell of a benchmark is to be measured on:

    target(dataset) = union of what the panel has already been asked  (results/budget/_target.json)
    block K of (dataset, tag) = up to `n` of target(dataset) that this cell does not yet have

Two properties fall out. Blocks of a cell are disjoint because the exclusion set is read off that
cell's own files on disk at draw time, so it holds no matter how many processes or seeds produced
the history. And the panel converges on IDENTICAL molecules per benchmark, so a model-to-model
difference is a model effect rather than a sample effect -- which the offset design could not
deliver, since it left 2-3 different samples per benchmark. Homogenising costs nothing extra: the
target is the union of molecules already paid for, so each cell buys only what it is missing.

`--block K` names the block; re-running the same K skips finished cells, so a resume never buys a
second sample. A cell that owes more than the cap says so and finishes in the next block. To add
depth rather than matching, re-freeze with `--grow N` and every cell tops up toward the larger set.

    python src/run_budget_sweep.py --freeze-target --models ...   # define the target, then
    python src/run_budget_sweep.py --dry --n 250 --block 1        # show what each cell owes
    python src/run_budget_sweep.py --n 250 --block 1 ...          # buy it
    python src/run_budget_sweep.py --verify                       # blocks disjoint? matched?
    python src/run_budget_sweep.py --probe --budget 1             # 1 call per model, prices it

`--variant random` asks the same molecules with a DIFFERENT VALID SMILES for each of them
(`smiles_random`, written by `prepare_datasets.py`). It is the chemistry-invariance control: if
recall survives a rewriting of the string, the key is the molecule; if it collapses, the key is
the string. It is deliberately NOT drawn disjointly from the canonical arm -- the comparison is
only paired if both arms ask about the same molecules, which is what the frozen target and the
per-arm `purchased()` scoping already give.

The variant gets its own arm tag (`t1024r`), for the reason `--arm registry` needed one: two arms
that ask the SAME molecules must never be concatenated into one cell, and `purchased()` must not
see the canonical arm's history or the random arm would draw nothing at all. `analyze_budget.arm_of`
reads `meta.variant` for the same reason -- the filename alone is not what the analysis keys on.

Output: results/budget/<dataset>__<tag>__t<cap>[r]__b<K>n<n>.json
"""
import argparse, glob, hashlib, json, os, sys, time, urllib.error
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_reasoning_ladder as L      # key(), call(), parse_number(), the spend lock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG, SCREEN = os.path.join(ROOT, "src", "registry"), os.path.join(ROOT, "data", "screening")
OUT = os.path.join(ROOT, "results", "budget")
TARGET = os.path.join(OUT, "_target.json")   # the molecule set every cell is measured on

# Fixed for the lifetime of the study. Changing it invalidates every block boundary already
# run, because disjointness is a property of this permutation and nothing else.
SEED = 20260728

# `reasoning.max_tokens` turns out to be ADVISORY, not binding. Measured on one ESOL molecule
# with a 12,000-token completion cap and a 1,024-token budget requested:
#
#     gpt55   6,120 thinking   $0.092/call        sol      2,841   $0.043/call
#     gpt5    5,824 thinking   $0.059/call        glm5     3,214   $0.008/call
#     kimik26 8,388 thinking   $0.023/call        dsv4pro  3,594   $0.003/call
#
# Every one of them answers -- the earlier blank cells were the COMPLETION cap being eaten by
# thinking, not a refusal -- but the OpenAI family ignores the budget so completely that one
# model alone would cost $65 for a 700-call block. For those, an effort label is both bounded
# and honoured (`low` gives gpt55 516 thinking tokens at $0.008/call).
#
# So the budget is requested where it is respected and an effort label is used where it is not.
# The arms are not token-matched, which is fine: the claim is that recall rises with
# deliberation, not that every model deliberates identically. What matters is that the setting
# is RECORDED per cell and the emitted tokens are measured per call, which they are.
EFFORT_FALLBACK = {"gpt55": "low", "gpt5": "low", "sol": "low", "terra": "low", "luna": "low"}


def spec_for(tag, cap, mode="budget", registry=None):
    """The reasoning setting this model is actually asked for.

    Two modes, because the budget arm as first designed confounds two changes at once -- a
    thinking budget AND an extra sentence in the prompt -- and the zero-thinking rows proved
    the prompt does real work on its own. Claude Sonnet 5 emitted no thinking tokens and still
    took LD50 from 0.45% to 5.69%, R23 from 8.0 to 38.9. So the prompt alone can flip a cell.

        mode="budget"     1024-token budget (or the effort fallback) + the new prompt
        mode="registry"   whatever the ORIGINAL map used for this model + the new prompt

    The second is the missing cell of the 2x2: it isolates the prompt from the budget, using
    the same molecules, the same models and the same detector. It is also nearly free, because
    the thinking tokens are where the money went.
    """
    if mode == "registry":
        return (registry or {}).get(tag, {}).get("reasoning", "none")
    return EFFORT_FALLBACK.get(tag, f"max_tokens:{cap}")


# Which column of data/screening/<ds>.csv the prompt is built from. `smiles_random` is a
# different valid SMILES for the SAME molecule, generated once by `prepare_datasets.py` and
# frozen there -- re-generating it with `--force` unpairs every randomised result already
# collected, which is why nothing here writes to it.
VARIANT_COL = {"canonical": "smiles", "random": "smiles_random"}


def arm_tag(cap, mode, variant="canonical"):
    """Filename component, so the arms cannot overwrite each other.

    Three things vary independently and all three must be in the tag: the thinking cap, whether
    the setting is the budget or the registry minimum, and which structure string is asked about.
    Leaving the variant out would let a randomised block land in the canonical cell's `purchased()`
    history, at which point the random arm draws nothing (the bug `--arm registry` hit) and the
    disjointness guard fires on a pairing that is intentional.
    """
    base = f"t{cap}" if mode == "budget" else "reg"
    return base if variant == "canonical" else f"{base}r"

# The seven benchmarks that carry a claim in the paper: three recalled, two clean, two controls.
# BACE/AqSolDB/Caco-2/PPBR/QM9 are at pilot depth and support no verdict, so re-running them at
# a thinking budget buys nothing but tokens.
DEFAULT_DS = "esol,freesolv,ld50,lipophilicity,qm8,antiviral,boilingpoint"


def prompts(ds, smiles):
    """The screening prompt, plus an explicit precision instruction.

    The added sentence is not cosmetic, and it does NOT do what it was added to do.

    hit3 is computed over answers carrying three significant figures, and emission is causally
    coupled to whether the model is reciting, which makes the denominator a collider on the
    treatment. The instruction was added to break that coupling. Measured, it does not:
    Claude Opus 5 under this exact prompt emits three figures on 76% of ESOL, 76% of FreeSolv
    and 59% of LD50 -- and 5% of Lipophilicity, the one benchmark of the four it does not
    recall. The model complies where it is reciting and declines where it is guessing.

    Worse, the instruction is not inert: Claude Sonnet 5, emitting ZERO thinking tokens, went
    from 0.45% to 5.69% on LD50 under it. Telling a model the answer HAS three significant
    figures is itself a cue that a specific published value is wanted. So this prompt is a
    treatment, not a measurement convenience, and the `--arm registry` run exists to separate
    it from the thinking budget.

    Either way this sweep's hit3 is NOT numerically comparable to the original map's; the arms
    are comparable by VERDICT only.
    """
    sysmsg = (f"You are an {ds['prompt_expert']} and know the {ds['prompt_dsname']} dataset very "
              f"well. You are given a SMILES string of a molecule. Your task is to predict the "
              f"{ds['prompt_property']} of that molecule in {ds['prompt_unit']} as reported in "
              f"that dataset. Provide only the numerical value as output, without any "
              f"additional text. Report the value with 3 significant figures.")
    usr = (f"What is the {ds['prompt_property']} in {ds['prompt_unit']} of the molecule with the "
           f"following SMILES string in the {ds['prompt_dsname']} dataset: {smiles}? Provide "
           f"only the numerical value as output, without any additional text. Report the value "
           f"with 3 significant figures.")
    return sysmsg, usr


def _dsseed(dk):
    """A per-dataset seed that is stable ACROSS PROCESSES.

    This used to be `SEED + abs(hash(dk)) % 10_000`, and that was a real bug: Python randomises
    `hash()` on str per interpreter process unless PYTHONHASHSEED is fixed. Block 0 and block 1
    therefore drew DIFFERENT permutations of the same benchmark, and slices [0:250] and [250:500]
    of two unrelated permutations overlap by chance -- measured at 60 shared molecules on ESOL,
    95 on FreeSolv, 68 on antiviral. Disjointness was guaranteed by construction only within a
    single process, which is exactly the case the original unit check happened to exercise.

    An overlapping molecule would be double-weighted inside the molecule-level null, which is the
    one place this pipeline cannot absorb it. `analyze_budget.cells()` raises on overlap, so the
    error surfaced before it could reach a verdict -- but the guard is not the fix.

    Fixing the seed does NOT retroactively align block 0, which was already bought under the old
    one: 2-3 distinct molecule sets per benchmark survive on disk. Blocks are therefore drawn by
    exclusion (see `draw`), not by offset, and this seed only fixes the ORDER in which unseen
    molecules are taken.
    """
    return SEED + int(hashlib.sha256(dk.encode()).hexdigest()[:8], 16) % 10_000


def purchased(dk, tag, skip=None, arm=None):
    """Every mol_id this cell has already been asked about, over its blocks on disk.

    `arm` scopes the history to ONE arm (`t1024`, `reg`, ...). Disjointness is a within-arm
    property -- two blocks of the same arm must not buy the same molecule twice -- but ACROSS
    arms the opposite is wanted: the registry arm is only a paired comparison if it is asked
    about exactly the molecules the budget arm was asked about. Globbing every arm made the
    second arm's cells look complete before they were run, so `--arm registry` drew nothing at
    all. `arm=None` keeps the old behaviour and is what the target manifest is built from,
    since the target is the union of everything the panel has ever been asked.
    """
    ids = set()
    pat = f"{dk}__{tag}__{arm}__*.json" if arm else f"{dk}__{tag}__*.json"
    for p in sorted(glob.glob(os.path.join(OUT, pat))):
        if skip and os.path.basename(p) == os.path.basename(skip):
            continue
        for c in json.load(open(p))["calls"]:
            ids.add(str(c["mol_id"]).strip())
    return ids


def ordered(dk):
    """This dataset's rows in the one fixed order every draw here uses."""
    df = pd.read_csv(os.path.join(SCREEN, f"{dk}.csv")).dropna(subset=["value", "smiles"])
    df = df.reset_index(drop=True)
    return df.iloc[np.random.default_rng(_dsseed(dk)).permutation(len(df))]


def freeze_target(keys, tags, size=0):
    """Write the molecule set every cell of a benchmark is to be measured on.

    Block 0 left 2-3 different 250-molecule samples per benchmark (see `_dsseed`), so the panel is
    not molecule-matched and a model-to-model rate difference inside a benchmark mixes model
    effect with sample effect. The target starts as the UNION of what the panel has already been
    asked, which makes homogenising a matter of buying only what each cell is missing -- no
    molecule is paid for twice and nothing already bought is wasted.

    `size` then tops the target up to a round depth with molecules nobody has seen. Matching comes
    FIRST and the top-up second, so the money goes to making cells comparable before it goes to
    making them deeper; a benchmark with fewer rows than `size` simply keeps all of them.
    """
    # MERGE, never replace. The manifest is the record of what every benchmark of the arm is
    # measured on, and re-freezing for one new benchmark must not silently drop the seven that
    # are already bought -- `--verify` reads this file to check the matching claim, and an
    # entry that vanishes reads as "never matched" rather than as "lost the manifest".
    prev = json.load(open(TARGET)) if os.path.exists(TARGET) else {}
    tgt = dict(prev.get("target", {}))
    for dk in keys:
        have = {t: purchased(dk, t) for t in tags}
        seq = ordered(dk).mol_id.astype(str).str.strip()
        union = set().union(*have.values()) if have else set()
        n_union = len(union)
        if size and len(union) < size:
            union |= set(seq[~seq.isin(union)].head(size - len(union)))
        tgt[dk] = list(seq[seq.isin(union)])          # fixed order, so the manifest reads like a draw
        miss = [len(union - v) for v in have.values()]
        print(f"  {dk:14s} target {len(tgt[dk]):5d} = {n_union:4d} matched "
              f"+ {len(union) - n_union:4d} new   cells {len(have):3d}   "
              f"owed per cell: max {max(miss or [0]):4d}  total {sum(miss):6d}")
    json.dump(dict(panel=sorted(set(prev.get("panel", [])) | set(tags)), size=size, target=tgt),
              open(TARGET, "w"), indent=1)
    print(f"\nwrote {TARGET}\n  total calls to complete: "
          f"{sum(len(set(v) - purchased(dk, t)) for dk, v in tgt.items() for t in tags):,}")


def load_target():
    if not os.path.exists(TARGET):
        return None
    return json.load(open(TARGET))["target"]


def draw(dk, n, exclude=(), target=None):
    """Up to `n` molecules this cell still owes, in the dataset's fixed order.

    With a frozen target, a cell draws only from that set, so every cell of a benchmark converges
    on the SAME molecules and `n` is a per-block cap rather than a sample size. Without one, it
    falls back to taking unseen molecules in the fixed order.
    """
    df, exclude = ordered(dk), set(exclude)
    mid = df.mol_id.astype(str).str.strip()
    keep = ~mid.isin(exclude)
    if target is not None:
        keep &= mid.isin(set(target))
    return df[keep].head(n)


def one(k, md, ds, row, cap, slack, retries, budget, mode="budget", registry=None,
        variant="canonical"):
    """One call, with the one retry that a truncated empty answer earns."""
    sysmsg, usr = prompts(ds, str(row[VARIANT_COL[variant]]).strip())
    level = spec_for(md["tag"], cap, mode, registry)
    for attempt in range(retries):
        with L._lock:
            if L._spent >= budget:
                return None
        try:
            extra = slack * (2 if attempt else 1)
            # 300 s (L.call's default) x 4 retries x 20 threads is 20 minutes of total silence per
            # job if the endpoint stalls, and the progress line only fires every 50 results -- a
            # stalled sweep then looks identical to a slow one. 90 s is well clear of the slowest
            # legitimate call measured in this arm (kimik3, ~40 s at 9.9k thinking tokens).
            r = L.call(k, md, level, sysmsg, usr, cap + extra, timeout=90)
            with L._lock:
                L._spent += r["cost"]
            r["mol_id"] = str(row["mol_id"]).strip()
            r["truth"] = float(row["value"])
            r["truncated"] = (r["finish"] == "length")
            # A truncated answer that still parsed is fine -- the number is emitted before the
            # trailing prose. Only a truncated MISSING answer is worth paying to retry.
            if r["value"] is None and r["truncated"] and attempt + 1 < retries:
                continue
            return r
        except urllib.error.HTTPError as e:
            if e.code == 400:
                raise RuntimeError(f"HTTP 400: endpoint rejects a {cap}-token thinking budget")
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def run_cell(k, md, ds, n, blk, cap, slack, iters, threads, retries, budget, force,
             mode="budget", registry=None, target=None, variant="canonical"):
    arm = arm_tag(cap, mode, variant)
    p = os.path.join(OUT, f"{ds['key']}__{md['tag']}__{arm}__b{blk}n{n}.json")
    if os.path.exists(p) and not force:
        print(f"  [skip] {os.path.basename(p)} exists")
        return 0.0
    # Drawn here, per cell, against what this cell already owns -- `--force` re-buys THIS block,
    # so its own molecules must not count as already purchased.
    already = purchased(ds["key"], md["tag"], skip=p, arm=arm)
    tg = None if target is None else target.get(ds["key"])
    data = draw(ds["key"], n, already, tg)
    owed = 0 if tg is None else len(set(tg) - already)
    if not len(data):
        print(f"  [==] {ds['key']}/{md['tag']}: already complete "
              f"({len(already)} molecules, nothing owed)" if tg is not None else
              f"  [--] {ds['key']}/{md['tag']}: benchmark exhausted ({len(already)} bought)")
        return 0.0
    if tg is not None and len(data) < owed:
        print(f"  [!] {ds['key']}/{md['tag']}: cap {n} short of the {owed} it owes; "
              f"run another block to finish")
    # A molecule whose randomisation round-trips to the published string is not a control, it is
    # the canonical arm run twice. It is still QUERIED -- dropping it would break the molecule
    # matching that makes the comparison paired -- but it is named here so the analysis can
    # exclude it from the paired contrast rather than silently averaging it in.
    unchanged = []
    if variant != "canonical":
        col = VARIANT_COL[variant]
        unchanged = [str(r["mol_id"]).strip() for _, r in data.iterrows()
                     if str(r[col]).strip() == str(r["smiles"]).strip()]
        print(f"       {ds['key']}/{md['tag']}: {len(data) - len(unchanged)}/{len(data)} "
              f"molecules have a distinct rewriting"
              + (f", {len(unchanged)} do not (recorded, not dropped)" if unchanged else ""))
    jobs = [row for _, row in data.iterrows() for _ in range(iters)]
    t0, before = time.time(), L._spent
    calls = []
    try:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for i, r in enumerate(ex.map(
                    lambda row: one(k, md, ds, row, cap, slack, retries, budget, mode,
                                    registry, variant), jobs), 1):
                if r is not None:
                    calls.append(r)
                if i % 50 == 0:
                    print(f"       .. {ds['key']}/{md['tag']} {i}/{len(jobs)} "
                          f"{sum(1 for c in calls if c['value'] is None)} unanswered "
                          f"${L._spent - before:.3f}  {time.time() - t0:.0f}s", flush=True)
    except RuntimeError as e:
        print(f"  [--] {ds['key']}/{md['tag']}: {e}")
        return 0.0
    spent = L._spent - before
    ok = sum(1 for c in calls if c["value"] is not None)
    trunc = sum(1 for c in calls if c.get("truncated"))
    rt = sorted(c["reasoning_tokens"] for c in calls) or [0]
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(meta=dict(dataset=ds["key"], tag=md["tag"], endpoint=md["endpoint"],
                             thinking_cap=cap, arm=mode,
                             variant=variant, smiles_column=VARIANT_COL[variant],
                             unchanged_ids=unchanged,
                             reasoning=spec_for(md["tag"], cap, mode, registry),
                             max_tokens=cap + slack, seed=SEED,
                             block=blk, n_excluded=len(already), n_owed=owed,
                             homogenised=tg is not None,
                             n_mol=len(data), iters=iters,
                             mol_ids=[str(x).strip() for x in data.mol_id],
                             prompt_precision="3 significant figures",
                             cost=spent, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")),
                   calls=calls), open(p, "w"), indent=1)
    print(f"  [ok] {ds['key']:14s} {md['tag']:16s} {ok}/{len(jobs)} parsed  "
          f"{100 * trunc / max(len(calls), 1):4.1f}% truncated  "
          f"median thinking {rt[len(rt) // 2]:5d}/{cap}  ${spent:7.4f}  "
          f"{time.time() - t0:5.0f}s", flush=True)
    return spent


def verify():
    """Confirm that every pair of blocks for a cell really is disjoint."""
    files = sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []
    seen, bad, retried = {}, 0, 0
    for f in files:
        if not f.endswith(".json") or f.startswith("_"):   # _target.json is a manifest, not a block
            continue
        try:
            d = json.load(open(os.path.join(OUT, f)))
        except json.JSONDecodeError:
            # A block is only written when it completes, so a truncated or zero-byte file is a
            # run that was killed mid-cell -- `qm9__gem36flash__t1024__b0n500.json` is one, left
            # by the 4 August stop. `analyze_budget.cells()` already skips these; this did not,
            # so the verifier died on the one file it had nothing to say about.
            print(f"  [skip] {f}: empty or half-written (killed run)")
            continue
        m = d["meta"]
        # Keyed on the ARM as well, because disjointness is a within-arm property. Two arms ask
        # the same molecules on purpose -- that pairing is the whole design -- so a key of
        # (dataset, tag) reports every deliberate pairing as an OVERLAP and buries a real one.
        k = (m["dataset"], m["tag"],
             arm_tag(m.get("thinking_cap", 1024), m.get("arm", "budget"),
                     m.get("variant", "canonical")))
        ids = set(m.get("mol_ids", []))
        # A block excludes what its cell has ANSWERED, not what it was asked, so a molecule that
        # failed in an earlier block is legitimately drawn again -- that recovers it rather than
        # double-counting it, and the analysis keys on answers so it never sees a duplicate.
        # Only a repeat of an ANSWERED molecule is a real overlap.
        ans = {str(c["mol_id"]).strip() for c in d["calls"]}
        for other, oids, oans in seen.get(k, []):
            hard, soft = ans & oans, (ids & oids) - (ans & oans)
            if hard:
                bad += 1
                print(f"  OVERLAP {f} vs {other}: {len(hard)} molecules answered twice")
            retried += len(soft)
        seen.setdefault(k, []).append((f, ids, ans))
    tot = sum(len(v) for v in seen.values())
    print(f"{tot} blocks over {len(seen)} cells; "
          f"{'NO OVERLAPS' if not bad else str(bad) + ' OVERLAPPING PAIRS'}"
          + (f"; {retried} molecule(s) re-drawn after failing in an earlier block" if retried
             else ""))

    # Disjointness within a cell is only half of it: the arm also claims every cell of a benchmark
    # is measured on the SAME molecules, and that claim is checkable here rather than asserted.
    if not os.path.exists(TARGET):
        print("\nno target manifest -- cells are NOT molecule-matched")
        return
    T = json.load(open(TARGET))
    tgt, panel = T["target"], set(T["panel"])
    # Only the panel was ever meant to reach the target; deferred models keep their block 0 and
    # are reported separately rather than counted as incomplete.
    print(f"\n{'benchmark':14s}{'target':>7}{'panel':>7}{'matched':>9}   off-panel cells")
    for dk, ids in sorted(tgt.items()):
        # The matching claim is about the canonical budget arm; the other arms are paired to it
        # by construction, so reporting them here would just triple every row.
        have = {t: set().union(*[x for _, x, _ in v])
                for (d, t, a), v in seen.items() if d == dk and a == "t1024"}
        if not have:
            continue
        pan = {t: len(set(ids) - s) for t, s in have.items() if t in panel}
        off = [t for t in have if t not in panel]
        done = sum(1 for v in pan.values() if v == 0)
        print(f"  {dk:12s}{len(ids):7d}{len(pan):7d}{done:9d}   "
              + ("all matched" if done == len(pan) else f"MAX {max(pan.values())} STILL OWED")
              + (f"   ({', '.join(off)} at block 0 only)" if off else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=DEFAULT_DS)
    ap.add_argument("--models", default="", help="default: every enabled model in the registry")
    ap.add_argument("--n", type=int, default=100, help="molecules in THIS block")
    ap.add_argument("--block", type=int, default=0,
                    help="names this block in the file name; re-running the same K skips "
                         "finished cells instead of buying a second sample")
    ap.add_argument("--iters", type=int, default=1,
                    help="repeats per molecule; 1 is right unless you want self-consistency")
    ap.add_argument("--thinking", type=int, default=1024, help="thinking-token cap per call")
    ap.add_argument("--answer-slack", type=int, default=11000,
                    help="completion tokens reserved for the answer on top of the cap")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--budget", type=float, default=25.0, help="hard ceiling in USD")
    ap.add_argument("--probe", action="store_true", help="one call per model; prices the sweep")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--arm", default="budget", choices=["budget", "registry"],
                    help="'budget' = thinking budget + new prompt; 'registry' = the original "
                         "map's reasoning setting + new prompt, which isolates the prompt")
    ap.add_argument("--variant", default="canonical", choices=list(VARIANT_COL),
                    help="'canonical' = the published SMILES; 'random' = a different valid SMILES "
                         "for the same molecule (the chemistry-invariance control). The variant "
                         "gets its own arm tag, so the two are paired rather than concatenated")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--freeze-target", action="store_true",
                    help="write the molecule set every cell of a benchmark is to be measured on, "
                         "as the union of what the panel already has")
    ap.add_argument("--target-size", type=int, default=0,
                    help="with --freeze-target, top the target up to this many molecules per "
                         "benchmark after matching; 0 leaves it at the matched union")
    ap.add_argument("--no-target", action="store_true",
                    help="ignore the manifest and draw unseen molecules in the fixed order")
    args = ap.parse_args()

    if args.verify:
        verify()
        return

    models = {m["tag"]: m for m in json.load(open(os.path.join(REG, "models.json")))["models"]
              if m.get("enabled", True)}
    dsets = {d["key"]: d for d in json.load(open(os.path.join(REG, "datasets.json")))["datasets"]}
    tags = args.models.split(",") if args.models else list(models)
    keys = args.datasets.split(",")

    if args.freeze_target:
        freeze_target(keys, tags, args.target_size)
        return
    target = None if args.no_target else load_target()
    # Sizes only -- the actual molecules are drawn per cell, against that cell's own history.
    avail = {dk: len(pd.read_csv(os.path.join(SCREEN, f"{dk}.csv"))
                     .dropna(subset=["value", "smiles"])) for dk in keys}

    # Fail here rather than three hours into a sweep: a benchmark without the variant column has
    # never been through `prepare_datasets.py` with the transforms, and there is no recovery from
    # it mid-run that does not also unpair whatever has already been bought.
    col = VARIANT_COL[args.variant]
    for dk in keys:
        head = pd.read_csv(os.path.join(SCREEN, f"{dk}.csv"), nrows=1)
        if col not in head.columns:
            sys.exit(f"{dk}.csv has no '{col}' column -- run prepare_datasets.py for it first")

    arm = arm_tag(args.thinking, args.arm, args.variant)
    plan = {(dk, t): len(draw(dk, args.n, purchased(dk, t, arm=arm),
                              None if target is None else target.get(dk)))
            for dk in keys for t in tags}
    n_calls = sum(plan.values()) * args.iters
    print("=" * 84)
    print(f"models    {len(tags)}   arm '{args.arm}'   variant '{args.variant}' "
          f"(column '{col}', file tag '{arm}')   "
          f"{'thinking cap %d tok' % args.thinking if args.arm == 'budget' else 'reasoning as in the original map'}"
          f"   (completion cap {args.thinking + args.answer_slack})")
    if target is None:
        print(f"sampling  unseen molecules in the fixed order, cap {args.n} per cell (NOT matched)")
        print(f"datasets  {', '.join(f'{k}({avail[k]} available)' for k in keys)}")
    else:
        print(f"sampling  homogenised -- every cell converges on the frozen target, "
              f"cap {args.n} per block")
        print(f"datasets  {', '.join(f'{k}(target {len(target.get(k, []))})' for k in keys)}")
    print(f"block     b{args.block}   "
          f"{sum(1 for v in plan.values() if v == 0)} of {len(plan)} cells already complete")
    print(f"{len(plan)} cells, {n_calls:,} calls, ceiling ${args.budget:.2f}")
    print("=" * 84)

    if args.dry:
        for dk in keys:
            owed = [len(set(target[dk]) - purchased(dk, t, arm=arm)) for t in tags] if target else []
            print(f"  {dk:14s} " + (f"target {len(target[dk]):4d}   owed per cell "
                                    f"min {min(owed)} max {max(owed)}   "
                                    f"this block buys {sum(plan[(dk, t)] for t in tags):5d}"
                                    if target else f"buys {sum(plan[(dk, t)] for t in tags):5d}"))
        dk = keys[0]
        d = draw(dk, args.n, purchased(dk, tags[0], arm=arm),
                 None if target is None else target.get(dk))
        print(f"\n  {dk}/{tags[0]} first 4: {list(d.mol_id[:4])}")
        if args.variant != "canonical":
            same = sum(1 for _, r in d.iterrows()
                       if str(r[col]).strip() == str(r["smiles"]).strip())
            print(f"  {len(d) - same}/{len(d)} of them have a distinct rewriting; "
                  f"{same} round-trip to the published string (queried, flagged in meta)")
            print(f"  published : {str(d.iloc[0]['smiles']).strip()}")
            print(f"  variant   : {str(d.iloc[0][col]).strip()}")
        s, u = prompts(dsets[dk], str(d.iloc[0][col]).strip())
        print(f"\n--- SYSTEM ---\n{s}\n\n--- USER ---\n{u}")
        print("DRY RUN: nothing queried.")
        return

    k = L.key()

    if args.probe:
        # One real call per model on one benchmark, at the real cap, to measure what each
        # endpoint emits and whether it honours the budget at all. Cost is cents.
        dk = keys[0]
        row = draw(dk, 1, purchased(dk, tags[0], arm=arm),
                   None if target is None else target.get(dk)).iloc[0]
        rows = []
        for t in tags:
            try:
                r = L.call(k, models[t], spec_for(t, args.thinking, args.arm, models),
                           *prompts(dsets[dk], str(row[col]).strip()),
                           args.thinking + args.answer_slack)
                L._spent += r["cost"]
                rows.append(dict(tag=t, spec=spec_for(t, args.thinking, args.arm, models),
                                 thinking=r["reasoning_tokens"], completion=r["completion"],
                                 cost=r["cost"], finish=r["finish"],
                                 answered=r["value"] is not None, text=r["text"]))
                print(f"  {t:16s} {spec_for(t, args.thinking, args.arm, models):16s} "
                      f"thinking={r['reasoning_tokens']:>5} "
                      f"completion={r['completion']:>5} ${r['cost']:.5f} [{r['finish']}] "
                      f"{'' if r['value'] is not None else '<< NO ANSWER'}", flush=True)
            except urllib.error.HTTPError as e:
                print(f"  {t:16s} HTTP {e.code} -- budget form rejected")
                rows.append(dict(tag=t, spec=spec_for(t, args.thinking, args.arm, models),
                                 thinking=None, completion=None, cost=0.0,
                                 finish=f"http{e.code}", answered=False, text=""))
            except Exception as e:
                print(f"  {t:16s} ERR {type(e).__name__}")
        P = pd.DataFrame(rows)
        os.makedirs(os.path.join(ROOT, "results", "meta"), exist_ok=True)
        P.to_csv(os.path.join(ROOT, "results", "meta", "budget_probe.csv"), index=False)
        per_model = n_calls / max(len(tags), 1)
        print(f"\nprobe spent ${L._spent:.3f}")
        print(f"projected block at n={args.n}, iters={args.iters}, "
              f"{len(keys)} benchmarks:  ${P.cost.sum() * per_model:.2f}")
        print(f"  refused the budget form: "
              f"{', '.join(P[P.finish.astype(str).str.startswith('http')].tag) or 'none'}")
        print(f"  gave no answer:          "
              f"{', '.join(P[~P.answered & ~P.finish.astype(str).str.startswith('http')].tag) or 'none'}")
        return

    total = 0.0
    for t in tags:
        for dk in keys:
            if L._spent >= args.budget:
                print(f"\nBUDGET ${args.budget:.2f} reached (${L._spent:.2f}) -- stopping. "
                      f"Re-run with the same --block/--n to continue; finished cells are "
                      f"skipped.")
                return
            total += run_cell(k, models[t], dsets[dk], args.n, args.block, args.thinking,
                              args.answer_slack, args.iters, args.threads, args.retries,
                              args.budget, args.force, args.arm, models, target, args.variant)
    print(f"\nDONE. spent ${L._spent:.2f} of ${args.budget:.2f}")


if __name__ == "__main__":
    main()
