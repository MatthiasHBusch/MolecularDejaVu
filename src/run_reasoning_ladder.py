"""
A controlled reasoning ladder: does verbatim recall rise with the deliberation budget?

The evidence so far is an accident. The pilot ran before the reasoning calibration and the deep
sweep after, so 72 cells happen to hold two settings -- but those branches also differ in
molecule count, iteration count and date, and only two of them are powered. This runs the
experiment properly:

  * ONE fixed molecule subset, the same rows in the same order for every cell;
  * the same iteration count everywhere;
  * every level of a model queried back to back, so endpoint drift cannot masquerade as an
    effect;
  * `max_tokens` raised well above the thinking budget, because a truncated completion returns
    an empty answer and would be scored as a miss -- which is how a high-reasoning cell can be
    made to look clean;
  * the emitted reasoning tokens recorded PER CALL, so the dose is measured rather than
    requested. That matters: asked for 128 thinking tokens, Gemini 3.1 Pro emitted 641.

Prompt and sampling are identical to Run_ZeroShot.jl, so results join directly onto the audit.

    python src/run_reasoning_ladder.py --probe                 # 1 call per cell, prices the sweep
    python src/run_reasoning_ladder.py --dry                   # print the plan, spend nothing
    python src/run_reasoning_ladder.py --budget 70             # run, hard ceiling in USD

Output: results/ladder/<dataset>__<tag>__<level>.json
        {"meta": {...}, "calls": [{mol_id, value, reasoning_tokens, completion, cost}, ...]}
"""
import argparse, json, os, re, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG = os.path.join(ROOT, "src", "registry")
SCREEN = os.path.join(ROOT, "data", "screening")
OUT = os.path.join(ROOT, "results", "ladder")
API = "https://openrouter.ai/api/v1/chat/completions"

# Ordered by deliberation. "none" is not accepted by every endpoint; a cell that 400s is
# recorded as unsupported rather than retried, since the point is the ladder, not coverage.
LEVELS = ["none", "minimal", "low", "medium", "high"]

NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_lock = threading.Lock()
_spent = 0.0


def key():
    """The OpenRouter API key.

    $OPENROUTER_API_KEY if it is set. Otherwise the key is sliced out of a Julia LLM library,
    whose path is $LLM_KEYS_JL and which defaults to the key-free src/lib/LLMs.jl shipped here --
    that fallback exists so a checkout that keeps its credentials in the shared Julia library can
    run unchanged. If neither yields a key, exit with an instruction rather than a traceback.
    """
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if k:
        return k
    jl = os.environ.get("LLM_KEYS_JL", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "lib", "LLMs.jl"))
    try:
        src = open(jl, encoding="utf8").read()
        for var in ("key_openrouter_moi", "key_openrouter"):
            if f'{var} = "' in src:
                k = src.split(f'{var} = "')[1].split('"')[0].strip()
                if k and not k.startswith("YOUR_"):
                    return k
    except OSError:
        pass
    sys.exit("No OpenRouter key. Set OPENROUTER_API_KEY, or point LLM_KEYS_JL at an LLMs.jl "
             "that defines key_openrouter.")


def parse_number(txt):
    """Last number in the reply, scientific notation first -- matches Run_ZeroShot.parse_number."""
    t = txt.replace("\u2212", "-").replace("\u2013", "-").replace(",", "")
    m = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+", t)
    if m:
        try:
            return float(m[-1])
        except ValueError:
            pass
    m = NUM.findall(t)
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def prompts(ds, smiles):
    sysmsg = (f"You are an {ds['prompt_expert']} and know the {ds['prompt_dsname']} dataset very "
              f"well. You are given a SMILES string of a molecule. Your task is to predict the "
              f"{ds['prompt_property']} of that molecule in {ds['prompt_unit']} as reported in "
              f"that dataset. Provide only the numerical value as output, without any "
              f"additional text.")
    usr = (f"What is the {ds['prompt_property']} in {ds['prompt_unit']} of the molecule with the "
           f"following SMILES string in the {ds['prompt_dsname']} dataset: {smiles}? Provide "
           f"only the numerical value as output, without any additional text.")
    return sysmsg, usr


def call(k, md, level, sysmsg, usr, max_tokens, timeout=300):
    # Two reasoning forms, as the registry uses both: a named effort ("none".."high") and a
    # fixed thinking budget ("max_tokens:N"). Sending the budget string as an effort label is
    # an instant HTTP 400 on every call -- which costs nothing but silently produces an empty
    # cell, so it is worth handling rather than assuming effort labels everywhere.
    reasoning = ({"max_tokens": int(level.split(":", 1)[1])}
                 if isinstance(level, str) and level.startswith("max_tokens:")
                 else {"effort": level})
    body = {"model": md["endpoint"],
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": usr}],
            "max_tokens": max_tokens, "temperature": 1.0, "top_p": 1.0,
            "usage": {"include": True},
            "reasoning": reasoning}
    if md.get("providers"):
        body["provider"] = {"order": md["providers"]}
    if md.get("service_tier"):
        body["service_tier"] = md["service_tier"]
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {k}"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    u = d.get("usage", {}) or {}
    det = u.get("completion_tokens_details", {}) or {}
    txt = (d["choices"][0]["message"].get("content") or "").strip()
    fin = (d["choices"][0].get("finish_reason") or "")
    return dict(text=txt[:80], value=parse_number(txt),
                reasoning_tokens=int(det.get("reasoning_tokens", 0) or 0),
                completion=int(u.get("completion_tokens", 0) or 0),
                cost=float(u.get("cost", 0) or 0), finish=fin)


def one(k, md, level, ds, row, max_tokens, retries, budget):
    global _spent
    sysmsg, usr = prompts(ds, str(row["smiles"]).strip())
    for attempt in range(retries):
        with _lock:
            if _spent >= budget:
                return None
        try:
            r = call(k, md, level, sysmsg, usr, max_tokens)
            with _lock:
                _spent += r["cost"]
            r["mol_id"] = str(row["mol_id"]).strip()
            return r
        except urllib.error.HTTPError as e:
            if e.code == 400:
                raise RuntimeError(f"HTTP 400 (level '{level}' rejected by endpoint)")
            time.sleep(1.5 * (attempt + 1))
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def run_cell(k, md, level, ds, data, iters, threads, max_tokens, retries, budget, force):
    out_file = os.path.join(OUT, f"{ds['key']}__{md['tag']}__{level}.json")
    if os.path.exists(out_file) and not force:
        try:
            got = len(json.load(open(out_file))["calls"])
            if got >= len(data) * iters:
                print(f"  [skip] {ds['key']}/{md['tag']}/{level}: {got} calls already")
                return 0.0
        except Exception:
            pass
    jobs = [row for _, row in data.iterrows() for _ in range(iters)]
    t0, before = time.time(), _spent
    calls = []
    try:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for i, r in enumerate(ex.map(
                    lambda row: one(k, md, level, ds, row, max_tokens, retries, budget), jobs), 1):
                if r is not None:
                    calls.append(r)
                # A high-effort cell can take half an hour; without a heartbeat there is no way
                # to tell a slow endpoint from a hung one except by waiting for both.
                if i % 60 == 0:
                    print(f"       .. {ds['key']}/{md['tag']}/{level} {i}/{len(jobs)} "
                          f"({i / max(time.time() - t0, 1e-9):.2f} calls/s, "
                          f"${_spent - before:.3f})", flush=True)
    except RuntimeError as e:
        print(f"  [--] {ds['key']}/{md['tag']}/{level}: {e}")
        return 0.0
    spent = _spent - before
    ok = sum(1 for c in calls if c["value"] is not None)
    rt = sorted(c["reasoning_tokens"] for c in calls)
    med = rt[len(rt) // 2] if rt else 0
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(meta=dict(dataset=ds["key"], tag=md["tag"], endpoint=md["endpoint"],
                             level=level, n_mol=len(data), iters=iters,
                             max_tokens=max_tokens, cost=spent,
                             timestamp=time.strftime("%Y-%m-%dT%H:%M:%S")),
                   calls=calls), open(out_file, "w"), indent=1)
    # A cell that thinks until it hits `max_tokens` returns an empty completion, which parses as
    # a miss -- so the cell with the MOST deliberation reads as the cleanest. This is the failure
    # the module docstring warns about, and it happened anyway: GPT-5.5 at `high` on BACE,
    # Caco-2 and AqSolDB emitted a median of exactly 8,000 tokens and parsed 14-24%. Silence is
    # not acceptable here, because the damage is invisible in the summary.
    #
    # The thresholds were 25% truncated / 75% parsed on the first pass and qm7/gpt55/medium
    # walked straight through them at 24% and 76%. A cell that loses a quarter of its answers is
    # already compromised, and the loss is not random -- it falls on the calls that thought
    # longest. The parse rate is the signal that matters, because a truncated completion whose
    # number still came through costs nothing.
    trunc = sum(1 for c in calls if c.get("finish") == "length")
    if trunc > 0.10 * len(jobs) or ok < 0.95 * len(jobs):
        print(f"  [WARN] {ds['key']}/{md['tag']}/{level}: {trunc}/{len(calls)} completions hit "
              f"the {max_tokens}-token cap and only {ok}/{len(jobs)} parsed. This cell is "
              f"truncation-limited, not clean -- raise --max-tokens or drop the rung.", flush=True)
    print(f"  [ok] {ds['key']:14s} {md['tag']:16s} {level:8s} {ok}/{len(jobs)} parsed  "
          f"median reasoning {med:5d} tok  ${spent:7.4f}  {time.time()-t0:5.0f}s", flush=True)
    return spent


def main():
    global _spent
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gem3flash,gem35flash,gem31flashlite,gpt55")
    ap.add_argument("--datasets", default="ld50,lipophilicity,boilingpoint")
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--n", type=int, default=120, help="molecules per cell")
    # The original ladder took the FIRST --n rows, and that turned out to be a biased sample:
    # measured on the deep map at a fixed setting, the first 60 molecules of a benchmark are
    # recalled 12-18x more often than the rest (ld50/gem35flash 9.88% vs 0.54%; ld50/gem3flash
    # 13.14% vs 0.97%; esol/gem35flash 4.84% vs 0.39%). The files lead with small canonical
    # compounds -- LD50's first 60 average 17.1 SMILES characters against 27.6 for the rest,
    # index-vs-length rho = +0.25, and QM8 opens with methane, ammonia, water, acetylene --
    # and short famous molecules are the ones a model has seen most often.
    #
    # The ladder's DOSE-RESPONSE is unaffected, because every effort level in a cell used the
    # same molecules and the comparison is paired. Its ABSOLUTE rates do not generalise to the
    # benchmark. `head` is kept as the default only so the published cells stay reproducible;
    # anything new should use `random`.
    ap.add_argument("--sample", default="head", choices=["head", "random"],
                    help="'head' reproduces the published ladder (biased, see comment); "
                         "'random' draws a representative sample")
    ap.add_argument("--sample-seed", type=int, default=20260728)
    # A ladder cell normally takes the first --n rows so that it overlaps the main sweep.
    # When the question is about a specific downstream study's test set -- "is the model that
    # study used contaminated on the molecules it was scored on?" -- the molecules have to be
    # named instead, or the overlap is whatever the two samplings happen to share.
    ap.add_argument("--mol-ids-from", default=None,
                    help="CSV with a mol_id (or sample_id) column; restricts the cell to those "
                         "molecules instead of the first --n rows")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=6000,
                    help="completion cap; must exceed the thinking budget or answers truncate")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--budget", type=float, default=5.0, help="hard ceiling in USD")
    ap.add_argument("--probe", action="store_true",
                    help="one call per (model, level); measures the dose and prices the sweep")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = {m["tag"]: m for m in json.load(open(os.path.join(REG, "models.json")))["models"]}
    dsets = {d["key"]: d for d in json.load(open(os.path.join(REG, "datasets.json")))["datasets"]}
    tags = args.models.split(",")
    keys = args.datasets.split(",")
    levels = args.levels.split(",")

    wanted = None
    if args.mol_ids_from:
        w = pd.read_csv(args.mol_ids_from)
        col = "mol_id" if "mol_id" in w.columns else "sample_id"
        wanted = {str(x).strip() for x in w[col].dropna().unique()}
        print(f"restricting to {len(wanted)} named molecules from "
              f"{os.path.basename(args.mol_ids_from)}")

    frames = {}
    for dk in keys:
        df = pd.read_csv(os.path.join(SCREEN, f"{dk}.csv")).dropna(subset=["value", "smiles"])
        if wanted is not None:
            df = df[df.mol_id.astype(str).str.strip().isin(wanted)]
            if df.empty:
                sys.exit(f"none of the named molecules are in data/screening/{dk}.csv")
            frames[dk] = df
        elif args.sample == "random":
            import numpy as _np
            idx = _np.random.default_rng(args.sample_seed + abs(hash(dk)) % 10_000)
            frames[dk] = df.iloc[idx.permutation(len(df))[:args.n]]
        else:
            frames[dk] = df.head(args.n)

    n_cells = len(tags) * len(keys) * len(levels)
    n_calls = sum(len(frames[dk]) for dk in keys) * len(tags) * len(levels) * args.iters
    print("=" * 78)
    print(f"models   {', '.join(tags)}")
    print(f"datasets {', '.join(f'{k}({len(frames[k])})' for k in keys)}")
    print(f"levels   {', '.join(levels)}")
    print(f"{n_cells} cells, up to {n_calls} calls, budget ceiling ${args.budget:.2f}")
    print("=" * 78)

    if args.dry:
        dk = keys[0]
        s, u = prompts(dsets[dk], str(frames[dk].iloc[0]["smiles"]).strip())
        print(f"\n--- SYSTEM ---\n{s}\n--- USER ---\n{u}\n\nDRY RUN: nothing queried.")
        return

    k = key()

    if args.probe:
        # One call per (model, level) at the real max_tokens, to measure what each endpoint
        # actually emits before committing to a sweep. Cost is a few cents.
        dk = keys[0]
        row = frames[dk].iloc[0]
        rows = []
        for t in tags:
            for lv in levels:
                try:
                    r = call(k, models[t], lv, *prompts(dsets[dk], str(row["smiles"]).strip()),
                             args.max_tokens)
                    _spent += r["cost"]
                    rows.append(dict(tag=t, level=lv, **{x: r[x] for x in
                                ("reasoning_tokens", "completion", "cost", "finish")},
                                answered=r["value"] is not None, text=r["text"]))
                    print(f"  {t:16s} {lv:8s} reasoning={r['reasoning_tokens']:>6} "
                          f"completion={r['completion']:>6} ${r['cost']:.5f} "
                          f"{'' if r['value'] is not None else 'NO ANSWER'} [{r['finish']}]",
                          flush=True)
                except urllib.error.HTTPError as e:
                    print(f"  {t:16s} {lv:8s} HTTP {e.code} -- level not supported")
                    rows.append(dict(tag=t, level=lv, reasoning_tokens=None, completion=None,
                                     cost=0.0, finish=f"http{e.code}", answered=False, text=""))
                except Exception as e:
                    print(f"  {t:16s} {lv:8s} ERR {type(e).__name__}")
        P = pd.DataFrame(rows)
        os.makedirs(os.path.join(ROOT, "results", "meta"), exist_ok=True)
        P.to_csv(os.path.join(ROOT, "results", "meta", "ladder_probe.csv"), index=False)
        per_cell = sum(len(frames[dk]) for dk in keys) / len(keys) * args.iters
        proj = P.dropna(subset=["cost"]).groupby("tag").cost.sum() * per_cell * len(keys)
        print(f"\nprobe spent ${_spent:.3f}")
        print(f"projected full sweep at n={args.n}, iters={args.iters}:")
        for t, v in proj.items():
            print(f"  {t:16s} ${v:8.2f}")
        print(f"  {'TOTAL':16s} ${proj.sum():8.2f}")
        return

    total = 0.0
    for t in tags:
        md = models[t]
        for dk in keys:
            for lv in levels:
                if _spent >= args.budget:
                    print(f"\nBUDGET ${args.budget:.2f} reached (${_spent:.2f} spent) -- stopping.")
                    return
                total += run_cell(k, md, lv, dsets[dk], frames[dk], args.iters, args.threads,
                                  args.max_tokens, args.retries, args.budget, args.force)
    print(f"\nDONE. spent ${_spent:.2f} of ${args.budget:.2f}")


if __name__ == "__main__":
    main()
