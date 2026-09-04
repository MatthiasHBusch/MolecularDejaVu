"""
Re-query a handful of molecules and keep the model's reasoning trace.

The ladder stores parsed values and the first 80 characters of each reply, which is enough to
count hits and not enough to say what the model did. This asks the direct question: on the
molecules whose recall switches on with the thinking budget, and on matched molecules where it
does not, what is actually in the trace?

Selection is made from an existing ladder cell, so the two groups are defined by measured
behaviour rather than by guesswork:

    switchers      missed at the low level, reproduced verbatim at the high one
    non-switchers  missed at both

    python src/capture_traces.py --dataset ld50 --tag gem3flash --n 8
    python src/capture_traces.py --dataset ld50 --tag gem3flash --level medium --budget 2
"""
import argparse, json, os, sys, urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memodetect import sig_figs, sig_round
import run_reasoning_ladder as L

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LADDER, SCREEN, REG = (os.path.join(ROOT, "results", "ladder"),
                       os.path.join(ROOT, "data", "screening"),
                       os.path.join(ROOT, "src", "registry"))
OUT = os.path.join(ROOT, "results", "traces")


def call_with_trace(k, md, level, sysmsg, usr, max_tokens=8000):
    body = {"model": md["endpoint"],
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": usr}],
            "max_tokens": max_tokens, "temperature": 1.0, "top_p": 1.0,
            "usage": {"include": True},
            "reasoning": {"effort": level}}
    if md.get("providers"):
        body["provider"] = {"order": md["providers"]}
    if md.get("service_tier"):
        body["service_tier"] = md["service_tier"]
    req = urllib.request.Request(L.API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {k}"})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    msg = d["choices"][0]["message"]
    u = d.get("usage", {}) or {}
    det = u.get("completion_tokens_details", {}) or {}
    # Providers expose the trace under different keys, and some summarise it rather than
    # returning it verbatim. Whatever comes back is recorded as-is.
    trace = msg.get("reasoning") or ""
    if not trace and isinstance(msg.get("reasoning_details"), list):
        trace = "\n".join(str(x.get("text", "")) for x in msg["reasoning_details"])
    return dict(text=(msg.get("content") or "").strip(), trace=trace,
                reasoning_tokens=int(det.get("reasoning_tokens", 0) or 0),
                cost=float(u.get("cost", 0) or 0))


def classify(dskey, tag, lo, hi, lut):
    """Split the molecules of an existing ladder cell into switchers and non-switchers."""
    def load(level):
        p = os.path.join(LADDER, f"{dskey}__{tag}__{level}.json")
        if not os.path.exists(p):
            sys.exit(f"missing ladder cell {os.path.basename(p)}")
        out = {}
        for c in json.load(open(p))["calls"]:
            out.setdefault(c["mol_id"], []).append(c["value"])
        return out

    A, B = load(lo), load(hi)
    hit = lambda vals, t: any(v is not None and sig_figs(t) >= 3 and sig_figs(v) >= 3
                              and sig_round(v, 3) == sig_round(t, 3) for v in vals)
    sw, non = [], []
    for mol in sorted(set(A) & set(B) & set(lut)):
        t = lut[mol]
        a, b = hit(A[mol], t), hit(B[mol], t)
        (sw if (b and not a) else non if (not a and not b) else []).append(mol)
    return sw, non


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ld50")
    ap.add_argument("--tag", default="gem3flash")
    ap.add_argument("--lo", default="none")
    ap.add_argument("--level", default="high", help="level to capture traces at")
    ap.add_argument("--n", type=int, default=8, help="molecules per group")
    ap.add_argument("--budget", type=float, default=3.0)
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(SCREEN, f"{args.dataset}.csv")).dropna(subset=["value"])
    lut = {str(k).strip(): float(v) for k, v in zip(df["mol_id"], df["value"])}
    smi = {str(k).strip(): str(s) for k, s in zip(df["mol_id"], df["smiles"])}
    ds = {d["key"]: d for d in json.load(open(os.path.join(REG, "datasets.json")))["datasets"]}
    md = {m["tag"]: m for m in json.load(open(os.path.join(REG, "models.json")))["models"]}

    sw, non = classify(args.dataset, args.tag, args.lo, args.level, lut)
    print(f"{args.dataset}/{args.tag}: {len(sw)} switchers, {len(non)} non-switchers; "
          f"capturing {args.n} of each at effort='{args.level}'\n")

    k = L.key()
    spent, records = 0.0, []
    for group, mols in [("switcher", sw[:args.n]), ("non-switcher", non[:args.n])]:
        for mol in mols:
            if spent >= args.budget:
                print(f"budget ${args.budget} reached"); break
            s, u = L.prompts(ds[args.dataset], smi[mol].strip())
            try:
                r = call_with_trace(k, md[args.tag], args.level, s, u)
            except Exception as e:
                print(f"  {mol[:40]}: {type(e).__name__}")
                continue
            spent += r["cost"]
            records.append(dict(group=group, mol_id=mol, smiles=smi[mol],
                                published=lut[mol], **r))
            print("=" * 100)
            print(f"[{group}] {mol[:70]}")
            print(f"  SMILES {smi[mol][:88]}")
            print(f"  published {lut[mol]}   model answered {r['text'][:60]!r}   "
                  f"({r['reasoning_tokens']} thinking tokens)")
            if r["trace"]:
                print("  --- trace ---")
                for line in r["trace"].strip().splitlines():
                    print(f"  {line[:110]}")
            else:
                print("  (no trace returned by this endpoint)")
    # Never overwrite. An earlier version wrote to a fixed path; running it twice on the same
    # cell destroyed the first capture, and two observations that had already been quoted in a
    # draft could not afterwards be produced. A capture tool that clobbers is a tool that
    # deletes evidence between the observation and the write-up.
    os.makedirs(OUT, exist_ok=True)
    base = f"{args.dataset}__{args.tag}__{args.level}"
    p = os.path.join(OUT, f"{base}.json")
    if os.path.exists(p):
        n = 2
        while os.path.exists(os.path.join(OUT, f"{base}__run{n}.json")):
            n += 1
        p = os.path.join(OUT, f"{base}__run{n}.json")
        print(f"  ({base}.json exists -- writing run {n} instead of overwriting)")
    json.dump(records, open(p, "w"), indent=1)
    print(f"\nspent ${spent:.3f}; saved {p}")


if __name__ == "__main__":
    main()
