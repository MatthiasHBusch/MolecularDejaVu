"""
Find, per model, the reasoning setting that actually suppresses deliberation.

Why this exists. The probe is a test of RECALL, so the model must not be given room to
*compute* an estimate. The natural way to enforce that is a fixed thinking budget, and the
shared library exposes one as `reasoning={"max_tokens": N}`. Measurement shows that several
endpoints ignore it:

    Gemini 3.1 Pro   asked 128 -> emitted 641 reasoning tokens
    Gemini 3 Flash   asked 128 -> emitted 765
    Gemini 3.6 Flash asked 128 -> emitted 427
    GPT-5.5          asked 128 -> emitted 800 and returned an EMPTY answer, because the
                                  800-token completion cap was exhausted by thinking

while `reasoning={"effort": "minimal"}` reaches genuinely zero reasoning tokens on the two
Flash models and costs up to 40x less. A budget that is silently multiplied by five is not a
controlled condition, so the setting has to be chosen by measurement rather than assumed.

For each model this tries the candidate settings in order of increasing deliberation and
records reasoning tokens, total cost and whether a parseable number came back. The chosen
setting is the first that returns an answer, preferring fewer reasoning tokens.

    python src/calibrate_reasoning.py                 # probe all enabled models
    python src/calibrate_reasoning.py --apply         # ...and write it into the registry
"""
import argparse, json, os, re, sys, urllib.error, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REG = os.path.join(ROOT, "src", "registry", "models.json")
OUT = os.path.join(ROOT, "results", "meta", "reasoning_calibration.json")

CANDIDATES = [("none", {"effort": "none"}),
              ("minimal", {"effort": "minimal"}),
              ("low", {"effort": "low"}),
              ("max_tokens:128", {"max_tokens": 128})]

SYS = ("You are an expert chemist and know the Delaney (ESOL) dataset very well. You are given "
       "a SMILES string of a molecule. Your task is to predict the measured aqueous solubility "
       "of that molecule in log(mol/L) as reported in that dataset. Provide only the numerical "
       "value as output, without any additional text.")
USR = ("What is the measured aqueous solubility in log(mol/L) of the molecule with the following "
       "SMILES string in the Delaney (ESOL) dataset: Cc1occc1C(=O)Nc2ccccc2? Provide only the "
       "numerical value as output, without any additional text.")

NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# The key resolution lives in one place; see run_reasoning_ladder.key().
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_reasoning_ladder import key  # noqa: E402


def call(k, model, provs, tier, reasoning, max_tokens=800):
    body = {"model": model,
            "messages": [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                         {"role": "user", "content": [{"type": "text", "text": USR}]}],
            "max_tokens": max_tokens, "usage": {"include": True}}
    if reasoning is not None:
        body["reasoning"] = reasoning
    if provs:
        body["provider"] = {"order": provs}
    if tier:
        body["service_tier"] = tier
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {k}"})
    d = json.load(urllib.request.urlopen(req, timeout=240))
    u = d.get("usage", {}) or {}
    det = u.get("completion_tokens_details", {}) or {}
    txt = (d["choices"][0]["message"].get("content") or "").strip()
    return dict(reasoning_tokens=det.get("reasoning_tokens", 0) or 0,
                completion=u.get("completion_tokens", 0), cost=float(u.get("cost", 0) or 0),
                answered=bool(NUM.search(txt)), text=txt[:40])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    k = key()
    reg = json.load(open(REG, encoding="utf8"))
    models = [m for m in reg["models"] if m.get("enabled", True)]
    if args.only:
        models = [m for m in models if m["tag"] in args.only]

    report, total = {}, 0.0
    for m in models:
        rows = []
        for name, rs in CANDIDATES:
            try:
                r = call(k, m["endpoint"], m.get("providers", []), m.get("service_tier", ""), rs)
                r["setting"] = name
                rows.append(r)
                total += r["cost"]
                flag = "" if r["answered"] else "  NO ANSWER"
                print(f"  {m['tag']:15s} {name:15s} reasoning={r['reasoning_tokens']:>5} "
                      f"completion={r['completion']:>5} ${r['cost']:.5f}{flag}", flush=True)
            except urllib.error.HTTPError as e:
                print(f"  {m['tag']:15s} {name:15s} HTTP {e.code} (rejected)", flush=True)
            except Exception as e:
                print(f"  {m['tag']:15s} {name:15s} ERR {type(e).__name__}", flush=True)

        ok = [r for r in rows if r["answered"]]
        if not ok:
            print(f"  {m['tag']:15s} -> NO SETTING PRODUCED AN ANSWER")
            report[m["tag"]] = dict(chosen=None, rows=rows)
            continue
        # Fewest reasoning tokens wins; cost breaks ties. Both are the same objective here --
        # suppress deliberation -- but cost is the tiebreak that matters for a 400-cell sweep.
        best = min(ok, key=lambda r: (r["reasoning_tokens"], r["cost"]))
        report[m["tag"]] = dict(chosen=best["setting"], reasoning_tokens=best["reasoning_tokens"],
                                cost=best["cost"], rows=rows)
        print(f"  {m['tag']:15s} -> CHOSEN {best['setting']} "
              f"({best['reasoning_tokens']} reasoning tokens, ${best['cost']:.5f}/call)\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=1)
    print(f"\nprobe cost ${total:.3f}; saved {OUT}")

    if args.apply:
        for m in reg["models"]:
            r = report.get(m["tag"])
            if r and r["chosen"]:
                m["reasoning"] = r["chosen"]
                m["reasoning_tokens_measured"] = r["reasoning_tokens"]
        json.dump(reg, open(REG, "w", encoding="utf8"), indent=2)
        print("registry updated")
    else:
        print("re-run with --apply to write the chosen settings into registry/models.json")


if __name__ == "__main__":
    main()
