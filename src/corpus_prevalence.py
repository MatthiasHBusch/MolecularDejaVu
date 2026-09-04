"""
How often does each benchmark actually appear in a web-scale pre-training corpus?

Every contamination audit infers exposure from model behaviour. This measures the other
side directly, in public corpora, using the infini-gram suffix-array API over

    v4_olmo-mix-1124_llama   OLMo-2 pre-training mix   4.6T tokens
    v4_dclm-baseline_llama   DCLM-baseline             4.3T tokens
    v4_dolma-v1_7_llama      Dolma v1.7                2.6T tokens

None of these is the training set of Claude, Gemini or GPT. They are large public crawls
of the same web, so they estimate the PREVALENCE of a benchmark on the open internet --
which is the mechanism by which a closed model would have ingested it. Prevalence measured
here is therefore a predictor of, not a substitute for, the behavioural audit.

One model in the panel, OLMo 3, is trained on a corpus in this list, so for that model the
measurement is not a proxy at all: it is a direct check of whether the benchmark was in the
training data. That is the control the prior blinding study named as impossible.

Three measurements per dataset:
  header_count   occurrences of the benchmark's distinctive column header or name. Detects
                 the CSV itself being present.
  smiles_rate    fraction of a random sample of the dataset's SMILES strings that occur at
                 least once. Detects the molecules being present in any context.
  row_rate       fraction of sampled molecules whose SMILES *and* published value occur in
                 the same document (CNF query). This is the one that matters: co-occurrence
                 in one document is what makes a value recallable from a structure.

Free, no API key. Rate-limited, so a sample rather than the full dataset is queried.

    python src/corpus_prevalence.py                     # all datasets, 60 molecules each
    python src/corpus_prevalence.py --sample 150 --index v4_olmo-mix-1124_llama
"""
import argparse, json, os, sys, time, urllib.error, urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCREEN = os.path.join(ROOT, "data", "screening")
REG = os.path.join(ROOT, "src", "registry", "datasets.json")
OUT = os.path.join(ROOT, "results", "corpus_prevalence.csv")
API = "https://api.infini-gram.io/"

INDEXES = ["v4_olmo-mix-1124_llama", "v4_dclm-baseline_llama", "v4_dolma-v1_7_llama"]

# Strings unique enough that a hit means the benchmark FILE, not the topic. Generic phrases
# are deliberately excluded: "antiviral potency" occurs 7,635 times in Dolma and "pIC50"
# tens of thousands, none of which is evidence that the benchmark table was crawled. Only
# column headers and distributed filenames qualify.
HEADERS = {
    "esol":          ["measured log solubility in mols per litre", "delaney-processed"],
    "freesolv":      ["FreeSolv", "SAMPL.csv"],
    "lipophilicity": ["CMPD_CHEMBLID", "Lipophilicity.csv"],
    "bace":          ["bace.csv", "ChiralCenterCountAllPossible"],
    "aqsoldb":       ["AqSolDB"],
    "caco2":         ["Caco2_Wang"],
    "ld50":          ["LD50_Zhu"],
    "ppbr":          ["PPBR_AZ"],
    "qm7":           ["u0_atom", "qm7.csv"],
    "qm8":           ["E1-CC2"],
    "antiviral":     ["ASAP Discovery", "antiviral_potency.csv"],
    "boilingpoint":  ["known_boiling_points"],
}


def query(payload, tries=4, pause=1.5):
    for k in range(tries):
        try:
            req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=180))
            if isinstance(r, dict) and "error" in r:
                return None
            return r
        except Exception:
            time.sleep(pause * (k + 1))
    return None


def count(index, s):
    r = query({"index": index, "query_type": "count", "query": s})
    return None if r is None else int(r.get("count", 0))


def cnf_count(index, a, b):
    """Documents containing both strings. The API's CNF syntax is 'X AND Y'."""
    r = query({"index": index, "query_type": "find_cnf", "query": f"{a} AND {b}"})
    return None if r is None else int(r.get("cnt", 0))


def fmt_value(v: float) -> str:
    """The value as it is written in the published CSV -- that is the string in the corpus."""
    s = f"{v:.10g}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--index", default=None, help="restrict to one index")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    idxs = [args.index] if args.index else INDEXES
    reg = json.load(open(REG))["datasets"]
    if args.only:
        reg = [d for d in reg if d["key"] in args.only]

    rows = []
    for cfg in reg:
        key = cfg["key"]
        f = os.path.join(SCREEN, f"{key}.csv")
        if not os.path.exists(f):
            print(f"  {key}: no prepared CSV, skipping")
            continue
        df = pd.read_csv(f)
        rng = np.random.default_rng(args.seed)
        take = rng.choice(len(df), size=min(args.sample, len(df)), replace=False)
        sub = df.iloc[take]

        for index in idxs:
            hdr = {}
            for h in HEADERS.get(key, []):
                hdr[h] = count(index, h)
                time.sleep(0.3)

            smi_hits = row_hits = 0
            n_ok = 0
            for _, r in sub.iterrows():
                smi = str(r["smiles"]).strip()
                # infini-gram tokenises with a leading-space marker, so extremely short
                # strings ("C", "O") match essentially every document and are meaningless.
                if len(smi) < 8:
                    continue
                n_ok += 1
                c = count(index, smi)
                time.sleep(0.25)
                if c:
                    smi_hits += 1
                    if cnf_count(index, smi, fmt_value(float(r["value"]))):
                        row_hits += 1
                    time.sleep(0.25)

            rows.append(dict(
                dataset=key, dataset_name=cfg["name"], ds_class=cfg["class"], index=index,
                n_queried=n_ok,
                header_count=max([v for v in hdr.values() if v is not None], default=None),
                header_detail=json.dumps(hdr),
                smiles_present=smi_hits, smiles_rate=100.0 * smi_hits / max(n_ok, 1),
                row_present=row_hits, row_rate=100.0 * row_hits / max(n_ok, 1)))
            print(f"  {key:14s} {index:26s} header={rows[-1]['header_count']}"
                  f"  smiles {smi_hits}/{n_ok} ({rows[-1]['smiles_rate']:.1f}%)"
                  f"  row {row_hits}/{n_ok} ({rows[-1]['row_rate']:.1f}%)", flush=True)

    if not rows:
        sys.exit("nothing measured")
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
