"""
SUPERSEDED -- kept only to reproduce the numbers in the first version of this manuscript.

Use src/analyze_screening.py instead. This script tests against the GLOBAL permutation null
only, which destroys the model's accuracy along with the molecule-value pairing and therefore
reports any accurate model as contaminated: on ESOL it flags GPT-5.6 terra, Gemini 2.5 Pro and
Claude Haiku 4.5, all three of which sit at chance once the null holds accuracy fixed. It also
applies no multiplicity correction across the matrix, and it counts a cell as clean when the
retention ratio is merely unpowered (QM7/GPT-4.1 has 2,980 usable pairs and m1 = 0).

Chemistry-invariance control: canonical vs randomized SMILES on Delaney.

Same molecules, different (valid, non-canonical) SMILES strings. Chemistry is invariant to the
rewriting, so a structure-based predictor must score identically. A lookup keyed to the string as
it appears in the source corpus is not invariant. Collapse under randomization therefore excludes
chemical calculation as the source of digit-level agreement.
"""
import glob, json, os, numpy as np, pandas as pd
from scipy.stats import pearsonr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, DATA = os.path.join(ROOT, "results"), os.path.join(ROOT, "data")
TAG_NAME = {
 "gem25pro":"Gemini 2.5 Pro","gem3flash":"Gemini 3 Flash","gem31flashlite":"Gemini 3.1 Flash-Lite",
 "gem35flash":"Gemini 3.5 Flash","gem31pro":"Gemini 3.1 Pro","opus48":"Claude Opus 4.8",
 "sonnet5":"Claude Sonnet 5","haiku45":"Claude Haiku 4.5","terra":"GPT-5.6 terra",
 "gpt5":"GPT-5","gpt41":"GPT-4.1","qwen3":"Qwen3-235B","glm52":"GLM 5.2",
}
ORDER = ["opus48","sonnet5","gem35flash","gem31pro","gem3flash","glm52","terra",
         "gem25pro","gem31flashlite","gpt5","gpt41","haiku45","qwen3"]
N_PERM = 300

def sig_round(x,n): return "0" if x==0 else f"{x:.{n}g}"
def sig_figs(v):
    if v==0: return 1
    s=f"{v:.10g}".lstrip("-").replace(".","").lstrip("0")
    return max(1,len(s.rstrip("0"))) if s else 1
def leaves(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if isinstance(v,list) and v and all(isinstance(x,(int,float)) for x in v): yield k,v
            else: yield from leaves(v)
def load_preds(p):
    d=json.load(open(p)); out={}
    for mol,vals in leaves(d):
        nums=[float(x) for x in vals if x is not None and not (isinstance(x,float) and np.isnan(x))]
        if nums: out.setdefault(str(mol).strip(),[]).extend(nums)
    return out
def nested(T,P):
    m1=m2=m3=0
    for tv,pv in zip(T,P):
        if sig_round(pv,1)==sig_round(tv,1):
            m1+=1
            if sig_round(pv,2)==sig_round(tv,2):
                m2+=1
                if sig_round(pv,3)==sig_round(tv,3): m3+=1
    return m1,m2,m3

df = pd.read_csv(os.path.join(DATA,"delaney-processed.csv")).dropna(subset=["measured log solubility in mols per litre"])
lut = {str(k).strip(): float(v) for k,v in zip(df["Compound ID"], df["measured log solubility in mols per litre"])}
rng = np.random.default_rng(0)

def stats(path):
    ally=[];allp=[];T=[];P=[]
    for mol,plist in load_preds(path).items():
        tv=lut.get(mol)
        if tv is None: continue
        for pv in plist:
            ally.append(tv); allp.append(pv)
            if sig_figs(tv)>=3 and sig_figs(pv)>=3: T.append(tv);P.append(pv)
    ally=np.array(ally);allp=np.array(allp);T=np.array(T);P=np.array(P)
    if len(T)==0: return None
    m1,m2,m3=nested(T,P)
    cm3=np.array([nested(T,P[rng.permutation(len(P))])[2] for _ in range(N_PERM)])
    return dict(n3=len(T), r=pearsonr(allp,ally)[0], rmse=float(np.sqrt(np.mean((allp-ally)**2))),
                hit3=100*m3/len(T), chance3=100*cm3.mean()/len(T),
                R12=100*m2/m1 if m1 else float("nan"), R23=100*m3/m2 if m2 else float("nan"))

rows=[]
for tag in ORDER:
    c = os.path.join(RES,f"LLM_ZeroShot_delaney_{tag}.json")
    r_ = os.path.join(RES,f"LLM_ZeroShot_delaney_{tag}_rand.json")
    if not (os.path.exists(c) and os.path.exists(r_)): continue
    a, b = stats(c), stats(r_)
    if a is None or b is None: continue
    rows.append(dict(model=TAG_NAME[tag],
                     r_can=round(a["r"],3), r_rand=round(b["r"],3),
                     rmse_can=round(a["rmse"],3), rmse_rand=round(b["rmse"],3),
                     hit3_can=round(a["hit3"],2), hit3_rand=round(b["hit3"],2), chance=round(b["chance3"],2),
                     R23_can=round(a["R23"],1), R23_rand=round(b["R23"],1),
                     drop=round(a["hit3"]-b["hit3"],2)))
R=pd.DataFrame(rows)
if R.empty:
    print("no paired canonical/randomized files yet"); raise SystemExit
print("Delaney: CANONICAL vs RANDOMIZED SMILES (identical molecules, different strings)\n")
print(f"{'model':22s}{'r can':>7}{'r rnd':>7}{'RMSE can':>10}{'RMSE rnd':>10}"
      f"{'hit3 can':>10}{'hit3 rnd':>10}{'chance':>8}{'R23 can':>9}{'R23 rnd':>9}")
for _,x in R.iterrows():
    print(f"{x['model']:22s}{x['r_can']:>7}{x['r_rand']:>7}{x['rmse_can']:>10}{x['rmse_rand']:>10}"
          f"{x['hit3_can']:>10}{x['hit3_rand']:>10}{x['chance']:>8}{x['R23_can']:>9}{x['R23_rand']:>9}")
R.to_csv(os.path.join(RES,"randomization_control.csv"), index=False)
print("\nSaved results/randomization_control.csv")
print("Chemistry is invariant to SMILES rewriting; a string-keyed lookup is not.")
