#
# Blinding-convergence experiment.
#
# The screening establishes WHICH (model, benchmark) pairs are contaminated. This asks what
# that contamination is worth: if a model's advantage on a benchmark comes from recall
# rather than chemistry, then disabling recall should collapse the advantage, and a
# retrieving and a non-retrieving model of comparable chemical competence should converge
# to the same accuracy.
#
# Five conditions, ordered by how much of the retrieval path they close:
#
#   L1  clear      published SMILES, named dataset and property, published values
#                  -> every route open: recall, chemical knowledge, in-context learning
#   L2  randomSMI  randomised SMILES, named dataset and property
#                  -> string-level lookup closed; molecule-level recall still open
#   L3  generic    published SMILES, property named only as "a molecular property"
#                  -> the property-name cue is closed
#   L5  agnostic   character-substituted SMILES, "sample property", published values
#                  -> molecule-level recall closed; only in-context learning remains
#   L6  agnostic+  character-substituted SMILES, "sample property", TRANSFORMED values
#                  -> value-scale recognition closed as well
#
# Levels are numbered to match the six-level framework of the prior blinding study, whose
# transformed_smiles / transformed_solubility columns are reused as smiles_blind / value_blind.
#
# NOT verbatim any more, and a resume hazard. `smiles_blind` was regenerated under BLIND_MAP
# version 2 (see prepare_datasets.py): version 1 had no entry for hydrogen, so `[nH]` blinded to
# `{dH}` and leaked the atom. The 8 `substituted` cells in results/convergence/ were bought on
# 28 July under version 1, so their prompts CANNOT be rebuilt from the current column -- resuming
# or extending that axis would mix two string sets inside one arm. Re-buy the substituted axis
# from scratch, or leave it as the version-1 record it is; do not top it up.
# The published and randomised axes are unaffected: neither column was touched.
#
# Unlike the zero-shot probe, these conditions need in-context examples: at L5/L6 there is
# no other information in the prompt. Each query therefore carries K training pairs drawn
# from a fixed pool disjoint from the test molecules.
#
# The prediction under test: at L1 the retriever beats the non-retriever by a wide margin;
# at L5/L6 the gap closes. If it does not close, the retriever's advantage was chemistry
# after all, and the contamination finding does not license discounting its benchmark score.
#
# Usage
#   julia src/Run_Convergence.jl --dry
#   julia src/Run_Convergence.jl --models opus48,gem31flashlite --dataset esol --ntest 100 --shots 60
#
using CSV, DataFrames, JSON, Dates, Printf, Random, Statistics, HTTP
# Shared LLM helper library. Defaults to src/lib inside this repository; point
# LLM_JULIA_LIB at another checkout to use one kept outside the repo.
const LLMLIB = get(ENV, "LLM_JULIA_LIB", joinpath(@__DIR__, "lib"))
include(joinpath(LLMLIB, "LLMs.jl"))
include(joinpath(LLMLIB, "FileWritingHelpers.jl"))

const SCRIPT_DIR = @__DIR__
const ROOT = normpath(joinpath(SCRIPT_DIR, ".."))

# RQ2 is a FACTORIAL, not a ladder. "How deep does retrieval go?" decomposes into two
# independent axes, and the interesting question is which one actually closes the retrieval
# path -- a single ordered sequence of levels confounds them.
#
#   structure axis   published -> randomised SMILES -> character-substituted SMILES
#   label axis       true values -> affine -> binned-permutation -> sine
#
# The label axis is the discriminating one. An affine relabel is MONOTONIC, so a model that
# recalls a molecule's true value can recover the map from a few in-context anchors and push
# its recollection through it; the two non-monotonic relabels cannot be inverted that way,
# but remain learnable in context (the binned permutation is invertible; the sine is
# many-to-one and merely lowers the ceiling). If accuracy survives affine relabeling and
# collapses under non-monotonic relabeling, that gap is retrieval, not chemistry.
const SMILES_AXIS = ["published", "randomised", "substituted"]
const LABEL_AXIS  = ["true", "affine", "nonmono", "sine"]

const SMILES_COL = Dict("published" => :smiles, "randomised" => :smiles_random,
                        "substituted" => :smiles_blind)
const LABEL_COL  = Dict("true" => :value, "affine" => :value_affine,
                        "nonmono" => :value_nonmono, "sine" => :value_sine)

"Prompt framing follows the STRUCTURE axis: a substituted string cannot be called a molecule."
function level_spec(smi_lvl, lab_lvl, ds)
    mode = smi_lvl == "substituted" ? :agnostic : :chem
    if mode == :agnostic
        return (SMILES_COL[smi_lvl], LABEL_COL[lab_lvl], "sample property", "arbitrary units",
                nothing, :agnostic)
    end
    # Once the labels are transformed the property name and unit are no longer truthful,
    # so the task is stated generically even though the structures are still real molecules.
    if lab_lvl == "true"
        return (SMILES_COL[smi_lvl], LABEL_COL[lab_lvl], ds["prompt_property"],
                ds["prompt_unit"], ds["prompt_dsname"], :chem)
    end
    return (SMILES_COL[smi_lvl], LABEL_COL[lab_lvl], "molecular property",
            "arbitrary units", nothing, :generic)
end

function build_prompts(smi_lvl, lab_lvl, ds, train::DataFrame, test_row)
    smcol, vcol, prop, unit, dsname, mode = level_spec(smi_lvl, lab_lvl, ds)

    examples = join(["$(strip(string(r[smcol]))) -> $(r[vcol])" for r in eachrow(train)], "\n")
    target = strip(string(test_row[smcol]))

    if mode == :agnostic
        sys = """You are a professional machine learning model with expert knowledge in regression. \
You are given string based structure representations of samples together with their known sample \
property, and must predict the sample property of a new sample. Provide only the numerical value \
as output, without any additional text."""
        usr = """Training samples (structure string -> sample property):
$examples

Predict the sample property of this sample: $target
Provide only the numerical value as output, without any additional text."""
    elseif mode == :generic
        sys = """You are an expert chemist. You are given SMILES strings of molecules together with \
a known molecular property, and must predict that property for a new molecule. Provide only the \
numerical value as output, without any additional text."""
        usr = """Training molecules (SMILES -> molecular property):
$examples

Predict the molecular property of this molecule: $target
Provide only the numerical value as output, without any additional text."""
    else
        sys = """You are an $(ds["prompt_expert"]) and know the $(dsname) dataset very well. You are \
given SMILES strings of molecules together with their $(prop) in $(unit), and must predict the \
$(prop) of a new molecule. Provide only the numerical value as output, without any additional text."""
        usr = """Training molecules from the $(dsname) dataset (SMILES -> $(prop) in $(unit)):
$examples

Predict the $(prop) in $(unit) of this molecule: $target
Provide only the numerical value as output, without any additional text."""
    end
    return sys, usr
end

function parse_number(s::AbstractString)
    txt = replace(String(s), "\u2212" => "-", "\u2013" => "-", "," => "")
    m = match(r"[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+", txt)
    if m !== nothing
        v = tryparse(Float64, m.match); v === nothing || return v
    end
    return search_for_last_number_in_string(String(txt))
end

function parse_args(argv)
    o = Dict{String,Any}("models" => ["opus48", "gem31flashlite"], "dataset" => "esol",
                         "ntest" => 100, "shots" => 60, "iters" => 2, "threads" => 8,
                         "dry" => false, "smiles" => SMILES_AXIS, "labels" => LABEL_AXIS,
                         "seed" => 7)
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--dry"; o["dry"] = true; i += 1
        elseif a == "--models";  o["models"] = String.(split(argv[i+1], ",")); i += 2
        elseif a == "--dataset"; o["dataset"] = argv[i+1]; i += 2
        elseif a == "--smiles";  o["smiles"] = String.(split(argv[i+1], ",")); i += 2
        elseif a == "--labels";  o["labels"] = String.(split(argv[i+1], ",")); i += 2
        elseif a == "--ntest";   o["ntest"] = parse(Int, argv[i+1]); i += 2
        elseif a == "--shots";   o["shots"] = parse(Int, argv[i+1]); i += 2
        elseif a == "--iters";   o["iters"] = parse(Int, argv[i+1]); i += 2
        elseif a == "--threads"; o["threads"] = parse(Int, argv[i+1]); i += 2
        elseif a == "--seed";    o["seed"] = parse(Int, argv[i+1]); i += 2
        else error("unknown flag $a")
        end
    end
    o
end

function main()
    o = parse_args(ARGS)
    dsall = JSON.parsefile(joinpath(SCRIPT_DIR, "registry", "datasets.json"))["datasets"]
    mdall = JSON.parsefile(joinpath(SCRIPT_DIR, "registry", "models.json"))["models"]
    ds = dsall[findfirst(d -> d["key"] == o["dataset"], dsall)]

    data = DataFrame(CSV.File(joinpath(ROOT, "data", "screening", "$(o["dataset"]).csv")))
    dropmissing!(data, [:smiles, :smiles_random, :smiles_blind, :value,
                        :value_affine, :value_nonmono, :value_sine])

    # One fixed split for every (model, level) cell. Varying the split across conditions
    # would confound the level effect with the difficulty of whichever molecules were drawn.
    rng = MersenneTwister(o["seed"])
    perm = randperm(rng, nrow(data))
    test_idx = perm[1:min(o["ntest"], nrow(data))]
    pool_idx = perm[(length(test_idx)+1):end]
    train = data[pool_idx[1:min(o["shots"], length(pool_idx))], :]

    outdir = joinpath(ROOT, "results", "convergence")
    mkpath(outdir)
    ncells = length(o["models"]) * length(o["smiles"]) * length(o["labels"])
    println("dataset=$(ds["name"])  test=$(length(test_idx))  shots=$(nrow(train))  " *
            "models=$(join(o["models"], ","))\nsmiles=$(join(o["smiles"], ","))  " *
            "labels=$(join(o["labels"], ","))  -> $ncells cells, " *
            "$(ncells * length(test_idx) * o["iters"]) calls")

    for tag in o["models"]
        md = mdall[findfirst(m -> m["tag"] == tag, mdall)]
        llm = LLMAccessOpenRouter(key_openrouter, md["endpoint"],
                                  String[String(p) for p in get(md, "providers", [])],
                                  get(md, "service_tier", ""))
        for smi_lvl in o["smiles"], lab_lvl in o["labels"]
            level = "$(smi_lvl)__$(lab_lvl)"
            out_file = joinpath(outdir, "$(o["dataset"])__$(tag)__$(level).json")
            if isfile(out_file) && get(ENV, "CV_FORCE", "0") != "1"
                println("[CV] $tag $level already done -- skipped"); continue
            end
            convs = Vector{Vector{Tuple}}()
            ids = String[]
            for ti in test_idx
                sys, usr = build_prompts(smi_lvl, lab_lvl, ds, train, data[ti, :])
                for _ in 1:o["iters"]
                    push!(convs, [("system", sys), ("user", usr)])
                end
                push!(ids, string(data[ti, :mol_id]))
            end
            @printf("\n[CV] %-16s %-24s %d calls\n", tag, level, length(convs))
            if o["dry"]
                s, u = build_prompts(smi_lvl, lab_lvl, ds,
                                     train[1:min(3, nrow(train)), :], data[test_idx[1], :])
                println("--- SYSTEM ---\n$s\n--- USER (3 shots shown) ---\n$u\n")
                continue
            end
            # temperature = top_p = 1 for every model, matching Run_ZeroShot.jl: the shared
            # library's 0.7 / 0.95 default is not honoured identically across this panel.
            answers = ask_gpt_threaded(llm, convs; num_threads=o["threads"],
                                       temperature=1.0, top_p=1.0,
                                       reasoning_effort=md["reasoning"], retries=8)
            keys_list = Vector{Vector{String}}(); vals = []
            for (j, mol) in enumerate(ids), i in 1:o["iters"]
                k = (j - 1) * o["iters"] + i
                k <= length(answers) || continue
                push!(keys_list, String[level, md["endpoint"], mol])
                push!(vals, parse_number(answers[k]))
            end
            append_values_to_json(out_file, keys_list, vals)
            println("[CV] saved -> $(basename(out_file))")
        end
    end
end

main()
