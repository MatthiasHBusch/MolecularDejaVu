#
# Unified zero-shot recall probe.
#
# Replaces Run_Delaney_ZeroShot.jl / Run_Lipophilicity_ZeroShot.jl / Run_QM7_ZeroShot.jl,
# which were three copies of the same code differing only in the prompt strings and the
# column names. Everything that varies now lives in src/registry/{datasets,models}.json,
# so adding a benchmark or a model is a data change, not a code change.
#
# The probe itself is unchanged in substance: fully unblinded, SMILES only, no in-context
# examples, deliberation suppressed where the provider allows it. Under that protocol the
# only route to digit-level agreement with the published value is having seen it.
#
# Environment
#   ZS_MODELTAG   model tag from registry/models.json            (required)
#   ZS_DATASET    dataset key from registry/datasets.json        (required)
#   ZS_VARIANT    canonical | random | blind                     (default canonical)
#                   canonical -> the SMILES exactly as published (the string in the corpus)
#                   random    -> a different valid SMILES for the same molecule
#                   blind     -> character-substituted SMILES (prior study, levels 5/6)
#   ZS_N          number of molecules, from the top of the file  (default all)
#   ZS_ITERS      repeats per molecule                           (default 3)
#   ZS_THREADS    concurrent requests                            (default 10)
#   ZS_OUTDIR     output directory                               (default ../results/zs)
#   ZS_FORCE      "1" to re-query a cell that already has a complete result file
#
# Output: <ZS_OUTDIR>/<dataset>__<modeltag>__<variant>.json, nested
#         {variant -> reasoning -> endpoint -> mol_id -> [values...]}
# Cost:   one row appended to <ZS_OUTDIR>/../usage_log.csv per cell.
#
using CSV, DataFrames, JSON, Dates, Printf, HTTP
# Shared LLM helper library. Defaults to src/lib inside this repository; point
# LLM_JULIA_LIB at another checkout to use one kept outside the repo.
const LLMLIB = get(ENV, "LLM_JULIA_LIB", joinpath(@__DIR__, "lib"))
include(joinpath(LLMLIB, "LLMs.jl"))
include(joinpath(LLMLIB, "FileWritingHelpers.jl"))

const SCRIPT_DIR = @__DIR__
const ROOT = normpath(joinpath(SCRIPT_DIR, ".."))

"""
    parse_number(s)

Extract the model's numeric answer.

`search_for_last_number_in_string` from the shared library does not understand scientific
notation: given "4.33e-01" its regex matches the trailing "01" and returns 1.0. That is
silent and catastrophic for benchmarks whose values are naturally written in exponent form
(QM8 excitation energies, QM9 energies), so scientific notation is handled here first and
the library routine is used only as the fallback for plain decimals.
"""
function parse_number(s::AbstractString)
    txt = replace(String(s), "\u2212" => "-", "\u2013" => "-", "," => "")
    m = match(r"[-+]?(?:\d+\.?\d*|\.\d+)[eE][-+]?\d+", txt)
    if m !== nothing
        v = tryparse(Float64, m.match)
        v === nothing || return v
    end
    return search_for_last_number_in_string(String(txt))
end

function load_registry(file, key, id)
    entries = JSON.parsefile(joinpath(SCRIPT_DIR, "registry", file))[key]
    idx = findfirst(e -> e[id[1]] == id[2], entries)
    idx === nothing && error("$(id[2]) not found in registry/$file")
    return entries[idx]
end

"""
    run_cell(dskey, tag; variant, n, iters, threads, outdir, force, dryrun, reasoning)

Query one (dataset, model, variant) cell. Called both by the CLI entry point below and,
in-process, by run_matrix.jl -- looping inside one Julia session avoids paying the ~20 s
startup and package-load cost several hundred times over.
"""
function run_cell(dskey::AbstractString, tag::AbstractString;
                  variant::AbstractString = get(ENV, "ZS_VARIANT", "canonical"),
                  n::Union{Int,Nothing} = nothing,
                  iters::Int = parse(Int, get(ENV, "ZS_ITERS", "3")),
                  threads::Int = parse(Int, get(ENV, "ZS_THREADS", "10")),
                  outdir::AbstractString = get(ENV, "ZS_OUTDIR", joinpath(ROOT, "results", "zs")),
                  force::Bool = get(ENV, "ZS_FORCE", "0") == "1",
                  dryrun::Bool = get(ENV, "ZS_DRYRUN", "0") == "1",
                  reasoning_override::Union{String,Nothing} = nothing)
    variant in ("canonical", "random", "blind") || error("variant must be canonical|random|blind")

    ds = load_registry("datasets.json", "datasets", ("key", dskey))
    md = load_registry("models.json", "models", ("tag", tag))

    mkpath(outdir)
    out_file = joinpath(outdir, "$(dskey)__$(tag)__$(variant).json")

    data_file = joinpath(ROOT, "data", "screening", "$(dskey).csv")
    isfile(data_file) || error("missing $data_file -- run: python src/prepare_datasets.py")
    data = DataFrame(CSV.File(data_file))

    n_cap = n === nothing ? parse(Int, get(ENV, "ZS_N", string(nrow(data)))) : n
    data = data[1:min(n_cap, nrow(data)), :]
    iterations = iters

    # Resumability: the matrix is large and endpoints fail. A cell that already holds the
    # expected number of predictions is skipped, so an interrupted sweep can simply be re-run.
    if isfile(out_file) && !force
        got = 0
        try
            walk(o) = o isa Dict ? sum(walk(v) for v in values(o); init=0) :
                      o isa Vector ? length(o) : 0
            got = walk(JSON.parsefile(out_file))
        catch
        end
        if got >= nrow(data) * iterations
            println("[ZS] $(dskey)/$(tag)/$(variant): already complete ($got predictions) -- skipped")
            return 0.0
        end
        println("[ZS] $(dskey)/$(tag)/$(variant): incomplete ($got predictions) -- re-querying")
    end

    smiles_col = variant == "canonical" ? :smiles :
                 variant == "random"    ? :smiles_random : :smiles_blind

    # The blinded variant must not name the dataset or the property: the whole point of the
    # level-5/6 condition is that neither the structure nor the task is recognisable.
    local system_prompt::String
    local make_prompt
    if variant == "blind"
        system_prompt = """You are a professional machine learning model with expert knowledge in regression. \
You are given a string based structure representation of a sample. Your task is to predict the \
numerical sample property. Provide only the numerical value as output, without any additional text."""
        make_prompt = s -> "What is the sample property of the sample with the following structure " *
                           "representation string: $s? Provide only the numerical value as output, " *
                           "without any additional text."
    else
        dsname = ds["prompt_dsname"]; prop = ds["prompt_property"]
        unit = ds["prompt_unit"]; expert = ds["prompt_expert"]
        system_prompt = """You are an $(expert) and know the $(dsname) dataset very well. You are given a \
SMILES string of a molecule. Your task is to predict the $(prop) of that molecule in $(unit) as reported \
in that dataset. Provide only the numerical value as output, without any additional text."""
        make_prompt = s -> "What is the $(prop) in $(unit) of the molecule with the following SMILES " *
                           "string in the $(dsname) dataset: $s? Provide only the numerical value as " *
                           "output, without any additional text."
    end

    # Flex tier is half price for the endpoints that expose one (Google and OpenAI
    # terra/luna/sol/5.5; verified per-endpoint, see registry/models.json). It buys that with
    # higher latency and lower availability, so the retry budget matters more here. Requesting
    # flex on a model with no flex endpoint routes normally at standard rates rather than
    # failing, but the registry only sets it where an endpoint actually exists.
    tier = get(md, "service_tier", "")
    provs = String[String(p) for p in get(md, "providers", [])]
    llm = LLMAccessOpenRouter(key_openrouter, md["endpoint"], provs, tier)
    reasoning = reasoning_override === nothing ? get(ENV, "ZS_REASONING", md["reasoning"]) :
                reasoning_override

    conversations = Vector{Vector{Tuple}}()
    mol_ids = String[]
    for row in eachrow(data)
        s = string(row[smiles_col])
        (ismissing(s) || isempty(strip(s)) || s == "missing") && continue
        p = make_prompt(strip(s))
        for _ in 1:iterations
            push!(conversations, [("system", system_prompt), ("user", p)])
        end
        push!(mol_ids, string(row.mol_id))
    end

    println("[ZS] $(ds["name"]) / $(md["name"]) / $(variant): $(length(mol_ids)) molecules " *
            "x $(iterations) iters = $(length(conversations)) calls, reasoning=$(reasoning)" *
            (isempty(tier) ? "" : ", tier=$(tier)"))

    # ZS_DRYRUN prints the exact prompts and exits without touching the API. Every sweep
    # should be dry-run first: a wrong column or a mis-templated prompt is otherwise only
    # discovered after the whole matrix has been paid for.
    if dryrun
        println("\n--- SYSTEM ---\n$system_prompt\n--- USER (first) ---\n$(conversations[1][2][2])")
        println("--- USER (last) ---\n$(conversations[end][2][2])")
        println("\n[ZS] DRY RUN: no API calls made. Would write -> $out_file")
        return 0.0
    end

    usage_before = current_usage()
    t0 = now()

    # Sampling is pinned to the neutral setting, temperature = top_p = 1, for every model.
    # The shared library defaults to 0.7 / 0.95, but several endpoints in this panel silently
    # ignore or reject non-default sampling parameters, so those defaults would mean the
    # models are not actually being compared under the same conditions. 1/1 is the one setting
    # every provider honours identically.
    answers = ask_gpt_threaded(llm, conversations;
                               num_threads=threads, temperature=1.0, top_p=1.0,
                               reasoning_effort=reasoning, retries=10)

    keys_list = Vector{Vector{String}}()
    results_list = []
    for (j, mol) in enumerate(mol_ids)
        for i in 1:iterations
            idx = (j - 1) * iterations + i
            idx <= length(answers) || continue
            push!(keys_list, String[variant, reasoning, md["endpoint"], mol])
            push!(results_list, parse_number(answers[idx]))
        end
    end
    append_values_to_json(out_file, keys_list, results_list)

    spent = current_usage() - usage_before
    secs = Dates.value(now() - t0) / 1000
    n_ok = count(x -> x isa Number && !isnan(x), results_list)
    log_usage(outdir, dskey, tag, variant, length(conversations), spent, secs, n_ok)
    @printf("[ZS] saved -> %s   (%d/%d parsed, \$%.4f, %.0f s)\n",
            basename(out_file), n_ok, length(conversations), spent, secs)
    return spent
end

"Cumulative OpenRouter spend recorded by the shared library, in USD."
function current_usage()
    f = joinpath(LLMLIB, "LLMUtilsData", "LLMUsage.json")
    isfile(f) || return 0.0
    try
        return Float64(get(JSON.parsefile(f), "total_cost_usd", 0.0))
    catch
        return 0.0
    end
end

"Append one row per cell so estimate_cost.py can project the deep runs from real numbers."
function log_usage(outdir, dskey, tag, variant, ncalls, cost, secs, nparsed)
    f = joinpath(dirname(outdir), "usage_log.csv")
    new = !isfile(f)
    open(f, "a") do io
        new && println(io, "timestamp,dataset,model,variant,n_calls,n_parsed,cost_usd,seconds")
        @printf(io, "%s,%s,%s,%s,%d,%d,%.6f,%.1f\n",
                now(), dskey, tag, variant, ncalls, nparsed, cost, secs)
    end
end

# Only act as a CLI when executed directly; run_matrix.jl includes this file for its
# definitions and drives run_cell itself.
if abspath(PROGRAM_FILE) == @__FILE__
    run_cell(get(ENV, "ZS_DATASET", ""), get(ENV, "ZS_MODELTAG", ""))
end
