#
# Sweep driver for the memorization screening.
#
# Loops over (dataset, model, variant) cells inside a single Julia session, so the ~20 s
# startup cost is paid once rather than several hundred times, and prints a live running
# total against a hard budget ceiling.
#
# The study is run in three stages, per the design:
#
#   1. pilot   -- every model x every dataset at a small molecule count. Cheap; its purpose
#                 is to produce real per-cell cost data and a first ranking of which cells
#                 show retrieval, so stages 2 and 3 can be aimed rather than guessed.
#   2. deep-ds -- the heaviest-retrieving model(s) against every dataset, at full depth.
#                 Answers "which benchmarks has this model absorbed?"
#   3. deep-mdl-- every model against the datasets that stage 1 flagged, at full depth.
#                 Answers "which models have absorbed this benchmark?"
#
# Usage
#   julia src/run_matrix.jl pilot
#   julia src/run_matrix.jl deep-ds  --models opus48,gem35flash
#   julia src/run_matrix.jl deep-mdl --datasets esol,freesolv,aqsoldb
#   julia src/run_matrix.jl random   --datasets esol --models opus48,gpt41
#
# Flags
#   --n N          molecules per cell            (stage defaults: pilot 60, deep 1000)
#   --iters K      repeats per molecule          (stage defaults: pilot 1,  deep 3)
#   --variant V    canonical | random | blind    (default canonical)
#   --budget USD   abort before starting a cell once spend exceeds this   (default 600)
#   --threads T    concurrent requests           (default 10)
#   --dry          print the plan and the first prompt of each cell, spend nothing
#   --force        re-query cells that already have complete result files
#
# ALWAYS run with --dry first. A mis-templated prompt is otherwise discovered only after
# the entire matrix has been paid for.
#
using JSON, Printf, Dates
include(joinpath(@__DIR__, "Run_ZeroShot.jl"))

function parse_args(argv)
    stage = isempty(argv) ? "pilot" : argv[1]
    o = Dict{String,Any}("stage" => stage, "variant" => "canonical", "budget" => 600.0,
                         "threads" => 10, "dry" => false, "force" => false,
                         "models" => nothing, "datasets" => nothing,
                         "n" => nothing, "iters" => nothing, "reasoning" => nothing)
    i = 2
    while i <= length(argv)
        a = argv[i]
        if a == "--dry";        o["dry"] = true; i += 1
        elseif a == "--force";  o["force"] = true; i += 1
        elseif a == "--n";       o["n"] = parse(Int, argv[i+1]); i += 2
        elseif a == "--iters";   o["iters"] = parse(Int, argv[i+1]); i += 2
        elseif a == "--threads"; o["threads"] = parse(Int, argv[i+1]); i += 2
        elseif a == "--budget";  o["budget"] = parse(Float64, argv[i+1]); i += 2
        elseif a == "--variant"; o["variant"] = argv[i+1]; i += 2
        elseif a == "--models";   o["models"] = split(argv[i+1], ","); i += 2
        elseif a == "--datasets"; o["datasets"] = split(argv[i+1], ","); i += 2
        # Overrides the per-model registry setting for every cell in the sweep. Results are
        # keyed by reasoning setting inside the JSON, so a re-sweep at a second setting adds
        # a branch rather than overwriting one -- but the resumability check counts every
        # prediction in the file and will otherwise skip the cell, so this needs --force.
        elseif a == "--reasoning"; o["reasoning"] = argv[i+1]; i += 2
        else error("unknown flag: $a")
        end
    end
    return o
end

function main()
    o = parse_args(ARGS)
    stage = o["stage"]
    stage in ("pilot", "deep-ds", "deep-mdl", "random", "custom") ||
        error("stage must be pilot | deep-ds | deep-mdl | random | custom")

    models_all = JSON.parsefile(joinpath(@__DIR__, "registry", "models.json"))["models"]
    ds_all     = JSON.parsefile(joinpath(@__DIR__, "registry", "datasets.json"))["datasets"]

    # Models can be parked in the registry with enabled=false (e.g. OLMo 3, catalogued on
    # OpenRouter but with zero serving endpoints, so every call 404s). An explicit --models
    # list overrides the flag, so a disabled model can still be tried deliberately.
    tags = o["models"] === nothing ?
           [m["tag"] for m in models_all if get(m, "enabled", true)] : String.(o["models"])
    skipped = [m["tag"] for m in models_all if !get(m, "enabled", true)]
    if o["models"] === nothing && !isempty(skipped)
        println("skipping disabled models: $(join(skipped, ", "))")
    end
    keys_ = o["datasets"] === nothing ? [d["key"] for d in ds_all]     : String.(o["datasets"])

    # Stage defaults. The pilot deliberately trades depth for breadth: 60 molecules cannot
    # resolve the weak "partial exposure" regime, and is not meant to -- it exists to price
    # the matrix and to rank cells, not to classify them.
    n     = o["n"]     !== nothing ? o["n"]     : (stage == "pilot" ? 60 : 1000)
    iters = o["iters"] !== nothing ? o["iters"] : (stage == "pilot" ? 1  : 3)
    variant = stage == "random" ? "random" : o["variant"]

    cells = [(d, t) for t in tags for d in keys_]
    total_calls = length(cells) * n * iters

    println("=" ^ 78)
    println("stage=$stage  variant=$variant  n=$n  iters=$iters" *
            (o["reasoning"] === nothing ? "" : "  reasoning=$(o["reasoning"]) (override)"))
    if o["reasoning"] !== nothing && !o["force"] && !o["dry"]
        error("--reasoning without --force: every target cell already has predictions, so " *
              "the resumability check would skip all of them and nothing would be queried.")
    end
    println("$(length(tags)) models x $(length(keys_)) datasets = $(length(cells)) cells, " *
            "up to $(total_calls) calls")
    println("budget ceiling \$$(o["budget"]);  starting spend \$$(round(current_usage(), digits=2))")
    println("=" ^ 78)

    start_usage = current_usage()
    logf = joinpath(ROOT, "results", "matrix_$(stage)_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log")
    mkpath(dirname(logf))
    open(logf, "a") do io
        println(io, "USAGE_START $start_usage  stage=$stage n=$n iters=$iters variant=$variant")
    end

    done = 0
    for (dskey, tag) in cells
        done += 1
        spent_total = current_usage() - start_usage
        if !o["dry"] && spent_total > o["budget"]
            msg = "BUDGET CEILING \$$(o["budget"]) reached after $done cells " *
                  "(spent \$$(round(spent_total, digits=2))) -- stopping."
            println(msg); open(logf, "a") do io; println(io, msg) end
            break
        end
        @printf("\n[%3d/%3d] %-14s %-16s  (spent \$%.2f)\n",
                done, length(cells), dskey, tag, spent_total)
        try
            run_cell(dskey, tag; variant=variant, n=n, iters=iters,
                     threads=o["threads"], force=o["force"], dryrun=o["dry"],
                     reasoning_override=o["reasoning"])
        catch e
            # One dead endpoint must not abort a 400-cell sweep.
            msg = "CELL FAILED $dskey/$tag/$variant: $(sprint(showerror, e))"
            println(msg); open(logf, "a") do io; println(io, msg) end
        end
    end

    final = current_usage()
    println("\n" * "=" ^ 78)
    @printf("DONE. stage=%s  spent \$%.2f  (total account \$%.2f)\n",
            stage, final - start_usage, final)
    open(logf, "a") do io
        println(io, "USAGE_END $final  spent $(final - start_usage)")
    end
end

main()
