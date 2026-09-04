using CSV
using DataFrames
using JSON
using Dates
using Printf
using HTTP
# Shared LLM helper library. Defaults to src/lib inside this repository; point
# LLM_JULIA_LIB at another checkout to use one kept outside the repo.
const LLMLIB = get(ENV, "LLM_JULIA_LIB", joinpath(@__DIR__, "lib"))
include(joinpath(LLMLIB, "LLMs.jl"))
include(joinpath(LLMLIB, "FileWritingHelpers.jl"))

# Zero-shot memorization probe (adapted). Unblinded, SMILES only, minimal reasoning.
function main()
    script_dir = @__DIR__
    data_file = joinpath(script_dir, "../data", "Lipophilicity.csv")
    modeltag = get(ENV, "ZS_MODELTAG", "gem31pro")
    out_file = joinpath(script_dir, "../results", "LLM_ZeroShot_lipophilicity_$(modeltag).json")

    llm = getfield(Main, Symbol(get(ENV, "ZS_MODEL", "gemini_3_1_pro_flex")))

    system_prompt = """You are an expert chemist and know the Lipophilicity dataset very well. You are given a SMILES string of a molecule. Your task is to predict the lipophilicity of that molecule in logD octanol/water partition coefficient at pH 7.4. Provide only the numerical value as output, without any additional text."""
    get_prediction_prompt(smiles) = "What is the lipophilicity of the molecule with the following SMILES string: $smiles? Provide only the numerical value as output, without any additional text."

    data = DataFrame(CSV.File(data_file))
    n_cap = parse(Int, get(ENV, "ZS_N", string(nrow(data))))
    data = data[1:min(n_cap, nrow(data)), :]
    iterations = parse(Int, get(ENV, "ZS_ITERS", "2"))

    conversations = Vector{Vector{Tuple}}()
    for row in eachrow(data)
        prompt = get_prediction_prompt(row.smiles)
        for i in 1:iterations
            push!(conversations, [("system", system_prompt), ("user", prompt)])
        end
    end
    reasoning = get(ENV, "ZS_REASONING", "max_tokens:128")
    println("[ZS] Lipophilicity: $(nrow(data)) molecules x $(iterations) iters = $(length(conversations)) calls, reasoning=$(reasoning)")
    answers = ask_gpt_threaded(llm, conversations; num_threads=parse(Int, get(ENV,"ZS_THREADS","10")), reasoning_effort=reasoning, retries=10)

    keys_list = Vector{Vector{String}}()
    results_list = []
    for j in 1:(Int(round(length(answers) / iterations)))
        for i in 1:iterations
            result = search_for_last_number_in_string(answers[(j-1)*iterations+i])
            push!(keys_list, Vector{String}(["Smiles", "NoReasoning", llm.model, data[j, "smiles"]]))
            push!(results_list, result)
        end
    end
    append_values_to_json(out_file, keys_list, results_list)
    println("[ZS] saved -> $out_file")
end

main()
