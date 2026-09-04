#
# Model handles for the runners in ../ -- key-free by construction.
#
# The original of this file lives outside any repository and holds real API keys. Nothing of
# the sort is committed here: the key is read from the OPENROUTER_API_KEY environment variable
# at load time, and every model handle is built from src/registry/models.json, which carries the
# OpenRouter endpoint id, the provider routing order and the service tier for each model.
#
#   export OPENROUTER_API_KEY=sk-or-...        # bash
#   $env:OPENROUTER_API_KEY = "sk-or-..."      # PowerShell
#
# Nothing here is needed to regenerate the paper's figures or tables -- those read only the
# pre-computed files in results/. This file exists so the query runners can be re-run.
#
include(joinpath(@__DIR__, "LLMUtils.jl"))

using JSON

const key_openrouter = get(ENV, "OPENROUTER_API_KEY", "YOUR_OPENROUTER_API_KEY")

# Kept for source compatibility with the private original, which distinguished two accounts.
const key_openrouter_moi = key_openrouter

# --- Azure OpenAI (only the three legacy Part-I runners can use these) ---------------------
const key = get(ENV, "AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY")
const endpoint = get(ENV, "AZURE_OPENAI_ENDPOINT", "YOUR_AZURE_ENDPOINT")
const version = "2024-10-01-preview"
const version_new = "2025-01-01-preview"
const version_new2 = "2024-12-01-preview"
const version_old = "2024-05-01-preview"

# --- OpenRouter handles, one per registry tag ---------------------------------------------
# `gem31pro`, `opus48`, ... exactly as they appear in src/registry/models.json. The unified
# runner (Run_ZeroShot.jl) does not use these -- it builds its own access object from the
# registry -- but the three legacy Part-I runners resolve ZS_MODEL through getfield(Main, ...),
# so the handles have to exist as globals.
let reg = JSON.parsefile(joinpath(@__DIR__, "..", "registry", "models.json"))["models"]
    for m in reg
        access = LLMAccessOpenRouter(key_openrouter, m["endpoint"],
                                     Vector{String}(get(m, "providers", String[])),
                                     String(get(m, "service_tier", "")))
        @eval Main const $(Symbol(m["tag"])) = $access
    end
end

# Aliases used as defaults by the legacy runners (Run_{Delaney,Lipophilicity,QM7}_ZeroShot.jl).
const gemini_3_1_pro_flex = Main.gem31pro
