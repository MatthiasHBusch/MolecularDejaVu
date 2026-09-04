using HTTP
using JSON
using DataFrames
using Dates
using Base.Threads: @threads, nthreads, threadid

"""
An abstract base type representing a generic interface for accessing Language Model (LLM) services.

This abstract type serves as a blueprint for creating concrete implementations of LLM access 
mechanisms. It ensures that any specific LLM access struct provides a consistent interface 
for interacting with different language model providers.

# Implementation Guidelines
- Concrete subtypes should implement the necessary fields and methods required to 
  authenticate and interact with a specific LLM service.
- Users should create instances via specific structs or by deriving their own struct 
  from this abstract type.

# Example
```julia
# Creating a custom LLM access struct
struct MyCustomLLMAccess <: LLMAccess
    # Define necessary fields for authentication and configuration
    api_key::String
    endpoint::String
    # ... other required fields
end
```
"""
abstract type LLMAccess end

"""
A struct for accessing OpenRouter models through OpenRouter's OpenAI API (openai.azure.com).

# Fields
- `key::String`: The API key required for authentication with the Azure OpenAI service.
- `model::String`: The specific model to use with the provider upfront (e.g., "google/gemini-2.5-flash-lite").
- `providers_order::Vector{String}`: The order in which to try providers.
- `service_tier::String`: The service tier to use. Valid options are "" (standard), "flex" (reduces most costs by 50%), "priority" (increase costs by variable amount factor ~2)

# Usage
```julia
azure_access = LLMAccessOpenRouter(
    "your-api-key", 
    "google/gemini-2.5-flash-lite"
)
```
"""
struct LLMAccessOpenRouter <: LLMAccess
    key::String
    model::String
    providers_order::Vector{String}
    service_tier::String
end

function LLMAccessOpenRouter(key::String, model::String; providers_order::Vector{String}=[])
    return LLMAccessOpenRouter(key, model, providers_order, "")
end

# Positional 3-arg form (back-compat with pre-service_tier LLMs.jl entries).
function LLMAccessOpenRouter(key::String, model::String, providers_order::Vector{String})
    return LLMAccessOpenRouter(key, model, providers_order, "")
end

"""
A struct for accessing OpenAI models through Azure's OpenAI API (openai.azure.com).

# Fields
- `key::String`: The API key required for authentication with the Azure OpenAI service.
- `model::String`: The specific model to use (e.g., "gpt-3.5-turbo", "gpt-4").
- `version::String`: The API version to use (e.g., "2024-02-15").
- `endpoint::String`: The Azure resource name used in the base URL.

# URL Construction Example
If the full URL looks like "https://xxxxx.openai.azure.com/openai/deployments/",
then `endpoint` would be the "xxxxx" part of the URL.

# Usage
```julia
azure_access = LLMAccessAzureOpenAI(
    "your-api-key", 
    "gpt-3.5-turbo", 
    "2024-02-15", 
    "your-resource-name"
)
```
"""
struct LLMAccessAzureOpenAI <: LLMAccess
    key::String
    model::String
    version::String
    endpoint::String
    deployment::String
end

function LLMAccessAzureOpenAI(key::String, model::String, version::String, endpoint::String)
    # Set default deployment name if not provided
    return LLMAccessAzureOpenAI(key, model, version, endpoint, "")
end

"""
A struct for accessing Azure AI Language Models through the Azure Services API (services.ai.azure.com).

# Fields
- `key::String`: The API key required for authentication with the Azure AI service.
- `model::String`: The specific model to use (e.g., "gpt-3.5-turbo", "custom-model").
- `version::String`: The API version to use (e.g., "2024-02-15").
- `endpoint::String`: The personal endpoint used in the base URL.

# URL Construction Example
If the full URL looks like "https://xxxxx.services.ai.azure.com/models",
then `endpoint` would be the "xxxxx" part of the URL.

# Usage
```julia
azure_services_access = LLMAccessAzureServices(
    "your-api-key", 
    "your-model", 
    "2024-02-15", 
    "your-endpoint-name"
)
```
"""
struct LLMAccessAzureServices <: LLMAccess
    key::String
    model::String
    version::String
    endpoint::String
end

"""
Dynamically selects the appropriate Azure LLM access struct based on the model name.

This experimental function provides a convenient way to create the correct LLM access 
struct by inferring the type based on the model name.

# Arguments
- `key::String`: The API key for authentication.
- `model::String`: The model to be used (e.g., "gpt-3.5-turbo", "custom-model").
- `version::String`: The API version to use.
- `endpoint::String`: The endpoint for the Azure service.

# Returns
- `LLMAccessAzureOpenAI`: If the model name contains "gpt" or "o1".
- `LLMAccessAzureServices`: For other model types.

# Usage
```julia
# Automatically selects the appropriate struct
access = LLMAccessAzure(
    "your-api-key", 
    "gpt-3.5-turbo", 
    "2024-02-15", 
    "your-endpoint-name"
)
```

# Notes
- The function uses a simple string matching approach to determine the struct type.
- Future versions may implement more sophisticated type selection logic.
"""
function LLMAccessAzure(key::String, model::String, version::String, endpoint::String)
    if occursin("gpt", model) || occursin("o1", model)
        return LLMAccessAzureOpenAI(key, model, version, endpoint)
    end
    return LLMAccessAzureServices(key, model, version, endpoint)
end

"""
A struct that bundles the necessary information to access an LLM via the Azure Api (inference.ai.azure.com)
key: The API key for the OpenAI API
model: The model to use for the API (eg. "Llama")
version: The version of the API to use (eg. "2024-02-15")
endpoint: "your-host-name.your-azure-region" for the base url (eg. "xxxxx.eastus2" where xxxxx is your unique model deployment host name)
"""
struct LLMAccessAzureInference <: LLMAccess
    key::String
    #model::String
    #version::String
    endpoint::String
end

"""
A struct that bundles the necessary information to access an LLM via the DeepSeek API (https://api.deepseek.com). 
key: The API key for the DeepSeek API
model: The model to use for the API (currently "deepseek-chat" or "deepseek-reasoner")
"""
struct LLMAccessDeepSeek <: LLMAccess
    key::String
    model::String
    #version::String
    #endpoint::String
end



"""
A struct that bundles the necessary information to access an LLM via the OpenAI API
key: The API key for the OpenAI API
model: The model to use for the API (eg. "gpt-3.5-turbo")
version: The version of the API to use (eg. "2024-02-15")
base_url: The base URL for the API (eg. "https://xxxxxxxxxx/openai/deployments" or just everything infront of the model name)
"""
struct LLMAccessOpenAI <: LLMAccess
    key::String
    model::String
    version::String
    base_url::String
end

function LLMAccessOpenAI(key::String, model::String)
    return LLMAccessOpenAI(key, model, "", "https://api.openai.com/v1")
end


"""
key: The API key for the Llama API
model: The model to use for the API (eg. "gpt-3.5-turbo")
version: The version of the API to use (eg. "2024-02-15")
base_url: The base URL for the API ("https://api.llama-api.com")
"""
struct LLMAccessLlama <: LLMAccess
    key::String
    model::String
    base_url::String
end

function LLMAccessLlama(key::String, model::String)
    return LLMAccessLlama(key, model, "https://api.llama-api.com")
end

"""
A mutable struct to manage a chat session with an LLM.
conversation: A vector of tuples representing the conversation history.
max_tokens: The maximum number of tokens for the response.
temperature: The sampling temperature for the response.
top_p: The nucleus sampling parameter.
seed: An optional seed for reproducibility.
"""
mutable struct LLMChat
    llmaccess::LLMAccess
    conversation::Vector{Tuple}
    max_tokens::Int
    seed::Union{Int,Missing}
    temperature::Float64
    top_p::Float64
    retries::Int
    reasoning_effort::Union{String,Missing}
    verbosity::Union{String,Missing}
end

function LLMChat(
    llmaccess::LLMAccess;
    conversation::Vector{Tuple}=Vector{Tuple}(),
    max_tokens::Int=1000,
    seed::Union{Int,Missing}=missing,
    temperature::Float64=0.7,
    top_p::Float64=0.95,
    retries::Int=0,
    reasoning_effort::Union{String,Missing}=missing,
    verbosity::Union{String,Missing}=missing
)
    LLMChat(llmaccess, conversation, max_tokens, seed, temperature, top_p, retries, reasoning_effort, verbosity)
end

function Base.show(io::IO, llmaccess::LLMAccess)
    deployment_string = ""
    if llmaccess isa LLMAccessAzureOpenAI
        if llmaccess.deployment != ""
            deployment_string = "_" * llmaccess.deployment
        end
    end
    # switch between accesses that have a version vs those that don't
    if hasfield(typeof(llmaccess), :version)
        print(io, string(llmaccess.model) * "_" * string(llmaccess.version) * deployment_string)
    else
        print(io, string(llmaccess.model) * deployment_string)
    end
end

"""
returns the URL for the API request for the given llmaccess and request_type (eg. "/chat/completions")
"""
function get_url(llmaccess::LLMAccessAzureOpenAI, request_type::String)
    return "https://" * llmaccess.endpoint * ".openai.azure.com/openai/deployments/" * llmaccess.model * request_type * "?api-version=" * llmaccess.version
end

function get_url(llmaccess::LLMAccessAzureServices, request_type::String)
    return "https://" * llmaccess.endpoint * ".services.ai.azure.com/models" * request_type * "?api-version=" * llmaccess.version
end

function get_url(llmaccess::LLMAccessAzureInference, request_type::String)
    return "https://" * llmaccess.endpoint * ".inference.ai.azure.com" * request_type
end

function get_url(llmaccess::LLMAccessDeepSeek, request_type::String)
    return "https://api.deepseek.com" * request_type
end

function get_url(llmaccess::LLMAccessOpenRouter, request_type::String)
    return "https://openrouter.ai/api/v1" * request_type
end

function get_url(llmaccess::LLMAccessOpenAI, request_type::String)
    return llmaccess.base_url * request_type
end

function get_url(llmaccess::LLMAccessLlama, request_type::String)
    return llmaccess.base_url * request_type
end

"""
returns the URL for batch API request for the given llmaccess
"""
function get_batch_url(llmaccess::LLMAccessAzureServices)
    return "https://" * llmaccess.endpoint * ".openai.azure.com/openai/files?api-version=" * llmaccess.version
end

function get_batch_url(llmaccess::LLMAccessAzureOpenAI)
    return "https://" * llmaccess.endpoint * ".openai.azure.com/openai/files?api-version=" * llmaccess.version
end

"""
returns the header for the API request for the given llmaccess
"""
function get_header(llmaccess::Union{LLMAccessAzureOpenAI,LLMAccessAzureServices})
    return Dict("Content-Type" => "application/json", "api-key" => llmaccess.key)
end

function get_header(llmaccess::LLMAccessAzureInference)
    return Dict("Content-Type" => "application/json", "Authorization" => llmaccess.key)
end

function get_header(llmaccess::LLMAccessDeepSeek)
    return Dict("Content-Type" => "application/json", "Accept" => "application/json", "Authorization" => "Bearer " * llmaccess.key)
end

function get_header(llmaccess::LLMAccessOpenRouter)
    return Dict("Content-Type" => "application/json", "Authorization" => "Bearer " * llmaccess.key)
end

function get_header(llmaccess::LLMAccessOpenAI)
    return Dict("Content-Type" => "application/json", "Authorization" => "Bearer " * llmaccess.key)
end

function get_header(llmaccess::LLMAccessLlama)
    return Dict("Content-Type" => "application/json", "Authorization" => "Bearer " * llmaccess.key)
end

function get_batch_header(llmaccess::LLMAccessAzureOpenAI)
    return Dict("api-key" => llmaccess.key) #"Content-Type" => "multipart/form-data", automatically set by HTTP.jl
end


"""
returns the body for the API request for the given conversation. 
"""
function get_body(llmaccess::LLMAccessAzureOpenAI, conversation::Vector{Tuple}; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing, reasoning_effort=missing, verbosity=missing)
    messages = []
    for (role, content) in conversation
        if occursin("o3", llmaccess.model) && role == "system"
            push!(messages, Dict("role" => "developer", "content" => [Dict("type" => "text", "text" => content)]))
        else
            push!(messages, Dict("role" => role, "content" => [Dict("type" => "text", "text" => content)]))
        end
    end
    ret_dict = Dict("messages" => messages, "temperature" => temperature, "top_p" => top_p)
    if llmaccess.deployment != ""
        ret_dict["model"] = llmaccess.deployment
    end
    # if it is a reasoning model, use max_completion_tokens instead of max_tokens and set temperature to 1.0 and top_p to 1.0 (might later be changed)
    if occursin("o1", llmaccess.model)
        ret_dict["max_completion_tokens"] = max_tokens
        ret_dict["temperature"] = 1.0
        ret_dict["top_p"] = 1.0
    elseif occursin("o3", llmaccess.model)
        ret_dict["max_completion_tokens"] = max_tokens
        if !ismissing(reasoning_effort)
            ret_dict["reasoning_effort"] = reasoning_effort
        else
            ret_dict["reasoning_effort"] = "high"
        end
        ret_dict["temperature"] = 1.0
        ret_dict["top_p"] = 1.0
    elseif occursin("gpt-5", llmaccess.model)
        ret_dict["max_completion_tokens"] = max_tokens
        if !ismissing(reasoning_effort)
            ret_dict["reasoning_effort"] = reasoning_effort
        else
            ret_dict["reasoning_effort"] = "high"
        end
        if !ismissing(verbosity)
            ret_dict["verbosity"] = verbosity
        else
            ret_dict["verbosity"] = "medium"
        end
        ret_dict["temperature"] = 1.0
        ret_dict["top_p"] = 1.0
    else
        ret_dict["max_tokens"] = max_tokens
        if !ismissing(seed)
            ret_dict["seed"] = seed
        end
    end
    return JSON.json(ret_dict)
end

function get_body(llmaccess::LLMAccessAzureServices, conversation::Vector{Tuple}; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing, reasoning_effort=missing)
    messages = []
    for (role, content) in conversation
        push!(messages, Dict("role" => role, "content" => [Dict("type" => "text", "text" => content)]))
    end
    ret_dict = Dict("messages" => messages, "temperature" => temperature, "top_p" => top_p, "max_tokens" => max_tokens)
    if !ismissing(seed)
        ret_dict["seed"] = seed
    end
    ret_dict["model"] = llmaccess.model
    return JSON.json(ret_dict)
end

function get_body(llmaccess::LLMAccessDeepSeek, conversation::Vector{Tuple}; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing)
    messages = []
    for (role, content) in conversation
        push!(messages, Dict("role" => role, "content" => [Dict("type" => "text", "text" => content)]))
    end
    ret_dict = Dict("messages" => messages, "temperature" => temperature, "top_p" => top_p, "max_tokens" => max_tokens)
    if !ismissing(seed)
        ret_dict["seed"] = seed
    end
    ret_dict["model"] = llmaccess.model
    return JSON.json(ret_dict)
end

function get_body(llmaccess::LLMAccessAzureInference, conversation::Vector{Tuple}; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing)
    # not finished or tested yet
    messages = []
    for (role, content) in conversation
        push!(messages, Dict("role" => role, "content" => [Dict("type" => "text", "text" => content)]))
    end
    return JSON.json(Dict("messages" => messages, "temperature" => temperature, "top_p" => top_p, "max_tokens" => max_tokens))
end

function get_body(llmaccess::LLMAccessOpenRouter, conversation::Vector{Tuple}; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing, reasoning_effort=missing, verbosity=missing)
    messages = []
    for msg in conversation
        role = msg[1]
        content = msg[2]
        if length(msg) > 2
            cache_msg = msg[3]
            push!(messages, Dict("role" => role, "content" => [Dict("type" => "text", "text" => content, "cache_control" => Dict("type" => cache_msg))]))
        else
            push!(messages, Dict("role" => role, "content" => [Dict("type" => "text", "text" => content)]))
        end
    end
    bodyJSON = Dict("model" => llmaccess.model, "messages" => messages, "max_tokens" => max_tokens)
    if !ismissing(llmaccess.service_tier) && llmaccess.service_tier != ""
        bodyJSON["service_tier"] = llmaccess.service_tier
    end
    if temperature != 1.0
        bodyJSON["temperature"] = temperature
    end
    if top_p != 1.0
        bodyJSON["top_p"] = top_p
    end
    # always include usage in answer for internal cost tracking
    bodyJSON["usage"] = Dict("include" => true)
    if length(llmaccess.providers_order) > 0
        bodyJSON["provider"] = Dict("order" => llmaccess.providers_order)
    end
    if !ismissing(reasoning_effort)
        # Two accepted forms:
        #   "low"/"medium"/"high"/"none"  -> reasoning {effort: ...}
        #   "max_tokens:N"                -> reasoning {max_tokens: N}: a FIXED thinking
        #      budget. This is the unified, cross-provider way to put every model on the
        #      same reasoning footing (effort labels are interpreted very differently per
        #      provider). For models that only accept an effort level (e.g. OpenAI),
        #      OpenRouter converts the token budget to the nearest effort bucket.
        if startswith(reasoning_effort, "max_tokens:")
            budget = parse(Int, strip(split(reasoning_effort, ":")[2]))
            bodyJSON["reasoning"] = Dict{String,Any}("max_tokens" => budget)
        else
            bodyJSON["reasoning"] = Dict{String,Any}("effort" => reasoning_effort)
            if reasoning_effort == "none" && llmaccess.model == "google/gemini-2.5-pro"
                delete!(bodyJSON["reasoning"], "effort")
                bodyJSON["reasoning"]["max_tokens"] = 128
            end
        end
    end
    if !ismissing(verbosity)
        bodyJSON["verbosity"] = verbosity
    end
    return JSON.json(bodyJSON)
end

function get_body(llmaccess::LLMAccessOpenAI, conversation::Vector{Tuple}; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing, reasoning_effort=missing)
    messages = []
    for (role, content) in conversation
        push!(messages, Dict("role" => role, "content" => [Dict("type" => "text", "text" => content)]))
    end
    model_ = llmaccess.model
    if llmaccess.version != ""
        model_ = llmaccess.model * "-" * llmaccess.version
    end
    return JSON.json(Dict("model" => model_, "messages" => messages, "temperature" => temperature, "top_p" => top_p, "max_tokens" => max_tokens))
end

function get_body(llmaccess::LLMAccessLlama, conversation::Vector{Tuple}; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing)
    messages = []
    for (role, content) in conversation
        push!(messages, Dict("role" => role, "content" => content))
    end
    return JSON.json(Dict("messages" => messages, "stream" => false, "function_call" => "none", "temperature" => temperature, "top_p" => top_p, "max_tokens" => max_tokens))
end


"""
returns the body for the API request for the given conversation. 
"""
function get_batch_body(llmaccess::LLMAccessAzureOpenAI, conversation::Vector{Tuple}; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing, reasoning_effort=missing, verbosity=missing)
    messages = []
    for (role, content) in conversation
        if occursin("o3", llmaccess.model) && role == "system"
            push!(messages, Dict("role" => "developer", "content" => [Dict("type" => "text", "text" => content)]))
        else
            push!(messages, Dict("role" => role, "content" => [Dict("type" => "text", "text" => content)]))
        end
    end
    ret_dict = Dict("messages" => messages, "temperature" => temperature, "top_p" => top_p)
    ret_dict["model"] = llmaccess.deployment
    # if it is a reasoning model, use max_completion_tokens instead of max_tokens and set temperature to 1.0 and top_p to 1.0 (might later be changed)
    if occursin("o1", llmaccess.model)
        ret_dict["max_completion_tokens"] = max_tokens
        ret_dict["temperature"] = 1.0
        ret_dict["top_p"] = 1.0
    elseif occursin("o3", llmaccess.model)
        ret_dict["max_completion_tokens"] = max_tokens
        if !ismissing(reasoning_effort)
            ret_dict["reasoning_effort"] = reasoning_effort
        else
            ret_dict["reasoning_effort"] = "high"
        end
        ret_dict["temperature"] = 1.0
        ret_dict["top_p"] = 1.0
    elseif occursin("gpt-5", llmaccess.model)
        ret_dict["max_completion_tokens"] = max_tokens
        if !ismissing(reasoning_effort)
            ret_dict["reasoning_effort"] = reasoning_effort
        else
            ret_dict["reasoning_effort"] = "high"
        end
        if !ismissing(verbosity)
            ret_dict["verbosity"] = verbosity
        else
            ret_dict["verbosity"] = "medium"
        end
        ret_dict["temperature"] = 1.0
        ret_dict["top_p"] = 1.0
    else
        ret_dict["max_tokens"] = max_tokens
        if !ismissing(seed)
            ret_dict["seed"] = seed
        end
    end
    #return JSON.json(ret_dict)
    return ret_dict
end

"""
    ask_gpt_batch(llmaccess::Union{LLMAccessAzureOpenAI,LLMAccessAzureServices}, batch_conversations::Vector{Vector{Tuple}}; kwargs...)

Process multiple conversations in a single batch request to an Azure LLM service.

# Arguments
- `llmaccess::Union{LLMAccessAzureOpenAI,LLMAccessAzureServices}`: The Azure LLM access configuration
- `batch_conversations::Vector{Vector{Tuple}}`: Array of conversations to process
- `temperature::Real=0.7`: Controls randomness of the output (0.0-1.0)
- `top_p::Real=0.95`: Nucleus sampling probability threshold
- `max_tokens::Int=800`: Maximum number of tokens in the response
- `seed::Union{Int,Missing}=missing`: Random seed for reproducibility
- `retries::Int=0`: Number of retry attempts for failed requests
- `reasoning_effort::Union{String,Missing}=missing`: Reasoning effort for the model (applicable to certain models)
- `ttl_days::Int=14`: Time-to-live for the batch job in days (14-30)
- `use_last_file_with_id::Union{String,Missing}=missing`: Read the results from an existing batch job with this ID instead of running a new one

# Returns
- `Vector{String}`: Array of responses corresponding to each conversation in the batch

# Example
```julia
# Create multiple conversations
conversations = [
    [("user", "What is the capital of France?")],
    [("user", "What is the capital of Germany?")],
    [("user", "What is the capital of Italy?")]
]

# Process them in a single batch request
responses = ask_gpt_batch(my_azure_llm, conversations)
```

# Notes
- Only available for Azure OpenAI and Azure Services
- More efficient for processing multiple requests
- Handles API rate limiting and automatic retries
- If supplied with multiple LLMChats, it will take the LLM parameters of the first chat for all of them (since they must be constant for the batch). Just the conversations are processed
"""
function ask_gpt_batch(llmaccess::Union{LLMAccessAzureOpenAI,LLMAccessAzureServices}, batch_conversations::Vector{Vector{Tuple}}; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing, reasoning_effort=missing, verbosity=missing, ttl_days=14, use_last_file_with_id=missing, retries=0)
    if ismissing(use_last_file_with_id)

        # check cost of the request
        input_tokens = 0
        for conversation in batch_conversations
            input_tokens += estimate_tokens(conversation)
        end
        quota_ok = check_llm_cost_limit(llmaccess.model, input_tokens, div(max_tokens, 2) * length(batch_conversations))
        if !quota_ok
            error("Quota might exceed limit for the LLM access. Please check your usage limits.")
        end

        # check if ttl_days is between 14 and 30
        if ttl_days < 14 || ttl_days > 30
            error("ttl_days must be between 14 and 30 days")
        end
        url = get_batch_url(llmaccess)
        jsonl_path = mktempdir()
        # Create a JSONL file in the temp dir
        identifier = replace(string(now()), "." => "_", ":" => "_")
        jsonl_path = joinpath(jsonl_path, "batch_input_$(identifier).jsonl")
        # Create the JSONL file with the batch conversations
        open(jsonl_path, "w") do io
            for (i, conv) in enumerate(batch_conversations)
                body = get_batch_body(llmaccess, conv; temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed, reasoning_effort=reasoning_effort, verbosity=verbosity)
                jsonl_line_dict = Dict("custom_id" => string(i), "method" => "POST", "url" => "/chat/completions", "body" => body)
                write(io, JSON.json(jsonl_line_dict) * "\n")
            end
        end
        header = get_batch_header(llmaccess)

        # Make the batch request
        retries_left = retries
        println("Uploading input file...")

        file_id = nothing
        uploaded = false
        while retries_left >= 0 && !uploaded
            try
                file_io = open(jsonl_path, "r")  # ← this is what file_bytes is
                filename = split(jsonl_path, "\\") |> last  # get just the file name
                form = HTTP.Form(Dict(
                    "purpose" => "batch",
                    "file" => HTTP.Multipart(filename, file_io, "application/json"),
                    "expires_after.seconds" => ttl_days * 86400,
                    "expires_after.anchor" => "created_at"
                ))
                file_info = HTTP.request("POST", url, headers=header, body=form)
                file_id = JSON.parse(String(file_info.body))["id"]
                println("Waiting for file to be uploaded...")

                while true
                    sleep(5)
                    status = JSON.parse(String(HTTP.get(
                        "https://$(llmaccess.endpoint).openai.azure.com/openai/files/$(file_id)?api-version=$(llmaccess.version)",
                        headers=Dict("api-key" => llmaccess.key)
                    ).body))["status"]
                    if status == "processed"
                        uploaded = true
                        break
                    end
                end
            catch e
                println("Failed to upload file: $(e)")
                if retries_left > 0
                    retries_left -= 1
                    sleep(60)
                    println("Retrying... ($retries_left retries left)")
                else
                    rethrow(e)
                end
            end
        end
        println("File is uploaded successfully with ID: $file_id")

        # Create batch job
        bid = nothing
        while retries_left >= 0
            try
                response = HTTP.request("POST",
                    "https://$(llmaccess.endpoint).openai.azure.com/openai/batches?api-version=$(llmaccess.version)",
                    headers=Dict("Content-Type" => "application/json", "api-key" => llmaccess.key),
                    body=JSON.json(Dict(
                        "input_file_id" => file_id,
                        "endpoint" => "/chat/completions",
                        "completion_window" => string(ttl_days, "h")
                    ))
                )
                bid = JSON.parse(String(response.body))["id"]
                println("Batch job successfully created. Waiting for result...")
                break
            catch e
                println("Failed to submit batch job: $(e)")
                if retries_left > 0
                    retries_left -= 1
                    sleep(60)
                    println("Retrying... ($retries_left retries left)")
                else
                    rethrow(e)
                end
            end
        end
    else
        bid = use_last_file_with_id
        println("Using existing batch job with ID: $bid")
    end
    # Wait for the batch job to complete
    try
        while true
            sleep(10)
            jb = JSON.parse(String(HTTP.get(
                "https://$(llmaccess.endpoint).openai.azure.com/openai/batches/$(bid)?api-version=$(llmaccess.version)",
                headers=Dict("api-key" => llmaccess.key)
            ).body))
            if jb["status"] in ["completed", "failed", "cancelled", "expired"]
                if jb["output_file_id"] !== nothing
                    ofid = jb["output_file_id"]
                    data = String(HTTP.get(
                        "https://$(llmaccess.endpoint).openai.azure.com/openai/files/$(ofid)/content?api-version=$(llmaccess.version)",
                        headers=Dict("api-key" => llmaccess.key)
                    ).body)
                    # for debugging: save the output to a file
                    open("batch_output.jsonl", "w") do io
                        write(io, data)
                    end
                    data = split(data, '\n')
                    # update usage information
                    dicts = [JSON.parse(line; dicttype=Dict) for line in data if !isempty(line)]
                    response_jsons = [dict["response"]["body"] for dict in dicts if haskey(dict, "response") && haskey(dict["response"], "body")]
                    # convert response bodies to dicts
                    #response_dicts = [JSON.parse(String(JSON.json(dict["response"]["body"]))) for dict in dicts]
                    update_llm_usage_from_response(llmaccess.model, response_jsons, true)
                    # order the responses by the custom_ids
                    custom_ids = [parse(Int, dict["custom_id"]) for dict in dicts]
                    sorted_indices = sortperm(custom_ids)
                    sorted_dicts = [dicts[i] for i in sorted_indices]
                    complete_dicts = []
                    sorted_dicts_cntr = 1
                    for i in 1:length(batch_conversations)
                        if i in custom_ids
                            push!(complete_dicts, sorted_dicts[sorted_dicts_cntr])
                            sorted_dicts_cntr += 1
                        else
                            @warn "No response for custom_id $i"
                            push!(complete_dicts, Dict("response" => Dict("body" => Dict("choices" => [Dict("message" => Dict("content" => "ERROR: NO RESPONSE"))]))))
                        end
                    end
                    return [dict["response"]["body"]["choices"][1]["message"]["content"] for dict in complete_dicts]
                else
                    error("Batch failed or cancelled")
                end
            end
        end
    catch e
        println("Failed fetch results file: $(e)")
        rethrow(e)
    end
end

function ask_gpt_batch(llmaccess::Union{LLMAccessAzureOpenAI,LLMAccessAzureServices}, questions::Vector{String}; kwargs...)
    # Convert each question to a simple conversation format
    batch_conversations = [
        [("user", question)] for question in questions
    ]

    return ask_gpt_batch(llmaccess, batch_conversations; kwargs...)
end

"""
    delete_old_batch_files(llmaccess::Union{LLMAccessAzureOpenAI,LLMAccessAzureServices}, num_to_delete::Int)

Deletes the oldest batch files from the Azure OpenAI storage to free up quota.

# Arguments
- `llmaccess::Union{LLMAccessAzureOpenAI,LLMAccessAzureServices}`: The Azure LLM access configuration
- `num_to_delete::Int`: The maximum number of files to delete

# Returns
- `Tuple{Int, Int}`: A tuple containing (number of files remaining, number of files deleted)
"""
function delete_old_batch_files(llmaccess::Union{LLMAccessAzureOpenAI,LLMAccessAzureServices}, num_to_delete::Int)
    # CLEANUP: Ensure endpoint is just the resource name, not a full URL
    clean_endpoint = replace(llmaccess.endpoint, r"^https?://" => "")

    # OPTIMIZATION: Added '&purpose=batch' to filter at the source
    # example: https://$(clean_endpoint).openai.azure.com/openai/files/$(file_id)?api-version=$(llmaccess.version)
    base_url = "https://$(clean_endpoint).openai.azure.com/openai/files"
    list_url = "$(base_url)?api-version=$(llmaccess.version)&purpose=batch"

    headers = Dict("api-key" => llmaccess.key)

    try
        println("Fetching batch files from Azure...")
        response = HTTP.get(list_url, headers=headers)
        response_data = JSON.parse(String(response.body))

        if !haskey(response_data, "data")
            println("Error: No 'data' field in response.")
            return (0, 0)
        end

        # The list is already filtered to 'batch' by Azure
        batch_files = response_data["data"]

        # Sort by created_at (oldest first)
        sort!(batch_files, by=x -> get(x, "created_at", 0))

        actual_delete_count = min(length(batch_files), num_to_delete)

        if actual_delete_count == 0
            println("No batch files found to delete.")
            return (length(batch_files), 0)
        end

        println("Deleting $actual_delete_count oldest batch files of $(length(batch_files))...")
        deleted_count = 0

        for file in first(batch_files, actual_delete_count)
            file_id = get(file, "id", "")
            delete_url = "$(base_url)/$(file_id)?api-version=$(llmaccess.version)"

            try
                HTTP.delete(delete_url, headers=headers)
                deleted_count += 1
                println("✓ Deleted: $file_id")
            catch e
                println("✗ Failed: $file_id ($e)")
            end
            sleep(0.2) # Rate limit safety
        end

        return (length(batch_files) - deleted_count, deleted_count)

    catch e
        println("Critical API Error: $e")
        rethrow(e)
    end
end

function ask_gpt_batch(llmchat::LLMChat, questions::Vector{String})
    # Convert each question to a simple conversation format by using the LLMChat's conversation history
    conversations = [
        vcat(llmchat.conversation, [("user", question)]) for question in questions
    ]
    return ask_gpt_batch(
        llmchat.llmaccess,
        conversations;
        temperature=llmchat.temperature,
        top_p=llmchat.top_p,
        max_tokens=llmchat.max_tokens,
        seed=llmchat.seed,
        retries=llmchat.retries,
        reasoning_effort=llmchat.reasoning_effort,
        verbosity=llmchat.verbosity
    )
end

function ask_gpt_batch(llmchats::Vector{LLMChat}; kwargs...)
    # Convert each question to a simple conversation format by using the LLMChat's conversation history
    conversations = [llmchat.conversation for llmchat in llmchats]
    return ask_gpt_batch(
        llmchats[1].llmaccess,
        conversations;
        temperature=llmchats[1].temperature,
        top_p=llmchats[1].top_p,
        max_tokens=llmchats[1].max_tokens,
        seed=llmchats[1].seed,
        retries=llmchats[1].retries,
        reasoning_effort=llmchats[1].reasoning_effort,
        verbosity=llmchats[1].verbosity,
        kwargs...
    )
end

""" 
    ask_gpt(llmaccess::LLMAccess, question::String; kwargs...)
    ask_gpt(llmaccess::LLMAccess, question::String, conversation::Vector{Tuple}; kwargs...)
    ask_gpt(llmaccess::LLMAccess, conversation::Vector{Tuple}; kwargs...)
    ask_gpt(llmaccess::LLMAccess, conversation::Union{String,Vector{Tuple}}, callback_function::Function; kwargs...)
    ask_gpt(llmchat::LLMChat, question::String)
    ask_gpt(llmchat::LLMChat)
    ask_gpt!(llmchat::LLMChat, question::String)
    ask_gpt!(llmchat::LLMChat)

Interact with a Language Model (LLM) by sending a question or conversation, with multiple dispatch options for flexibility.

# Variants

## Simple Question Submission
Send a single question to the LLM without previous context.
```julia
# Basic usage
response = ask_gpt(my_llm_access, "What is the capital of France?")
response = ask_gpt(my_llm_chat, "What is the capital of France?")
```

## Contextual Conversation
Submit a question with previous conversation context.
```julia
# Conversation with context
conversation = [
    ("user", "Tell me about Julia programming"),
    ("assistant", "Julia is a high-performance programming language...")
]
response = ask_gpt(my_llm_access, "What are its main advantages?", conversation)
```

## LLMChat Interaction
Interact with an LLMChat object, which bundles LLM access and conversation settings.
```julia
# Using ask_gpt with LLMChat
response = ask_gpt(my_llm_chat)  # Continue conversation
response = ask_gpt(my_llm_chat, "What are its main advantages?")

# Using ask_gpt! to automatically update conversation history
response = ask_gpt!(my_llm_chat)  # Continue conversation
response = ask_gpt!(my_llm_chat, "What are its main advantages?")
```

## Callback-Based Interaction
Continuously query the LLM until a callback function validates the response.
```julia
# Example with a validation callback
function is_answer_valid(answer::String)
    # Custom validation logic
    return length(answer) > 50 && contains(answer, "key information")
end
response = ask_gpt(my_llm_access, "Explain quantum computing", is_answer_valid)
```

# Parameters
- `llmaccess::LLMAccess`: An access object for the Language Model API
- `llmchat::LLMChat`: A chat session object bundling LLM access and conversation settings
- `question::String`: The prompt or question to send to the LLM
- `conversation::Vector{Tuple}`: Previous conversation history
  - First element of tuple is the role ("user" or "assistant")
  - Second element is the message content
- `callback_function::Function`: Optional function to validate LLM responses

# Keyword Arguments
- `temperature::Real=0.7`: Controls randomness of the output (0.0-1.0)
- `top_p::Real=0.95`: Nucleus sampling probability threshold
- `max_tokens::Int=800`: Maximum number of tokens in the response
- `seed::Union{Int,Missing}=missing`: Random seed for reproducibility
- `retries::Int=0`: Number of retry attempts for failed requests
- `correction_prompt::String`: Custom prompt for requesting corrections

# LLMChat Specifics
When using `ask_gpt` or `ask_gpt!` with an `LLMChat` object:
- The function uses the `LLMChat`'s bundled settings (temperature, top_p, etc.)
- `ask_gpt!` automatically updates the conversation history
- Without a question, it continues the existing conversation
- With a question, it sends the new question in the context of previous conversation

# Notes
- Handles API rate limiting and automatic retries
- Supports various LLM access methods through the LLMAccess abstraction
- Provides flexible interaction modes: single query, contextual conversation, or validated responses

# Errors
- Throws exceptions for persistent API access failures
- Handles rate limit scenarios with automatic waiting and retrying

# See Also
- `LLMAccess`: The base type for LLM API access
- API-specific subtypes for different LLM providers
"""
function ask_gpt(llmaccess::LLMAccess, question::String; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing, retries=0, reasoning_effort=missing, verbosity=missing)
    return ask_gpt(llmaccess, [("user", question)]; temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed, retries=retries, reasoning_effort=reasoning_effort, verbosity=verbosity)
end

function ask_gpt(llmaccess::LLMAccess, question::String, conversation::Vector{Tuple}; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing, retries=0, reasoning_effort=missing, verbosity=missing)
    conv = deepcopy(conversation)
    push!(conv, ("user", question))
    return ask_gpt(llmaccess, conv; temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed, retries=retries, reasoning_effort=reasoning_effort, verbosity=verbosity)
end

function ask_gpt(llmaccess::LLMAccess, conversation::Vector{Tuple}; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing, retries=0, reasoning_effort=missing, verbosity=missing)
    url = get_url(llmaccess, "/chat/completions")
    body = get_body(llmaccess, conversation; temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed, reasoning_effort=reasoning_effort, verbosity=verbosity)
    header = get_header(llmaccess)
    retries_left = retries + 1
    # check cost of the request
    input_tokens = estimate_tokens(conversation)
    quota_ok = check_llm_cost_limit(llmaccess.model, input_tokens, max_tokens)
    if !quota_ok
        error("Quota exceeded for the LLM access. Please check your usage limits.")
    end

    upstream_retries_left = 3  # dedicated retries for upstream error bodies (independent of `retries`)
    while true
        retries_left -= 1
        try
            # Make the request
            answer = HTTP.request("POST", url, headers=header, body=body)
            body_str = String(answer.body)
            json_body = JSON.parse(body_str)

            # Upstream/provider errors (e.g. flex-tier 504 "operation was aborted" on large
            # contexts) come back as HTTP 200 with an {"error": ...} body and NO "choices".
            # Without this, the KeyError below turns into an empty answer -> NaN prediction.
            # Retry a few times with backoff before giving up.
            if !haskey(json_body, "choices")
                errinfo = get(json_body, "error", json_body)
                if upstream_retries_left > 0
                    upstream_retries_left -= 1
                    println("Upstream error (no choices): $errinfo. Retrying after 12s... ($upstream_retries_left upstream retries left)")
                    sleep(12)
                    continue
                else
                    error("Upstream error (no choices) after retries: $errinfo")
                end
            end

            # update usage information
            update_llm_usage_from_response(llmaccess.model, json_body)

            # If successful, return the content
            try
                return json_body["choices"][1]["message"]["content"]
            catch e
                println("Failed to parse response: $json_body; Error: $e")
                rethrow(e)
            end
        catch e
            if e isa HTTP.ExceptionRequest.StatusError
                try
                    if e.status == 429 && retries_left > 0  # Too Many Requests (rate limit exceeded)
                        response = string(e)
                        regex_retry_after = r"Retry-After: (\d+)"
                        regex_please_wait = r"Please wait (\d+) seconds"
                        if occursin(regex_retry_after, response)
                            retry_after = parse(Int, match(regex_retry_after, response).captures[1])
                        elseif occursin(regex_please_wait, response)
                            retry_after = parse(Int, match(regex_please_wait, response).captures[1])
                        else
                            retry_after = 10  # Default wait time if not specified
                        end

                        try
                            regex_ratelimit_reset_tokens = r"ratelimit-reset-tokens: (\d+)"
                            ratelimit_reset_tokens = parse(Int, match(regex_ratelimit_reset_tokens, response).captures[1])
                            println("Rate limit hit. Rate limit reset tokens: $ratelimit_reset_tokens")
                        catch
                            println("Rate limit hit. No rate limit reset tokens found.")
                        end
                        println("Waiting for $retry_after seconds...")
                        sleep(retry_after)
                        continue
                    end
                catch
                end
            end
            # Raise other unexpected errors if no retries are left
            if retries_left <= 0
                println("An error occurred: $e")
                rethrow(e)
            else
                println("Error: $e")
                println("Retrying after 10 seconds...")
                sleep(10)
            end
        end
    end
    error("This should not happen")
end
CORRECTION_PROMPT = "Your previous answer to this prompt was not sufficient. Please correct your previous answer. 
Check all your previous assumptions, calculations, and conclusions if applicable. "

function ask_gpt(llmaccess::LLMAccess, conversation::Union{String,Vector{Tuple}}, callback_function::Function; temperature=0.7, top_p=0.95, max_tokens=800, seed=missing, correction_prompt::String=CORRECTION_PROMPT, retries=0, reasoning_effort=missing, verbosity=missing)
    conv = deepcopy(conversation)
    if typeof(conv) == String
        conv = [("user", conv)]
    end
    question = conv[end][2]
    answers = []
    solution_found = false
    while !solution_found
        answer = ask_gpt(llmaccess, conv; temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed, retries=retries, reasoning_effort=reasoning_effort, verbosity=verbosity)
        push!(answers, answer)
        push!(conv, ("assistant", answer))
        push!(conv, ("user", "This was the prompt for you: \n" * question * " \n\n" * correction_prompt))
        solution_found = callback_function(answer)
    end
    return answer
end

function ask_gpt(llmchat::LLMChat, question::String)
    return ask_gpt(llmchat.llmaccess, question, llmchat.conversation; temperature=llmchat.temperature, top_p=llmchat.top_p, max_tokens=llmchat.max_tokens, seed=llmchat.seed, retries=llmchat.retries, reasoning_effort=llmchat.reasoning_effort, verbosity=llmchat.verbosity)
end

function ask_gpt(llmchat::LLMChat)
    return ask_gpt(llmchat.llmaccess, llmchat.conversation; temperature=llmchat.temperature, top_p=llmchat.top_p, max_tokens=llmchat.max_tokens, seed=llmchat.seed, retries=llmchat.retries, reasoning_effort=llmchat.reasoning_effort, verbosity=llmchat.verbosity)
end


""" 
    ask_gpt(llmaccess::LLMAccess, question::String; kwargs...)
    ask_gpt(llmaccess::LLMAccess, question::String, conversation::Vector{Tuple}; kwargs...)
    ask_gpt(llmaccess::LLMAccess, conversation::Vector{Tuple}; kwargs...)
    ask_gpt(llmaccess::LLMAccess, conversation::Union{String,Vector{Tuple}}, callback_function::Function; kwargs...)
    ask_gpt(llmchat::LLMChat, question::String)
    ask_gpt(llmchat::LLMChat)
    ask_gpt!(llmchat::LLMChat, question::String)
    ask_gpt!(llmchat::LLMChat)

Interact with a Language Model (LLM) by sending a question or conversation, with multiple dispatch options for flexibility.

# Variants

## Simple Question Submission
Send a single question to the LLM without previous context.
```julia
# Basic usage
response = ask_gpt(my_llm_access, "What is the capital of France?")
response = ask_gpt(my_llm_chat, "What is the capital of France?")
```

## Contextual Conversation
Submit a question with previous conversation context.
```julia
# Conversation with context
conversation = [
    ("user", "Tell me about Julia programming"),
    ("assistant", "Julia is a high-performance programming language...")
]
response = ask_gpt(my_llm_access, "What are its main advantages?", conversation)
```

## LLMChat Interaction
Interact with an LLMChat object, which bundles LLM access and conversation settings.
```julia
# Using ask_gpt with LLMChat
response = ask_gpt(my_llm_chat)  # Continue conversation
response = ask_gpt(my_llm_chat, "What are its main advantages?")

# Using ask_gpt! to automatically update conversation history
response = ask_gpt!(my_llm_chat)  # Continue conversation
response = ask_gpt!(my_llm_chat, "What are its main advantages?")
```

## Callback-Based Interaction
Continuously query the LLM until a callback function validates the response.
```julia
# Example with a validation callback
function is_answer_valid(answer::String)
    # Custom validation logic
    return length(answer) > 50 && contains(answer, "key information")
end
response = ask_gpt(my_llm_access, "Explain quantum computing", is_answer_valid)
```

# Parameters
- `llmaccess::LLMAccess`: An access object for the Language Model API
- `llmchat::LLMChat`: A chat session object bundling LLM access and conversation settings
- `question::String`: The prompt or question to send to the LLM
- `conversation::Vector{Tuple}`: Previous conversation history
  - First element of tuple is the role ("user" or "assistant")
  - Second element is the message content
- `callback_function::Function`: Optional function to validate LLM responses

# Keyword Arguments
- `temperature::Real=0.7`: Controls randomness of the output (0.0-1.0)
- `top_p::Real=0.95`: Nucleus sampling probability threshold
- `max_tokens::Int=800`: Maximum number of tokens in the response
- `seed::Union{Int,Missing}=missing`: Random seed for reproducibility
- `retries::Int=0`: Number of retry attempts for failed requests
- `correction_prompt::String`: Custom prompt for requesting corrections

# LLMChat Specifics
When using `ask_gpt` or `ask_gpt!` with an `LLMChat` object:
- The function uses the `LLMChat`'s bundled settings (temperature, top_p, etc.)
- `ask_gpt!` automatically updates the conversation history
- Without a question, it continues the existing conversation
- With a question, it sends the new question in the context of previous conversation

# Notes
- Handles API rate limiting and automatic retries
- Supports various LLM access methods through the LLMAccess abstraction
- Provides flexible interaction modes: single query, contextual conversation, or validated responses

# Errors
- Throws exceptions for persistent API access failures
- Handles rate limit scenarios with automatic waiting and retrying

# See Also
- `LLMAccess`: The base type for LLM API access
- API-specific subtypes for different LLM providers
"""
function ask_gpt!(llmchat::LLMChat, question::String)
    answer = ask_gpt(llmchat.llmaccess, question; temperature=llmchat.temperature, top_p=llmchat.top_p, max_tokens=llmchat.max_tokens, seed=llmchat.seed, retries=llmchat.retries, reasoning_effort=llmchat.reasoning_effort, verbosity=llmchat.verbosity)
    push!(llmchat.conversation, ("user", question))
    push!(llmchat.conversation, ("assistant", answer))
    return answer
end

function ask_gpt!(llmchat::LLMChat)
    answer = ask_gpt(llmchat.llmaccess, llmchat.conversation; temperature=llmchat.temperature, top_p=llmchat.top_p, max_tokens=llmchat.max_tokens, seed=llmchat.seed, retries=llmchat.retries, reasoning_effort=llmchat.reasoning_effort, verbosity=llmchat.verbosity)
    push!(llmchat.conversation, ("assistant", answer))
    return answer
end


function search_for_numbers(text::String)
    parts = split(text, ['\n', ' ', ':', ';', '?', '!', '%', '[', ']', '~', '{', '}', '"', '\'', '\\'])
    numbers = []
    for part in parts
        try
            part = strip(part, ['*', '(', ')', ',', '.', ':', ';', '?', '!', '%', '[', ']', '~', '{', '}', '"', '\''])
            push!(numbers, parse(Float64, part))
        catch
        end
    end
    return numbers
end

"""
Creates a JSON string from a DataFrame where a Vector of the columns is created. This basically transposes the DataFrame:
[col_name1: col1_as_vector, 
col_name2: col2_as_vector, 
...]
"""
function json_colwise(df::DataFrame)
    return "[" * join([nm * ": [" * join([string(x) for x in df[!, nm]], ", ") for nm in names(df)], "]\n") * "]]"
end

"""
Creates a JSON string from a DataFrame or Matrix where the first row is the column names and the following rows are the data:
[[col_name1, col_name2, ...],
[row1_col1, row1_col2, ...],
[row2_col1, row2_col2, ...]]
"""
function json_rowwise(df::DataFrame)
    return "[[" * join([string(x) for x in names(df)], ",") * "]\n[" * join([join([string(x) for x in row], ", ") for row in eachrow(df)], "]\n[") * "]]"
end

function json_rowwise(mat::Matrix)
    return "[[" * join([string(i) for i in 1:size(mat, 2)], ",") * "]\n[" * join([join([string(x) for x in row], ", ") for row in eachrow(mat)], "]\n[") * "]]"
end

"""
    ask_gpt_threaded(llmaccess::LLMAccess, questions::Vector{String}; kwargs...)

Process multiple questions in parallel using threads.
This provides similar functionality to batch processing but works with any LLM service by 
running standard API calls in parallel threads.

# Arguments
- `llmaccess::LLMAccess`: The LLM access configuration, can be replaced by LLMChat
- `batch_conversations::Vector{Vector{Tuple}}`: Array of conversations to process in parallel (each conversation is a vector of (role, content) tuples)
- `num_threads::Int=10`: Number of threads to use for parallel processing
- `temperature::Real=0.7`: Controls randomness of the output (0.0-1.0)
- `top_p::Real=0.95`: Nucleus sampling probability threshold
- `max_tokens::Int=800`: Maximum number of tokens in the response
- `seed::Union{Int,Missing}=missing`: Random seed for reproducibility
- `retries::Int=0`: Number of retry attempts for failed requests
- `reasoning_effort::Union{String,Missing}=missing`: Reasoning effort for the model

# Returns
- `Vector{String}`: Array of responses corresponding to each question

# Example
```julia
questions = ["What is the capital of France?", "What is the capital of Germany?", "What is the capital of Italy?"]
responses = ask_gpt_threaded(my_llm, questions)

# Using thread oversubscription (useful for I/O-bound workloads)
responses = ask_gpt_threaded(my_llm, questions, num_threads=100)
```

# Notes
- Uses Julia's built-in threading capabilities
- Works with any LLM access type
- Provides parallelism without requiring batch API support
- Thread safety is handled automatically with locks
- `num_threads` allows using more concurrent threads than CPU cores
"""
function ask_gpt_threaded(llmaccess::LLMAccess, batch_conversations::Vector{Vector{Tuple}}; num_threads::Int=10, kwargs...)

    n = length(batch_conversations)
    if n == 0
        return String[]
    end


    # Use the smaller of: available threads (with oversubscription), or number of conversations
    thread_count = min(
        num_threads,
        n
    )

    if thread_count < 2
        # If only one thread is available or requested, fall back to sequential processing
        return [ask_gpt(llmaccess, conv; kwargs...) for conv in batch_conversations]
    end

    println("Processing $(length(batch_conversations)) requests using $thread_count threads")


    responses = Vector{String}(undef, n)

    # Create a thread-safe counter for progress tracking
    progress_lock = ReentrantLock()
    completed = Threads.Atomic{Int}(0)

    # Create a task channel for work distribution
    task_channel = Channel{Int}(thread_count)

    # Launch worker tasks to process conversations
    worker_tasks = []
    for i in 1:thread_count
        task = @async begin
            while true
                idx = try
                    take!(task_channel)
                catch e
                    # Channel closed
                    break
                end

                try
                    # Process the conversation
                    responses[idx] = ask_gpt(llmaccess, batch_conversations[idx]; kwargs...)

                    # Update progress
                    lock(progress_lock) do
                        Threads.atomic_add!(completed, 1)
                        println("Worker task $i: Completed $(completed[]) of $n requests")
                    end
                catch e
                    lock(progress_lock) do
                        println("Worker task $i: Error processing conversation $(idx): $(e)")
                    end
                    responses[idx] = "ERROR: NO ANSWER"
                end
            end
        end
        push!(worker_tasks, task)
    end

    # Feed tasks into the channel
    for i in 1:n
        put!(task_channel, i)
    end

    # Close the channel when all tasks are distributed
    close(task_channel)

    # Wait for all worker tasks to complete
    for task in worker_tasks
        wait(task)
    end

    return responses
end

function ask_gpt_threaded(llmaccess::LLMAccess, questions::Vector{String}; num_threads::Int=10, kwargs...)
    conversations = [
        [("user", question)] for question in questions
    ]
    return ask_gpt_threaded(llmaccess, conversations; num_threads=num_threads, kwargs...)
end

function ask_gpt_threaded(llmchat::LLMChat, questions::Vector{String};
    num_threads::Int=10)
    conversations = []
    for question in questions
        conversation = deepcopy(llmchat.conversation)
        push!(conversation, ("user", question))
        push!(conversations, conversation)
    end
    return ask_gpt_threaded(
        llmchat.llmaccess,
        conversations;
        temperature=llmchat.temperature,
        top_p=llmchat.top_p,
        max_tokens=llmchat.max_tokens,
        seed=llmchat.seed,
        retries=llmchat.retries,
        reasoning_effort=llmchat.reasoning_effort,
        verbosity=llmchat.verbosity,
        num_threads=num_threads
    )
end

function ask_gpt_threaded(llmchats::Vector{LLMChat}; num_threads::Int=10)
    n = length(llmchats)
    if n == 0
        return String[]
    end


    # Use the smaller of: available threads (with oversubscription), or number of conversations
    thread_count = min(
        num_threads,
        n
    )

    if thread_count < 2
        # If only one thread is available or requested, fall back to sequential processing
        return [ask_gpt(llmchat) for llmchat in llmchats]
    end

    println("Processing $(n) requests using $thread_count threads")


    responses = Vector{String}(undef, n)

    # Create a thread-safe counter for progress tracking
    progress_lock = ReentrantLock()
    completed = Threads.Atomic{Int}(0)

    # Create a task channel for work distribution
    task_channel = Channel{Int}(thread_count)

    # Launch worker tasks to process conversations
    worker_tasks = []
    for i in 1:thread_count
        task = @async begin
            while true
                idx = try
                    take!(task_channel)
                catch e
                    # Channel closed
                    break
                end

                try
                    # Process the conversation
                    responses[idx] = ask_gpt(llmchats[idx])

                    # Update progress
                    lock(progress_lock) do
                        Threads.atomic_add!(completed, 1)
                        println("Worker task $i: Completed $(completed[]) of $n requests")
                    end
                catch e
                    lock(progress_lock) do
                        println("Worker task $i: Error processing conversation $(idx): $(e)")
                    end
                    responses[idx] = ""
                end
            end
        end
        push!(worker_tasks, task)
    end

    # Feed tasks into the channel
    for i in 1:n
        put!(task_channel, i)
    end

    # Close the channel when all tasks are distributed
    close(task_channel)

    # Wait for all worker tasks to complete
    for task in worker_tasks
        wait(task)
    end

    # Guard against any remaining #undef slots (e.g. if a worker task itself
    # died before hitting the catch block). Callers skip empty answers.
    for i in 1:n
        if !isassigned(responses, i)
            responses[i] = ""
        end
    end

    return responses
end

"""
    estimate_tokens(conversation::Vector{Tuple}) -> Int

Gibt eine grobe Schätzung der Tokenanzahl für eine Konversation zurück.
`conversation` ist ein Vektor von (role, content)-Tupeln, z. B.:

    [("user", "Hallo!"), ("assistant", "Hi, wie kann ich helfen?")]

Die Tokenzahl wird angenähert als `round(length(text) / avg_chars_per_token)`.
"""
function estimate_tokens(conversation::Vector{Tuple})::Int
    avg_chars_per_token = 4.0  # typische Näherung für Englisch & Deutsch bei OpenAI-Modellen
    total_chars = 0
    for (role, content) in conversation
        total_chars += length(role) + length(content)
    end
    return Int(round(total_chars / avg_chars_per_token))
end


# --- Robust cost/usage file IO -------------------------------------------------
# The cost/usage JSON files are written from many calls and were corrupted in the
# past when a run was force-killed mid-write (a stray trailing `}` -> the file no
# longer parses). A corrupt file made check_llm_cost_limit() return false, which
# turned EVERY subsequent call into a "Quota exceeded" error -> empty answers -> NaN
# predictions. These helpers prevent that: writes are atomic (temp file + rename, so
# a kill can never leave a half-written file in place) and reads self-heal a trailing
# brace/garbage by truncating to the last balanced top-level value.

function _heal_json_string(s::AbstractString)
    depth = 0; instr = false; esc = false; count = 0; keep = 0
    for c in s
        count += 1
        if instr
            if esc
                esc = false
            elseif c == '\\'
                esc = true
            elseif c == '"'
                instr = false
            end
        else
            if c == '"'
                instr = true
            elseif c == '{' || c == '['
                depth += 1
            elseif c == '}' || c == ']'
                depth -= 1
                if depth == 0
                    keep = count
                    break
                end
            end
        end
    end
    return keep > 0 ? first(s, keep) : s
end

function _atomic_write_json(path::AbstractString, data)
    # Per-process temp name avoids any cross-process collision on the temp file.
    tmp = string(path, ".tmp.", getpid())
    open(tmp, "w") do io
        JSON.print(io, data)
    end
    # Base.Filesystem.rename atomically REPLACES an existing destination (libuv
    # uv_fs_rename -> MoveFileEx/SetFileInformationByHandle on Windows). Do NOT use
    # mv(force=true): it does rm(dst)+rename, leaving a window where the file is
    # missing -- a concurrent reader or a kill there loses the file entirely, and
    # check_llm_cost_limit would then recreate a fresh total_cost_usd=0 file.
    Base.Filesystem.rename(tmp, path)
end

function _read_json_file_tolerant(path::AbstractString)
    str = read(path, String)
    try
        return JSON.parse(str)
    catch
        parsed = JSON.parse(_heal_json_string(str))  # rethrows if unrecoverable
        try
            _atomic_write_json(path, parsed)
            @warn "Repaired corrupt JSON file (stray trailing data removed): $path"
        catch
        end
        return parsed
    end
end


function check_llm_cost_limit(llm::Union{String,Nothing}=nothing, input_tokens::Int=0, output_tokens::Int=0)::Bool
    # Ensure the data directory exists
    SCRIPT_DIR = dirname(@__FILE__)
    data_dir = joinpath(SCRIPT_DIR, "LLMUtilsData")
    if !isdir(data_dir)
        mkpath(data_dir)
    end

    lockfile = joinpath(data_dir, "lockfile.lock")
    costs_file = joinpath(data_dir, "LLMCosts.json")
    usage_file = joinpath(data_dir, "LLMUsage.json")

    try
        # Ensure the costs file exists
        if !isfile(costs_file)
            @warn "LLMCosts.json nicht gefunden. Erstelle Standarddatei."
            default_costs = Dict(
                "limit" => 100.0,
                "gpt-4" => Dict("input_per_1m_tokens_usd" => 30.0, "output_per_1m_tokens_usd" => 60.0),
                "gpt-3.5-turbo" => Dict("input_per_1m_tokens_usd" => 1.0, "output_per_1m_tokens_usd" => 2.0)
            )
            _atomic_write_json(costs_file, default_costs)
        end

        # Ensure the usage file exists
        if !isfile(usage_file)
            default_usage = Dict("total_cost_usd" => 0.0, "last_updated" => string(Dates.now()))
            _atomic_write_json(usage_file, default_usage)
        end

        open(lockfile, "w") do lock_io
            lock(lock_io)

            usage_data = _read_json_file_tolerant(usage_file)
            cost_data = _read_json_file_tolerant(costs_file)
            limit = cost_data["limit"]
            current_cost = get(usage_data, "total_cost_usd", 0.0)

            if llm !== nothing && haskey(cost_data, llm)
                input_rate = cost_data[llm]["input_per_1m_tokens_usd"]
                output_rate = cost_data[llm]["output_per_1m_tokens_usd"]
                estimate_cost = (input_tokens / 1000000) * input_rate + (output_tokens / 1000000) * output_rate
                current_cost += estimate_cost
            end

            return current_cost <= limit
        end
    catch e
        @warn "Fehler beim Lesen der Kostendateien: $e"
        return false  # Im Fehlerfall lieber stoppen
    end
end

function update_llm_usage_from_response(llm::String, response_json::AbstractDict, batch=false)

    # Ensure the data directory exists
    SCRIPT_DIR = dirname(@__FILE__)
    data_dir = joinpath(SCRIPT_DIR, "LLMUtilsData")
    if !isdir(data_dir)
        mkpath(data_dir)
    end

    lockfile = joinpath(data_dir, "lockfile.lock")
    costs_file = joinpath(data_dir, "LLMCosts.json")
    usage_file = joinpath(data_dir, "LLMUsage.json")

    try
        open(lockfile, "w") do lock_io
            lock(lock_io)

            # Ensure costs file exists
            if !isfile(costs_file)
                @warn "LLMCosts.json nicht gefunden für update_llm_usage_from_response"
                return
            end

            usage_data = _read_json_file_tolerant(usage_file)
            cost_data = Dict()
            if !isfile(costs_file)
                @warn "LLMCosts.json nicht gefunden für update_llm_usage_from_response"
            else
                cost_data = _read_json_file_tolerant(costs_file)
            end

            input_tokens = response_json["usage"]["prompt_tokens"]
            cached_tokens = get(response_json["usage"]["prompt_tokens_details"], "cached_tokens", 0)
            input_tokens -= cached_tokens
            #println("Input tokens (nach Cache-Abzug): $input_tokens")
            #println("Cached tokens: $cached_tokens")
            output_tokens = response_json["usage"]["completion_tokens"]
            #try to get cost directly from response_json
            cost = get(response_json["usage"], "cost", -1)
            if batch
                # For batch responses, costs are half so we just divide token numbers by 2
                input_tokens = div(input_tokens, 2)
                output_tokens = div(output_tokens, 2)
            end

            # Check if the LLM model exists in cost data
            if !haskey(cost_data, llm) && cost < 0
                @warn "LLM model '$llm' nicht in LLMCosts.json gefunden und Kosten nicht in Response enthalten"
                return
            end


            if cost < 0
                input_rate = cost_data[llm]["input_per_1m_tokens_usd"]
                cached_rate = get(cost_data[llm], "cached_per_1m_tokens_usd", 0)
                output_rate = cost_data[llm]["output_per_1m_tokens_usd"]
                cost = (input_tokens / 1000000) * input_rate + (cached_tokens / 1000000) * cached_rate + (output_tokens / 1000000) * output_rate
            end # else cost is already exactly given by provider

            if !haskey(usage_data, llm)
                usage_data[llm] = Dict("input_tokens" => 0, "cached_tokens" => 0, "output_tokens" => 0)
            end
            if !haskey(usage_data[llm], "cached_tokens")
                usage_data[llm]["cached_tokens"] = 0
            end

            usage_data[llm]["input_tokens"] += input_tokens
            usage_data[llm]["cached_tokens"] += cached_tokens
            usage_data[llm]["output_tokens"] += output_tokens
            usage_data["total_cost_usd"] = get(usage_data, "total_cost_usd", 0.0) + cost
            usage_data["last_updated"] = string(Dates.now())

            _atomic_write_json(usage_file, usage_data)
        end
    catch e
        @warn "Fehler beim Aktualisieren der Nutzungsdaten: $e"
        println(response_json)
    end
end

function update_llm_usage_from_response(llm::String, response_jsons::AbstractVector{<:AbstractDict}, batch=false)
    # Ensure the data directory exists
    for response_json in response_jsons
        update_llm_usage_from_response(llm, response_json, batch)
    end
end