using JSON

"""
    append_value_to_json(
        file_name::String, 
        keys::Vector{String}, 
        value; 
        overwrite::Bool=false, 
        create_new::Bool=false,
        nice_print::Bool=true
    )

Append a value to a JSON file at a specified nested key path, with flexibility for different scenarios.

# Parameters
- `file_name::String`: Path to the JSON file to be modified.
- `keys::Vector{String}`: A vector of keys representing the nested path to the target location.
- `value`: The value to be added to the JSON file.
- `overwrite::Bool=false`: If true, replaces the existing value(s) at the specified key path.
- `create_new::Bool=false`: If true, creates a new JSON file overwriting any existing one.
- `nice_print::Bool=true`: If true, the JSON file will be written with custom formatting.

# Behavior
- Creates nested dictionary structures as needed along the specified key path.
- If the target key doesn't exist, initializes it with a list containing the value.
- When appending:
  - Adds the value to an existing list if the key already contains a list.
  - Converts a non-list value to a list containing both the original and new value.
- Handles type mismatches by converting existing values to a list.

# Error Handling
- Warns and creates a new file if the existing JSON cannot be parsed.
- Provides warnings for file writing failures.

# Examples
```julia
# Create a new JSON file with a nested structure
append_value_to_json("data.json", ["users", "active"], "john_doe", create_new=true)

# Append a value to an existing list
append_value_to_json("data.json", ["users", "active"], "jane_smith")

# Overwrite existing values
append_value_to_json("data.json", ["users", "active"], "new_user", overwrite=true)
```

# Warnings
- Provides warnings for parsing and writing failures.
"""
function append_value_to_json(
    file_name::String,
    keys::Vector{String},
    value;
    overwrite::Bool=false,
    create_new::Bool=false,
    nice_print::Bool=true
)
    # Initialize or load JSON data
    json_data = Dict{String,Any}()
    if create_new || !isfile(file_name)
        println("Creating new JSON file: ", file_name)
    else
        try
            json_data = open(file_name, "r") do io
                JSON.parse(read(io, String), allownan=true; dicttype=Dict)
            end
        catch e
            # if file exists but cannot be parsed, raise the error
            if isfile(file_name)
                @warn "File exists but failed to read: $e"
                throw(e)
            end
            # if file does not exist, create a new one
            @warn "File does not exist, creating a new one: $e"
        end
    end

    # Traverse the hierarchy to the target dictionary
    parent = json_data
    for key in keys[1:end-1]
        if !haskey(parent, key) || !(parent[key] isa Dict)
            parent[key] = Dict{String,Any}()
        end
        parent = parent[key]
    end

    # Update the target key
    target_key = keys[end]
    if !haskey(parent, target_key) || overwrite
        # Overwrite or initialize with a new list containing the value
        parent[target_key] = [value]
    else
        # Append to the existing list
        if parent[target_key] isa AbstractVector
            push!(parent[target_key], value)
        else
            @warn "Type mismatch: $parent[$target_key] is not a list"
            # Handle type mismatch by converting to a list
            parent[target_key] = [parent[target_key], value]
        end
    end
    json=""
    if nice_print
        json=json_custom(json_data)
    else
        json=JSON.json(json_data, allownan=true)
    end
    # Write the updated JSON back to the file with custom formatting
    try
        open(file_name, "w") do io
            write(io, json)
        end
    catch e
        @warn "Failed to write JSON file: $e"
        println("Path: ", file_name)
        println("JSON: ", json_data)
    end
end

# Custom function to recursively write JSON with single-line arrays
function json_custom(data, indent_level=0)
    json_string=""
    indent = " "^(4 * indent_level)  # Indentation (4 spaces per level)
    if isa(data, Dict)
        json_string *= "{\n"
        for (i, (k, v)) in enumerate(collect(pairs(data)))
            json_string *= indent * "    " * JSON.json(k, allownan=true) * ": "
            json_string *= json_custom(v, indent_level + 1)
            if i < length(data)  # Add a comma if it's not the last key-value pair
                json_string *=  ",\n"
            else # add a line break if it is the last key-value pair
                json_string *= "\n"
            end
        end
        json_string *= indent * "}\n"
    else
        # Write other JSON-compatible data types
        json_string *= JSON.json(data, allownan=true)
    end
    return json_string
end

"""
    write_json(file_name::String, dict::AbstractDict)

Write a dictionary to a JSON file with custom formatting.

# Arguments
- `file_name::String`: Path and name of the file to write the JSON data to
- `dict::AbstractDict`: Dictionary containing the data to be serialized to JSON

# Details
Uses `json_custom` for JSON serialization with custom formatting. If the write operation 
fails, a warning is logged with the error message and additional debug information 
(file path and dictionary content) is printed to stdout.

# Example
```julia
data = Dict("name" => "Alice", "age" => 30)
write_json("output.json", data)
```

# Throws
- A warning if the JSON file cannot be written
"""
function write_json(file_name::String, dict::AbstractDict)
    json = json_custom(dict)
    # Write the JSON to the file with custom formatting
    try
        open(file_name, "w") do io
            write(io, json)
        end
    catch e
        @warn "Failed to write JSON file: $e"
        println("Path: ", file_name)
        println("JSON: ", dict)
    end
end

"""
    read_json(file_name::String)

Stable read and parse a JSON file into a Dict. If the file is not found or cannot be parsed, an error is raised.

# Arguments
- `file_name::String`: Path and name of the JSON file to read

# Returns
- A Dict containing the JSON data from the file

# Throws
- An error if the file is not found or cannot be parsed
"""
function read_json(file_name::String)
    """
    Reads a JSON file and returns its contents as a Dict.
    If the file doesn't exist or cannot be parsed, an error is raised.
    """
    if !isfile(file_name)
        error("File not found: $file_name")
    end

    try
        return open(file_name, "r") do io
            JSON.parse(read(io, String), allownan=true; dicttype=Dict)
        end
    catch e
        error("Failed to read or parse JSON file: $e")
    end
end



"""
This function searches from the end of the string for the first occurence of a number (float or integer) and returns it as a float.
"""
function search_for_last_number_in_string(str::String)
    """
    Extracts the last valid number from the input string and returns it as a Float64.
    
    A valid number is defined as follows:
      - It is preceded by either the beginning of the string or a character that is not a digit or a period.
      - It consists of an optional minus sign, followed by either:
            • one or more digits optionally followed by a decimal point and one or more digits, or
            • a decimal point followed by one or more digits.
      - It is followed by either a character that is not a digit or period, or the end of the string.
    
    Examples of valid matches:
      " The result is -12.3"          → "-12.3"
      " Test 01.2 and 0.1"            → "0.1" (last occurrence)
      " 0 is the answer"              → "0"
      "The inhibition efficiency for 6,4-Pyridinedicarboxylic acid is approximately 37.0."  → "37.0"
      " Just .45"                    → ".45"
    
    Examples of invalid cases (no match):
      " Answer: 0,1 "                → (no match, because comma is not allowed)
    
    Returns the last matching number as a Float64 or NaN if no valid number is found.
    """
    # Regex breakdown:
    #   (?:(?<=^)|(?<=[^0-9.])) ensures that before the number we have either the start of the string
    #       or a character that is neither a digit nor a period.
    #   ([-–]? ... ) allows for an optional minus sign or en-dash.
    #   \d+(?:\.\d+)? matches one or more digits optionally followed by a decimal point and one or more digits.
    #       Alternatively, |\.\d+ handles numbers starting with a decimal point.
    #   (?=[^0-9]|$) ensures that after the number there is either a character that is not a digit
    #       or the end of the string.
    regex = r"(?:(?<=^)|(?<=[^0-9.]))([-–]?(?:\d+(?:\.\d+)?|\.\d+))(?=[^0-9]|$)"
    # matches = collect(Base.eachmatch(r"-?\d+(\.\d+)?", str))

    matches = collect(eachmatch(regex, str))
    if !isempty(matches)
        # Use the last match (i.e. first when scanning backward)
        m = matches[end].match
        try
            return parse(Float64, m)
        catch
        end
        try
            return parse(Float64, m * ".0")
        catch
        end
        try
            return parse(Float64, replace(m, "–" => "-"))
        catch
        end
    end
    return NaN
end

"""
    append_values_to_json(
        file_name::String,
        keys_list::Vector{Vector{String}},
        values::AbstractVector;
        overwrite::Bool=false,
        create_new::Bool=false,
        nice_print::Bool=true
    )

Append multiple values to a JSON file using a batch of key paths and values.

This behaves like `append_value_to_json` but accepts a vector of `keys` vectors
and a vector of `values` and performs all in-memory updates before writing
the file once at the end to avoid repeated IO.

# Arguments
- `file_name::String`: Path to the JSON file to be modified.
- `keys_list::Vector{Vector{String}}`: Vector of key vectors specifying nested paths.
- `values::AbstractVector`: Vector of values to append (must have same length as `keys_list`).

Throws an error if `keys_list` and `values` have different lengths.
"""
function append_values_to_json(
    file_name::String,
    keys_list::Vector{Vector{String}},
    values::AbstractVector;
    overwrite::Bool=false,
    create_new::Bool=false,
    nice_print::Bool=true
)
    if length(keys_list) != length(values)
        error("keys_list and values must have the same length")
    end

    # Initialize or load JSON data (single read)
    json_data = Dict{String,Any}()
    if create_new || !isfile(file_name)
        println("Creating new JSON file: ", file_name)
    else
        try
            json_data = open(file_name, "r") do io
                JSON.parse(read(io, String), allownan=true; dicttype=Dict)
            end
        catch e
            if isfile(file_name)
                @warn "File exists but failed to read: $e"
                throw(e)
            end
            @warn "File does not exist, creating a new one: $e"
        end
    end

    # Process all updates in memory
    for (keys, value) in zip(keys_list, values)
        if isempty(keys)
            @warn "Empty keys vector provided; skipping"
            continue
        end

        parent = json_data
        for key in keys[1:end-1]
            if !haskey(parent, key) || !(parent[key] isa Dict)
                parent[key] = Dict{String,Any}()
            end
            parent = parent[key]
        end

        target_key = keys[end]
        if !haskey(parent, target_key) || overwrite
            parent[target_key] = [value]
        else
            if parent[target_key] isa AbstractVector
                push!(parent[target_key], value)
            else
                @warn "Type mismatch: $parent[$target_key] is not a list"
                parent[target_key] = [parent[target_key], value]
            end
        end
    end

    # Write the updated JSON back to the file once
    json = nice_print ? json_custom(json_data) : JSON.json(json_data, allownan=true)
    try
        open(file_name, "w") do io
            write(io, json)
        end
    catch e
        @warn "Failed to write JSON file: $e"
        println("Path: ", file_name)
        println("JSON: ", json_data)
    end
end