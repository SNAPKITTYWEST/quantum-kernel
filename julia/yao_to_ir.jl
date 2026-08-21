# yao_to_ir.jl
#
# Lower Yao.jl block tree to QuantumIR flat op list.
# Produces Dict with mandatory `unsupported` semantics tracking.

include("yao_types.jl")

# -----------------------------------------------------------------------
# IR Lowering
# -----------------------------------------------------------------------

"""
    yao_to_ir(block::ChainBlock) -> Dict

Flatten a Yao ChainBlock tree into QuantumIR format.
Tracks semantic losses in metadata.unsupported.
"""
function yao_to_ir(block::ChainBlock)::Dict{String,Any}
    ops = Dict{String,Any}[]
    unsupported = String[]
    n_qubits = block.n

    flatten_block!(ops, unsupported, block, collect(0:n_qubits-1))

    gate_count = count(op -> op["type"] == "gate", ops)

    return Dict{String,Any}(
        "version" => "0.1.0",
        "source_lang" => "yao",
        "qubits" => n_qubits,
        "cbits" => n_qubits,
        "ops" => ops,
        "metadata" => Dict{String,Any}(
            "source_lang" => "yao",
            "version" => "0.1.0",
            "unsupported" => unsupported
        ),
        "resources" => Dict{String,Any}(
            "gate_count" => gate_count,
            "depth" => estimate_depth(ops, n_qubits),
            "t_count" => count(op -> get(op, "name", "") in ("T", "Tdg"), ops),
            "width" => n_qubits
        )
    )
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::ChainBlock, qubit_map::Vector{Int})
    for sub in block.blocks
        flatten_block!(ops, unsupported, sub, qubit_map)
    end
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::KronBlock, qubit_map::Vector{Int})
    if length(block.locs_blocks) > 1
        if !("KronBlock parallelism (serialized to sequential in QIR)" in unsupported)
            push!(unsupported, "KronBlock parallelism (serialized to sequential in QIR)")
        end
    end
    for (loc, sub) in block.locs_blocks
        flatten_block!(ops, unsupported, sub, [qubit_map[loc]])
    end
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::PutBlock, qubit_map::Vector{Int})
    mapped = [qubit_map[l] for l in block.locs]
    flatten_block!(ops, unsupported, block.content, mapped)
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::ControlBlock, qubit_map::Vector{Int})
    ctrl_qubits = [qubit_map[c] for c in block.ctrl_locs]
    target_qubit = qubit_map[block.target_loc]

    if block.content isa Z
        push!(ops, Dict{String,Any}("type" => "gate", "name" => "CZ", "params" => Float64[], "qubits" => [ctrl_qubits[1], target_qubit]))
    elseif block.content isa X
        push!(ops, Dict{String,Any}("type" => "gate", "name" => "CX", "params" => Float64[], "qubits" => [ctrl_qubits[1], target_qubit]))
    else
        push!(ops, Dict{String,Any}("type" => "gate", "name" => "C-$(typeof(block.content))", "params" => Float64[], "qubits" => vcat(ctrl_qubits, [target_qubit])))
    end
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::MeasureBlock, qubit_map::Vector{Int})
    for (i, loc) in enumerate(block.locs)
        q = qubit_map[min(loc, length(qubit_map))]
        push!(ops, Dict{String,Any}("type" => "measure", "qubit" => q, "cbit" => q))
    end
end

# Single-qubit gates
function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::Rz, qubit_map::Vector{Int})
    push!(ops, Dict{String,Any}("type" => "gate", "name" => "Rz", "params" => [block.theta], "qubits" => [qubit_map[1]]))
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::Ry, qubit_map::Vector{Int})
    push!(ops, Dict{String,Any}("type" => "gate", "name" => "Ry", "params" => [block.theta], "qubits" => [qubit_map[1]]))
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::Rx, qubit_map::Vector{Int})
    push!(ops, Dict{String,Any}("type" => "gate", "name" => "Rx", "params" => [block.theta], "qubits" => [qubit_map[1]]))
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::H, qubit_map::Vector{Int})
    push!(ops, Dict{String,Any}("type" => "gate", "name" => "H", "params" => Float64[], "qubits" => [qubit_map[1]]))
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::X, qubit_map::Vector{Int})
    push!(ops, Dict{String,Any}("type" => "gate", "name" => "X", "params" => Float64[], "qubits" => [qubit_map[1]]))
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::Z, qubit_map::Vector{Int})
    push!(ops, Dict{String,Any}("type" => "gate", "name" => "Z", "params" => Float64[], "qubits" => [qubit_map[1]]))
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::S, qubit_map::Vector{Int})
    push!(ops, Dict{String,Any}("type" => "gate", "name" => "S", "params" => Float64[], "qubits" => [qubit_map[1]]))
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::Sdg, qubit_map::Vector{Int})
    push!(ops, Dict{String,Any}("type" => "gate", "name" => "Sdg", "params" => Float64[], "qubits" => [qubit_map[1]]))
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::T, qubit_map::Vector{Int})
    push!(ops, Dict{String,Any}("type" => "gate", "name" => "T", "params" => Float64[], "qubits" => [qubit_map[1]]))
end

function flatten_block!(ops::Vector{Dict{String,Any}}, unsupported::Vector{String},
                        block::SX, qubit_map::Vector{Int})
    push!(ops, Dict{String,Any}("type" => "gate", "name" => "SX", "params" => Float64[], "qubits" => [qubit_map[1]]))
end

# -----------------------------------------------------------------------
# Depth Estimation
# -----------------------------------------------------------------------

function estimate_depth(ops::Vector{Dict{String,Any}}, n_qubits::Int)::Int
    qubit_depth = zeros(Int, n_qubits)
    for op in ops
        if op["type"] == "gate"
            qubits = op["qubits"]
            max_d = maximum(qubit_depth[q+1] for q in qubits; init=0)
            for q in qubits
                qubit_depth[q+1] = max_d + 1
            end
        end
    end
    return maximum(qubit_depth; init=0)
end
