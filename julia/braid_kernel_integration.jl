# braid_kernel_integration.jl — Braid Feature Map + VQC + QNTK

module BraidKernelIntegration

using LinearAlgebra
using Random
using Statistics

export BraidKernelEngine, compute_braid_kernel_matrix, build_braid_feature_map_ops

# ═══════════════════════════════════════════════════════════════════════
# Braid Kernel Engine
# ═══════════════════════════════════════════════════════════════════════

struct BraidKernelEngine
    n_qubits::Int
    n_strands::Int
    n_layers::Int
    encoding::Symbol
    shots::Int
    zne_factors::Vector{Float64}
    use_markov::Bool
    use_lattice_surgery::Bool
end

function BraidKernelEngine(; n_qubits=4, n_strands=4, n_layers=2,
                            encoding=:braid, shots=1000,
                            zne_factors=[1.0,1.5,2.0,3.0],
                            use_markov=true, use_lattice_surgery=false)
    BraidKernelEngine(n_qubits, n_strands, n_layers, encoding, shots, zne_factors,
                      use_markov, use_lattice_surgery)
end

# ═══════════════════════════════════════════════════════════════════════
# Types (self-contained for module independence)
# ═══════════════════════════════════════════════════════════════════════

struct BraidWord
    generators::Vector{Int}
    edge_indices::Vector{Int}
    n_strands::Int
end

BraidWord(n_strands::Int) = BraidWord(Int[], Int[], n_strands)

const HERON_EDGES_0 = [
    (0, 1), (1, 2),
    (0, 3), (1, 3), (1, 4), (2, 4), (2, 5),
    (3, 4), (4, 5), (5, 6),
    (3, 7), (4, 7), (4, 8), (5, 8), (5, 9), (6, 9),
    (7, 8), (8, 9)
]

const HERON_EDGE_INDEX = Dict(edge => i for (i, edge) in enumerate(HERON_EDGES_0))

struct FeatureMapParams
    data::Array{Float64,3}
    n_layers::Int
    n_qubits::Int
end

function FeatureMapParams(n_layers::Int, n_qubits::Int; init_scale::Float64=0.1)
    data = randn(n_layers, n_qubits, 3) * init_scale .+ 1.0
    FeatureMapParams(data, n_layers, n_qubits)
end

Base.getindex(p::FeatureMapParams, i...) = p.data[i...]

struct CircuitOp
    gate::String
    qubits::Vector{Int}
    params::Vector{Float64}
end

# ═══════════════════════════════════════════════════════════════════════
# Braid Feature Map
# ═══════════════════════════════════════════════════════════════════════

"""
    build_braid_feature_map_ops(engine, features, params)

Feature map U_Φ(x) using braid encoding:
1. Feature diff → BraidWord
2. Free reduction (cancel σσ⁻¹)
3. BraidWord → CX/H sequences on heavy-hex
4. Variational rotation layers
"""
function build_braid_feature_map_ops(engine::BraidKernelEngine,
                                      features::Vector{Float64},
                                      params::FeatureMapParams)::Vector{CircuitOp}
    ops = CircuitOp[]

    # Feature → Braid
    bw = feature_to_braid(features, engine.n_strands)

    # Free reduction
    if engine.use_markov
        bw = free_reduce(bw)
    end

    # Braid → circuit ops
    for (gen, edge_idx) in zip(bw.generators, bw.edge_indices)
        if edge_idx > length(HERON_EDGES_0)
            continue
        end
        q1, q2 = HERON_EDGES_0[edge_idx]
        if q1 >= engine.n_qubits || q2 >= engine.n_qubits
            continue
        end

        if gen > 0
            push!(ops, CircuitOp("H", [q2], Float64[]))
            push!(ops, CircuitOp("CX", [q1, q2], Float64[]))
            push!(ops, CircuitOp("H", [q2], Float64[]))
            push!(ops, CircuitOp("CX", [q1, q2], Float64[]))
            push!(ops, CircuitOp("H", [q2], Float64[]))
        else
            push!(ops, CircuitOp("H", [q2], Float64[]))
            push!(ops, CircuitOp("CX", [q2, q1], Float64[]))
            push!(ops, CircuitOp("H", [q2], Float64[]))
            push!(ops, CircuitOp("CX", [q2, q1], Float64[]))
            push!(ops, CircuitOp("H", [q2], Float64[]))
        end
    end

    # Variational layers
    for layer in 1:engine.n_layers
        for q in 0:engine.n_qubits-1
            θz1 = params[layer, q+1, 1]
            θy = params[layer, q+1, 2]
            θz2 = params[layer, q+1, 3]
            push!(ops, CircuitOp("Rz", [q], [θz1]))
            push!(ops, CircuitOp("Ry", [q], [θy]))
            push!(ops, CircuitOp("Rz", [q], [θz2]))
        end

        # Entangling on heavy-hex
        for (q1, q2) in HERON_EDGES_0
            if q1 < engine.n_qubits && q2 < engine.n_qubits
                push!(ops, CircuitOp("CZ", [q1, q2], Float64[]))
            end
        end
    end

    return ops
end

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

function feature_to_braid(features::Vector{Float64}, n_strands::Int;
                           epsilon::Float64=0.5)::BraidWord
    generators = Int[]
    edge_indices = Int[]
    n_gens = min(n_strands - 1, length(HERON_EDGES_0))

    for (i, f) in enumerate(features)
        if abs(f) < epsilon
            continue
        end
        gen_idx = (i - 1) % n_gens + 1
        edge = HERON_EDGES_0[gen_idx]
        edge_idx = HERON_EDGE_INDEX[edge]
        sign = f > 0 ? 1 : -1
        repeats = min(max(1, Int(round(abs(f) * 2))), 3)
        for _ in 1:repeats
            push!(generators, sign * gen_idx)
            push!(edge_indices, edge_idx)
        end
    end

    isempty(generators) ? BraidWord(n_strands) : BraidWord(generators, edge_indices, n_strands)
end

function free_reduce(bw::BraidWord)::BraidWord
    stack = Tuple{Int,Int}[]
    for (gen, edge) in zip(bw.generators, bw.edge_indices)
        if !isempty(stack) && stack[end] == (-gen, edge)
            pop!(stack)
        else
            push!(stack, (gen, edge))
        end
    end
    gens = [s[1] for s in stack]
    edges = [s[2] for s in stack]
    BraidWord(gens, edges, bw.n_strands)
end

# ═══════════════════════════════════════════════════════════════════════
# Kernel Matrix Computation
# ═══════════════════════════════════════════════════════════════════════

"""
    compute_braid_kernel_matrix(engine, dataset, params, anu_bases)

Compute K_ij = |⟨0|U_Φ(x_i) U_Φ(x_j)†|0⟩|² using DFE protocol.
"""
function compute_braid_kernel_matrix(engine::BraidKernelEngine,
                                      dataset::Vector{Vector{Float64}},
                                      params::FeatureMapParams,
                                      anu_bases::Vector{Vector{Char}})::Matrix{Float64}
    n = length(dataset)
    K = Matrix{Float64}(undef, n, n)

    for i in 1:n
        for j in i:n
            ops_i = build_braid_feature_map_ops(engine, dataset[i], params)
            ops_j = build_braid_feature_map_ops(engine, dataset[j], params)

            # DFE fidelity estimation (placeholder — real execution in Rust)
            braid_i = feature_to_braid(dataset[i], engine.n_strands)
            braid_j = feature_to_braid(dataset[j], engine.n_strands)

            # Topological distance: shorter combined braid = higher kernel
            combined = free_reduce(BraidWord(
                vcat(braid_i.generators, reverse(-braid_j.generators)),
                vcat(braid_i.edge_indices, reverse(braid_j.edge_indices)),
                engine.n_strands
            ))
            complexity = length(combined.generators)
            fidelity = exp(-0.1 * complexity)

            K[i,j] = fidelity
            K[j,i] = fidelity
        end
    end

    return K
end

end # module BraidKernelIntegration
