# braid_diff.jl — Differentiable Artin Generators on Heavy-Hex

module BraidDiff

using LinearAlgebra
using Random

export BraidWord, braid_to_circuit, gumbel_softmax_braid, markov_loss
export apply_braid_relations

# ═══════════════════════════════════════════════════════════════════════
# Types
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

# ═══════════════════════════════════════════════════════════════════════
# Braid Word Operations
# ═══════════════════════════════════════════════════════════════════════

Base.length(bw::BraidWord) = length(bw.generators)

function Base.:(==)(bw1::BraidWord, bw2::BraidWord)
    bw1.generators == bw2.generators && bw1.edge_indices == bw2.edge_indices
end

function Base.hash(bw::BraidWord, h::UInt)
    hash(bw.generators, hash(bw.edge_indices, hash(bw.n_strands, h)))
end

function Base.inv(bw::BraidWord)::BraidWord
    BraidWord(reverse(-bw.generators), reverse(bw.edge_indices), bw.n_strands)
end

function Base.:*(bw1::BraidWord, bw2::BraidWord)::BraidWord
    @assert bw1.n_strands == bw2.n_strands
    BraidWord(vcat(bw1.generators, bw2.generators),
              vcat(bw1.edge_indices, bw2.edge_indices),
              bw1.n_strands)
end

# ═══════════════════════════════════════════════════════════════════════
# Braid → Circuit (CX/H sequences on Heron edges)
# ═══════════════════════════════════════════════════════════════════════

struct BraidCircuitOp
    gate::String
    qubits::Vector{Int}
    params::Vector{Float64}
end

"""
    braid_to_circuit_ops(bw::BraidWord, n_qubits::Int) -> Vector{BraidCircuitOp}

Map Artin generators to SWAP/CX sequences on Heron edges.
σ_i → H(t) · CX(c,t) · H(t) · CX(c,t) · H(t)
σ_i⁻¹ → inverse sequence
"""
function braid_to_circuit_ops(bw::BraidWord, n_qubits::Int)::Vector{BraidCircuitOp}
    ops = BraidCircuitOp[]

    for (gen, edge_idx) in zip(bw.generators, bw.edge_indices)
        if edge_idx > length(HERON_EDGES_0)
            continue
        end
        q1, q2 = HERON_EDGES_0[edge_idx]
        if q1 >= n_qubits || q2 >= n_qubits
            continue
        end

        if gen > 0
            push!(ops, BraidCircuitOp("H", [q2], Float64[]))
            push!(ops, BraidCircuitOp("CX", [q1, q2], Float64[]))
            push!(ops, BraidCircuitOp("H", [q2], Float64[]))
            push!(ops, BraidCircuitOp("CX", [q1, q2], Float64[]))
            push!(ops, BraidCircuitOp("H", [q2], Float64[]))
        else
            push!(ops, BraidCircuitOp("H", [q2], Float64[]))
            push!(ops, BraidCircuitOp("CX", [q2, q1], Float64[]))
            push!(ops, BraidCircuitOp("H", [q2], Float64[]))
            push!(ops, BraidCircuitOp("CX", [q2, q1], Float64[]))
            push!(ops, BraidCircuitOp("H", [q2], Float64[]))
        end
    end

    return ops
end

# ═══════════════════════════════════════════════════════════════════════
# Gumbel-Softmax Braid (Differentiable Selection)
# ═══════════════════════════════════════════════════════════════════════

"""
    gumbel_softmax_braid(logits, τ=1.0)

Differentiable braid generator selection via Gumbel-Softmax.
logits: [n_generators, n_positions]
"""
function gumbel_softmax_braid(logits::Matrix{Float64}, τ::Float64=1.0)::BraidWord
    n_gens, n_pos = size(logits)
    generators = Int[]
    edge_indices = Int[]

    for pos in 1:n_pos
        gumbel = -log.(-log.(rand(n_gens) .+ 1e-20) .+ 1e-20)
        y = (logits[:, pos] .+ gumbel) ./ τ
        y_max = maximum(y)
        probs = exp.(y .- y_max) ./ sum(exp.(y .- y_max))

        gen_idx = argmax(probs)
        sign = rand() < 0.5 ? 1 : -1

        push!(generators, sign * gen_idx)
        if gen_idx <= length(HERON_EDGES_0)
            edge = HERON_EDGES_0[gen_idx]
            push!(edge_indices, HERON_EDGE_INDEX[edge])
        else
            push!(edge_indices, 1)
        end
    end

    BraidWord(generators, edge_indices, n_gens + 1)
end

# ═══════════════════════════════════════════════════════════════════════
# Markov Loss
# ═══════════════════════════════════════════════════════════════════════

function markov_loss(bw::BraidWord, kernel_fidelity::Float64, gate_count::Int;
                      λ_length::Float64=0.01, λ_gates::Float64=0.001)::Float64
    length_penalty = λ_length * length(bw)
    gate_penalty = λ_gates * gate_count
    return -kernel_fidelity + length_penalty + gate_penalty
end

# ═══════════════════════════════════════════════════════════════════════
# Braid Group Relations (Artin Presentation)
# ═══════════════════════════════════════════════════════════════════════

function shares_vertex(e1::Int, e2::Int)::Bool
    if e1 > length(HERON_EDGES_0) || e2 > length(HERON_EDGES_0)
        return false
    end
    q1a, q1b = HERON_EDGES_0[e1]
    q2a, q2b = HERON_EDGES_0[e2]
    return q1a == q2a || q1a == q2b || q1b == q2a || q1b == q2b
end

"""
    apply_braid_relations(bw::BraidWord)

Apply Artin relations:
1. σ_i σ_j = σ_j σ_i for |i-j| > 1 (far commutativity)
2. σ_i σ_{i+1} σ_i = σ_{i+1} σ_i σ_{i+1} (braid relation)
"""
function apply_braid_relations(bw::BraidWord)::BraidWord
    gens = copy(bw.generators)
    edges = copy(bw.edge_indices)
    changed = true

    while changed
        changed = false
        i = 1
        while i <= length(gens) - 1
            e1, e2 = edges[i], edges[i+1]

            if !shares_vertex(e1, e2)
                gens[i], gens[i+1] = gens[i+1], gens[i]
                edges[i], edges[i+1] = edges[i+1], edges[i]
                changed = true
                i += 1
            elseif shares_vertex(e1, e2) && i <= length(gens) - 2
                g1, g3 = gens[i], gens[i+2]
                e3 = edges[i+2]
                if g1 == g3 && e1 == e3
                    g2 = gens[i+1]
                    gens[i], gens[i+1], gens[i+2] = g2, g1, g2
                    edges[i], edges[i+1], edges[i+2] = e2, e1, e2
                    changed = true
                    i += 2
                else
                    i += 1
                end
            else
                i += 1
            end
        end
    end

    BraidWord(gens, edges, bw.n_strands)
end

end # module BraidDiff
