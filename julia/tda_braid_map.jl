# tda_braid_map.jl — Barcodes → BraidWords on Heavy-Hex

module TDABraidMap

using LinearAlgebra
using Random

export barcode_to_braid_word, feature_diff_to_braid, heavy_hex_braid_generators
export pairwise_braid_words

# ═══════════════════════════════════════════════════════════════════════
# Types (imported from YaoTypes in full build)
# ═══════════════════════════════════════════════════════════════════════

struct BraidWord
    generators::Vector{Int}
    edge_indices::Vector{Int}
    n_strands::Int

    function BraidWord(gens::Vector{Int}, edges::Vector{Int}, n_strands::Int)
        @assert length(gens) == length(edges)
        new(gens, edges, n_strands)
    end
end

BraidWord(n_strands::Int) = BraidWord(Int[], Int[], n_strands)

struct PersistenceInterval
    dim::Int
    birth::Float64
    death::Float64
end

struct Barcode
    H0::Vector{PersistenceInterval}
    H1::Vector{PersistenceInterval}
end

const HERON_EDGES_0 = [
    (0, 1), (1, 2),
    (0, 3), (1, 3), (1, 4), (2, 4), (2, 5),
    (3, 4), (4, 5), (5, 6),
    (3, 7), (4, 7), (4, 8), (5, 8), (5, 9), (6, 9),
    (7, 8), (8, 9)
]

const HERON_EDGE_INDEX = Dict(edge => i for (i, edge) in enumerate(HERON_EDGES_0))

# ═══════════════════════════════════════════════════════════════════════
# Heavy-Hex Braid Generators
# ═══════════════════════════════════════════════════════════════════════

function heavy_hex_braid_generators(n_strands::Int)::Dict{Int, Tuple{Int,Int}}
    gens = Dict{Int, Tuple{Int,Int}}()
    for i in 1:min(n_strands-1, length(HERON_EDGES_0))
        gens[i] = HERON_EDGES_0[i]
    end
    return gens
end

# ═══════════════════════════════════════════════════════════════════════
# Barcode → Braid Word
# ═══════════════════════════════════════════════════════════════════════

"""
    barcode_to_braid_word(bc::Barcode, n_strands::Int; persistence_threshold=0.1)

Map persistent homology intervals to Artin generators.
High-persistence H1 features → over-crossings (σ)
Low-persistence / noise → under-crossings (σ⁻¹) or identity
"""
function barcode_to_braid_word(bc::Barcode, n_strands::Int;
                                persistence_threshold::Float64=0.1)::BraidWord
    generators = Int[]
    edge_indices = Int[]

    gens_map = heavy_hex_braid_generators(n_strands)
    n_gens = length(gens_map)

    for (idx, intv) in enumerate(bc.H1)
        pers = intv.death - intv.birth
        if pers < persistence_threshold
            continue
        end

        gen_idx = (idx - 1) % n_gens + 1
        edge = gens_map[gen_idx]
        edge_idx = HERON_EDGE_INDEX[edge]

        sign = (idx % 2 == 1) ? 1 : -1

        push!(generators, sign * gen_idx)
        push!(edge_indices, edge_idx)
    end

    if isempty(generators)
        return BraidWord(n_strands)
    end

    BraidWord(generators, edge_indices, n_strands)
end

"""
    feature_diff_to_braid(x, x′, n_strands; epsilon=0.5)

Direct mapping: feature difference Δ = x - x' → braid word.
K(x,x') = ⟨0|U_Φ(x) U_Φ(x')†|0⟩ where U_Φ encodes braid.
"""
function feature_diff_to_braid(x::Vector{Float64}, x′::Vector{Float64},
                                n_strands::Int; epsilon::Float64=0.5)::BraidWord
    Δ = x - x′
    generators = Int[]
    edge_indices = Int[]

    gens_map = heavy_hex_braid_generators(n_strands)
    n_gens = length(gens_map)

    for (i, δ) in enumerate(Δ)
        if abs(δ) < epsilon
            continue
        end

        gen_idx = (i - 1) % n_gens + 1
        edge = gens_map[gen_idx]
        edge_idx = HERON_EDGE_INDEX[edge]

        sign = δ > 0 ? 1 : -1

        repeats = min(max(1, Int(round(abs(δ) * 2))), 3)
        for _ in 1:repeats
            push!(generators, sign * gen_idx)
            push!(edge_indices, edge_idx)
        end
    end

    if isempty(generators)
        return BraidWord(n_strands)
    end

    BraidWord(generators, edge_indices, n_strands)
end

# ═══════════════════════════════════════════════════════════════════════
# Batch Operations
# ═══════════════════════════════════════════════════════════════════════

function pairwise_braid_words(X::Matrix{Float64}, n_strands::Int;
                               epsilon::Float64=0.5)::Matrix{BraidWord}
    n_samples = size(X, 2)
    braids = Matrix{BraidWord}(undef, n_samples, n_samples)
    for i in 1:n_samples, j in 1:n_samples
        braids[i,j] = feature_diff_to_braid(X[:,i], X[:,j], n_strands; epsilon=epsilon)
    end
    return braids
end

end # module TDABraidMap
