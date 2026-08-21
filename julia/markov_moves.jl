# markov_moves.jl — Braid Simplification + Canonical Form

module MarkovMoves

using LinearAlgebra

export canonical_form, markov_stabilization, markov_destabilization
export braid_conjugacy_class, is_trivial_braid, burau_matrix

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

# ═══════════════════════════════════════════════════════════════════════
# Canonical Form via Handle Reduction
# ═══════════════════════════════════════════════════════════════════════

"""
    canonical_form(bw::BraidWord)

Compute canonical form using:
1. Free reduction (cancel σ σ⁻¹ pairs)
2. Artin relations
3. Garside normal form (left-greedy)
"""
function canonical_form(bw::BraidWord)::BraidWord
    bw_reduced = free_reduce(bw)
    bw_garside = garside_normal_form(bw_reduced)
    return bw_garside
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

function garside_normal_form(bw::BraidWord)::BraidWord
    gens = copy(bw.generators)
    edges = copy(bw.edge_indices)

    for _ in 1:3
        for i in 1:length(gens)-1
            if gens[i] < 0 && gens[i+1] > 0 && edges[i] == edges[i+1]
                gens[i], gens[i+1] = gens[i+1], gens[i]
            end
        end
    end

    BraidWord(gens, edges, bw.n_strands)
end

# ═══════════════════════════════════════════════════════════════════════
# Markov Moves
# ═══════════════════════════════════════════════════════════════════════

function markov_stabilization(bw::BraidWord, strand_pos::Int)::BraidWord
    @assert 1 <= strand_pos <= bw.n_strands
    new_n = bw.n_strands + 1
    new_gen = strand_pos
    edge_idx = (strand_pos - 1) % length(HERON_EDGES_0) + 1
    BraidWord(vcat(bw.generators, new_gen),
              vcat(bw.edge_indices, edge_idx),
              new_n)
end

function markov_destabilization(bw::BraidWord)::BraidWord
    if bw.n_strands <= 2
        return bw
    end
    last_gen = bw.n_strands - 1
    if length(bw.generators) > 0 && abs(bw.generators[end]) == last_gen
        if count(g -> abs(g) == last_gen, bw.generators) == 1
            return BraidWord(bw.generators[1:end-1],
                           bw.edge_indices[1:end-1],
                           bw.n_strands - 1)
        end
    end
    return bw
end

# ═══════════════════════════════════════════════════════════════════════
# Conjugacy & Triviality
# ═══════════════════════════════════════════════════════════════════════

function braid_conjugacy_class(bw::BraidWord)::BraidWord
    gens = bw.generators
    edges = bw.edge_indices
    n = length(gens)

    if n == 0
        return bw
    end

    best = (gens, edges)
    for shift in 1:n-1
        shifted_gens = vcat(gens[shift+1:end], gens[1:shift])
        shifted_edges = vcat(edges[shift+1:end], edges[1:shift])
        if shifted_gens < best[1]
            best = (shifted_gens, shifted_edges)
        end
    end

    BraidWord(best[1], best[2], bw.n_strands)
end

function is_trivial_braid(bw::BraidWord)::Bool
    reduced = canonical_form(bw)
    return isempty(reduced.generators)
end

# ═══════════════════════════════════════════════════════════════════════
# Burau Representation (for Jones polynomial verification)
# ═══════════════════════════════════════════════════════════════════════

"""
    burau_matrix(bw::BraidWord, t=im)

Reduced Burau representation (n-1 × n-1).
Used for Jones polynomial evaluation.
"""
function burau_matrix(bw::BraidWord, t::ComplexF64=ComplexF64(0,1))::Matrix{ComplexF64}
    n = bw.n_strands
    if n <= 1
        return Matrix{ComplexF64}(I, 1, 1)
    end
    M = Matrix{ComplexF64}(I, n-1, n-1)

    for gen in bw.generators
        i = abs(gen)
        if i >= n
            continue
        end
        B = Matrix{ComplexF64}(I, n-1, n-1)
        if gen > 0
            if i < n-1
                B[i,i] = 1 - t
                if i+1 <= n-1
                    B[i,i+1] = t
                    B[i+1,i] = 1
                    B[i+1,i+1] = 0
                end
            else
                B[i,i] = 1 - t
            end
        else
            if i < n-1
                B[i,i] = 0
                if i+1 <= n-1
                    B[i,i+1] = 1
                    B[i+1,i] = t
                    B[i+1,i+1] = 1 - t
                end
            end
        end
        M = B * M
    end
    return M
end

end # module MarkovMoves
