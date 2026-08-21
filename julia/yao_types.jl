# yao_types.jl — Core Yao.jl block hierarchy + Topological types

module YaoTypes

using LinearAlgebra

# ═══════════════════════════════════════════════════════════════════════
# Abstract Block Hierarchy
# ═══════════════════════════════════════════════════════════════════════

abstract type AbstractBlock end
abstract type CompositeBlock <: AbstractBlock end

struct PrimitiveGate <: AbstractBlock
    name::String
    params::Vector{Float64}
    nqubits::Int
    mat::Matrix{ComplexF64}
end

struct ChainBlock <: CompositeBlock
    nqubits::Int
    blocks::Vector{AbstractBlock}
end

struct KronBlock <: CompositeBlock
    nqubits::Int
    locs::Vector{Int}
    blocks::Vector{AbstractBlock}
end

struct PutBlock <: CompositeBlock
    nqubits::Int
    locs::Vector{Int}
    block::AbstractBlock
end

struct ControlBlock <: CompositeBlock
    nqubits::Int
    ctrl_locs::Vector{Int}
    ctrl_bits::Vector{Int}
    block::AbstractBlock
end

struct MeasureBlock <: AbstractBlock
    nqubits::Int
    locs::Vector{Int}
end

# ═══════════════════════════════════════════════════════════════════════
# TOPOLOGICAL TYPES: Braids, Defects, Anyons
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

struct DefectPair
    id::String
    anyon_type::Symbol
    smooth_defect::Tuple{Int,Int}
    rough_defect::Tuple{Int,Int}
    braid_trajectory::Vector{Tuple{Int,Int}}
end

mutable struct DefectTracker
    defects::Dict{String, DefectPair}
    fusion_rules::Dict{Tuple{Symbol,Symbol}, Vector{Symbol}}
    lattice_size::Tuple{Int,Int}
    time_step::Int

    function DefectTracker(lattice_size::Tuple{Int,Int}=(20,20))
        rules = Dict(
            (:fibonacci, :fibonacci) => [:vacuum, :fibonacci],
            (:ising, :ising) => [:vacuum, :fermion],
            (:toric, :toric) => [:vacuum],
        )
        new(Dict{String, DefectPair}(), rules, lattice_size, 0)
    end
end

struct LatticeSurgeryOp
    op_type::Symbol
    defect_ids::Vector{String}
    basis::Symbol
    ancilla_id::Union{String, Nothing}
end

function allocate_defect_pair!(tracker::DefectTracker, id::String, anyon_type::Symbol,
                                smooth_pos::Tuple{Int,Int}, rough_pos::Tuple{Int,Int})
    pair = DefectPair(id, anyon_type, smooth_pos, rough_pos, [smooth_pos, rough_pos])
    tracker.defects[id] = pair
    return pair
end

function braid_defects!(tracker::DefectTracker, id1::String, id2::String, direction::Int)
    d1 = tracker.defects[id1]
    d2 = tracker.defects[id2]
    new_traj1 = vcat(d1.braid_trajectory, [d2.rough_defect])
    new_traj2 = vcat(d2.braid_trajectory, [d1.rough_defect])
    tracker.defects[id1] = DefectPair(d1.id, d1.anyon_type, d1.smooth_defect,
                                       d2.rough_defect, new_traj1)
    tracker.defects[id2] = DefectPair(d2.id, d2.anyon_type, d2.smooth_defect,
                                       d1.rough_defect, new_traj2)
    tracker.time_step += 1
end

# ═══════════════════════════════════════════════════════════════════════
# Accessors & Constructors
# ═══════════════════════════════════════════════════════════════════════

nqubits(b::AbstractBlock) = b.nqubits
nqubits(b::PrimitiveGate) = b.nqubits

const SQRT2 = sqrt(2.0)
const IM = ComplexF64(0, 1)

H() = PrimitiveGate("H", Float64[], 1, ComplexF64[1 1; 1 -1] / SQRT2)
X() = PrimitiveGate("X", Float64[], 1, ComplexF64[0 1; 1 0])
Y() = PrimitiveGate("Y", Float64[], 1, ComplexF64[0 -IM; IM 0])
Z() = PrimitiveGate("Z", Float64[], 1, ComplexF64[1 0; 0 -1])
S() = PrimitiveGate("S", Float64[], 1, ComplexF64[1 0; 0 IM])
Sdg() = PrimitiveGate("Sdg", Float64[], 1, ComplexF64[1 0; 0 -IM])
T() = PrimitiveGate("T", Float64[], 1, ComplexF64[1 0; 0 exp(IM*π/4)])
Tdg() = PrimitiveGate("Tdg", Float64[], 1, ComplexF64[1 0; 0 exp(-IM*π/4)])
SX() = PrimitiveGate("SX", Float64[], 1, ComplexF64[(1+IM)/2 (1-IM)/2; (1-IM)/2 (1+IM)/2])

Rx(θ) = PrimitiveGate("Rx", [θ], 1, ComplexF64[cos(θ/2) -IM*sin(θ/2); -IM*sin(θ/2) cos(θ/2)])
Ry(θ) = PrimitiveGate("Ry", [θ], 1, ComplexF64[cos(θ/2) -sin(θ/2); sin(θ/2) cos(θ/2)])
Rz(θ) = PrimitiveGate("Rz", [θ], 1, ComplexF64[exp(-IM*θ/2) 0; 0 exp(IM*θ/2)])

CNOT() = PrimitiveGate("CX", Float64[], 2, ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0])
CZ() = PrimitiveGate("CZ", Float64[], 2, ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1])

function chain(nq::Int, blocks::AbstractBlock...)
    ChainBlock(nq, collect(blocks))
end
chain(nq::Int, blocks::Vector{<:AbstractBlock}) = ChainBlock(nq, blocks)

function kron(nq::Int, pairs::Pair{Int,<:AbstractBlock}...)
    locs = [p.first for p in pairs]
    blks = [p.second for p in pairs]
    KronBlock(nq, locs, blks)
end

put(nq::Int, locs::Vector{Int}, block::AbstractBlock) = PutBlock(nq, locs, block)

function control(nq::Int, ctrl_locs::Vector{Int}, target::Pair{Int,<:AbstractBlock})
    ControlBlock(nq, ctrl_locs, ones(Int, length(ctrl_locs)), PutBlock(nq, [target.first], target.second))
end
control(nq::Int, ctrl_locs::Vector{Int}, block::AbstractBlock) = ControlBlock(nq, ctrl_locs, ones(Int, length(ctrl_locs)), block)

measure(nq::Int, locs::Vector{Int}) = MeasureBlock(nq, locs)

# ═══════════════════════════════════════════════════════════════════════
# Heron-Native Decomposition
# ═══════════════════════════════════════════════════════════════════════

function decompose_to_heron(block::AbstractBlock)::AbstractBlock
    if block isa PrimitiveGate
        return _decompose_primitive(block)
    elseif block isa ChainBlock
        return ChainBlock(block.nqubits, decompose_to_heron.(block.blocks))
    elseif block isa KronBlock
        return KronBlock(block.nqubits, block.locs, decompose_to_heron.(block.blocks))
    elseif block isa PutBlock
        return PutBlock(block.nqubits, block.locs, decompose_to_heron(block.block))
    elseif block isa ControlBlock
        return ControlBlock(block.nqubits, block.ctrl_locs, block.ctrl_bits, decompose_to_heron(block.block))
    elseif block isa MeasureBlock
        return block
    else
        error("Unknown block type: $(typeof(block))")
    end
end

function _decompose_primitive(g::PrimitiveGate)::AbstractBlock
    name = g.name
    params = g.params

    if name in ("Rz", "SX", "CX", "CNOT")
        return g
    elseif name == "Ry"
        θ = params[1]
        return chain(1, Rz(π/2), SX(), Rz(θ), SX(), Rz(-π/2))
    elseif name == "Rx"
        θ = params[1]
        return chain(1, Rz(-π/2), SX(), Rz(θ), SX(), Rz(π/2))
    elseif name == "H"
        return chain(1, Rz(π/2), SX(), Rz(π/2), SX(), Rz(π/2))
    elseif name == "X"
        return chain(1, SX(), SX())
    elseif name == "Y"
        return chain(1, SX(), Rz(π), SX())
    elseif name == "Z"
        return Rz(π)
    elseif name == "S"
        return Rz(π/2)
    elseif name == "Sdg"
        return Rz(-π/2)
    elseif name == "T"
        return Rz(π/4)
    elseif name == "Tdg"
        return Rz(-π/4)
    elseif name == "CZ"
        return g
    else
        return g
    end
end

# ═══════════════════════════════════════════════════════════════════════
# Feature Map Parameters
# ═══════════════════════════════════════════════════════════════════════

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
Base.setindex!(p::FeatureMapParams, v, i...) = (p.data[i...] = v)
Base.size(p::FeatureMapParams) = size(p.data)
Base.length(p::FeatureMapParams) = length(p.data)
Base.eachindex(p::FeatureMapParams) = eachindex(p.data)

# ═══════════════════════════════════════════════════════════════════════
# Pauli String & Heavy-Hex Topology
# ═══════════════════════════════════════════════════════════════════════

struct PauliString
    paulis::Vector{Char}
end
PauliString(n::Int) = PauliString(rand(['I','X','Y','Z'], n))

const HERON_EDGES_0 = [
    (0, 1), (1, 2),
    (0, 3), (1, 3), (1, 4), (2, 4), (2, 5),
    (3, 4), (4, 5), (5, 6),
    (3, 7), (4, 7), (4, 8), (5, 8), (5, 9), (6, 9),
    (7, 8), (8, 9)
]

const HERON_EDGE_INDEX = Dict(edge => i for (i, edge) in enumerate(HERON_EDGES_0))

# ═══════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════

export AbstractBlock, PrimitiveGate, ChainBlock, KronBlock, PutBlock, ControlBlock, MeasureBlock
export nqubits
export H, X, Y, Z, S, Sdg, T, Tdg, SX, Rx, Ry, Rz, CNOT, CZ
export chain, kron, put, control, measure
export decompose_to_heron
export FeatureMapParams, PauliString
export HERON_EDGES_0, HERON_EDGE_INDEX
export BraidWord, DefectPair, DefectTracker, LatticeSurgeryOp
export allocate_defect_pair!, braid_defects!

end # module YaoTypes
