# yao_types.jl
#
# Core type definitions for Yao.jl circuit representation.
# Standalone types — no external Yao dependency required.

abstract type AbstractBlock end

struct Rz <: AbstractBlock
    theta::Float64
end

struct Ry <: AbstractBlock
    theta::Float64
end

struct Rx <: AbstractBlock
    theta::Float64
end

struct H <: AbstractBlock end
struct X <: AbstractBlock end
struct Y <: AbstractBlock end
struct Z <: AbstractBlock end
struct S <: AbstractBlock end
struct Sdg <: AbstractBlock end
struct T <: AbstractBlock end
struct Tdg <: AbstractBlock end
struct SX <: AbstractBlock end
struct CX <: AbstractBlock end
struct CZ <: AbstractBlock end
struct CCX <: AbstractBlock end
struct Swap <: AbstractBlock end

struct ChainBlock <: AbstractBlock
    n::Int
    blocks::Vector{AbstractBlock}
end

struct KronBlock <: AbstractBlock
    n::Int
    locs_blocks::Vector{Pair{Int,AbstractBlock}}
end

struct PutBlock <: AbstractBlock
    n::Int
    locs::Vector{Int}
    content::AbstractBlock
end

struct ControlBlock <: AbstractBlock
    n::Int
    ctrl_locs::Vector{Int}
    target_loc::Int
    content::AbstractBlock
end

struct MeasureBlock <: AbstractBlock
    n::Int
    locs::Vector{Int}
end

# Constructors matching Yao.jl API

function chain(n::Int, blocks::AbstractBlock...)
    ChainBlock(n, collect(blocks))
end

function chain(n::Int, blocks::Vector{<:AbstractBlock})
    ChainBlock(n, blocks)
end

function kron(n::Int, pairs::Pair{Int,AbstractBlock}...)
    KronBlock(n, collect(pairs))
end

function put(n::Int, locs::Vector{Int}, block::AbstractBlock)
    PutBlock(n, locs, block)
end

function control(n::Int, ctrl_locs::Vector{Int}, target_pair::Pair{Int,AbstractBlock})
    ControlBlock(n, ctrl_locs, target_pair.first, target_pair.second)
end

function measure(n::Int, locs::Vector{Int})
    MeasureBlock(n, locs)
end
