# lattice_surgery.jl — Map CZ Layers to Smooth/Rough Defect Operations

module LatticeSurgery

using LinearAlgebra

export LatticeSurgeryOp, DefectPair, DefectTracker
export logical_cz, defect_braid_to_circuit_ops, syndrome_extraction_ops
export allocate_defect_pair!, braid_defects!

# ═══════════════════════════════════════════════════════════════════════
# Defect Types
# ═══════════════════════════════════════════════════════════════════════

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
end

function DefectTracker(lattice_size::Tuple{Int,Int}=(20,20))
    rules = Dict(
        (:fibonacci, :fibonacci) => [:vacuum, :fibonacci],
        (:ising, :ising) => [:vacuum, :fermion],
        (:toric, :toric) => [:vacuum],
    )
    DefectTracker(Dict{String, DefectPair}(), rules, lattice_size, 0)
end

struct LatticeSurgeryOp
    op_type::Symbol
    defect_ids::Vector{String}
    basis::Symbol
    ancilla_id::Union{String, Nothing}
end

# ═══════════════════════════════════════════════════════════════════════
# Defect Allocation & Braiding
# ═══════════════════════════════════════════════════════════════════════

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
# Logical CZ via Lattice Surgery
# ═══════════════════════════════════════════════════════════════════════

"""
    logical_cz(tracker, id1, id2)

Implement logical CZ between two defect-encoded qubits:
1. Merge rough defects (Z-basis merge)
2. Measure joint Z operator
3. Split defects
"""
function logical_cz(tracker::DefectTracker, id1::String, id2::String)::Vector{LatticeSurgeryOp}
    ops = LatticeSurgeryOp[]

    push!(ops, LatticeSurgeryOp(:merge, [id1, id2], :Z, nothing))

    ancilla = "ancilla_$(id1)_$(id2)"
    allocate_defect_pair!(tracker, ancilla, :toric, (0,0), (0,0))
    push!(ops, LatticeSurgeryOp(:measure, [id1, id2, ancilla], :Z, ancilla))

    push!(ops, LatticeSurgeryOp(:split, [id1, id2], :Z, nothing))

    return ops
end

# ═══════════════════════════════════════════════════════════════════════
# Defect Braiding → Circuit Ops
# ═══════════════════════════════════════════════════════════════════════

const HERON_EDGES_0 = [
    (0, 1), (1, 2),
    (0, 3), (1, 3), (1, 4), (2, 4), (2, 5),
    (3, 4), (4, 5), (5, 6),
    (3, 7), (4, 7), (4, 8), (5, 8), (5, 9), (6, 9),
    (7, 8), (8, 9)
]

struct CircuitOp
    gate::String
    qubits::Vector{Int}
end

"""
    defect_braid_to_circuit_ops(bw_gens, bw_edges, n_qubits)

Compile braid word to physical circuit ops using defect trajectories.
Each braid generator → defect exchange via lattice surgery moves.
"""
function defect_braid_to_circuit_ops(generators::Vector{Int}, edge_indices::Vector{Int},
                                      n_qubits::Int)::Vector{CircuitOp}
    ops = CircuitOp[]

    for (gen, edge_idx) in zip(generators, edge_indices)
        if edge_idx > length(HERON_EDGES_0)
            continue
        end
        q1, q2 = HERON_EDGES_0[edge_idx]
        if q1 >= n_qubits || q2 >= n_qubits
            continue
        end

        if gen > 0
            push!(ops, CircuitOp("H", [q1]))
            push!(ops, CircuitOp("CX", [q1, q2]))
            push!(ops, CircuitOp("H", [q2]))
            push!(ops, CircuitOp("CX", [q2, q1]))
            push!(ops, CircuitOp("H", [q1]))
            push!(ops, CircuitOp("CX", [q1, q2]))
            push!(ops, CircuitOp("H", [q2]))
        else
            push!(ops, CircuitOp("H", [q2]))
            push!(ops, CircuitOp("CX", [q2, q1]))
            push!(ops, CircuitOp("H", [q1]))
            push!(ops, CircuitOp("CX", [q1, q2]))
            push!(ops, CircuitOp("H", [q2]))
            push!(ops, CircuitOp("CX", [q2, q1]))
            push!(ops, CircuitOp("H", [q1]))
        end
    end

    return ops
end

# ═══════════════════════════════════════════════════════════════════════
# Syndrome Extraction
# ═══════════════════════════════════════════════════════════════════════

function syndrome_extraction_ops(tracker::DefectTracker, basis::Symbol=:Z)::Vector{CircuitOp}
    ops = CircuitOp[]

    for (id, defect) in tracker.defects
        q = basis == :Z ? defect.rough_defect[1] : defect.smooth_defect[1]
        push!(ops, CircuitOp("H", [q]))
        push!(ops, CircuitOp("CX", [q, q+1]))
        push!(ops, CircuitOp("H", [q]))
        push!(ops, CircuitOp("MEASURE", [q]))
    end

    return ops
end

end # module LatticeSurgery
