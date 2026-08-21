# yao_kernel.jl
#
# Complete Yao.jl circuit construction for Quantum Kernel Architecture:
# Feature Map U_Φ(x) + Inverse U_Φ(x')† + DFE Measurement + Classical Feedforward

include("yao_types.jl")
include("yao_circuit.jl")
include("yao_to_ir.jl")

using LinearAlgebra
using Random

# -----------------------------------------------------------------------
# Heron Topology & Qubit Mapping
# -----------------------------------------------------------------------

"""
    HERON_HEAVY_HEX_EDGES

Heavy-hex connectivity for 10-qubit subset (from diagram):
q0-q1-q2
|/|/|/|
q3 q4 q5 q6
|\\|\\|\\|
q7-q8-q9
"""
const HERON_HEAVY_HEX_EDGES = [
    (1,2), (2,3),
    (1,4), (2,4), (2,5), (3,5), (3,6),
    (4,5), (5,6), (6,7),
    (4,8), (5,8), (5,9), (6,9), (6,10),
    (8,9), (9,10)
]

const HERON_EDGES_0 = [(a-1, b-1) for (a,b) in HERON_HEAVY_HEX_EDGES]

# -----------------------------------------------------------------------
# Feature Map Parameters
# -----------------------------------------------------------------------

"""
    FeatureMapParams

Trainable parameters θ for feature map.
Shape: (n_layers, n_qubits, 3) for [θz1, θy, θz2] per qubit per layer.
"""
struct FeatureMapParams
    data::Array{Float64,3}
end

function FeatureMapParams(n_layers::Int, n_qubits::Int; init_scale::Float64=0.1)
    data = randn(n_layers, n_qubits, 3) * init_scale .+ 1.0
    return FeatureMapParams(data)
end

Base.getindex(p::FeatureMapParams, layer::Int, qubit::Int, param::Int) = p.data[layer, qubit, param]
Base.setindex!(p::FeatureMapParams, val, layer::Int, qubit::Int, param::Int) = p.data[layer, qubit, param] = val

# -----------------------------------------------------------------------
# Single-Qubit Data Encoding Block
# -----------------------------------------------------------------------

"""
    data_encoding_block(qubit::Int, x::Float64, θz1::Float64, θy::Float64, θz2::Float64)

R_Z(2xθz1) · R_Y(2xθy) · R_Z(2xθz2) on a single qubit.
"""
function data_encoding_block(qubit::Int, x::Float64, θz1::Float64, θy::Float64, θz2::Float64)
    return chain(1,
        put(1, [1], Rz(2x * θz1)),
        put(1, [1], Ry(2x * θy)),
        put(1, [1], Rz(2x * θz2))
    )
end

# -----------------------------------------------------------------------
# Feature Map U_Φ(x) Construction
# -----------------------------------------------------------------------

"""
    build_feature_map(n_qubits, n_layers, features, params, ent_edges)

Build U_Φ(x) = ∏_l [U_ent · U_rot(x)] as a Yao.jl ChainBlock.
"""
function build_feature_map(n_qubits::Int, n_layers::Int,
                           features::Vector{Float64},
                           params::FeatureMapParams,
                           ent_edges::Vector{Tuple{Int,Int}}=HERON_EDGES_0)::ChainBlock

    layers = AbstractBlock[]

    for layer in 0:n_layers-1
        # Parallel single-qubit data encoding (KronBlock = true parallelism)
        encoding_blocks = Pair{Int,AbstractBlock}[]
        for q in 0:n_qubits-1
            x = features[(q % length(features)) + 1]
            θz1 = params[layer+1, q+1, 1]
            θy = params[layer+1, q+1, 2]
            θz2 = params[layer+1, q+1, 3]

            enc_block = data_encoding_block(1, x, θz1, θy, θz2)
            push!(encoding_blocks, q+1 => enc_block)
        end
        push!(layers, kron(n_qubits, encoding_blocks...))

        # Entangling layer on heavy-hex edges (sequential)
        ent_blocks = AbstractBlock[]
        for (q1, q2) in ent_edges
            if q1 < n_qubits && q2 < n_qubits
                cz_block = control(n_qubits, [q1+1], q2+1 => Z())
                push!(ent_blocks, cz_block)
            end
        end
        if !isempty(ent_blocks)
            push!(layers, chain(n_qubits, ent_blocks...))
        end
    end

    return chain(n_qubits, layers...)
end

# -----------------------------------------------------------------------
# Inverse Feature Map U_Φ(x)†
# -----------------------------------------------------------------------

"""
    build_inverse_feature_map(n_qubits, n_layers, features, params, ent_edges)

Build U_Φ(x)† = ∏_l [U_ent† · U_rot(x)†] with reversed layer order and negative angles.
"""
function build_inverse_feature_map(n_qubits::Int, n_layers::Int,
                                   features::Vector{Float64},
                                   params::FeatureMapParams,
                                   ent_edges::Vector{Tuple{Int,Int}}=HERON_EDGES_0)::ChainBlock

    layers = AbstractBlock[]

    for layer in n_layers-1:-1:0
        # Entangling layer (CZ is self-adjoint)
        ent_blocks = AbstractBlock[]
        for (q1, q2) in ent_edges
            if q1 < n_qubits && q2 < n_qubits
                cz_block = control(n_qubits, [q1+1], q2+1 => Z())
                push!(ent_blocks, cz_block)
            end
        end
        if !isempty(ent_blocks)
            push!(layers, chain(n_qubits, ent_blocks...))
        end

        # Single-qubit adjoint: reverse order, negative angles
        encoding_blocks = Pair{Int,AbstractBlock}[]
        for q in n_qubits-1:-1:0
            x = features[(q % length(features)) + 1]
            θz1 = params[layer+1, q+1, 1]
            θy = params[layer+1, q+1, 2]
            θz2 = params[layer+1, q+1, 3]

            # Adjoint: RZ(-2xθz2) · RY(-2xθy) · RZ(-2xθz1)
            enc_block = chain(1,
                put(1, [1], Rz(-2x * θz2)),
                put(1, [1], Ry(-2x * θy)),
                put(1, [1], Rz(-2x * θz1))
            )
            push!(encoding_blocks, q+1 => enc_block)
        end
        push!(layers, kron(n_qubits, encoding_blocks...))
    end

    return chain(n_qubits, layers...)
end

# -----------------------------------------------------------------------
# DFE Measurement Circuit
# -----------------------------------------------------------------------

"""
    build_dfe_measurement(n_qubits, pauli_basis)

Build mid-circuit measurement in Pauli basis with conditional reset.
"""
function build_dfe_measurement(n_qubits::Int, pauli_basis::Vector{Char})::ChainBlock
    blocks = AbstractBlock[]

    # Pauli basis rotation
    rotation_blocks = Pair{Int,AbstractBlock}[]
    for (q, pauli) in enumerate(pauli_basis)
        if pauli == 'X'
            push!(rotation_blocks, q => chain(1, put(1, [1], H())))
        elseif pauli == 'Y'
            push!(rotation_blocks, q => chain(1, put(1, [1], Sdg()), put(1, [1], H())))
        end
    end
    if !isempty(rotation_blocks)
        push!(blocks, kron(n_qubits, rotation_blocks...))
    end

    # Mid-circuit measurement
    meas_locs = collect(1:n_qubits)
    push!(blocks, measure(n_qubits, meas_locs))

    # Conditional reset is handled in QASM emission (classical feedforward)
    # Yao.jl doesn't directly support classical feedforward in blocks

    return chain(n_qubits, blocks...)
end

# -----------------------------------------------------------------------
# Full DFE Kernel Circuit
# -----------------------------------------------------------------------

"""
    build_dfe_kernel_circuit(n_qubits, n_layers, features_a, features_b, params, pauli_basis, ent_edges)

Build complete DFE kernel circuit for one shot:
U_Φ(x) · U_Φ(x')† · Pauli_rotation · Measure
"""
function build_dfe_kernel_circuit(n_qubits::Int, n_layers::Int,
                                  features_a::Vector{Float64},
                                  features_b::Vector{Float64},
                                  params::FeatureMapParams,
                                  pauli_basis::Vector{Char},
                                  ent_edges::Vector{Tuple{Int,Int}}=HERON_EDGES_0)::ChainBlock

    blocks = AbstractBlock[]

    # U_Φ(x)
    push!(blocks, build_feature_map(n_qubits, n_layers, features_a, params, ent_edges))

    # U_Φ(x')†
    push!(blocks, build_inverse_feature_map(n_qubits, n_layers, features_b, params, ent_edges))

    # Pauli basis rotation + measurement
    push!(blocks, build_dfe_measurement(n_qubits, pauli_basis))

    return chain(n_qubits, blocks...)
end

# -----------------------------------------------------------------------
# Batch Kernel Matrix Circuit Generation
# -----------------------------------------------------------------------

"""
    generate_kernel_circuits(dataset, params, n_layers, shots_per_entry, anu_bases)

Generate Yao circuits for all kernel matrix entries with ANU QRNG bases.
Returns Dict mapping (i,j) -> Vector{ChainBlock} (one per shot).
"""
function generate_kernel_circuits(dataset::Vector{Vector{Float64}},
                                  params::FeatureMapParams,
                                  n_layers::Int,
                                  shots_per_entry::Int,
                                  anu_bases::Vector{Vector{Char}};
                                  ent_edges::Vector{Tuple{Int,Int}}=HERON_EDGES_0)

    n_samples = length(dataset)
    n_qubits = size(params.data, 2)
    circuits = Dict{Tuple{Int,Int}, Vector{ChainBlock}}()

    for i in 1:n_samples
        for j in i:n_samples
            shot_circuits = ChainBlock[]
            for shot in 1:shots_per_entry
                basis_idx = (i-1)*n_samples + (j-1)
                basis_idx = (basis_idx * shots_per_entry + shot - 1) % length(anu_bases) + 1
                basis = anu_bases[basis_idx]

                circuit = build_dfe_kernel_circuit(
                    n_qubits, n_layers, dataset[i], dataset[j], params, basis, ent_edges
                )
                push!(shot_circuits, circuit)
            end
            circuits[(i,j)] = shot_circuits
        end
    end

    return circuits
end

# -----------------------------------------------------------------------
# Lower All Circuits to QuantumIR
# -----------------------------------------------------------------------

"""
    lower_kernel_to_ir(circuits) -> Vector{Dict}

Lower all kernel circuits to QuantumIR JSON format.
"""
function lower_kernel_to_ir(circuits::Dict{Tuple{Int,Int}, Vector{ChainBlock}})
    ir_list = Dict{String,Any}[]

    for ((i,j), shot_circuits) in circuits
        for (shot, circuit) in enumerate(shot_circuits)
            ir = yao_to_ir(circuit)
            ir["metadata"]["kernel_entry"] = [i, j]
            ir["metadata"]["shot"] = shot
            push!(ir_list, ir)
        end
    end

    return ir_list
end

# -----------------------------------------------------------------------
# Example: Generate Kernel for Circles Dataset
# -----------------------------------------------------------------------

function generate_circles_dataset(n::Int; noise::Float64=0.1)
    X = Vector{Vector{Float64}}(undef, n)
    y = Vector{Float64}(undef, n)
    for i in 1:n
        r = rand()
        θ = rand() * 2π
        if i ≤ n÷2
            r = 0.5 + r * 0.3
            y[i] = -1.0
        else
            r = 1.0 + r * 0.3
            y[i] = 1.0
        end
        X[i] = [r * cos(θ), r * sin(θ)]
        X[i] .+= randn(2) * noise
    end
    return X, y
end

function demo_kernel_generation()
    # Dataset
    X, y = generate_circles_dataset(20, noise=0.1)

    # Parameters
    n_qubits = 4
    n_layers = 2
    shots = 100
    params = FeatureMapParams(n_layers, n_qubits)

    # ANU QRNG bases (simulated for demo)
    anu_bases = [rand(['I','X','Y','Z'], n_qubits) for _ in 1:10000]

    # Generate circuits
    circuits = generate_kernel_circuits(X, params, n_layers, shots, anu_bases)

    # Lower to QuantumIR
    ir_list = lower_kernel_to_ir(circuits)

    # Save
    open("kernel_ir.json", "w") do f
        JSON3.write(f, ir_list)
    end

    println("Generated $(length(ir_list)) QuantumIR circuits")
    println("Saved to kernel_ir.json")

    # Convert first circuit to OpenQASM 3.0
    first_ir = ir_list[1]
    qasm = qir_to_openqasm3(first_ir; zne_factors=[1.0, 1.5, 2.0, 3.0],
                            anu_bases=anu_bases[1:shots],
                            dynamic_shots=true)
    write("kernel.qasm3", qasm)
    println("Written kernel.qasm3")
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo_kernel_generation()
end
