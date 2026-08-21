# yao_circuit.jl
#
# Circuit simulation utilities for standalone Yao-compatible execution.
# Statevector simulation without external dependencies.

include("yao_types.jl")

using LinearAlgebra

# -----------------------------------------------------------------------
# Statevector Simulation
# -----------------------------------------------------------------------

struct StateVector
    n_qubits::Int
    amplitudes::Vector{ComplexF64}
end

function StateVector(n_qubits::Int)
    amps = zeros(ComplexF64, 2^n_qubits)
    amps[1] = 1.0 + 0.0im
    return StateVector(n_qubits, amps)
end

function apply_gate!(sv::StateVector, gate::AbstractBlock, target::Int)
    mat = gate_matrix(gate)
    n = sv.n_qubits
    dim = 2^n

    for i in 0:dim-1
        if (i >> target) & 1 == 0
            j = i | (1 << target)
            a0 = sv.amplitudes[i+1]
            a1 = sv.amplitudes[j+1]
            sv.amplitudes[i+1] = mat[1,1] * a0 + mat[1,2] * a1
            sv.amplitudes[j+1] = mat[2,1] * a0 + mat[2,2] * a1
        end
    end
end

function apply_controlled!(sv::StateVector, gate::AbstractBlock, control::Int, target::Int)
    mat = gate_matrix(gate)
    n = sv.n_qubits
    dim = 2^n

    for i in 0:dim-1
        if ((i >> control) & 1 == 1) && ((i >> target) & 1 == 0)
            j = i | (1 << target)
            a0 = sv.amplitudes[i+1]
            a1 = sv.amplitudes[j+1]
            sv.amplitudes[i+1] = mat[1,1] * a0 + mat[1,2] * a1
            sv.amplitudes[j+1] = mat[2,1] * a0 + mat[2,2] * a1
        end
    end
end

# -----------------------------------------------------------------------
# Gate Matrices
# -----------------------------------------------------------------------

function gate_matrix(::H)
    s = 1.0/sqrt(2.0)
    ComplexF64[s s; s -s]
end

function gate_matrix(::X)
    ComplexF64[0 1; 1 0]
end

function gate_matrix(::Y)
    ComplexF64[0 -im; im 0]
end

function gate_matrix(::Z)
    ComplexF64[1 0; 0 -1]
end

function gate_matrix(::S)
    ComplexF64[1 0; 0 im]
end

function gate_matrix(::Sdg)
    ComplexF64[1 0; 0 -im]
end

function gate_matrix(::T)
    ComplexF64[1 0; 0 exp(im*π/4)]
end

function gate_matrix(::Tdg)
    ComplexF64[1 0; 0 exp(-im*π/4)]
end

function gate_matrix(::SX)
    ComplexF64[(1+im)/2 (1-im)/2; (1-im)/2 (1+im)/2]
end

function gate_matrix(g::Rz)
    θ = g.theta
    ComplexF64[exp(-im*θ/2) 0; 0 exp(im*θ/2)]
end

function gate_matrix(g::Ry)
    θ = g.theta
    c = cos(θ/2)
    s = sin(θ/2)
    ComplexF64[c -s; s c]
end

function gate_matrix(g::Rx)
    θ = g.theta
    c = cos(θ/2)
    s = sin(θ/2)
    ComplexF64[c -im*s; -im*s c]
end

# -----------------------------------------------------------------------
# Measurement
# -----------------------------------------------------------------------

function measure_qubit!(sv::StateVector, target::Int)::Int
    n = sv.n_qubits
    dim = 2^n

    prob_0 = 0.0
    for i in 0:dim-1
        if (i >> target) & 1 == 0
            prob_0 += abs2(sv.amplitudes[i+1])
        end
    end

    outcome = rand() < prob_0 ? 0 : 1

    # Collapse
    norm_factor = outcome == 0 ? sqrt(prob_0) : sqrt(1.0 - prob_0)
    for i in 0:dim-1
        bit = (i >> target) & 1
        if bit == outcome
            sv.amplitudes[i+1] /= norm_factor
        else
            sv.amplitudes[i+1] = 0.0 + 0.0im
        end
    end

    return outcome
end

function fidelity(sv1::StateVector, sv2::StateVector)::Float64
    return abs2(dot(sv1.amplitudes, sv2.amplitudes))
end
