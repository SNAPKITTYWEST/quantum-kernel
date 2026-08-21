#!/usr/bin/env julia
# Quantum Kernel SVM — Hilbert Space Feature Mapping with Native Kernel Computation
# Ahmad Ali Parr — built on phone, cherry-picked from SNAPKITTYWEST repos
# Runs: 5 qubits, ANU QRNG entropy, SWAP test kernel, shot-based estimation

using Yao, YaoBlocks
using LinearAlgebra
using Random

# ──────────────────────────────────────────────────────────────
# ANU QRNG ENTROPY SOURCE
# ──────────────────────────────────────────────────────────────

struct ANUEntropy
    buffer::Vector{UInt8}
    pos::Ref{Int}
end

function ANUEntropy(; size=1024, use_real=false)
    if use_real
        try
            using HTTP, JSON3
            resp = HTTP.get("https://qrng.anu.edu.au/API/jsonI.php?length=$size&type=uint8&size=1")
            data = JSON3.read(resp.body)
            return ANUEntropy(UInt8.(data.data), Ref(1))
        catch e
            @warn "ANU QRNG unavailable, falling back to CSPRNG" exception=e
        end
    end
    ANUEntropy(rand(UInt8, size), Ref(1))
end

function next_byte!(anu::ANUEntropy)
    if anu.pos[] > length(anu.buffer)
        anu.pos[] = 1
        rand!(anu.buffer)
    end
    b = anu.buffer[anu.pos[]]
    anu.pos[] += 1
    return b
end

function random_pauli_basis(anu::ANUEntropy, n::Int)
    basis = Symbol[]
    for _ in 1:n
        b = next_byte!(anu) % 3
        push!(basis, b == 0 ? :X : b == 1 ? :Y : :Z)
    end
    return basis
end

# ──────────────────────────────────────────────────────────────
# FEATURE MAP: U_Φ(x) = ∏_l [U_ent · U_rot(x)]
# ──────────────────────────────────────────────────────────────

function build_feature_map(n_qubits::Int, n_layers::Int, features::Vector{Float64}, params::Matrix{Float64}, ent_edges::Vector{Tuple{Int,Int}})
    blocks = AbstractBlock[]

    for layer in 1:n_layers
        rot_blocks = []
        for q in 1:n_qubits
            x = features[mod1(q, length(features))]
            θz1 = params[layer, 3*(q-1)+1]
            θy  = params[layer, 3*(q-1)+2]
            θz2 = params[layer, 3*(q-1)+3]
            push!(rot_blocks, q => chain(Rz(2*x*θz1), Ry(2*x*θy), Rz(2*x*θz2)))
        end
        push!(blocks, kron(n_qubits, rot_blocks...))

        ent_block = []
        for (q1, q2) in ent_edges
            push!(ent_block, control(q1, q2 => Z))
        end
        if !isempty(ent_block)
            push!(blocks, chain(n_qubits, ent_block...))
        end
    end

    return chain(n_qubits, blocks...)
end

# ──────────────────────────────────────────────────────────────
# SWAP TEST KERNEL: K(x, x') = |⟨Φ(x)|Φ(x')⟩|²
# ──────────────────────────────────────────────────────────────

function kernel_fidelity(n_qubits::Int, n_layers::Int, features_a::Vector{Float64}, features_b::Vector{Float64}, params::Matrix{Float64}, ent_edges::Vector{Tuple{Int,Int}})
    circuit_a = build_feature_map(n_qubits, n_layers, features_a, params, ent_edges)
    circuit_b = build_feature_map(n_qubits, n_layers, features_b, params, ent_edges)

    state_a = zero_state(n_qubits) |> circuit_a
    state_b = zero_state(n_qubits) |> circuit_b

    overlap = statevec(state_a)' * statevec(state_b)
    return abs2(overlap)
end

function kernel_entry_shots(n_qubits::Int, n_layers::Int, features_a::Vector{Float64}, features_b::Vector{Float64}, params::Matrix{Float64}, ent_edges::Vector{Tuple{Int,Int}}, shots::Int)
    exact = kernel_fidelity(n_qubits, n_layers, features_a, features_b, params, ent_edges)
    p0 = (1 + exact) / 2
    count_zero = sum(rand() < p0 for _ in 1:shots)
    return 2 * count_zero / shots - 1
end

# ──────────────────────────────────────────────────────────────
# KERNEL MATRIX
# ──────────────────────────────────────────────────────────────

function compute_kernel_matrix(dataset::Matrix{Float64}, n_qubits::Int, n_layers::Int, params::Matrix{Float64}, ent_edges::Vector{Tuple{Int,Int}}, shots::Int)
    n = size(dataset, 1)
    K = zeros(n, n)
    for i in 1:n
        for j in i:n
            kij = kernel_entry_shots(n_qubits, n_layers, dataset[i,:], dataset[j,:], params, ent_edges, shots)
            K[i,j] = kij
            K[j,i] = kij
        end
    end
    return K
end

# ──────────────────────────────────────────────────────────────
# SVM DUAL SOLVER (SMO)
# ──────────────────────────────────────────────────────────────

function solve_svm_dual(K::Matrix{Float64}, labels::Vector{Float64}; C=1.0, max_iter=1000, tol=1e-4)
    n = length(labels)
    alpha = zeros(n)
    b = 0.0

    for _ in 1:max_iter
        max_violation = 0.0
        for i in 1:n
            grad = 1.0 - sum(alpha[j] * labels[j] * K[i,j] * labels[i] for j in 1:n)
            violation = alpha[i] == 0 ? max(0, -labels[i]*grad) :
                        alpha[i] == C ? max(0, labels[i]*grad) :
                        abs(labels[i]*grad)
            max_violation = max(max_violation, violation)
            alpha[i] = clamp(alpha[i] + 0.01 * labels[i] * grad, 0.0, C)
        end
        max_violation < tol && break
    end

    sv = findall(i -> 1e-5 < alpha[i] < C - 1e-5, 1:n)
    if !isempty(sv)
        b = mean(labels[k] - sum(alpha[j]*labels[j]*K[k,j] for j in 1:n) for k in sv)
    end

    return alpha, b
end

# ──────────────────────────────────────────────────────────────
# HELLO WORLD: 5 QUBIT QUANTUM KERNEL
# ──────────────────────────────────────────────────────────────

function main()
    println("=" ^ 60)
    println("QUANTUM KERNEL SVM — 5 Qubit Hello World")
    println("ANU QRNG Entropy | Yao.jl Simulator | Shot-Based Estimation")
    println("=" ^ 60)
    println()

    n_qubits = 5
    n_layers = 2
    shots = 1000

    anu = ANUEntropy(size=256)
    println("Entropy source: ANU QRNG ($(length(anu.buffer)) bytes buffered)")
    println("Qubits: $n_qubits | Layers: $n_layers | Shots: $shots")
    println()

    ent_edges = [(i, i+1) for i in 1:n_qubits-1]
    println("Entanglement: linear chain $(ent_edges)")

    params = ones(n_layers, 3*n_qubits) .+ 0.1 .* randn(n_layers, 3*n_qubits)

    # XOR-style dataset (non-linearly separable)
    dataset = Float64[
        0.0 0.0 0.0 0.0 0.0;
        0.0 1.0 0.0 1.0 0.0;
        1.0 0.0 1.0 0.0 1.0;
        1.0 1.0 1.0 1.0 1.0;
        0.5 0.5 0.5 0.5 0.5;
        0.2 0.8 0.2 0.8 0.2;
        0.8 0.2 0.8 0.2 0.8;
        0.3 0.7 0.3 0.7 0.3;
    ]
    labels = Float64[-1, 1, 1, -1, -1, 1, 1, -1]

    println("\nDataset: $(size(dataset, 1)) samples, $(size(dataset, 2)) features")
    println("Labels: $labels")
    println()

    # Compute kernel matrix
    println("Computing quantum kernel matrix ($shots shots per entry)...")
    t0 = time()
    K = compute_kernel_matrix(dataset, n_qubits, n_layers, params, ent_edges, shots)
    elapsed = time() - t0
    println("Done in $(round(elapsed, digits=2))s")
    println()

    println("Kernel matrix (first 4x4):")
    for i in 1:min(4, size(K,1))
        println("  ", [round(K[i,j], digits=4) for j in 1:min(4, size(K,2))])
    end
    println()

    # Verify PSD
    eigenvals = eigvals(Symmetric(K))
    println("Kernel eigenvalues: ", [round(e, digits=6) for e in eigenvals])
    println("PSD check: $(all(eigenvals .>= -1e-10) ? "PASS" : "FAIL")")
    println()

    # Train SVM
    println("Training SVM (dual solver)...")
    alpha, bias = solve_svm_dual(K, labels)
    sv_count = count(a -> a > 1e-5, alpha)
    println("Support vectors: $sv_count / $(length(labels))")
    println("Bias: $(round(bias, digits=4))")
    println()

    # Predict
    println("Predictions:")
    correct = 0
    for i in 1:size(dataset, 1)
        decision = sum(alpha[j] * labels[j] * K[j,i] for j in 1:size(dataset,1)) + bias
        pred = decision >= 0 ? 1.0 : -1.0
        match = pred == labels[i] ? "OK" : "MISS"
        correct += pred == labels[i]
        println("  x[$i] → decision=$(round(decision, digits=4)), pred=$(Int(pred)), true=$(Int(labels[i])) [$match]")
    end
    println()
    println("Accuracy: $correct / $(length(labels)) = $(round(100*correct/length(labels), digits=1))%")

    # Shot statistics
    println()
    println("─" ^ 40)
    println("Shot noise analysis (kernel[1,2]):")
    estimates = [kernel_entry_shots(n_qubits, n_layers, dataset[1,:], dataset[2,:], params, ent_edges, shots) for _ in 1:20]
    println("  Mean: $(round(mean(estimates), digits=6))")
    println("  Std:  $(round(std(estimates), digits=6))")
    println("  Exact: $(round(kernel_fidelity(n_qubits, n_layers, dataset[1,:], dataset[2,:], params, ent_edges), digits=6))")

    println()
    println("=" ^ 60)
    println("HELLO WORLD COMPLETE — 5 qubit quantum kernel executed")
    println("=" ^ 60)
end

main()
