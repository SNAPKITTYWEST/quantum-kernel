# Quantum Kernel SVM

Hilbert Space Feature Mapping with Native Kernel Computation.

5-qubit quantum kernel that encodes classical data into 2^5 = 32-dimensional
Hilbert space, computes kernel matrix via SWAP test with shot noise, and trains
a classical SVM on the quantum kernel.

## What This Does

1. Encodes classical features into quantum states via parameterized rotations + entanglement
2. Computes kernel K(x,x') = |<Phi(x)|Phi(x')>|^2 via SWAP test (shot-based)
3. Trains SVM dual problem on quantum kernel matrix
4. Classifies non-linearly separable data that linear SVMs fail on

## Run (Go)

```bash
cd go && go run main.go
```

## Run (Julia)

```bash
cd julia && julia --project=. -e 'using Pkg; Pkg.instantiate()' && julia quantum_kernel.jl
```

## Architecture

```
Classical Data (R^d)
    |
    v
Feature Map: U_Phi(x) = prod_l [U_ent * U_rot(x)]
    |
    v
Quantum State |Phi(x)> in (C^2)^{tensor n}  (dim = 2^n)
    |
    v
SWAP Test: K(x,x') = |<Phi(x)|Phi(x')>|^2  (shots -> statistics)
    |
    v
Kernel Matrix K (NxN, PSD by construction)
    |
    v
Classical SVM Dual: max_a sum(a_i) - 1/2 sum(a_i a_j y_i y_j K_ij)
    |
    v
Decision: f(x) = sign(sum(a_i y_i K(x_i, x)) + b)
```

## Key Properties

- Feature map unitarity: U^dag U = I (by construction)
- Kernel PSD: Gram matrix of quantum states
- SWAP test unbiased: E[K_hat] = K
- Concentration: P(|K_hat - K| > eps) <= 2*exp(-2*shots*eps^2)
- Entanglement necessity: without CZ layer, reduces to classical product kernel

## ANU QRNG

The Julia implementation connects to ANU Quantum Random Number Generator for
true vacuum-fluctuation randomness in measurement basis selection (DFE mode).
Falls back to CSPRNG when API unavailable.

## Hardware Targets

- IBM Heron r3 (133 qubits, heavy-hex, native: RZ+SX+CX)
- Compilation: feature map -> OpenQASM 3.0 -> Heron native gate set
- Error mitigation: Zero-Noise Extrapolation (ZNE) via CX stretching
- Mid-circuit measurement for Direct Fidelity Estimation

## License

BSL-1.1 / AGPL-3.0 / MPL-2.0 (tri-license)
