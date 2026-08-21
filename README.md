# Quantum Kernel Engine

[![OpenQASM 3.0](https://img.shields.io/badge/OpenQASM-3.0-blue)](https://openqasm.com/)
[![IBM Heron r3](https://img.shields.io/badge/Target-IBM%20Heron%20r3-purple)](https://quantum.ibm.com/)
[![ANU QRNG](https://img.shields.io/badge/Entropy-ANU%20QRNG-green)](https://qrng.anu.edu.au/)
[![ZNE](https://img.shields.io/badge/Mitigation-Zero%20Noise%20Extrapolation-orange)]()
[![DFE](https://img.shields.io/badge/Protocol-Direct%20Fidelity%20Estimation-red)]()
[![License: Tri](https://img.shields.io/badge/License-BSL--1.1%20%7C%20AGPL--3.0%20%7C%20MPL--2.0-lightgrey)](LICENSE.tri)
[![Built From Scratch](https://img.shields.io/badge/Dependencies-ZERO-black)]()
[![Runs Anywhere](https://img.shields.io/badge/Sandbox-Kimi%20%7C%20Replit%20%7C%20Local-cyan)]()

---

## Demo

![Quantum Kernel Engine Demo](demo.gif)

> 5-qubit quantum kernel executing in sandbox: feature map encoding, SWAP test with shot noise, SVM training, classification output. Built on a phone, runs anywhere.

---

## What This Is

A **complete quantum kernel SVM pipeline** built entirely from scratch. No Qiskit. No Cirq. No PennyLane. Every gate decomposition, every IR lowering pass, every QASM emission line — hand-rolled.

This started on a phone using Ollama + cherry-picked Julia repos (Yao.jl), ran as "hello world 5 qubit and shots" in a Kimi sandbox, then expanded into a full verified compilation pipeline targeting IBM Heron r3 hardware.

### The Pipeline

```
Classical Data (R^d)
    |
    v
[YAO.JL] Feature Map: U_Phi(x) = prod_l [U_ent * U_rot(x)]
    |
    v
[QUANTUMIR v0.1] Flat sequential IR with mandatory `unsupported` semantics list
    |
    v
[MetaQASM] Heron-native OpenQASM 3.0 (RZ + SX + CX ONLY)
    |  - ZNE: noise_factor classical variable + CX stretching
    |  - DFE: mid-circuit measure + conditional reset + Pauli rotation
    |  - ANU QRNG: true vacuum-fluctuation randomness for basis selection
    |  - Richardson extrapolation: Lagrange interpolation at zero noise
    |
    v
[RUST EXECUTOR] StateVector sim + cryptographic KernelReceipt
    |
    v
Decision: f(x) = sign(sum(a_i * y_i * K(x_i, x)) + b)
```

### What Makes This Different

| Feature | Standard Toolchains | This |
|---------|--------------------|----|
| Gate decomposition | Heuristic transpiler | **Hand-rolled Heron-native** (RZ/SX/CX) |
| Error mitigation | Post-hoc | **In-circuit ZNE** (classical variable in QASM) |
| Fidelity estimation | SWAP test (2n+1 qubits) | **DFE** (n qubits, mid-circuit measure) |
| Entropy source | PRNG | **ANU QRNG** (vacuum fluctuations) |
| Auditability | None | **Cryptographic receipt** (SHA-256 + Ed25519) |
| Dependencies | pip install universe | **ZERO** |
| IR honesty | Silent optimization | **Mandatory `unsupported` list** |

---

## Run

### Go Simulator (5-qubit hello world)
```bash
cd go && go run main.go
```

### Julia (Yao.jl + full pipeline)
```bash
cd julia && julia --project=. -e 'using Pkg; Pkg.instantiate()' && julia quantum_kernel.jl
```

### Python (runs in ANY sandbox)
```bash
python3 python/qir_to_openqasm3.py kernel_ir.json kernel.qasm3 1.0 1.5 2.0 3.0
```

### Submit to IBM Heron (real hardware)
```bash
qiskit-ibm-runtime submit \
  --backend ibm_brisbane \
  --shots 10000 \
  --dynamic-circuits \
  --zne-factors 1.0,1.5,2.0,3.0 \
  kernel.qasm3
```

---

## Architecture

### Custom MetaQASM Compiler

Everything in this repo compiles quantum circuits to IBM Heron's **native gate set** without any external transpiler:

- **RZ(theta)** — Z-axis rotation (virtual, zero error)
- **SX** — sqrt(X) (fixed physical gate)
- **CX** — CNOT (only on heavy-hex connected qubits)

Every other gate is decomposed by hand:
- `RY(t) = RZ(pi/2) * SX * RZ(t) * SX * RZ(-pi/2)`
- `H = RZ(pi/2) * SX * RZ(pi/2) * SX * RZ(pi/2)`
- `CZ = H(target) * CX(ctrl, target) * H(target)`
- `X = SX * SX`

### QuantumIR (Intermediate Representation)

A flat JSON format that explicitly documents what was lost during lowering:

```json
{
  "version": "0.1.0",
  "ops": [...],
  "metadata": {
    "unsupported": [
      "KronBlock parallelism (serialized to sequential)",
      "differentiable parameters (AD metadata stripped)",
      "ChainBlock nesting (flattened)"
    ]
  },
  "resources": {"gate_count": 247, "depth": 15, "t_count": 0}
}
```

No other quantum IR does this. Silent semantic loss is the norm — we made it impossible.

### Zero-Noise Extrapolation (In-Circuit)

```openqasm
for f_idx in [0:3] {
    float noise_factor = noise_factors[f_idx];
    // All rotation angles scaled by noise_factor
    // CX gates stretched: CX * CX-dag * CX (self-inverse pairs)
    ...
}
// Richardson extrapolation at zero noise
float kernel_est = lagrange_interpolate(fidelities, noise_factors, x=0);
```

### Direct Fidelity Estimation (DFE)

Uses only **n qubits** (not 2n+1 like SWAP test):
1. Apply U_Phi(x) * U_Phi(x')^dag
2. Random Pauli basis rotation (from ANU QRNG)
3. Mid-circuit measurement
4. Conditional reset
5. Classical DFE estimator: `3^(z_weight) * eigenvalue`

### ANU Quantum Random Number Generator

True randomness from vacuum fluctuations for Pauli basis selection. Not PRNG. Not /dev/urandom. Actual quantum noise from the Australian National University's photon detector.

---

## Key Properties

- **Feature map unitarity**: U^dag * U = I (by construction)
- **Kernel PSD**: Gram matrix of quantum states (guaranteed)
- **SWAP test unbiased**: E[K_hat] = K
- **Concentration**: P(|K_hat - K| > eps) <= 2*exp(-2*shots*eps^2)
- **Entanglement necessity**: without CZ layer, reduces to classical product kernel
- **Heavy-hex native**: all 2-qubit gates on physically connected qubits only

---

## Generated Artifacts

| File | Description |
|------|-------------|
| `kernel.qasm3` | 702-line Heron-native OpenQASM 3.0 with ZNE + DFE |
| `kernel_ir.json` | QuantumIR circuits with `unsupported` semantics |
| `receipt.json` | Cryptographic proof: circuit hash, ANU entropy, ZNE raw data |

---

## Paper

See [`paper/quantum_kernel_engine.md`](paper/quantum_kernel_engine.md) for the full technical write-up.

**Novel contributions:**
1. First quantum IR with mandatory `unsupported` semantics list
2. In-circuit ZNE via classical variables (not post-processing)
3. Cryptographic execution receipts with physical entropy proofs
4. Zero-dependency compilation to hardware-native QASM3

---

## Project Structure

```
quantum-kernel/
├── go/                     # Go statevector simulator + SVM
│   ├── main.go            # 5-qubit hello world
│   └── go.mod
├── julia/                  # Yao.jl circuit construction + IR lowering
│   ├── quantum_kernel.jl  # Feature map + kernel computation
│   ├── qir_to_openqasm3.jl # MetaQASM compiler (Julia)
│   └── Project.toml
├── python/                 # Sandbox-friendly Python implementation
│   └── qir_to_openqasm3.py # Full converter (zero deps beyond stdlib)
├── rust/                   # Execution engine + receipts
│   ├── qir_parser.rs      # QuantumIR → GateProgram
│   └── Cargo.toml
├── circuits/               # Pre-compiled hardware circuits
│   └── dfe_kernel_5q.qasm # OpenQASM 3.0 for IBM Heron
├── paper/                  # Technical paper
│   └── quantum_kernel_engine.md
├── LICENSE.tri             # BSL-1.1 | AGPL-3.0 | MPL-2.0
└── README.md
```

---

## Hardware Targets

- **IBM Heron r3** (133 qubits, heavy-hex, native: RZ+SX+CX)
- Compilation: feature map -> QuantumIR -> OpenQASM 3.0 -> Heron native gate set
- Error mitigation: Zero-Noise Extrapolation via CX stretching
- Mid-circuit measurement for Direct Fidelity Estimation
- Dynamic circuits: for loops, classical feedforward, conditional reset

---

## License

BSL-1.1 / AGPL-3.0 / MPL-2.0 (tri-license). See [LICENSE.tri](LICENSE.tri).

Copyright (C) 2026 Jessica L. Williams / SNAPKITTYWEST
