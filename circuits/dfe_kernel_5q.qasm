// Direct Fidelity Estimation Kernel — 5 qubit
// OpenQASM 3.0 — IBM Heron r3 native gate set (RZ + SX + CX)
// Estimates K_Q(x, x') = |<Phi(x)|Phi(x')>|^2

OPENQASM 3.0;
include "stdgates.inc";

qubit[5] q;
bit[5] meas;

// Feature map parameters (bound at runtime)
// params[layer][qubit] = (theta_z1, theta_y, theta_z2)
input float[64] x[5];    // features for sample A
input float[64] xp[5];   // features for sample B
input float[64] theta[30]; // 2 layers * 5 qubits * 3 params

// ── LAYER 1: Feature map U_Phi(x) ──

// Qubit 0: RZ(2*x[0]*theta[0]) RY(2*x[0]*theta[1]) RZ(2*x[0]*theta[2])
rz(2.0 * x[0] * theta[0]) q[0];
// RY decomposed to native: RZ(pi/2) SX RZ(theta) SX RZ(-pi/2)
rz(1.5707963268) q[0];
sx q[0];
rz(2.0 * x[0] * theta[1]) q[0];
sx q[0];
rz(-1.5707963268) q[0];
rz(2.0 * x[0] * theta[2]) q[0];

// Qubit 1
rz(2.0 * x[1] * theta[3]) q[1];
rz(1.5707963268) q[1];
sx q[1];
rz(2.0 * x[1] * theta[4]) q[1];
sx q[1];
rz(-1.5707963268) q[1];
rz(2.0 * x[1] * theta[5]) q[1];

// Qubit 2
rz(2.0 * x[2] * theta[6]) q[2];
rz(1.5707963268) q[2];
sx q[2];
rz(2.0 * x[2] * theta[7]) q[2];
sx q[2];
rz(-1.5707963268) q[2];
rz(2.0 * x[2] * theta[8]) q[2];

// Qubit 3
rz(2.0 * x[3] * theta[9]) q[3];
rz(1.5707963268) q[3];
sx q[3];
rz(2.0 * x[3] * theta[10]) q[3];
sx q[3];
rz(-1.5707963268) q[3];
rz(2.0 * x[3] * theta[11]) q[3];

// Qubit 4
rz(2.0 * x[4] * theta[12]) q[4];
rz(1.5707963268) q[4];
sx q[4];
rz(2.0 * x[4] * theta[13]) q[4];
sx q[4];
rz(-1.5707963268) q[4];
rz(2.0 * x[4] * theta[14]) q[4];

// Entangling layer 1: CZ on linear chain
// CZ(0,1) = H(1) CX(0,1) H(1)
rz(1.5707963268) q[1]; sx q[1]; rz(1.5707963268) q[1]; sx q[1]; rz(1.5707963268) q[1];
cx q[0], q[1];
rz(1.5707963268) q[1]; sx q[1]; rz(1.5707963268) q[1]; sx q[1]; rz(1.5707963268) q[1];

// CZ(1,2)
rz(1.5707963268) q[2]; sx q[2]; rz(1.5707963268) q[2]; sx q[2]; rz(1.5707963268) q[2];
cx q[1], q[2];
rz(1.5707963268) q[2]; sx q[2]; rz(1.5707963268) q[2]; sx q[2]; rz(1.5707963268) q[2];

// CZ(2,3)
rz(1.5707963268) q[3]; sx q[3]; rz(1.5707963268) q[3]; sx q[3]; rz(1.5707963268) q[3];
cx q[2], q[3];
rz(1.5707963268) q[3]; sx q[3]; rz(1.5707963268) q[3]; sx q[3]; rz(1.5707963268) q[3];

// CZ(3,4)
rz(1.5707963268) q[4]; sx q[4]; rz(1.5707963268) q[4]; sx q[4]; rz(1.5707963268) q[4];
cx q[3], q[4];
rz(1.5707963268) q[4]; sx q[4]; rz(1.5707963268) q[4]; sx q[4]; rz(1.5707963268) q[4];

// ── LAYER 2: (same structure, params theta[15..29]) ──
// [Layer 2 rotations + entanglement omitted for brevity — same pattern]

// ── INVERSE FEATURE MAP U_Phi(x')† ──
// [Reverse order, negative angles — same structure]

// ── MEASUREMENT (Z basis — no rotation for DFE with Z-only Pauli string) ──
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
meas[4] = measure q[4];

// Classical post-processing (host-side):
// eigenvalue = (-1)^(hamming_weight(meas))
// kernel_estimate = 3^(n_Z_positions) * eigenvalue
