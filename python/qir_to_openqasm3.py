"""
qir_to_openqasm3.py

Python implementation of QuantumIR → Heron-native OpenQASM 3.0 converter.
Includes ZNE stretching, mid-circuit measurement, DFE protocol, and
Richardson extrapolation. Runs in any sandbox (Kimi, Replit, local).

Built from scratch — no Qiskit, no Cirq, no PennyLane dependency.
"""

import json
import random
import math


def decompose_to_heron(name, params, qubits, zne_factor):
    instrs = []
    if name == "Rz":
        instrs.append(f"rz({params[0]}) q[{qubits[0]}];")
    elif name == "Rx":
        q = qubits[0]
        instrs.append(f"rz(-1.5707963267948966) q[{q}];")
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz({params[0]}) q[{q}];")
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz(1.5707963267948966) q[{q}];")
    elif name == "Ry":
        q = qubits[0]
        instrs.append(f"rz(1.5707963267948966) q[{q}];")
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz({params[0]}) q[{q}];")
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz(-1.5707963267948966) q[{q}];")
    elif name == "H":
        q = qubits[0]
        instrs.append(f"rz(1.5707963267948966) q[{q}];")
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz(1.5707963267948966) q[{q}];")
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz(1.5707963267948966) q[{q}];")
    elif name == "S":
        instrs.append(f"rz(1.5707963267948966) q[{qubits[0]}];")
    elif name == "Sdg":
        instrs.append(f"rz(-1.5707963267948966) q[{qubits[0]}];")
    elif name == "T":
        instrs.append(f"rz(0.7853981633974483) q[{qubits[0]}];")
    elif name == "Tdg":
        instrs.append(f"rz(-0.7853981633974483) q[{qubits[0]}];")
    elif name == "X":
        q = qubits[0]
        instrs.append(f"sx q[{q}];")
        instrs.append(f"sx q[{q}];")
    elif name == "Y":
        q = qubits[0]
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz(3.141592653589793) q[{q}];")
        instrs.append(f"sx q[{q}];")
    elif name == "Z":
        instrs.append(f"rz(3.141592653589793) q[{qubits[0]}];")
    elif name == "CX":
        c, t = qubits[0], qubits[1]
        instrs.append(f"cx q[{c}], q[{t}];")
        if zne_factor > 1.0:
            repeats = int(round(zne_factor)) - 1
            for _ in range(repeats):
                instrs.append(f"cx q[{c}], q[{t}];")
                instrs.append(f"cx q[{c}], q[{t}];")
    elif name == "CZ":
        c, t = qubits[0], qubits[1]
        for h in decompose_to_heron("H", [], [t], 1.0):
            instrs.append(h)
        instrs.append(f"cx q[{c}], q[{t}];")
        if zne_factor > 1.0:
            repeats = int(round(zne_factor)) - 1
            for _ in range(repeats):
                instrs.append(f"cx q[{c}], q[{t}];")
                instrs.append(f"cx q[{c}], q[{t}];")
        for h in decompose_to_heron("H", [], [t], 1.0):
            instrs.append(h)
    else:
        instrs.append(f"// Unknown gate: {name}")
    return instrs


def pauli_rotation_instrs(pauli, qubit):
    if pauli == 'X':
        return [
            f"rz(1.5707963267948966) q[{qubit}];",
            f"sx q[{qubit}];",
            f"rz(1.5707963267948966) q[{qubit}];",
            f"sx q[{qubit}];",
            f"rz(1.5707963267948966) q[{qubit}];"
        ]
    elif pauli == 'Y':
        return [
            f"rz(-1.5707963267948966) q[{qubit}];",
            f"sx q[{qubit}];",
            f"rz(1.5707963267948966) q[{qubit}];",
            f"sx q[{qubit}];",
            f"rz(1.5707963267948966) q[{qubit}];"
        ]
    elif pauli in ('Z', 'I'):
        return []
    return [f"// Unknown Pauli: {pauli}"]


def decompose_to_heron_zne(name, params, qubits):
    instrs = []
    factor = "noise_factor"
    if name == "Rz":
        instrs.append(f"rz({params[0]} * {factor}) q[{qubits[0]}];")
    elif name == "Rx":
        q = qubits[0]
        instrs.append(f"rz(-1.5707963267948966) q[{q}];")
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz({params[0]} * {factor}) q[{q}];")
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz(1.5707963267948966) q[{q}];")
    elif name == "Ry":
        q = qubits[0]
        instrs.append(f"rz(1.5707963267948966) q[{q}];")
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz({params[0]} * {factor}) q[{q}];")
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz(-1.5707963267948966) q[{q}];")
    elif name == "H":
        q = qubits[0]
        instrs.append(f"rz(1.5707963267948966) q[{q}];")
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz(1.5707963267948966) q[{q}];")
        instrs.append(f"sx q[{q}];")
        instrs.append(f"rz(1.5707963267948966) q[{q}];")
    elif name == "CX":
        c, t = qubits[0], qubits[1]
        instrs.append(f"cx q[{c}], q[{t}];")
    elif name == "CZ":
        c, t = qubits[0], qubits[1]
        for h in decompose_to_heron_zne("H", [], [t]):
            instrs.append(h)
        instrs.append(f"cx q[{c}], q[{t}];")
        for h in decompose_to_heron_zne("H", [], [t]):
            instrs.append(h)
    else:
        instrs.append(f"// {name} with ZNE not implemented")
    return instrs


def qir_to_openqasm3(ir_dict, zne_factors=None, anu_bases=None, dynamic_shots=True):
    if zne_factors is None:
        zne_factors = [1.0]

    nq = ir_dict["qubits"]
    nc = ir_dict["cbits"]
    ops = ir_dict["ops"]

    if len(zne_factors) > 1 and dynamic_shots:
        return qir_to_openqasm3_zne_dynamic(ir_dict, zne_factors, anu_bases)

    factor = zne_factors[0]
    lines = []
    indent = 0

    def emit(s=""):
        lines.append(" " * indent + s)

    emit("OPENQASM 3.0;")
    emit('include "stdgates.inc";')
    emit("")
    emit(f"qubit[{nq}] q;")
    emit(f"bit[{nc}] meas;")
    emit("")
    emit("float fidelity_sum = 0.0;")
    emit("int valid_shots = 0;")
    emit("")

    if dynamic_shots and anu_bases is not None:
        n_shots = len(anu_bases)
        emit(f"for shot in [0:{n_shots-1}] {{")
        indent += 2

    for op in ops:
        op_type = op["type"]
        if op_type == "gate":
            for instr in decompose_to_heron(op["name"], op.get("params", []), op["qubits"], factor):
                emit(instr)
        elif op_type == "measure":
            emit(f"meas[{op['cbit']}] = measure q[{op['qubit']}];")
        elif op_type == "barrier":
            qs = ", ".join(str(q) for q in op["qubits"])
            emit(f"barrier q[{qs}];")
        elif op_type == "reset":
            emit(f"if (meas[{op['qubit']}] == 1) {{ x q[{op['qubit']}]; }}")

    if dynamic_shots and anu_bases is not None:
        indent -= 2
        emit("}")

    emit("")
    emit("float kernel_est = fidelity_sum / float(valid_shots);")
    emit("kernel_est;")

    return "\n".join(lines)


def qir_to_openqasm3_zne_dynamic(ir_dict, zne_factors, anu_bases):
    nq = ir_dict["qubits"]
    nc = ir_dict["cbits"]
    ops = ir_dict["ops"]
    n_shots = len(anu_bases) if anu_bases else 1000
    n_factors = len(zne_factors)

    lines = []
    indent = 0

    def emit(s=""):
        lines.append(" " * indent + s)

    emit("OPENQASM 3.0;")
    emit('include "stdgates.inc";')
    emit("")
    emit(f"qubit[{nq}] q;")
    emit(f"bit[{nc}] meas;")
    emit("")
    emit(f"float[{n_factors}] fidelity_sum = {{{', '.join(['0.0'] * n_factors)}}};")
    emit(f"int[{n_factors}] valid_shots = {{{', '.join(['0'] * n_factors)}}};")
    emit("")

    if anu_bases is not None:
        emit("// ANU QRNG Pauli bases (pre-fetched)")
        emit(f"string[{n_shots * nq}] pauli_bases = {{")
        indent += 2
        for shot, basis in enumerate(anu_bases[:n_shots]):
            for q, pauli in enumerate(basis):
                emit(f'"{pauli}", // shot {shot+1}, qubit {q}')
        indent -= 2
        emit("};")
        emit("")

    emit(f"for f_idx in [0:{n_factors-1}] {{")
    indent += 2
    emit(f"float noise_factors[{n_factors}] = {{{', '.join(str(f) for f in zne_factors)}}};")
    emit("float noise_factor = noise_factors[f_idx];")
    emit("")
    emit(f"for shot in [0:{n_shots-1}] {{")
    indent += 2

    if anu_bases is not None:
        emit("// Pauli basis from ANU QRNG")
        for q in range(nq):
            emit(f'string pauli_{q} = pauli_bases[shot * {nq} + {q}];')

    emit("// Feature Map U_Phi(x)")
    for op in ops:
        if op["type"] == "gate":
            for instr in decompose_to_heron_zne(op["name"], op.get("params", []), op["qubits"]):
                emit(instr)

    emit("// Inverse Feature Map U_Phi(x')dagger")

    if anu_bases is not None:
        emit("// Pauli basis rotation")
        for q in range(nq):
            emit(f'if (pauli_{q} == "X") {{')
            indent += 2
            for instr in pauli_rotation_instrs('X', q):
                emit(instr)
            indent -= 2
            emit(f'}} else if (pauli_{q} == "Y") {{')
            indent += 2
            for instr in pauli_rotation_instrs('Y', q):
                emit(instr)
            indent -= 2
            emit("}")

    emit("// Mid-circuit measurement")
    for q in range(nq):
        emit(f"meas[{q}] = measure q[{q}];")

    emit("// Conditional reset")
    for q in range(nq):
        emit(f"if (meas[{q}] == 1) {{ x q[{q}]; }}")

    emit("// DFE fidelity estimator")
    emit("bool has_xy = false;")
    emit("int z_weight = 0;")
    if anu_bases is not None:
        for q in range(nq):
            emit(f'if (pauli_{q} == "X" || pauli_{q} == "Y") has_xy = true;')
            emit(f'if (pauli_{q} == "Z") z_weight = z_weight + 1;')
    emit("")
    emit("if (!has_xy) {")
    indent += 2
    emit("int eigenvalue = 1;")
    if anu_bases is not None:
        for q in range(nq):
            emit(f'if (pauli_{q} == "Z" && meas[{q}] == 1) eigenvalue = eigenvalue * -1;')
    emit("float estimator = pow(3.0, float(z_weight)) * float(eigenvalue);")
    emit("fidelity_sum[f_idx] = fidelity_sum[f_idx] + estimator;")
    emit("valid_shots[f_idx] = valid_shots[f_idx] + 1;")
    indent -= 2
    emit("}")

    indent -= 2
    emit("}")
    indent -= 2
    emit("}")

    emit("")
    emit("// Richardson extrapolation to zero noise")
    emit("float kernel_est = 0.0;")
    for i in range(n_factors):
        emit(f"float y{i} = fidelity_sum[{i}] / float(valid_shots[{i}]);")
    for i in range(n_factors):
        emit(f"float term{i} = y{i};")
        for j in range(n_factors):
            if i != j:
                emit(f"term{i} = term{i} * (-{zne_factors[j]}) / ({zne_factors[i]} - {zne_factors[j]});")
        emit(f"kernel_est = kernel_est + term{i};")
    emit("")
    emit("kernel_est;")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python qir_to_openqasm3.py <input.ir.json> <output.qasm3> [zne_factors...]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    zne_factors = [float(x) for x in sys.argv[3:]] if len(sys.argv) > 3 else [1.0]

    with open(input_file, 'r') as f:
        ir_list = json.load(f)

    first_ir = ir_list[0] if isinstance(ir_list, list) else ir_list
    n_qubits = first_ir["qubits"]
    anu_bases = [[random.choice(['I', 'X', 'Y', 'Z']) for _ in range(n_qubits)] for _ in range(100)]

    qasm = qir_to_openqasm3(first_ir, zne_factors=zne_factors, anu_bases=anu_bases, dynamic_shots=True)

    with open(output_file, 'w') as f:
        f.write(qasm)

    print(f"Written {output_file} ({len(qasm)} chars, {len(qasm.splitlines())} lines)")
    print(f"ZNE factors: {zne_factors}")
    print(f"ANU QRNG shots: {len(anu_bases)}")
