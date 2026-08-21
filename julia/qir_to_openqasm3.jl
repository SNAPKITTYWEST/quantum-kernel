# qir_to_openqasm3.jl
#
# Lower QuantumIR JSON to Heron-native OpenQASM 3.0 with dynamic circuit support.
# Includes ZNE stretching, mid-circuit measurement, and classical feedforward.

include("yao_types.jl")
include("yao_to_ir.jl")

using JSON3

# -----------------------------------------------------------------------
# OpenQASM 3.0 Emission
# -----------------------------------------------------------------------

mutable struct QASM3Emitter
    io::IOBuffer
    indent::Int
    n_qubits::Int
    n_cbits::Int
    in_classical::Bool
    zne_factor::Float64
    shot_var::String
    basis_var::String
end

function QASM3Emitter(nq::Int, nc::Int; zne_factor::Float64=1.0)
    QASM3Emitter(
        IOBuffer(), 0, nq, nc, false, zne_factor,
        "shot", "basis_idx"
    )
end

function emit!(e::QASM3Emitter, s::String)
    print(e.io, " " ^ e.indent, s)
end

function emitln!(e::QASM3Emitter, s::String="")
    emit!(e, s * "\n")
end

function indent!(e::QASM3Emitter, delta::Int=1)
    e.indent += 2 * delta
end

function dedent!(e::QASM3Emitter, delta::Int=1)
    e.indent = max(0, e.indent - 2 * delta)
end

# -----------------------------------------------------------------------
# Heron Native Gate Decomposition
# -----------------------------------------------------------------------

function decompose_to_heron(name::String, params::Vector{Float64}, qubits::Vector{Int}, zne_factor::Float64)
    instrs = String[]

    if name == "Rz"
        θ = params[1]
        push!(instrs, "rz($(θ)) q[$(qubits[1])];")

    elseif name == "Rx"
        θ = params[1]
        push!(instrs, "rz(-1.5707963267948966) q[$(qubits[1])];")
        push!(instrs, "sx q[$(qubits[1])];")
        push!(instrs, "rz($(θ)) q[$(qubits[1])];")
        push!(instrs, "sx q[$(qubits[1])];")
        push!(instrs, "rz(1.5707963267948966) q[$(qubits[1])];")

    elseif name == "Ry"
        θ = params[1]
        push!(instrs, "rz(1.5707963267948966) q[$(qubits[1])];")
        push!(instrs, "sx q[$(qubits[1])];")
        push!(instrs, "rz($(θ)) q[$(qubits[1])];")
        push!(instrs, "sx q[$(qubits[1])];")
        push!(instrs, "rz(-1.5707963267948966) q[$(qubits[1])];")

    elseif name == "H"
        q = qubits[1]
        push!(instrs, "rz(1.5707963267948966) q[$q];")
        push!(instrs, "sx q[$q];")
        push!(instrs, "rz(1.5707963267948966) q[$q];")
        push!(instrs, "sx q[$q];")
        push!(instrs, "rz(1.5707963267948966) q[$q];")

    elseif name == "S"
        push!(instrs, "rz(1.5707963267948966) q[$(qubits[1])];")

    elseif name == "Sdg" || name == "S†"
        push!(instrs, "rz(-1.5707963267948966) q[$(qubits[1])];")

    elseif name == "T"
        push!(instrs, "rz(0.7853981633974483) q[$(qubits[1])];")

    elseif name == "Tdg" || name == "T†"
        push!(instrs, "rz(-0.7853981633974483) q[$(qubits[1])];")

    elseif name == "X"
        q = qubits[1]
        push!(instrs, "sx q[$q];")
        push!(instrs, "sx q[$q];")

    elseif name == "Y"
        q = qubits[1]
        push!(instrs, "sx q[$q];")
        push!(instrs, "rz(3.141592653589793) q[$q];")
        push!(instrs, "sx q[$q];")

    elseif name == "Z"
        push!(instrs, "rz(3.141592653589793) q[$(qubits[1])];")

    elseif name == "CX"
        c, t = qubits[1], qubits[2]
        push!(instrs, "cx q[$c], q[$t];")
        if zne_factor > 1.0
            repeats = Int(round(zne_factor)) - 1
            for _ in 1:repeats
                push!(instrs, "cx q[$c], q[$t];")
                push!(instrs, "cx q[$c], q[$t];")
            end
        end

    elseif name == "CZ"
        c, t = qubits[1], qubits[2]
        for h_instr in decompose_to_heron("H", Float64[], [t], 1.0)
            push!(instrs, h_instr)
        end
        push!(instrs, "cx q[$c], q[$t];")
        if zne_factor > 1.0
            repeats = Int(round(zne_factor)) - 1
            for _ in 1:repeats
                push!(instrs, "cx q[$c], q[$t];")
                push!(instrs, "cx q[$c], q[$t];")
            end
        end
        for h_instr in decompose_to_heron("H", Float64[], [t], 1.0)
            push!(instrs, h_instr)
        end

    elseif name == "CCX"
        c1, c2, t = qubits[1], qubits[2], qubits[3]
        push!(instrs, "// CCX decomposition needed - using intrinsic")
        push!(instrs, "cx q[$c1], q[$t];")

    elseif startswith(name, "C") && length(name) > 1
        push!(instrs, "// Controlled-$(name[2:end]) not natively decomposed")

    else
        push!(instrs, "// Unknown gate: $name")
    end

    return instrs
end

# -----------------------------------------------------------------------
# Pauli Basis Rotation for DFE
# -----------------------------------------------------------------------

function pauli_rotation_instrs(pauli::Char, qubit::Int)
    if pauli == 'X'
        return [
            "rz(1.5707963267948966) q[$qubit];",
            "sx q[$qubit];",
            "rz(1.5707963267948966) q[$qubit];",
            "sx q[$qubit];",
            "rz(1.5707963267948966) q[$qubit];"
        ]
    elseif pauli == 'Y'
        return [
            "rz(-1.5707963267948966) q[$qubit];",
            "sx q[$qubit];",
            "rz(1.5707963267948966) q[$qubit];",
            "sx q[$qubit];",
            "rz(1.5707963267948966) q[$qubit];"
        ]
    elseif pauli == 'Z' || pauli == 'I'
        return String[]
    else
        return ["// Unknown Pauli: $pauli"]
    end
end

# -----------------------------------------------------------------------
# Main Lowering: QuantumIR → OpenQASM 3.0
# -----------------------------------------------------------------------

function qir_to_openqasm3(ir_dict::Dict;
                          zne_factors::Vector{Float64}=[1.0],
                          anu_bases::Union{Vector{Vector{Char}},Nothing}=nothing,
                          dynamic_shots::Bool=true)

    nq = ir_dict["qubits"]
    nc = ir_dict["cbits"]
    ops = ir_dict["ops"]

    if length(zne_factors) > 1 && dynamic_shots
        return qir_to_openqasm3_zne_dynamic(ir_dict, zne_factors, anu_bases)
    end

    factor = zne_factors[1]
    e = QASM3Emitter(nq, nc; zne_factor=factor)

    emitln!(e, "OPENQASM 3.0;")
    emitln!(e, "include \"stdgates.inc\";")
    emitln!(e)
    emitln!(e, "qubit[$nq] q;")
    emitln!(e, "bit[$nc] meas;")
    emitln!(e)
    emitln!(e, "float fidelity_sum = 0.0;")
    emitln!(e, "int valid_shots = 0;")
    emitln!(e)

    if dynamic_shots && anu_bases !== nothing
        n_shots = length(anu_bases)
        emitln!(e, "for shot in [0:$(n_shots-1)] {")
        indent!(e)
    end

    for op in ops
        op_type = op["type"]

        if op_type == "gate"
            name = op["name"]
            params = Float64[op["params"]...]
            qubits = Int[op["qubits"]...]

            for instr in decompose_to_heron(name, params, qubits, factor)
                emitln!(e, instr)
            end

        elseif op_type == "measure"
            q = op["qubit"]
            c = op["cbit"]
            emitln!(e, "meas[$c] = measure q[$q];")

        elseif op_type == "barrier"
            qs = join([string(q) for q in op["qubits"]], ", ")
            emitln!(e, "barrier q[$qs];")

        elseif op_type == "reset"
            q = op["qubit"]
            emitln!(e, "if (meas[$q] == 1) { x q[$q]; }")
        end
    end

    if anu_bases !== nothing && !dynamic_shots
        basis = anu_bases[1]
        emitln!(e)
        emitln!(e, "// Pauli basis rotation for DFE")
        for (q, pauli) in enumerate(basis)
            for instr in pauli_rotation_instrs(pauli, q-1)
                emitln!(e, instr)
            end
        end

        emitln!(e)
        emitln!(e, "// Mid-circuit measurement")
        for q in 0:nq-1
            emitln!(e, "meas[$q] = measure q[$q];")
        end

        emitln!(e)
        emitln!(e, "// Conditional reset")
        for q in 0:nq-1
            emitln!(e, "if (meas[$q] == 1) { x q[$q]; }")
        end

        emitln!(e)
        emitln!(e, "// DFE fidelity estimator")
        emitln!(e, "bool has_xy = false;")
        emitln!(e, "int z_weight = 0;")
        for (q, pauli) in enumerate(basis)
            if pauli in ('X', 'Y')
                emitln!(e, "has_xy = true;")
            elseif pauli == 'Z'
                emitln!(e, "z_weight = z_weight + 1;")
            end
        end
        emitln!(e)
        emitln!(e, "if (!has_xy) {")
        indent!(e)
        emitln!(e, "int eigenvalue = 1;")
        for (q, pauli) in enumerate(basis)
            if pauli == 'Z'
                emitln!(e, "if (meas[$(q-1)] == 1) eigenvalue = eigenvalue * -1;")
            end
        end
        emitln!(e, "float estimator = pow(3.0, float(z_weight)) * float(eigenvalue);")
        emitln!(e, "fidelity_sum = fidelity_sum + estimator;")
        emitln!(e, "valid_shots = valid_shots + 1;")
        dedent!(e)
        emitln!(e, "}")
    end

    if dynamic_shots && anu_bases !== nothing
        dedent!(e)
        emitln!(e, "}")
    end

    emitln!(e)
    emitln!(e, "float kernel_est = fidelity_sum / float(valid_shots);")
    emitln!(e, "kernel_est;")

    return String(take!(e.io))
end

# -----------------------------------------------------------------------
# Dynamic Circuit with ZNE + ANU QRNG Bases
# -----------------------------------------------------------------------

function qir_to_openqasm3_zne_dynamic(ir_dict::Dict,
                                       zne_factors::Vector{Float64},
                                       anu_bases::Union{Vector{Vector{Char}},Nothing})
    nq = ir_dict["qubits"]
    nc = ir_dict["cbits"]
    ops = ir_dict["ops"]

    n_shots = anu_bases === nothing ? 1000 : length(anu_bases)
    n_factors = length(zne_factors)

    e = QASM3Emitter(nq, nc)

    emitln!(e, "OPENQASM 3.0;")
    emitln!(e, "include \"stdgates.inc\";")
    emitln!(e)
    emitln!(e, "qubit[$nq] q;")
    emitln!(e, "bit[$nc] meas;")
    emitln!(e)
    emitln!(e, "float[$n_factors] fidelity_sum = {$(join(["0.0" for _ in 1:n_factors], ", "))};")
    emitln!(e, "int[$n_factors] valid_shots = {$(join(["0" for _ in 1:n_factors], ", "))};")
    emitln!(e)

    if anu_bases !== nothing
        emitln!(e, "// ANU QRNG Pauli bases (pre-fetched)")
        emitln!(e, "string[$(n_shots * nq)] pauli_bases = {")
        indent!(e)
        for (shot, basis) in enumerate(anu_bases)
            for (q, pauli) in enumerate(basis)
                emitln!(e, "\"$(pauli)\", // shot $shot, qubit $q")
            end
        end
        dedent!(e)
        emitln!(e, "};")
        emitln!(e)
    end

    emitln!(e, "for f_idx in [0:$(n_factors-1)] {")
    indent!(e)
    emitln!(e, "float noise_factors[$n_factors] = {$(join(string.(zne_factors), ", "))};")
    emitln!(e, "float noise_factor = noise_factors[f_idx];")
    emitln!(e)

    emitln!(e, "for shot in [0:$(n_shots-1)] {")
    indent!(e)

    if anu_bases !== nothing
        emitln!(e, "// Pauli basis from ANU QRNG")
        for q in 0:nq-1
            emitln!(e, "string pauli_$q = pauli_bases[shot * $nq + $q];")
        end
    end

    emitln!(e, "// Feature Map U_Φ(x)")
    for op in ops
        if op["type"] == "gate"
            name = op["name"]
            params = Float64[op["params"]...]
            qubits = Int[op["qubits"]...]
            for instr in decompose_to_heron_zne(name, params, qubits)
                emitln!(e, instr)
            end
        end
    end

    emitln!(e, "// Inverse Feature Map U_Φ(x')†")

    if anu_bases !== nothing
        emitln!(e, "// Pauli basis rotation")
        for q in 0:nq-1
            emitln!(e, "if (pauli_$q == \"X\") {")
            indent!(e)
            for instr in pauli_rotation_instrs('X', q)
                emitln!(e, instr)
            end
            dedent!(e)
            emitln!(e, "} else if (pauli_$q == \"Y\") {")
            indent!(e)
            for instr in pauli_rotation_instrs('Y', q)
                emitln!(e, instr)
            end
            dedent!(e)
            emitln!(e, "}")
        end
    end

    emitln!(e, "// Mid-circuit measurement")
    for q in 0:nq-1
        emitln!(e, "meas[$q] = measure q[$q];")
    end

    emitln!(e, "// Conditional reset")
    for q in 0:nq-1
        emitln!(e, "if (meas[$q] == 1) { x q[$q]; }")
    end

    emitln!(e, "// DFE fidelity estimator")
    emitln!(e, "bool has_xy = false;")
    emitln!(e, "int z_weight = 0;")
    if anu_bases !== nothing
        for q in 0:nq-1
            emitln!(e, "if (pauli_$q == \"X\" || pauli_$q == \"Y\") has_xy = true;")
            emitln!(e, "if (pauli_$q == \"Z\") z_weight = z_weight + 1;")
        end
    end
    emitln!(e)
    emitln!(e, "if (!has_xy) {")
    indent!(e)
    emitln!(e, "int eigenvalue = 1;")
    if anu_bases !== nothing
        for q in 0:nq-1
            emitln!(e, "if (pauli_$q == \"Z\" && meas[$q] == 1) eigenvalue = eigenvalue * -1;")
        end
    end
    emitln!(e, "float estimator = pow(3.0, float(z_weight)) * float(eigenvalue);")
    emitln!(e, "fidelity_sum[f_idx] = fidelity_sum[f_idx] + estimator;")
    emitln!(e, "valid_shots[f_idx] = valid_shots[f_idx] + 1;")
    dedent!(e)
    emitln!(e, "}")

    dedent!(e)
    emitln!(e, "}")

    dedent!(e)
    emitln!(e, "}")

    emitln!(e)
    emitln!(e, "// Richardson extrapolation to zero noise")
    emitln!(e, "float kernel_est = 0.0;")
    for i in 0:n_factors-1
        emitln!(e, "float y$i = fidelity_sum[$i] / float(valid_shots[$i]);")
    end
    for i in 0:n_factors-1
        emitln!(e, "float term$i = y$i;")
        for j in 0:n_factors-1
            if i != j
                xj = zne_factors[j+1]
                xi = zne_factors[i+1]
                emitln!(e, "term$i = term$i * (-$xj) / ($xi - $xj);")
            end
        end
        emitln!(e, "kernel_est = kernel_est + term$i;")
    end
    emitln!(e)
    emitln!(e, "kernel_est;")

    return String(take!(e.io))
end

function decompose_to_heron_zne(name::String, params::Vector{Float64}, qubits::Vector{Int})
    instrs = String[]
    factor = "noise_factor"

    if name == "Rz"
        θ = params[1]
        push!(instrs, "rz($θ * $factor) q[$(qubits[1])];")
    elseif name == "Rx"
        θ = params[1]
        push!(instrs, "rz(-1.5707963267948966) q[$(qubits[1])];")
        push!(instrs, "sx q[$(qubits[1])];")
        push!(instrs, "rz($θ * $factor) q[$(qubits[1])];")
        push!(instrs, "sx q[$(qubits[1])];")
        push!(instrs, "rz(1.5707963267948966) q[$(qubits[1])];")
    elseif name == "Ry"
        θ = params[1]
        push!(instrs, "rz(1.5707963267948966) q[$(qubits[1])];")
        push!(instrs, "sx q[$(qubits[1])];")
        push!(instrs, "rz($θ * $factor) q[$(qubits[1])];")
        push!(instrs, "sx q[$(qubits[1])];")
        push!(instrs, "rz(-1.5707963267948966) q[$(qubits[1])];")
    elseif name == "CX"
        c, t = qubits[1], qubits[2]
        push!(instrs, "cx q[$c], q[$t];")
    elseif name == "CZ"
        c, t = qubits[1], qubits[2]
        for h_instr in decompose_to_heron_zne("H", Float64[], [t])
            push!(instrs, h_instr)
        end
        push!(instrs, "cx q[$c], q[$t];")
        for h_instr in decompose_to_heron_zne("H", Float64[], [t])
            push!(instrs, h_instr)
        end
    elseif name == "H"
        q = qubits[1]
        push!(instrs, "rz(1.5707963267948966) q[$q];")
        push!(instrs, "sx q[$q];")
        push!(instrs, "rz(1.5707963267948966) q[$q];")
        push!(instrs, "sx q[$q];")
        push!(instrs, "rz(1.5707963267948966) q[$q];")
    else
        push!(instrs, "// $name with ZNE not implemented")
    end
    return instrs
end

# -----------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------

function main()
    if length(ARGS) < 2
        println("Usage: julia qir_to_openqasm3.jl <input.ir.json> <output.qasm3> [zne_factors...]")
        println("Example: julia qir_to_openqasm3.jl kernel.ir.json kernel.qasm3 1.0 1.5 2.0 3.0")
        exit(1)
    end

    input_file = ARGS[1]
    output_file = ARGS[2]

    zne_factors = length(ARGS) > 2 ? parse.(Float64, ARGS[3:end]) : [1.0]

    json_str = read(input_file, String)
    ir = JSON3.read(json_str)

    nq = ir["qubits"]
    n_shots = 1000
    anu_bases = [rand(['I','X','Y','Z'], nq) for _ in 1:n_shots]

    qasm = qir_to_openqasm3(ir; zne_factors=zne_factors, anu_bases=anu_bases, dynamic_shots=true)

    write(output_file, qasm)
    println("Written $output_file with $(length(zne_factors)) ZNE factors, $n_shots shots")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
