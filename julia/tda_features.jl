# tda_features.jl — Vietoris-Rips → Persistence Barcodes → Feature Vectors

module TDAFeatures

using LinearAlgebra
using Statistics

export VietorisRipsComplex, PersistenceDiagram, Barcode, barcode_to_feature_vector
export compute_persistence, wasserstein_distance, bottleneck_distance
export PersistenceInterval

# ═══════════════════════════════════════════════════════════════════════
# Vietoris-Rips Complex
# ═══════════════════════════════════════════════════════════════════════

struct VietorisRipsComplex
    points::Matrix{Float64}
    max_dim::Int
    epsilon::Float64
    simplices::Vector{Vector{Int}}
    filtration_values::Vector{Float64}
end

function VietorisRipsComplex(points::Matrix{Float64}, epsilon::Float64; max_dim::Int=2)
    n = size(points, 1)
    simplices = Vector{Int}[]
    filt_vals = Float64[]

    for i in 1:n
        push!(simplices, [i])
        push!(filt_vals, 0.0)
    end

    for i in 1:n, j in i+1:n
        d = norm(points[i,:] - points[j,:])
        if d <= epsilon
            push!(simplices, [i, j])
            push!(filt_vals, d)
        end
    end

    if max_dim >= 2
        for i in 1:n, j in i+1:n, k in j+1:n
            d_ij = norm(points[i,:] - points[j,:])
            d_jk = norm(points[j,:] - points[k,:])
            d_ik = norm(points[i,:] - points[k,:])
            if d_ij <= epsilon && d_jk <= epsilon && d_ik <= epsilon
                push!(simplices, [i, j, k])
                push!(filt_vals, max(d_ij, d_jk, d_ik))
            end
        end
    end

    VietorisRipsComplex(points, max_dim, epsilon, simplices, filt_vals)
end

# ═══════════════════════════════════════════════════════════════════════
# Persistent Homology (H0 and H1)
# ═══════════════════════════════════════════════════════════════════════

struct PersistenceInterval
    dim::Int
    birth::Float64
    death::Float64
end

struct PersistenceDiagram
    intervals::Vector{PersistenceInterval}
end

struct Barcode
    H0::Vector{PersistenceInterval}
    H1::Vector{PersistenceInterval}
end

function compute_persistence(vr::VietorisRipsComplex)::Barcode
    n = size(vr.points, 1)
    order = sortperm(vr.filtration_values)

    # H0: Connected components (union-find)
    parent = collect(1:n)
    rank = zeros(Int, n)

    function find(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end

    function union!(x, y)
        rx, ry = find(x), find(y)
        if rx != ry
            if rank[rx] < rank[ry]
                parent[rx] = ry
            elseif rank[rx] > rank[ry]
                parent[ry] = rx
            else
                parent[ry] = rx
                rank[rx] += 1
            end
            return true
        end
        return false
    end

    H0_intervals = PersistenceInterval[]
    for idx in order
        simp = vr.simplices[idx]
        val = vr.filtration_values[idx]
        if length(simp) == 2
            if union!(simp[1], simp[2])
                push!(H0_intervals, PersistenceInterval(0, 0.0, val))
            end
        end
    end

    max_filt = maximum(vr.filtration_values)
    for i in 1:n
        if find(i) == i
            push!(H0_intervals, PersistenceInterval(0, 0.0, max_filt))
        end
    end

    # H1: Cycles
    H1_intervals = PersistenceInterval[]
    parent_h1 = collect(1:n)
    function find_h1(x)
        while parent_h1[x] != x
            parent_h1[x] = parent_h1[parent_h1[x]]
            x = parent_h1[x]
        end
        return x
    end
    function union_h1!(x, y)
        rx, ry = find_h1(x), find_h1(y)
        if rx != ry
            parent_h1[rx] = ry
            return false
        end
        return true
    end

    for idx in order
        simp = vr.simplices[idx]
        val = vr.filtration_values[idx]
        if length(simp) == 2
            if union_h1!(simp[1], simp[2])
                push!(H1_intervals, PersistenceInterval(1, val, max_filt))
            end
        end
    end

    Barcode(H0_intervals, H1_intervals)
end

# ═══════════════════════════════════════════════════════════════════════
# Barcode → Feature Vector
# ═══════════════════════════════════════════════════════════════════════

function barcode_to_feature_vector(bc::Barcode; n_bins::Int=50, max_filt::Float64=1.0)::Vector{Float64}
    features = Float64[]

    for intervals in [bc.H0, bc.H1]
        if isempty(intervals)
            append!(features, zeros(n_bins))
            continue
        end

        landscape = zeros(n_bins)
        for intv in intervals
            mid = (intv.birth + intv.death) / 2
            half_pers = (intv.death - intv.birth) / 2
            for (i, t) in enumerate(range(0, max_filt, length=n_bins))
                val = max(0.0, half_pers - abs(t - mid))
                landscape[i] = max(landscape[i], val)
            end
        end
        append!(features, landscape)
    end

    push!(features, Float64(length(bc.H0)))
    push!(features, Float64(length(bc.H1)))
    push!(features, sum(i.death - i.birth for i in bc.H0))
    push!(features, sum(i.death - i.birth for i in bc.H1))
    push!(features, maximum([i.death - i.birth for i in bc.H1]; init=0.0))

    return features
end

# ═══════════════════════════════════════════════════════════════════════
# Distances Between Barcodes
# ═══════════════════════════════════════════════════════════════════════

function wasserstein_distance(bc1::Barcode, bc2::Barcode; p::Int=2)::Float64
    dist = 0.0
    for (intervals1, intervals2) in [(bc1.H0, bc2.H0), (bc1.H1, bc2.H1)]
        n1, n2 = length(intervals1), length(intervals2)
        if n1 == 0 && n2 == 0
            continue
        elseif n1 == 0
            dist += sum((i.death - i.birth)^p for i in intervals2)
        elseif n2 == 0
            dist += sum((i.death - i.birth)^p for i in intervals1)
        else
            sorted1 = sort(intervals1, by=i -> i.death - i.birth, rev=true)
            sorted2 = sort(intervals2, by=i -> i.death - i.birth, rev=true)
            for (i1, i2) in zip(sorted1, sorted2)
                dist += abs((i1.death - i1.birth) - (i2.death - i2.birth))^p
            end
        end
    end
    return dist^(1/p)
end

function bottleneck_distance(bc1::Barcode, bc2::Barcode)::Float64
    wasserstein_distance(bc1, bc2; p=100)
end

end # module TDAFeatures
