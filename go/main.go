package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Complex64 struct {
	Real float32
	Imag float32
}

type StateVector struct {
	data      []Complex64
	numQubits int
}

func NewStateVector(n int) *StateVector {
	size := 1 << n
	data := make([]Complex64, size)
	data[0] = Complex64{Real: 1.0, Imag: 0.0}
	return &StateVector{data: data, numQubits: n}
}

func (sv *StateVector) Apply1Qubit(q int, u [2][2]Complex64) {
	n := sv.numQubits
	block := 1 << (q + 1)
	stride := 1 << q
	for base := 0; base < (1 << n); base += block {
		for offset := 0; offset < stride; offset++ {
			i0 := base + offset
			i1 := i0 + stride
			a := sv.data[i0]
			b := sv.data[i1]
			sv.data[i0] = Complex64{
				Real: u[0][0].Real*a.Real - u[0][0].Imag*a.Imag + u[0][1].Real*b.Real - u[0][1].Imag*b.Imag,
				Imag: u[0][0].Real*a.Imag + u[0][0].Imag*a.Real + u[0][1].Real*b.Imag + u[0][1].Imag*b.Real,
			}
			sv.data[i1] = Complex64{
				Real: u[1][0].Real*a.Real - u[1][0].Imag*a.Imag + u[1][1].Real*b.Real - u[1][1].Imag*b.Imag,
				Imag: u[1][0].Real*a.Imag + u[1][0].Imag*a.Real + u[1][1].Real*b.Imag + u[1][1].Imag*b.Real,
			}
		}
	}
}

func (sv *StateVector) Apply2Qubit(q1, q2 int, u [4][4]Complex64) {
	n := sv.numQubits
	for i := 0; i < (1 << n); i++ {
		b1 := (i >> q1) & 1
		b2 := (i >> q2) & 1
		idx := b1*2 + b2
		if idx != 0 {
			continue
		}
		i00 := i
		i01 := i | (1 << q2)
		i10 := i | (1 << q1)
		i11 := i | (1 << q1) | (1 << q2)
		indices := [4]int{i00, i01, i10, i11}
		var vals [4]Complex64
		for k := 0; k < 4; k++ {
			vals[k] = sv.data[indices[k]]
		}
		for row := 0; row < 4; row++ {
			var sum Complex64
			for col := 0; col < 4; col++ {
				a := u[row][col]
				b := vals[col]
				sum.Real += a.Real*b.Real - a.Imag*b.Imag
				sum.Imag += a.Real*b.Imag + a.Imag*b.Real
			}
			sv.data[indices[row]] = sum
		}
	}
}

func (sv *StateVector) Copy() *StateVector {
	newData := make([]Complex64, len(sv.data))
	copy(newData, sv.data)
	return &StateVector{data: newData, numQubits: sv.numQubits}
}

func (sv *StateVector) InnerProduct(other *StateVector) Complex64 {
	var sum Complex64
	for i := range sv.data {
		a := sv.data[i]
		b := other.data[i]
		sum.Real += a.Real*b.Real + a.Imag*b.Imag
		sum.Imag += a.Real*b.Imag - a.Imag*b.Real
	}
	return sum
}

func RZGate(theta float64) [2][2]Complex64 {
	c := float32(math.Cos(theta / 2))
	s := float32(math.Sin(theta / 2))
	return [2][2]Complex64{
		{{Real: c, Imag: -s}, {Real: 0, Imag: 0}},
		{{Real: 0, Imag: 0}, {Real: c, Imag: s}},
	}
}

func RYGate(theta float64) [2][2]Complex64 {
	c := float32(math.Cos(theta / 2))
	s := float32(math.Sin(theta / 2))
	return [2][2]Complex64{
		{{Real: c, Imag: 0}, {Real: -s, Imag: 0}},
		{{Real: s, Imag: 0}, {Real: c, Imag: 0}},
	}
}

func CZGate() [4][4]Complex64 {
	return [4][4]Complex64{
		{{Real: 1}, {}, {}, {}},
		{{}, {Real: 1}, {}, {}},
		{{}, {}, {Real: 1}, {}},
		{{}, {}, {}, {Real: -1}},
	}
}

type QuantumKernelSVM struct {
	NQubits  int
	NLayers  int
	Shots    int
	EntGraph [][2]int
	C        float64
	Params   [][]float64
}

func NewQuantumKernelSVM(nQubits, nLayers, shots int) *QuantumKernelSVM {
	entGraph := make([][2]int, nQubits-1)
	for i := 0; i < nQubits-1; i++ {
		entGraph[i] = [2]int{i, i + 1}
	}

	params := make([][]float64, nLayers)
	for l := 0; l < nLayers; l++ {
		params[l] = make([]float64, 3*nQubits)
		for p := 0; p < 3*nQubits; p++ {
			params[l][p] = 1.0 + rand.Float64()*0.2 - 0.1
		}
	}

	return &QuantumKernelSVM{
		NQubits:  nQubits,
		NLayers:  nLayers,
		Shots:    shots,
		EntGraph: entGraph,
		C:        1.0,
		Params:   params,
	}
}

func (svm *QuantumKernelSVM) ApplyFeatureMap(sv *StateVector, features []float64) {
	for layer := 0; layer < svm.NLayers; layer++ {
		for q := 0; q < svm.NQubits; q++ {
			x := features[q%len(features)]
			tz1 := svm.Params[layer][3*q]
			ty := svm.Params[layer][3*q+1]
			tz2 := svm.Params[layer][3*q+2]
			sv.Apply1Qubit(q, RZGate(2*x*tz1))
			sv.Apply1Qubit(q, RYGate(2*x*ty))
			sv.Apply1Qubit(q, RZGate(2*x*tz2))
		}
		for _, edge := range svm.EntGraph {
			sv.Apply2Qubit(edge[0], edge[1], CZGate())
		}
	}
}

func (svm *QuantumKernelSVM) KernelExact(featuresA, featuresB []float64) float64 {
	svA := NewStateVector(svm.NQubits)
	svm.ApplyFeatureMap(svA, featuresA)
	svB := NewStateVector(svm.NQubits)
	svm.ApplyFeatureMap(svB, featuresB)
	ip := svA.InnerProduct(svB)
	return float64(ip.Real*ip.Real + ip.Imag*ip.Imag)
}

func (svm *QuantumKernelSVM) KernelShots(featuresA, featuresB []float64) float64 {
	exact := svm.KernelExact(featuresA, featuresB)
	p0 := (1.0 + exact) / 2.0
	countZero := 0
	for s := 0; s < svm.Shots; s++ {
		if rand.Float64() < p0 {
			countZero++
		}
	}
	return 2.0*float64(countZero)/float64(svm.Shots) - 1.0
}

func (svm *QuantumKernelSVM) ComputeKernelMatrix(dataset [][]float64) [][]float64 {
	n := len(dataset)
	K := make([][]float64, n)
	for i := range K {
		K[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			kij := svm.KernelShots(dataset[i], dataset[j])
			K[i][j] = kij
			K[j][i] = kij
		}
	}
	return K
}

func (svm *QuantumKernelSVM) SolveDual(K [][]float64, labels []float64) ([]float64, float64) {
	n := len(labels)
	alpha := make([]float64, n)
	b := 0.0

	for iter := 0; iter < 1000; iter++ {
		maxV := 0.0
		for i := 0; i < n; i++ {
			grad := 1.0
			for j := 0; j < n; j++ {
				grad -= alpha[j] * labels[j] * K[i][j] * labels[i]
			}
			v := math.Abs(grad)
			if v > maxV {
				maxV = v
			}
			alpha[i] = math.Max(0, math.Min(svm.C, alpha[i]+0.01*labels[i]*grad))
		}
		if maxV < 1e-4 {
			break
		}
	}

	svIndices := []int{}
	for i := 0; i < n; i++ {
		if alpha[i] > 1e-5 && alpha[i] < svm.C-1e-5 {
			svIndices = append(svIndices, i)
		}
	}
	if len(svIndices) > 0 {
		bSum := 0.0
		for _, k := range svIndices {
			sum := 0.0
			for j := 0; j < n; j++ {
				sum += alpha[j] * labels[j] * K[k][j]
			}
			bSum += labels[k] - sum
		}
		b = bSum / float64(len(svIndices))
	}
	return alpha, b
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("============================================================")
	fmt.Println("QUANTUM KERNEL SVM — 5 Qubit Hello World (Go Simulator)")
	fmt.Println("State Vector Engine | Shot-Based SWAP Test | SMO Solver")
	fmt.Println("============================================================")
	fmt.Println()

	nQubits := 5
	nLayers := 2
	shots := 1000

	svm := NewQuantumKernelSVM(nQubits, nLayers, shots)
	fmt.Printf("Qubits: %d | Layers: %d | Shots: %d\n", nQubits, nLayers, shots)
	fmt.Printf("Hilbert space dim: 2^%d = %d\n", nQubits, 1<<nQubits)
	fmt.Printf("Entanglement: linear chain %v\n", svm.EntGraph)
	fmt.Println()

	dataset := [][]float64{
		{0, 0, 0, 0, 0},
		{0, 1, 0, 1, 0},
		{1, 0, 1, 0, 1},
		{1, 1, 1, 1, 1},
		{0.5, 0.5, 0.5, 0.5, 0.5},
		{0.2, 0.8, 0.2, 0.8, 0.2},
		{0.8, 0.2, 0.8, 0.2, 0.8},
		{0.3, 0.7, 0.3, 0.7, 0.3},
	}
	labels := []float64{-1, 1, 1, -1, -1, 1, 1, -1}

	fmt.Printf("Dataset: %d samples, %d features\n", len(dataset), len(dataset[0]))
	fmt.Printf("Labels: %v\n", labels)
	fmt.Println()

	fmt.Println("Computing quantum kernel matrix...")
	start := time.Now()
	K := svm.ComputeKernelMatrix(dataset)
	elapsed := time.Since(start)
	fmt.Printf("Done in %v\n\n", elapsed)

	fmt.Println("Kernel matrix (4x4 corner):")
	for i := 0; i < 4; i++ {
		fmt.Printf("  [")
		for j := 0; j < 4; j++ {
			fmt.Printf(" %7.4f", K[i][j])
		}
		fmt.Println(" ]")
	}
	fmt.Println()

	fmt.Println("Training SVM (dual solver)...")
	alpha, bias := svm.SolveDual(K, labels)
	svCount := 0
	for _, a := range alpha {
		if a > 1e-5 {
			svCount++
		}
	}
	fmt.Printf("Support vectors: %d / %d\n", svCount, len(labels))
	fmt.Printf("Bias: %.4f\n\n", bias)

	fmt.Println("Predictions:")
	correct := 0
	for i := 0; i < len(dataset); i++ {
		decision := bias
		for j := 0; j < len(dataset); j++ {
			decision += alpha[j] * labels[j] * K[j][i]
		}
		pred := 1.0
		if decision < 0 {
			pred = -1.0
		}
		match := "OK"
		if pred != labels[i] {
			match = "MISS"
		} else {
			correct++
		}
		fmt.Printf("  x[%d] -> decision=%.4f, pred=%+.0f, true=%+.0f [%s]\n", i, decision, pred, labels[i], match)
	}
	fmt.Printf("\nAccuracy: %d / %d = %.1f%%\n", correct, len(labels), 100*float64(correct)/float64(len(labels)))

	fmt.Println()
	fmt.Println("------------------------------------------------------------")
	fmt.Println("Shot noise analysis (kernel[0,1]):")
	estimates := make([]float64, 20)
	for i := range estimates {
		estimates[i] = svm.KernelShots(dataset[0], dataset[1])
	}
	mean := 0.0
	for _, e := range estimates {
		mean += e
	}
	mean /= float64(len(estimates))
	variance := 0.0
	for _, e := range estimates {
		variance += (e - mean) * (e - mean)
	}
	variance /= float64(len(estimates))
	exact := svm.KernelExact(dataset[0], dataset[1])
	fmt.Printf("  Mean:  %.6f\n", mean)
	fmt.Printf("  Std:   %.6f\n", math.Sqrt(variance))
	fmt.Printf("  Exact: %.6f\n", exact)

	fmt.Println()
	fmt.Println("============================================================")
	fmt.Println("HELLO WORLD COMPLETE — 5 qubit quantum kernel executed")
	fmt.Println("============================================================")
}
