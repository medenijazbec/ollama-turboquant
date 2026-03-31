package turboquant

import (
	"math"
	"sync"
)

type Rotation struct {
	Dim    int
	Seed   uint64
	Matrix []float32 // row-major orthogonal matrix
}

type rotationCacheKey struct {
	dim  int
	seed uint64
}

var rotationCache sync.Map

func BuildRotation(dim int, seed uint64) Rotation {
	key := rotationCacheKey{dim: dim, seed: seed}
	if cached, ok := rotationCache.Load(key); ok {
		return cached.(Rotation)
	}

	rot := Rotation{
		Dim:    dim,
		Seed:   seed,
		Matrix: buildOrthogonalMatrix(dim, seed),
	}
	actual, _ := rotationCache.LoadOrStore(key, rot)
	return actual.(Rotation)
}

func ApplyRotation(x []float32, rot Rotation) []float32 {
	if len(x) != rot.Dim {
		panic("turboquant: vector length does not match rotation dimension")
	}

	out := make([]float32, rot.Dim)
	for row := 0; row < rot.Dim; row++ {
		base := row * rot.Dim
		var sum float32
		for col, value := range x {
			sum += rot.Matrix[base+col] * value
		}
		out[row] = sum
	}
	return out
}

func ApplyInverseRotation(y []float32, rot Rotation) []float32 {
	if len(y) != rot.Dim {
		panic("turboquant: vector length does not match rotation dimension")
	}

	out := make([]float32, rot.Dim)
	for row := 0; row < rot.Dim; row++ {
		yVal := y[row]
		base := row * rot.Dim
		for col := 0; col < rot.Dim; col++ {
			out[col] += rot.Matrix[base+col] * yVal
		}
	}
	return out
}

func buildOrthogonalMatrix(dim int, seed uint64) []float32 {
	if dim <= 0 {
		return nil
	}

	rows := make([][]float64, dim)
	for row := 0; row < dim; row++ {
		rows[row] = make([]float64, dim)
		rng := splitmix64(seed ^ uint64(dim)<<32 ^ uint64(row+1)*0x9e3779b97f4a7c15)
		for col := 0; col < dim; col++ {
			rows[row][col] = gaussianFloat64(&rng)
		}
	}

	for row := 0; row < dim; row++ {
		current := rows[row]
		for prev := 0; prev < row; prev++ {
			prevRow := rows[prev]
			proj := dotFloat64(current, prevRow)
			for col := 0; col < dim; col++ {
				current[col] -= proj * prevRow[col]
			}
		}

		norm := vectorNorm64(current)
		if norm < 1e-12 {
			current[row%dim] += 1
			norm = vectorNorm64(current)
		}
		for col := 0; col < dim; col++ {
			current[col] /= norm
		}

		for _, value := range current {
			if math.Abs(value) <= 1e-12 {
				continue
			}
			if value < 0 {
				for col := 0; col < dim; col++ {
					current[col] = -current[col]
				}
			}
			break
		}
	}

	out := make([]float32, dim*dim)
	for row := 0; row < dim; row++ {
		for col := 0; col < dim; col++ {
			out[row*dim+col] = float32(rows[row][col])
		}
	}
	return out
}

func dotFloat64(a, b []float64) float64 {
	var out float64
	for i := range a {
		out += a[i] * b[i]
	}
	return out
}

func vectorNorm64(values []float64) float64 {
	var sum float64
	for _, value := range values {
		sum += value * value
	}
	return math.Sqrt(sum)
}

func gaussianFloat64(rng *splitmix64) float64 {
	u1 := unitUniform64(rng)
	u2 := unitUniform64(rng)
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

func unitUniform64(rng *splitmix64) float64 {
	const scale = 1.0 / (1 << 53)
	return (float64(rng.next()>>11) + 0.5) * scale
}

func isPowerOfTwo(v int) bool {
	return v > 0 && (v&(v-1)) == 0
}

type splitmix64 uint64

func (s *splitmix64) next() uint64 {
	*s += 0x9e3779b97f4a7c15
	z := uint64(*s)
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}
