package turboquant

import (
	"math"
	"testing"
)

// testDims covers small, medium, and the primary production dims.
var testDims = []int{4, 8, 16, 32, 64, 128}

func TestBuildRotationDeterministic(t *testing.T) {
	for _, dim := range testDims {
		a := BuildRotation(dim, 123)
		b := BuildRotation(dim, 123)
		if a.Dim != b.Dim || a.Seed != b.Seed || len(a.Matrix) != len(b.Matrix) {
			t.Fatalf("rotation metadata mismatch for dim %d", dim)
		}
		for i := range a.Matrix {
			if a.Matrix[i] != b.Matrix[i] {
				t.Fatalf("rotation mismatch at dim=%d idx=%d", dim, i)
			}
		}
	}
}

func TestBuildRotationDifferentSeedsDiffer(t *testing.T) {
	a := BuildRotation(16, 111)
	b := BuildRotation(16, 222)
	different := false
	for i := range a.Matrix {
		if a.Matrix[i] != b.Matrix[i] {
			different = true
			break
		}
	}
	if !different {
		t.Fatal("different seeds produced the same orthogonal matrix")
	}
}

func TestApplyInverseRotation(t *testing.T) {
	for _, dim := range testDims {
		values := pseudoRandomVector(dim, uint64(dim)*17)
		rot := BuildRotation(dim, uint64(dim)*19)
		got := ApplyInverseRotation(ApplyRotation(values, rot), rot)
		for i := range values {
			if abs32(values[i]-got[i]) > 1e-4 {
				t.Fatalf("dim=%d idx=%d got=%v want=%v", dim, i, got[i], values[i])
			}
		}
	}
}

func TestRotationPreservesNorm(t *testing.T) {
	for _, dim := range testDims {
		values := pseudoRandomVector(dim, uint64(dim)*23)
		rot := BuildRotation(dim, uint64(dim)*29)
		before := vectorNorm(values)
		after := vectorNorm(ApplyRotation(values, rot))
		if abs32(before-after) > 1e-3 {
			t.Fatalf("dim=%d norm drift=%v", dim, abs32(before-after))
		}
	}
}

// TestBuildRotationIsOrthogonal verifies that Q satisfies Q*Q^T = I (rows are
// orthonormal). This is the core invariant required by the TurboQuant encoding
// and is guaranteed unconditionally by the Householder QR algorithm.
func TestBuildRotationIsOrthogonal(t *testing.T) {
	// Include dim=256 to exercise the algorithm well beyond typical head_dim.
	for _, dim := range append(testDims, 256) {
		rot := BuildRotation(dim, uint64(dim)*31)
		for i := 0; i < dim; i++ {
			for j := i; j < dim; j++ {
				var dot float32
				for k := 0; k < dim; k++ {
					dot += rot.Matrix[i*dim+k] * rot.Matrix[j*dim+k]
				}
				if i == j {
					if math.Abs(float64(dot-1)) > 5e-5 {
						t.Fatalf("dim=%d row %d: self-dot=%.6f, want 1.0", dim, i, dot)
					}
				} else if math.Abs(float64(dot)) > 5e-5 {
					t.Fatalf("dim=%d rows %d,%d: cross-dot=%.6f, want 0.0", dim, i, j, dot)
				}
			}
		}
	}
}
