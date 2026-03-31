package turboquant

import "testing"

func TestBuildRotationDeterministic(t *testing.T) {
	for _, dim := range []int{4, 8, 16, 32} {
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
	for _, dim := range []int{4, 8, 16, 32} {
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
	for _, dim := range []int{4, 8, 16, 32} {
		values := pseudoRandomVector(dim, uint64(dim)*23)
		rot := BuildRotation(dim, uint64(dim)*29)
		before := vectorNorm(values)
		after := vectorNorm(ApplyRotation(values, rot))
		if abs32(before-after) > 1e-3 {
			t.Fatalf("dim=%d norm drift=%v", dim, abs32(before-after))
		}
	}
}
