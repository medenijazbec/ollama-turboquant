package turboquant

import "testing"

func TestScalarCodebookDeterministic(t *testing.T) {
	for _, bits := range []int{2, 3} {
		codebookA, boundsA := scalarCodebook(128, bits)
		codebookB, boundsB := scalarCodebook(128, bits)
		if len(codebookA) != 1<<bits {
			t.Fatalf("bits=%d codebook len=%d", bits, len(codebookA))
		}
		for i := range codebookA {
			if codebookA[i] != codebookB[i] {
				t.Fatalf("bits=%d centroid mismatch at %d", bits, i)
			}
		}
		for i := range boundsA {
			if boundsA[i] != boundsB[i] {
				t.Fatalf("bits=%d boundary mismatch at %d", bits, i)
			}
		}
	}
}

func TestCodebookBoundariesMonotonic(t *testing.T) {
	for _, bits := range []int{2, 3} {
		_, bounds := scalarCodebook(128, bits)
		for i := 1; i < len(bounds); i++ {
			if bounds[i] <= bounds[i-1] {
				t.Fatalf("bits=%d boundaries are not monotonic", bits)
			}
		}
	}
}

func TestQuantizeScalarByBoundaryDeterministic(t *testing.T) {
	codebook, bounds := scalarCodebook(128, 3)
	mid := bounds[2]

	left := quantizeScalarByBoundary(mid-1e-6, codebook, bounds)
	right := quantizeScalarByBoundary(mid+1e-6, codebook, bounds)
	atBoundary := quantizeScalarByBoundary(mid, codebook, bounds)

	if left != 2 {
		t.Fatalf("left boundary bucket = %d, want 2", left)
	}
	if right != 3 {
		t.Fatalf("right boundary bucket = %d, want 3", right)
	}
	if atBoundary != 3 {
		t.Fatalf("exact boundary bucket = %d, want 3", atBoundary)
	}
}

func TestSelectOutliersStableOnTies(t *testing.T) {
	values := []float32{4, -4, 4, 1, -1}
	got := selectOutliers(values, 2)
	if len(got) != 2 || got[0] != 0 || got[1] != 1 {
		t.Fatalf("selectOutliers = %v, want [0 1]", got)
	}
}
