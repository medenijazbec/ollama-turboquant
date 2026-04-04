//go:build cuda

package ggml

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ollama/ollama/ml"
)

// ── helpers ──────────────────────────────────────────────────────────────────

func randSlice(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rng.NormFloat64())
	}
	return s
}

func l2norm32(v []float32) float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	return float32(math.Sqrt(sum))
}

// assertScoresClose fails the test if any element of got differs from want by
// more than tol. Large relative differences (> 1e-3) are highlighted.
func assertScoresClose(t *testing.T, want, got []float32, tol float64, label string) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: length mismatch want %d got %d", label, len(want), len(got))
	}
	maxDiff := 0.0
	maxIdx := -1
	for i, w := range want {
		d := math.Abs(float64(w - got[i]))
		if d > maxDiff {
			maxDiff = d
			maxIdx = i
		}
	}
	if maxDiff > tol {
		t.Errorf("%s: max absolute diff %.2e at cell %d (want %.6f got %.6f), tolerance %.2e",
			label, maxDiff, maxIdx, want[maxIdx], got[maxIdx], tol)
	}
}

// ── availability ─────────────────────────────────────────────────────────────

func TestTurboQuantCUDAAvailable(t *testing.T) {
	if !cudaAvailable() {
		t.Skip("CUDA device not accessible at runtime")
	}
	t.Log("CUDA device accessible — GPU scoring path enabled")
}

// ── kernel parity: MSE-only (no correction) ──────────────────────────────────

func TestTurboQuantCUDAKernelParityMSE(t *testing.T) {
	if !cudaAvailable() {
		t.Skip("CUDA not available")
	}

	rng := rand.New(rand.NewSource(0xc0ffee))
	const dim, nCells = 128, 512

	query := randSlice(rng, dim)
	dequant := randSlice(rng, nCells*dim)
	qnorm := l2norm32(query)

	cpuScores := make([]float32, nCells)
	cudaScores := make([]float32, nCells)

	scoreTurboQuantCells(query, qnorm, dequant, nil, nil, dim, nCells, cpuScores)
	scoreTurboQuantCellsCUDA(query, qnorm, dequant, nil, nil, dim, nCells, cudaScores)

	assertScoresClose(t, cpuScores, cudaScores, 1e-4, "MSE-only")
}

// ── kernel parity: product mode (with correction and Cauchy-Schwarz clamp) ───

func TestTurboQuantCUDAKernelParityProduct(t *testing.T) {
	if !cudaAvailable() {
		t.Skip("CUDA not available")
	}

	rng := rand.New(rand.NewSource(0xdeadbeef))
	const dim, nCells = 128, 512

	query := randSlice(rng, dim)
	dequant := randSlice(rng, nCells*dim)
	corr := randSlice(rng, nCells*dim)
	residNorms := make([]float32, nCells)
	for i := range residNorms {
		residNorms[i] = float32(rng.Float64() * 2) // random in [0, 2)
	}
	qnorm := l2norm32(query)

	cpuScores := make([]float32, nCells)
	cudaScores := make([]float32, nCells)

	scoreTurboQuantCells(query, qnorm, dequant, corr, residNorms, dim, nCells, cpuScores)
	scoreTurboQuantCellsCUDA(query, qnorm, dequant, corr, residNorms, dim, nCells, cudaScores)

	assertScoresClose(t, cpuScores, cudaScores, 1e-4, "product-mode")
}

// ── kernel parity: large dim (non-trivial stride loop) ───────────────────────

func TestTurboQuantCUDAKernelParityLargeDim(t *testing.T) {
	if !cudaAvailable() {
		t.Skip("CUDA not available")
	}

	rng := rand.New(rand.NewSource(42))
	const dim, nCells = 256, 256 // dim > TQ_CUDA_BLOCK_SIZE → stride loop

	query := randSlice(rng, dim)
	dequant := randSlice(rng, nCells*dim)
	corr := randSlice(rng, nCells*dim)
	residNorms := make([]float32, nCells)
	for i := range residNorms {
		residNorms[i] = float32(rng.Float64())
	}
	qnorm := l2norm32(query)

	cpuScores := make([]float32, nCells)
	cudaScores := make([]float32, nCells)

	scoreTurboQuantCells(query, qnorm, dequant, corr, residNorms, dim, nCells, cpuScores)
	scoreTurboQuantCellsCUDA(query, qnorm, dequant, corr, residNorms, dim, nCells, cudaScores)

	assertScoresClose(t, cpuScores, cudaScores, 1e-4, "large-dim-256")
}

// ── kernel parity: small dim (most threads idle, stride loop skips) ──────────

func TestTurboQuantCUDAKernelParitySmallDim(t *testing.T) {
	if !cudaAvailable() {
		t.Skip("CUDA not available")
	}

	rng := rand.New(rand.NewSource(13))
	const dim, nCells = 64, 128 // dim < TQ_CUDA_BLOCK_SIZE → upper half threads idle

	query := randSlice(rng, dim)
	dequant := randSlice(rng, nCells*dim)
	corr := randSlice(rng, nCells*dim)
	residNorms := make([]float32, nCells)
	for i := range residNorms {
		residNorms[i] = float32(rng.Float64())
	}
	qnorm := l2norm32(query)

	cpuScores := make([]float32, nCells)
	cudaScores := make([]float32, nCells)

	scoreTurboQuantCells(query, qnorm, dequant, corr, residNorms, dim, nCells, cpuScores)
	scoreTurboQuantCellsCUDA(query, qnorm, dequant, corr, residNorms, dim, nCells, cudaScores)

	assertScoresClose(t, cpuScores, cudaScores, 1e-4, "small-dim-64")
}

// ── kernel parity: tiny nCells=1 edge case ───────────────────────────────────

func TestTurboQuantCUDAKernelParitySingleCell(t *testing.T) {
	if !cudaAvailable() {
		t.Skip("CUDA not available")
	}

	rng := rand.New(rand.NewSource(1))
	const dim, nCells = 128, 1

	query := randSlice(rng, dim)
	dequant := randSlice(rng, nCells*dim)
	corr := randSlice(rng, nCells*dim)
	residNorms := []float32{0.5}
	qnorm := l2norm32(query)

	cpuScores := make([]float32, nCells)
	cudaScores := make([]float32, nCells)

	scoreTurboQuantCells(query, qnorm, dequant, corr, residNorms, dim, nCells, cpuScores)
	scoreTurboQuantCellsCUDA(query, qnorm, dequant, corr, residNorms, dim, nCells, cudaScores)

	assertScoresClose(t, cpuScores, cudaScores, 1e-5, "single-cell")
}

// ── kernel parity: zero residual norms (clamp is a no-op) ────────────────────

func TestTurboQuantCUDAKernelZeroResidNorm(t *testing.T) {
	if !cudaAvailable() {
		t.Skip("CUDA not available")
	}

	rng := rand.New(rand.NewSource(7))
	const dim, nCells = 128, 64

	query := randSlice(rng, dim)
	dequant := randSlice(rng, nCells*dim)
	corr := randSlice(rng, nCells*dim)
	residNorms := make([]float32, nCells) // all zero → clamp disabled
	qnorm := l2norm32(query)

	cpuScores := make([]float32, nCells)
	cudaScores := make([]float32, nCells)

	scoreTurboQuantCells(query, qnorm, dequant, corr, residNorms, dim, nCells, cpuScores)
	scoreTurboQuantCellsCUDA(query, qnorm, dequant, corr, residNorms, dim, nCells, cudaScores)

	assertScoresClose(t, cpuScores, cudaScores, 1e-4, "zero-resid-norms")
}

// ── CUDA capability assertions ────────────────────────────────────────────────

// TestTurboQuantCUDACapability verifies that the CUDA kernel is reachable and
// produces non-trivial results. It does NOT test TurboQuantSupport().CUDA
// because unit-test backends are CPU-only: TurboQuantSupport() checks
// b.schedBackends and only returns CUDA=true when model tensors are actually
// resident on the GPU (which requires loading a real model). The integration
// path is exercised by the benchmarks and E2E tests.
func TestTurboQuantCUDACapability(t *testing.T) {
	if !cudaAvailable() {
		t.Skip("CUDA not available")
	}

	// Smoke test: a single-call round-trip through the CUDA kernel must agree
	// with the CPU reference to within floating-point rounding tolerance.
	rng := rand.New(rand.NewSource(0xbadcafe))
	const dim, nCells = 128, 32

	query := randSlice(rng, dim)
	dequant := randSlice(rng, nCells*dim)
	qnorm := l2norm32(query)

	cpuScores := make([]float32, nCells)
	cudaScores := make([]float32, nCells)

	scoreTurboQuantCells(query, qnorm, dequant, nil, nil, dim, nCells, cpuScores)
	scoreTurboQuantCellsCUDA(query, qnorm, dequant, nil, nil, dim, nCells, cudaScores)

	assertScoresClose(t, cpuScores, cudaScores, 1e-5, "capability-smoke")

	// TurboQuantSupport on a CPU-only test backend must still report CPU=true
	// (not CUDA). The CUDA path only activates when model weights are on GPU.
	backend, _ := setupBackend(t, ml.BackendParams{FlashAttention: ml.FlashAttentionEnabled})
	tqb, ok := backend.(ml.TurboQuantBackend)
	if !ok {
		t.Fatal("backend does not implement TurboQuantBackend")
	}
	support := tqb.TurboQuantSupport()
	if !support.CPU || support.CUDA {
		t.Fatalf("CPU-only test backend: TurboQuantSupport() = %+v, want CPU=true CUDA=false", support)
	}
}

// ── benchmarks ───────────────────────────────────────────────────────────────

func benchmarkScoreCells(b *testing.B, nCells, dim int, useCUDA bool) {
	b.Helper()
	rng := rand.New(rand.NewSource(99))

	query := randSlice(rng, dim)
	dequant := randSlice(rng, nCells*dim)
	corr := randSlice(rng, nCells*dim)
	residNorms := make([]float32, nCells)
	for i := range residNorms {
		residNorms[i] = float32(rng.Float64())
	}
	qnorm := l2norm32(query)
	scores := make([]float32, nCells)

	b.ResetTimer()
	b.SetBytes(int64(nCells) * int64(dim) * 4 * 2) // dequant + corr bytes processed

	for i := 0; i < b.N; i++ {
		if useCUDA {
			scoreTurboQuantCellsCUDA(query, qnorm, dequant, corr, residNorms, dim, nCells, scores)
		} else {
			scoreTurboQuantCells(query, qnorm, dequant, corr, residNorms, dim, nCells, scores)
		}
	}
}

// CPU baselines — run without -tags cuda to get pure CPU numbers.
func BenchmarkTurboQuantCPU_1K_d128(b *testing.B)  { benchmarkScoreCells(b, 1024, 128, false) }
func BenchmarkTurboQuantCPU_4K_d128(b *testing.B)  { benchmarkScoreCells(b, 4096, 128, false) }
func BenchmarkTurboQuantCPU_16K_d128(b *testing.B) { benchmarkScoreCells(b, 16384, 128, false) }
func BenchmarkTurboQuantCPU_32K_d128(b *testing.B) { benchmarkScoreCells(b, 32768, 128, false) }

// CUDA path — only meaningful when CUDA is available.
func BenchmarkTurboQuantCUDA_1K_d128(b *testing.B) {
	if !cudaAvailable() {
		b.Skip("CUDA not available")
	}
	benchmarkScoreCells(b, 1024, 128, true)
}
func BenchmarkTurboQuantCUDA_4K_d128(b *testing.B) {
	if !cudaAvailable() {
		b.Skip("CUDA not available")
	}
	benchmarkScoreCells(b, 4096, 128, true)
}
func BenchmarkTurboQuantCUDA_16K_d128(b *testing.B) {
	if !cudaAvailable() {
		b.Skip("CUDA not available")
	}
	benchmarkScoreCells(b, 16384, 128, true)
}
func BenchmarkTurboQuantCUDA_32K_d128(b *testing.B) {
	if !cudaAvailable() {
		b.Skip("CUDA not available")
	}
	benchmarkScoreCells(b, 32768, 128, true)
}
