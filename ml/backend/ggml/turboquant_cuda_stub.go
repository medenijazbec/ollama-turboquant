//go:build !cuda

package ggml

// turboQuantCUDAThreshold is unused in non-CUDA builds but defined here to
// avoid conditional compilation of the dispatch site in turboQuantAttentionScores.
const turboQuantCUDAThreshold = 0

// cudaAvailable returns false in non-CUDA builds. The CUDA scoring path is
// completely compiled out; the CPU SIMD path (turboquant_kernel.c) is used
// unconditionally.
func cudaAvailable() bool { return false }

// scoreTurboQuantCellsCUDA is unreachable in non-CUDA builds (cudaAvailable
// always returns false) but must be declared so the call site in
// turboQuantAttentionScores compiles without build tags.
func scoreTurboQuantCellsCUDA(
	queryRotated []float32, _ float32,
	_, _ []float32, _ []float32,
	_, _ int,
	_ []float32,
) {
	panic("scoreTurboQuantCellsCUDA called in non-CUDA build")
}
