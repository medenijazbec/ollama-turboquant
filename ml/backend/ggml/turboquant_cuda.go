//go:build cuda

package ggml

// #cgo LDFLAGS: -L${SRCDIR} -lturboquant_cuda -lcudart
// #include "turboquant_kernel_cuda.h"
import "C"
import (
	"sync"
	"unsafe"
)

// turboQuantCUDAThreshold is the minimum number of cached cells at which the
// CUDA scoring path is preferred over the CPU SIMD path.
//
// Below this threshold the PCIe transfer overhead (H2D for dequant/corr flat
// arrays + D2H for scores) dominates execution time and the CPU SIMD kernel
// is faster. Estimated crossover on a Tesla P40 (PCIe 3.0 x16, ~16 GB/s)
// vs an AVX2 CPU at dim=128:
//   transfer ≈ n_cells × dim × 8B × 2 (dequant+corr) / 16 GB/s
//   cpu      ≈ n_cells × 2 × dim / (8 FLOPs/cycle × 3 GHz)
//   crossover ≈ 8 192 cells (approx)
//
// 16384 (16K tokens context) is used as a conservative threshold that avoids
// the overhead-dominated region. A GPU-resident implementation (where dequant
// and corr stay in device memory across decode steps) would eliminate the PCIe
// bottleneck and lower this threshold to near zero.
const turboQuantCUDAThreshold = 16384

// cudaMu serialises turboquant_score_cells_cuda calls from concurrent
// goroutines. Although CUDA runtime functions are individually thread-safe,
// serialising here avoids having N_heads goroutines simultaneously hammering
// cudaMalloc with multi-megabyte allocations, which can cause OOM errors
// under large contexts. The GPU itself serialises kernel launches to stream 0
// regardless, so this mutex adds no extra serialisation on the compute side.
var cudaMu sync.Mutex

// cudaAvailableOnce caches the result of the first cudaGetDeviceCount probe so
// that repeated calls from TurboQuantSupport() and turboQuantAttentionScores on
// every decode step incur only a single CGO/CUDA-runtime round-trip.
var cudaAvailableOnce = sync.OnceValue(func() bool {
	return C.turboquant_cuda_available() == 1
})

// cudaAvailable returns true when the binary was built with the cuda tag and
// at least one CUDA device is accessible at runtime. The probe is performed
// once; subsequent calls return the cached result.
func cudaAvailable() bool {
	return cudaAvailableOnce()
}

// scoreTurboQuantCellsCUDA is the GPU equivalent of scoreTurboQuantCells.
// It uploads the flat arrays to device memory, launches the CUDA kernel, and
// retrieves scores — all synchronously on the default stream.
//
// Parameters are identical to scoreTurboQuantCells; see that function for
// full documentation.
func scoreTurboQuantCellsCUDA(
	queryRotated []float32, queryNorm float32,
	dequantFlat, corrFlat []float32, residNorms []float32,
	encodedDim, nCells int,
	scores []float32,
) {
	var corrPtr *C.float
	var residPtr *C.float
	if len(corrFlat) > 0 {
		corrPtr = (*C.float)(unsafe.Pointer(&corrFlat[0]))
		residPtr = (*C.float)(unsafe.Pointer(&residNorms[0]))
	}

	cudaMu.Lock()
	defer cudaMu.Unlock()

	C.turboquant_score_cells_cuda(
		(*C.float)(unsafe.Pointer(&queryRotated[0])),
		(*C.float)(unsafe.Pointer(&dequantFlat[0])),
		corrPtr,
		residPtr,
		C.float(queryNorm),
		C.int(encodedDim),
		C.int(nCells),
		(*C.float)(unsafe.Pointer(&scores[0])),
	)
}
