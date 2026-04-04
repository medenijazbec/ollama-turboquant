// turboquant_kernel_cuda.cu — CUDA batch attention scoring for TurboQuant.
//
// Mirrors turboquant_kernel.c but executes on the GPU. One CUDA block per
// cached token (cell); threads within the block cooperatively compute the
// dot products via a parallel tree reduction in shared memory.
//
// Scoring model (identical to turboquant_kernel.c and decode.go):
//   score[i] = dot(query, dequant[i])
//            + clamp(dot(query, corr[i]), −rn[i]·qnorm, rn[i]·qnorm)
//
// Compile with:
//   nvcc -O2 -arch=sm_61 -Xcompiler "-fPIC" \
//        turboquant_kernel_cuda.cu -c -o turboquant_kernel_cuda.o
//   ar rcs libturboquant_cuda.a turboquant_kernel_cuda.o
//
// sm_61 targets the Tesla P40 (Pascal, compute 6.1). Adjust -arch as needed.

#include "turboquant_kernel_cuda.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Block size for the dot-product kernel. 128 threads exactly matches the
// standard head dimension (dim=128) so each thread handles one element in
// the common case, eliminating the stride loop overhead.
// For larger dims the stride loop handles the overflow correctly.
// Must be a power of two — the tree reduction depends on this invariant.
#define TQ_CUDA_BLOCK_SIZE 128
static_assert((TQ_CUDA_BLOCK_SIZE & (TQ_CUDA_BLOCK_SIZE - 1)) == 0,
              "TQ_CUDA_BLOCK_SIZE must be a power of two for tree reduction");

// turboquant_score_cells_kernel — one block per cached token.
//
// Each thread accumulates partial dot products for its assigned elements
// (using a stride loop when dim > blockDim.x), stores them to shared memory,
// then a tree reduction produces the final per-block scalar result.
// Thread 0 applies the Cauchy-Schwarz clamp and writes scores[cell].
__global__ void turboquant_score_cells_kernel(
    const float * __restrict__ query,
    const float * __restrict__ dequant,
    const float * __restrict__ corr,        // NULL → MSE-only
    const float * __restrict__ resid_norms,
    float                      query_norm,
    int                        dim,
    int                        n_cells,
    float       * __restrict__ scores
) {
    int cell = blockIdx.x;
    if (cell >= n_cells) return;

    int tid = threadIdx.x;
    int bsz = blockDim.x;

    // Shared memory layout: [0, bsz) primary, [bsz, 2*bsz) correction.
    // Always allocate 2*bsz floats (shared_bytes at launch) regardless of
    // whether corr is present; the second half is simply unused in MSE-only
    // mode. The allocation is fixed at kernel launch time by the host caller.
    extern __shared__ float shmem[];
    float *s_prim = shmem;
    float *s_corr = shmem + bsz;

    const float *d = dequant + (size_t)cell * dim;
    const float *c = corr    ? corr + (size_t)cell * dim : nullptr;

    // --- Accumulate partial dot products (stride loop) ---
    float prim_acc = 0.0f;
    float corr_acc = 0.0f;
    for (int j = tid; j < dim; j += bsz) {
        float q = query[j];
        prim_acc += q * d[j];
        if (c) corr_acc += q * c[j];
    }
    s_prim[tid] = prim_acc;
    if (c) s_corr[tid] = corr_acc;
    __syncthreads();

    // --- Tree reduction in shared memory ---
    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_prim[tid] += s_prim[tid + stride];
            if (c) s_corr[tid] += s_corr[tid + stride];
        }
        __syncthreads();
    }

    // --- Thread 0: apply clamp and write result ---
    if (tid == 0) {
        float result = s_prim[0];
        if (c) {
            float cv = s_corr[0];
            float rn = resid_norms[cell];
            if (rn >= 1e-6f) {
                float max_corr = rn * query_norm;
                if      (cv >  max_corr) cv =  max_corr;
                else if (cv < -max_corr) cv = -max_corr;
            }
            result += cv;
        }
        scores[cell] = result;
    }
}

// turboquant_score_cells_cuda — host-side launcher (called from Go via CGO).
//
// Allocates device memory, copies inputs host→device, launches the kernel,
// copies scores device→host, then frees device memory. All operations use
// the default CUDA stream (stream 0) which serializes across calls within a
// CUDA context.
//
// Thread safety: CUDA runtime calls (cudaMalloc, cudaMemcpy, kernel launches
// to stream 0) are internally serialized per device; multiple host threads
// may call this function concurrently but will serialize on the GPU.
void turboquant_score_cells_cuda(
    const float *h_query,
    const float *h_dequant,
    const float *h_corr,
    const float *h_resid_norms,
    float        query_norm,
    int          dim,
    int          n_cells,
    float       *h_scores
) {
    size_t query_bytes  = (size_t)dim    * sizeof(float);
    size_t mat_bytes    = (size_t)n_cells * (size_t)dim * sizeof(float);
    size_t norms_bytes  = (size_t)n_cells * sizeof(float);
    size_t scores_bytes = (size_t)n_cells * sizeof(float);

    // --- Allocate device memory ---
    float *d_query       = nullptr;
    float *d_dequant     = nullptr;
    float *d_corr        = nullptr;
    float *d_resid_norms = nullptr;
    float *d_scores      = nullptr;

    cudaError_t err;

    err = cudaMalloc(&d_query,   query_bytes);  if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_dequant, mat_bytes);    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_scores,  scores_bytes); if (err != cudaSuccess) goto cleanup;

    // --- Copy inputs host → device ---
    err = cudaMemcpy(d_query,   h_query,   query_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_dequant, h_dequant, mat_bytes,   cudaMemcpyHostToDevice); if (err != cudaSuccess) goto cleanup;

    if (h_corr != nullptr) {
        err = cudaMalloc(&d_corr,        mat_bytes);   if (err != cudaSuccess) goto cleanup;
        err = cudaMalloc(&d_resid_norms, norms_bytes); if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpy(d_corr,        h_corr,        mat_bytes,   cudaMemcpyHostToDevice); if (err != cudaSuccess) goto cleanup;
        err = cudaMemcpy(d_resid_norms, h_resid_norms, norms_bytes, cudaMemcpyHostToDevice); if (err != cudaSuccess) goto cleanup;
    }

    // --- Launch kernel: one block per cell, TQ_CUDA_BLOCK_SIZE threads ---
    {
        int  block_size  = TQ_CUDA_BLOCK_SIZE;
        int  grid_size   = n_cells;
        // Shared memory: primary + correction accumulators.
        size_t shared_bytes = (size_t)block_size * 2 * sizeof(float);

        turboquant_score_cells_kernel<<<grid_size, block_size, shared_bytes>>>(
            d_query, d_dequant, d_corr, d_resid_norms,
            query_norm, dim, n_cells, d_scores
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }

    // --- Copy scores device → host (synchronises with kernel completion) ---
    err = cudaMemcpy(h_scores, d_scores, scores_bytes, cudaMemcpyDeviceToHost);

cleanup:
    if (err != cudaSuccess) {
        fprintf(stderr, "turboquant_score_cells_cuda: CUDA error: %s\n",
                cudaGetErrorString(err));
        // Zero-fill scores on error so callers get a defined (if wrong) result
        // rather than uninitialised memory. The model will produce garbage
        // attention weights but won't crash.
        for (int i = 0; i < n_cells; i++) h_scores[i] = 0.0f;
    }

    cudaFree(d_query);
    cudaFree(d_dequant);
    cudaFree(d_scores);
    if (d_corr)        cudaFree(d_corr);
    if (d_resid_norms) cudaFree(d_resid_norms);
}

// turboquant_cuda_available: probes for at least one CUDA device.
// The Go caller (turboquant_cuda.go) caches the result with sync.OnceValue so
// the CGO boundary is crossed only once per process lifetime.
int turboquant_cuda_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) return 0;
    return 1;
}
