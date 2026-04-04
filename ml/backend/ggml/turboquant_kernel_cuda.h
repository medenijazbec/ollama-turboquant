// turboquant_kernel_cuda.h — C-compatible declarations for the CUDA scoring
// kernel. Included by both the CUDA implementation (.cu) and the CGO bridge
// (.go via cgo comment), so only standard C types are used here.
//
// Build: compile turboquant_kernel_cuda.cu with nvcc -O2, produce a static
// archive libturboquant_cuda.a, then link Go code with -tags cuda.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// turboquant_cuda_available: returns 1 when compiled with CUDA support and a
// CUDA device is accessible, 0 otherwise. Always 1 at compile time when this
// header is provided by a built libturboquant_cuda.a; runtime device errors
// are caught internally and cause the function to return 0.
int turboquant_cuda_available(void);

// turboquant_score_cells_cuda: GPU equivalent of turboquant_score_cells() in
// turboquant_kernel.c. Scores n_cells cached key vectors against a single
// pre-rotated query using the CUDA device.
//
// All pointers are host (CPU) memory. The function handles H2D and D2H
// transfers internally via the default CUDA stream.
//
//   h_query       [dim]           pre-rotated query (from ApplyRotation)
//   h_dequant     [n_cells × dim] row-major primary dequantized key values
//   h_corr        [n_cells × dim] precomputed QJL correction vectors;
//                                 NULL for MSE-only mode (no correction)
//   h_resid_norms [n_cells]       per-cell residualNorm for Cauchy-Schwarz
//                                 clamp; ignored when h_corr is NULL
//   query_norm    scalar          L2 norm of h_query
//   dim           vector length
//   n_cells       number of cached tokens to score
//   h_scores      [n_cells]       output scores (caller-allocated)
void turboquant_score_cells_cuda(
    const float *h_query,
    const float *h_dequant,
    const float *h_corr,
    const float *h_resid_norms,
    float        query_norm,
    int          dim,
    int          n_cells,
    float       *h_scores
);

#ifdef __cplusplus
}
#endif
