// turboquant_kernel.c — SIMD-eligible batch attention scoring for TurboQuant KV cache.
//
// Entry point: turboquant_score_cells
//
// Scoring model (mirrors turboquant/decode.go ScorePreparedBlockN):
//   score[i] = dot(query, dequant[i])                              // primary
//            + clamp(dot(query, corr[i]), -resid_norm[i]*qnorm,   // QJL correction
//                                          resid_norm[i]*qnorm)
//
// Both dot products are plain float32 accumulations over `dim` elements — the
// structure is identical to an FP32 GEMV row with no branching inside the loop,
// which lets GCC/Clang auto-vectorize with SSE/AVX/NEON.
//
// Build notes:
//   CGO compiles this file with the flags from ggml.go's CGO preamble.
//   Pass -O2 (or -O3) for auto-vectorization; no manual intrinsics needed.

#include <stdint.h>
#include <math.h>
#include <string.h>

// turboquant_score_cells: score `n_cells` cached keys against a single
// pre-rotated query vector.
//
// Parameters:
//   query        [dim]           pre-rotated query (from ApplyRotation)
//   dequant      [n_cells × dim] row-major primary dequantized key values
//   corr         [n_cells × dim] row-major precomputed QJL correction vectors;
//                                NULL when all cells are MSE-mode (no residual)
//   resid_norms  [n_cells]       per-cell residualNorm for Cauchy-Schwarz clamp;
//                                ignored when corr is NULL
//   query_norm   scalar          L2 norm of query; used for clamping
//   dim          vector length
//   n_cells      number of cached tokens to score
//   scores       [n_cells]       output
void turboquant_score_cells(
    const float *query,
    const float *dequant,
    const float *corr,
    const float *resid_norms,
    float        query_norm,
    int          dim,
    int          n_cells,
    float       *scores
) {
    for (int i = 0; i < n_cells; i++) {
        const float *d = dequant + (size_t)i * dim;

        // Primary dot product — auto-vectorized by the compiler.
        float s = 0.0f;
        for (int j = 0; j < dim; j++) {
            s += query[j] * d[j];
        }

        if (corr != NULL) {
            const float *c = corr + (size_t)i * dim;

            // QJL correction dot product — auto-vectorized by the compiler.
            float correction = 0.0f;
            for (int j = 0; j < dim; j++) {
                correction += query[j] * c[j];
            }

            // Cauchy-Schwarz clamp (matches Go ScorePreparedBlockN).
            float rn = resid_norms[i];
            if (rn >= 1e-6f) {
                float max_corr = rn * query_norm;
                if (correction > max_corr) {
                    correction = max_corr;
                } else if (correction < -max_corr) {
                    correction = -max_corr;
                }
            }

            s += correction;
        }

        scores[i] = s;
    }
}
