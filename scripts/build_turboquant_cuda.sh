#!/usr/bin/env bash
# build_turboquant_cuda.sh — compile the TurboQuant CUDA scoring kernel into
# a static archive that Go's CGO can link against when built with -tags cuda.
#
# Usage:
#   bash scripts/build_turboquant_cuda.sh [ARCH]
#
# ARCH defaults to sm_61 (Tesla P40 / Pascal). Override for other GPUs:
#   sm_70  → Volta (V100)
#   sm_75  → Turing (RTX 2000)
#   sm_80  → Ampere (A100)
#   sm_86  → Ampere (RTX 3000)
#   sm_89  → Ada Lovelace (RTX 4000)
#   native → auto-detect at build time (requires CUDA ≥ 11.6 + CMake ≥ 3.24)
#
# Output: ml/backend/ggml/libturboquant_cuda.a
#
# After building, run tests with:
#   go test -tags cuda ./ml/backend/ggml/...
#
# Run benchmarks (CPU vs CUDA side-by-side):
#   go test -tags cuda -run='^$' -bench=BenchmarkTurboQuant -benchtime=5s \
#       ./ml/backend/ggml/...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/ml/backend/ggml"
CU_FILE="${SRC_DIR}/turboquant_kernel_cuda.cu"
OBJ_FILE="${SRC_DIR}/turboquant_kernel_cuda.o"
LIB_FILE="${SRC_DIR}/libturboquant_cuda.a"

ARCH="${1:-sm_61}"

# ── Verify prerequisites ──────────────────────────────────────────────────────
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit and ensure nvcc is on PATH." >&2
    exit 1
fi

NVCC_VERSION=$(nvcc --version 2>&1 | grep -oP 'release \K[0-9.]+')
echo "nvcc ${NVCC_VERSION}, arch ${ARCH}"
echo "Source: ${CU_FILE}"
echo "Output: ${LIB_FILE}"

# ── Remove stale build artifacts ─────────────────────────────────────────────
rm -f "${LIB_FILE}" "${OBJ_FILE}"

# ── Compile .cu → .o ─────────────────────────────────────────────────────────
nvcc \
    -O2 \
    -arch="${ARCH}" \
    -Xcompiler "-fPIC,-O2" \
    -std=c++14 \
    "${CU_FILE}" \
    -c -o "${OBJ_FILE}"

echo "Compiled → ${OBJ_FILE}"

# ── Archive .o → .a ──────────────────────────────────────────────────────────
ar rcs "${LIB_FILE}" "${OBJ_FILE}"
rm -f "${OBJ_FILE}"

echo "Archived → ${LIB_FILE}"
echo ""
echo "Build complete. Run tests with:"
echo "  go test -tags cuda ./ml/backend/ggml/..."
echo ""
echo "Run CPU vs CUDA benchmarks:"
echo "  go test -tags cuda -run='^$' -bench=BenchmarkTurboQuant -benchtime=5s ./ml/backend/ggml/..."
