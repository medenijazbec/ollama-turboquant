package turboquant

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// PreparedBlock is a pre-parsed, pre-decoded key block ready for fast scoring.
// Use PrepareEncodedVector to create one, then ScorePreparedBlock per query.
//
// dequant holds the fully dequantized key values (codebook[index] * scale).
// correctionVec, when non-nil, is the precomputed QJL correction vector:
//
//	w = (√π/2 · residualNorm / sketchDim) · Σ_j sign_j · G_j
//
// Scoring reduces to two O(dim) dot products with no table lookups or
// per-query Gaussian sampling, letting the compiler and C/CUDA kernels SIMD-ize.
type PreparedBlock struct {
	block         Block
	dequant       []float32 // pre-dequantized: codebook[index] * scale per coordinate
	correctionVec []float32 // precomputed QJL correction vector (nil for MSE-only blocks)
	residualNorm  float32   // residual L2 norm; used for Cauchy-Schwarz clamping
	rotation      Rotation
}

// PrepareEncodedVector parses a key row once and pre-resolves the rotation,
// codebook, dequantized values, and QJL correction vector so that
// ScorePreparedBlock can score it against many queries with two plain float32
// dot products and no per-call allocations or Gaussian sampling.
func PrepareEncodedVector(data []byte) (PreparedBlock, Preset, error) {
	ev, err := UnmarshalEncodedVector(data)
	if err != nil {
		return PreparedBlock{}, Preset{}, err
	}
	if len(ev.Blocks) != 1 {
		return PreparedBlock{}, Preset{}, fmt.Errorf("PrepareEncodedVector expects exactly 1 block, got %d", len(ev.Blocks))
	}
	b := ev.Blocks[0]
	dim := int(b.OriginalDim)
	codebook, _ := scalarCodebook(dim, int(b.RegularBits))
	indices := unpackBits(b.RegularIndices, int(b.RegularBits), dim)
	dequant := make([]float32, len(indices))
	for i, idx := range indices {
		dequant[i] = dequantizeScalar(idx, codebook) * b.Scale
	}
	rotation := BuildRotation(dim, b.RotationSeed)
	var corrVec []float32
	var residNorm float32
	if vectorObjective(b.Objective) == objectiveProduct {
		corrVec = PrecomputeCorrectionVec(b.Residual, dim)
		residNorm = b.Residual.Scale
	}
	return PreparedBlock{
		block:         b,
		dequant:       dequant,
		correctionVec: corrVec,
		residualNorm:  residNorm,
		rotation:      rotation,
	}, ev.Preset, nil
}

// QueryNorm returns the L2 norm of a rotated query vector. Callers scoring
// many cells against the same query should compute this once and pass it to
// ScorePreparedBlockN to avoid recomputing it per cell.
func QueryNorm(queryRotated []float32) float32 {
	var sum float32
	for _, v := range queryRotated {
		sum += v * v
	}
	return float32(math.Sqrt(float64(sum)))
}

// ScorePreparedBlock computes dot(queryRotated, k) using a pre-rotated query
// and a pre-parsed key block. queryRotated must already be rotated with the
// same rotation matrix used during encoding (BuildRotation(dim, block.RotationSeed)).
//
// Both the primary dot product and the optional QJL correction are pure float32
// dot products — the Go compiler can auto-vectorize them and a C/CUDA kernel
// can use SIMD without any table lookups or per-call Gaussian sampling.
//
// For hot paths scoring many cells against the same query, prefer
// ScorePreparedBlockN which accepts a precomputed queryNorm.
func ScorePreparedBlock(queryRotated []float32, pb PreparedBlock) float32 {
	return ScorePreparedBlockN(queryRotated, QueryNorm(queryRotated), pb)
}

// ScorePreparedBlockN is like ScorePreparedBlock but accepts a precomputed
// queryNorm (from QueryNorm) to avoid recomputing it for every cached key.
func ScorePreparedBlockN(queryRotated []float32, queryNorm float32, pb PreparedBlock) float32 {
	var total float32
	for i, v := range pb.dequant {
		total += queryRotated[i] * v
	}
	if pb.correctionVec != nil {
		var correction float32
		for i, v := range pb.correctionVec {
			correction += queryRotated[i] * v
		}
		if pb.residualNorm >= 1e-6 {
			maxCorr := pb.residualNorm * queryNorm
			if correction > maxCorr {
				correction = maxCorr
			} else if correction < -maxCorr {
				correction = -maxCorr
			}
		}
		total += correction
	}
	return total
}

// RotationForBlock returns the rotation matrix for a prepared block, so the
// caller can pre-rotate a query once and reuse it across many ScorePreparedBlock calls.
func RotationForBlock(pb PreparedBlock) Rotation {
	return pb.rotation
}

// EncodedDim returns the original vector dimension stored in the block.
func (pb PreparedBlock) EncodedDim() int {
	return int(pb.block.OriginalDim)
}

// Dequant returns the pre-dequantized key values (codebook[index] × scale).
// The slice is valid for the lifetime of the PreparedBlock.
func (pb PreparedBlock) Dequant() []float32 { return pb.dequant }

// CorrectionVec returns the precomputed QJL correction vector, or nil for MSE-mode blocks.
// The slice is valid for the lifetime of the PreparedBlock.
func (pb PreparedBlock) CorrectionVec() []float32 { return pb.correctionVec }

// ResidualNorm returns the QJL residual L2 norm used for Cauchy-Schwarz clamping.
func (pb PreparedBlock) ResidualNorm() float32 { return pb.residualNorm }

// ScoreEncodedVector computes q^T k directly from the packed representation.
// For paper-product rows, this uses the primary rotated scalar codebook plus the
// asymmetric 1-bit QJL residual estimator.
//
// For high-throughput paths (many queries against the same cached keys), prefer
// PrepareEncodedVector + ScorePreparedBlock to avoid redundant parsing and rotation.
func ScoreEncodedVector(query []float32, data []byte) (float32, Preset, error) {
	ev, err := UnmarshalEncodedVector(data)
	if err != nil {
		return 0, Preset{}, err
	}
	if len(query) != ev.Dim {
		return 0, Preset{}, fmt.Errorf("query dim %d does not match encoded dim %d", len(query), ev.Dim)
	}

	var total float32
	offset := 0
	for _, block := range ev.Blocks {
		blockDim := int(block.OriginalDim)
		rotation := BuildRotation(blockDim, block.RotationSeed)
		queryRot := ApplyRotation(query[offset:offset+blockDim], rotation)
		codebook, _ := scalarCodebook(blockDim, int(block.RegularBits))
		indices := unpackBits(block.RegularIndices, int(block.RegularBits), blockDim)
		// Implemented checkpoint-visible V-path audit notes so inverse-WHT/dequant review can confirm FP32 accumulation sites instead of assuming half precision; idea source: @AmesianX.
		for i, idx := range indices {
			total += queryRot[i] * (dequantizeScalar(idx, codebook) * block.Scale)
		}
		if vectorObjective(block.Objective) == objectiveProduct {
			total += residualDotCorrection(queryRot, block.Residual)
		}
		offset += blockDim
	}

	return total, ev.Preset, nil
}

func DecodeVector(data []byte) ([]float32, Preset, error) {
	ev, err := UnmarshalEncodedVector(data)
	if err != nil {
		return nil, Preset{}, err
	}

	decoded := make([]float32, 0, ev.Dim)
	for _, block := range ev.Blocks {
		blockDim := int(block.OriginalDim)
		codebook, _ := scalarCodebook(blockDim, int(block.RegularBits))
		indices := unpackBits(block.RegularIndices, int(block.RegularBits), blockDim)
		rotated := make([]float32, blockDim)
		// The decode/reconstruct path accumulates in float32 here; any future half-precision backend mirror must preserve FP32-accumulate semantics for V reconstruction.
		for i, idx := range indices {
			rotated[i] = dequantizeScalar(idx, codebook) * block.Scale
		}
		if vectorObjective(block.Objective) == objectiveProduct {
			residual := reconstructResidual(blockDim, block.Residual)
			for i := range rotated {
				rotated[i] += residual[i]
			}
		}
		decoded = append(decoded, ApplyInverseRotation(rotated, BuildRotation(blockDim, block.RotationSeed))...)
	}

	if len(decoded) != ev.Dim {
		return nil, Preset{}, fmt.Errorf("decoded dim %d does not match header dim %d", len(decoded), ev.Dim)
	}
	return decoded, ev.Preset, nil
}

func UnmarshalEncodedVector(data []byte) (EncodedVector, error) {
	r := bytes.NewReader(data)
	var version uint8
	var presetID uint8
	var dim uint32
	var blockCount uint32
	for _, field := range []any{&version, &presetID, &dim, &blockCount} {
		if err := binary.Read(r, binary.LittleEndian, field); err != nil {
			return EncodedVector{}, err
		}
	}
	if version != BlockVersion {
		return EncodedVector{}, fmt.Errorf("unsupported encoded vector version %d", version)
	}

	preset, err := PresetByID(presetID)
	if err != nil {
		return EncodedVector{}, err
	}

	blocks := make([]Block, 0, blockCount)
	totalDim := 0
	for i := 0; i < int(blockCount); i++ {
		var blockLen uint32
		if err := binary.Read(r, binary.LittleEndian, &blockLen); err != nil {
			return EncodedVector{}, err
		}
		blockData := make([]byte, blockLen)
		if _, err := io.ReadFull(r, blockData); err != nil {
			return EncodedVector{}, err
		}
		var block Block
		if err := block.UnmarshalBinary(blockData); err != nil {
			return EncodedVector{}, err
		}
		if block.PresetID != preset.ID {
			return EncodedVector{}, fmt.Errorf("block preset id %d does not match encoded preset %d", block.PresetID, preset.ID)
		}
		if len(block.RegularIndices) != expectedPackedBytes(int(block.OriginalDim), int(block.RegularBits)) {
			return EncodedVector{}, fmt.Errorf("invalid primary index length %d for dim %d and bits %d", len(block.RegularIndices), block.OriginalDim, block.RegularBits)
		}
		if block.Residual.SketchDim != block.QJLRows {
			return EncodedVector{}, fmt.Errorf("residual sketch dim %d does not match qjl rows %d", block.Residual.SketchDim, block.QJLRows)
		}
		totalDim += int(block.OriginalDim)
		blocks = append(blocks, block)
	}
	if r.Len() != 0 {
		return EncodedVector{}, fmt.Errorf("unexpected trailing bytes in encoded vector: %d", r.Len())
	}
	if totalDim != int(dim) {
		return EncodedVector{}, fmt.Errorf("encoded vector dim mismatch: header=%d blocks=%d", dim, totalDim)
	}

	return EncodedVector{
		Version: version,
		Preset:  preset,
		Dim:     int(dim),
		Blocks:  blocks,
	}, nil
}
