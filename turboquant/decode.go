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
//
// For multi-block (outlier-split) encodings, isOriginalSpace is true and dequant
// and correctionVec are scattered into the full original-space dim. The query
// must NOT be rotated before scoring — dot(q, dequant_orig) equals dot(Rq, dequant_rot).
type PreparedBlock struct {
	block           Block
	dim             int       // full original vector dim (matches EncodedVector.Dim)
	dequant         []float32 // pre-dequantized: codebook[index] * scale per coordinate
	correctionVec   []float32 // precomputed QJL correction vector (nil for MSE-only blocks)
	residualNorm    float32   // residual L2 norm; used for Cauchy-Schwarz clamping
	rotation        Rotation  // valid only when !isOriginalSpace
	isOriginalSpace bool      // true for multi-block: dequant/corrVec are in original space
}

// IsOriginalSpace reports whether dequant and correctionVec are in original
// (unrotated) space. When true, the query must NOT be rotated before scoring.
func (pb PreparedBlock) IsOriginalSpace() bool { return pb.isOriginalSpace }

// PrepareEncodedVector parses a key row once and pre-resolves the rotation,
// codebook, dequantized values, and QJL correction vector so that
// ScorePreparedBlock can score it against many queries with two plain float32
// dot products and no per-call allocations or Gaussian sampling.
//
// For single-block encodings, dequant and corrVec are in rotated space and the
// caller must rotate the query with RotationForBlock before scoring.
//
// For multi-block (outlier-split) encodings, dequant and corrVec are scattered
// into the full original-space dim (IsOriginalSpace() == true) — the query must
// NOT be rotated before scoring.
func PrepareEncodedVector(data []byte) (PreparedBlock, Preset, error) {
	ev, err := UnmarshalEncodedVector(data)
	if err != nil {
		return PreparedBlock{}, Preset{}, err
	}
	if len(ev.Blocks) == 0 {
		return PreparedBlock{}, Preset{}, fmt.Errorf("PrepareEncodedVector: no blocks")
	}

	// Multi-block (outlier-split) path: scatter each sub-block's dequant and
	// corrVec into full-dim original-space arrays.
	isMultiBlock := len(ev.Blocks) > 1 || len(ev.Blocks[0].ChannelIndices) > 0
	if isMultiBlock {
		return prepareMultiBlock(ev)
	}

	// Single-block legacy path (unchanged behavior).
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
		dim:           ev.Dim,
		dequant:       dequant,
		correctionVec: corrVec,
		residualNorm:  residNorm,
		rotation:      rotation,
	}, ev.Preset, nil
}

// prepareMultiBlock handles the outlier-split path: for each sub-block, dequant
// and optional corrVec are computed in rotated space, inverse-rotated to original
// space, then scattered into full-dim arrays.
func prepareMultiBlock(ev EncodedVector) (PreparedBlock, Preset, error) {
	fullDequant := make([]float32, ev.Dim)
	var fullCorrVec []float32
	var residNormSq float64

	hasProduct := false
	for _, b := range ev.Blocks {
		if vectorObjective(b.Objective) == objectiveProduct {
			hasProduct = true
			break
		}
	}
	if hasProduct {
		fullCorrVec = make([]float32, ev.Dim)
	}

	for _, b := range ev.Blocks {
		blockDim := int(b.OriginalDim)
		codebook, _ := scalarCodebook(blockDim, int(b.RegularBits))
		indices := unpackBits(b.RegularIndices, int(b.RegularBits), blockDim)
		dequantRot := make([]float32, blockDim)
		for i, idx := range indices {
			dequantRot[i] = dequantizeScalar(idx, codebook) * b.Scale
		}

		rot := BuildRotation(blockDim, b.RotationSeed)
		dequantOrig := ApplyInverseRotation(dequantRot, rot)

		if len(b.ChannelIndices) == blockDim {
			for i, chIdx := range b.ChannelIndices {
				fullDequant[chIdx] = dequantOrig[i]
			}
		} else {
			// No ChannelIndices: block covers a contiguous prefix (shouldn't happen
			// in multi-block, but handle gracefully by treating as offset 0).
			copy(fullDequant, dequantOrig)
		}

		if vectorObjective(b.Objective) == objectiveProduct && b.Residual.Scale > 0 {
			corrRotated := PrecomputeCorrectionVec(b.Residual, blockDim)
			if corrRotated != nil {
				corrOrig := ApplyInverseRotation(corrRotated, rot)
				if len(b.ChannelIndices) == blockDim {
					for i, chIdx := range b.ChannelIndices {
						fullCorrVec[chIdx] = corrOrig[i]
					}
				} else {
					copy(fullCorrVec, corrOrig)
				}
			}
			residNormSq += float64(b.Residual.Scale) * float64(b.Residual.Scale)
		}
	}

	residNorm := float32(math.Sqrt(residNormSq))
	if !hasProduct || residNorm == 0 {
		fullCorrVec = nil
	}

	return PreparedBlock{
		block:           ev.Blocks[0],
		dim:             ev.Dim,
		dequant:         fullDequant,
		correctionVec:   fullCorrVec,
		residualNorm:    residNorm,
		rotation:        Rotation{}, // not used; caller should check IsOriginalSpace
		isOriginalSpace: true,
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
// For multi-block (outlier-split) blocks where IsOriginalSpace() is true, pass
// the original (unrotated) query instead — the caller must NOT rotate the query.
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
// Only valid when IsOriginalSpace() is false (single-block encodings).
func RotationForBlock(pb PreparedBlock) Rotation {
	return pb.rotation
}

// EncodedDim returns the original vector dimension stored in the prepared block.
func (pb PreparedBlock) EncodedDim() int {
	return pb.dim
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
	for _, block := range ev.Blocks {
		blockDim := int(block.OriginalDim)
		rotation := BuildRotation(blockDim, block.RotationSeed)

		// Gather the query channels for this sub-block.
		var subQuery []float32
		if len(block.ChannelIndices) == blockDim {
			subQuery = make([]float32, blockDim)
			for i, chIdx := range block.ChannelIndices {
				subQuery[i] = query[chIdx]
			}
		} else {
			// Legacy single-block: contiguous range starting at 0.
			subQuery = query[:blockDim]
		}

		queryRot := ApplyRotation(subQuery, rotation)
		codebook, _ := scalarCodebook(blockDim, int(block.RegularBits))
		indices := unpackBits(block.RegularIndices, int(block.RegularBits), blockDim)
		// FP32 accumulation audit: decode accumulates in float32; any future half-precision
		// backend must preserve FP32-accumulate semantics.
		for i, idx := range indices {
			total += queryRot[i] * (dequantizeScalar(idx, codebook) * block.Scale)
		}
		if vectorObjective(block.Objective) == objectiveProduct {
			total += residualDotCorrection(queryRot, block.Residual)
		}
	}

	return total, ev.Preset, nil
}

func DecodeVector(data []byte) ([]float32, Preset, error) {
	ev, err := UnmarshalEncodedVector(data)
	if err != nil {
		return nil, Preset{}, err
	}

	decoded := make([]float32, ev.Dim)
	offset := 0
	for _, block := range ev.Blocks {
		blockDim := int(block.OriginalDim)
		codebook, _ := scalarCodebook(blockDim, int(block.RegularBits))
		indices := unpackBits(block.RegularIndices, int(block.RegularBits), blockDim)
		rotated := make([]float32, blockDim)
		// FP32 accumulation audit: decode accumulates in float32; any future half-precision
		// backend must preserve FP32-accumulate semantics for V reconstruction.
		for i, idx := range indices {
			rotated[i] = dequantizeScalar(idx, codebook) * block.Scale
		}
		if vectorObjective(block.Objective) == objectiveProduct {
			residual := reconstructResidual(blockDim, block.Residual)
			for i := range rotated {
				rotated[i] += residual[i]
			}
		}
		original := ApplyInverseRotation(rotated, BuildRotation(blockDim, block.RotationSeed))

		if len(block.ChannelIndices) == blockDim {
			// Scatter to original channel positions.
			for i, chIdx := range block.ChannelIndices {
				decoded[chIdx] = original[i]
			}
		} else {
			// Legacy single-block: fill contiguous range.
			copy(decoded[offset:], original)
			offset += blockDim
		}
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
