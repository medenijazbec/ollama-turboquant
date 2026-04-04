package turboquant

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

// Rotation seed modifiers for outlier and regular sub-blocks.
// Using ASCII mnemonics: "OUTL1ER\0" and "REGULAR\0".
const (
	outlierSeedXOR = uint64(0x4f55544c31455200)
	regularSeedXOR = uint64(0x524547554c415200)
)

type EncodedVector struct {
	Version uint8
	Preset  Preset
	Dim     int
	Blocks  []Block
}

func EncodeVector(values []float32, preset Preset) (EncodedVector, error) {
	return encodeVector(values, preset, roleGeneric, objectiveMSE, preset.ValueBits)
}

func EncodeKeyVector(values []float32, preset Preset) (EncodedVector, error) {
	return encodeVector(values, preset, roleKey, objectiveProduct, preset.KeyPrimaryBits)
}

func EncodeValueVector(values []float32, preset Preset) (EncodedVector, error) {
	return encodeVector(values, preset, roleValue, objectiveMSE, preset.ValueBits)
}

func encodeVector(values []float32, preset Preset, role vectorRole, objective vectorObjective, bits int) (EncodedVector, error) {
	dim := len(values)
	blockDim := blockDimFor(preset, dim)
	if dim <= 0 || blockDim != dim {
		return EncodedVector{}, fmt.Errorf("invalid turboquant vector dim %d", dim)
	}
	if bits <= 0 || bits >= 8 {
		return EncodedVector{}, fmt.Errorf("invalid turboquant bit width %d", bits)
	}

	if preset.HasOutlierSplit() && dim > preset.OutlierCount {
		return encodeVectorWithOutliers(values, preset, role, objective, bits)
	}

	block, err := encodeSubBlock(values, nil, preset, role, objective, bits, preset.RotationSeed)
	if err != nil {
		return EncodedVector{}, err
	}
	return EncodedVector{
		Version: BlockVersion,
		Preset:  preset,
		Dim:     dim,
		Blocks:  []Block{block},
	}, nil
}

// encodeVectorWithOutliers splits the vector into outlier and regular channel
// sub-blocks, each encoded independently with its own rotation. The outlier
// block is stored first in Blocks.
func encodeVectorWithOutliers(values []float32, preset Preset, role vectorRole, objective vectorObjective, regularBits int) (EncodedVector, error) {
	split := SplitOutlierChannels(values, preset.OutlierCount)

	outlierBits := preset.OutlierBits
	if outlierBits <= 0 || outlierBits >= 8 {
		return EncodedVector{}, fmt.Errorf("invalid outlier bit width %d", outlierBits)
	}
	if regularBits <= 0 || regularBits >= 8 {
		return EncodedVector{}, fmt.Errorf("invalid regular bit width %d", regularBits)
	}

	outlierSeed := preset.RotationSeed ^ outlierSeedXOR
	regularSeed := preset.RotationSeed ^ regularSeedXOR

	outlierBlock, err := encodeSubBlock(split.OutlierValues, split.OutlierIndices, preset, role, objective, outlierBits, outlierSeed)
	if err != nil {
		return EncodedVector{}, fmt.Errorf("outlier block: %w", err)
	}

	// Regular block always uses MSE (no QJL sketch) regardless of the key/value
	// role. QJL is only applied to the outlier block, which concentrates the
	// residual correction budget on the highest-magnitude channels.
	regularBlock, err := encodeSubBlock(split.RegularValues, split.RegularIndices, preset, role, objectiveMSE, regularBits, regularSeed)
	if err != nil {
		return EncodedVector{}, fmt.Errorf("regular block: %w", err)
	}

	return EncodedVector{
		Version: BlockVersion,
		Preset:  preset,
		Dim:     len(values),
		Blocks:  []Block{outlierBlock, regularBlock},
	}, nil
}

// encodeSubBlock encodes a sub-vector (identified by channelIndices into the
// original full-dim vector) as a single Block. If channelIndices is nil, the
// block covers all channels (single-block legacy path).
func encodeSubBlock(values []float32, channelIndices []uint16, preset Preset, role vectorRole, objective vectorObjective, bits int, rotationSeed uint64) (Block, error) {
	dim := len(values)
	if dim <= 0 {
		return Block{}, fmt.Errorf("empty sub-block")
	}

	codebook, boundaries := scalarCodebook(dim, bits)
	rotation := BuildRotation(dim, rotationSeed)
	rotated := ApplyRotation(values, rotation)
	scale := blockScale(rotated)
	primaryCodes := make([]uint8, dim)
	reconRotated := make([]float32, dim)
	if scale == 0 {
		for i := range primaryCodes {
			primaryCodes[i] = quantizeScalarByBoundary(0, codebook, boundaries)
			reconRotated[i] = 0
		}
	} else {
		for i, value := range rotated {
			normalized := value / scale
			idx := quantizeScalarByBoundary(normalized, codebook, boundaries)
			primaryCodes[i] = idx
			reconRotated[i] = dequantizeScalar(idx, codebook) * scale
		}
	}

	qjlRows := 0
	if objective == objectiveProduct {
		qjlRows = preset.KeyQJLRows(dim)
	}

	block := Block{
		Version:        BlockVersion,
		PresetID:       preset.ID,
		Role:           uint8(role),
		Objective:      uint8(objective),
		OriginalDim:    uint16(dim),
		PaddedDim:      uint16(dim),
		BlockDim:       uint16(dim),
		RegularBits:    uint8(bits),
		RotationSeed:   rotationSeed,
		CodebookID:     uint16(bits),
		QJLRows:        uint16(qjlRows),
		AuxLayoutID:    1,
		ChannelIndices: channelIndices,
		Scale:          scale,
		RegularIndices: packBits(primaryCodes, bits),
		Residual:       encodeResidual(rotated, reconRotated, qjlRows, rotationSeed^0x9e3779b97f4a7c15),
	}
	return block, nil
}

func blockScale(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}
	var sumSquares float64
	for _, value := range values {
		sumSquares += float64(value * value)
	}
	if sumSquares < 1e-12 {
		return 0
	}
	return float32(math.Sqrt(sumSquares / float64(len(values))))
}

func (e EncodedVector) MarshalBinary() ([]byte, error) {
	var buf bytes.Buffer
	header := []any{
		e.Version,
		e.Preset.ID,
		uint32(e.Dim),
		uint32(len(e.Blocks)),
	}
	for _, field := range header {
		if err := binary.Write(&buf, binary.LittleEndian, field); err != nil {
			return nil, err
		}
	}
	for _, block := range e.Blocks {
		blockData, err := block.MarshalBinary()
		if err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, uint32(len(blockData))); err != nil {
			return nil, err
		}
		buf.Write(blockData)
	}
	return buf.Bytes(), nil
}
