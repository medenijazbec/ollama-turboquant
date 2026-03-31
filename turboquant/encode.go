package turboquant

import (
	"bytes"
	"encoding/binary"
	"fmt"
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

	codebook, boundaries := scalarCodebook(dim, bits)
	rotation := BuildRotation(dim, preset.RotationSeed)
	rotated := ApplyRotation(values, rotation)
	primaryCodes := make([]uint8, dim)
	reconRotated := make([]float32, dim)
	for i, value := range rotated {
		idx := quantizeScalarByBoundary(value, codebook, boundaries)
		primaryCodes[i] = idx
		reconRotated[i] = dequantizeScalar(idx, codebook)
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
		RotationSeed:   preset.RotationSeed,
		CodebookID:     uint16(bits),
		QJLRows:        uint16(qjlRows),
		AuxLayoutID:    1,
		Scale:          1,
		RegularIndices: packBits(primaryCodes, bits),
		Residual:       encodeResidual(rotated, reconRotated, qjlRows, preset.RotationSeed^0x9e3779b97f4a7c15),
	}

	return EncodedVector{
		Version: BlockVersion,
		Preset:  preset,
		Dim:     dim,
		Blocks:  []Block{block},
	}, nil
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
