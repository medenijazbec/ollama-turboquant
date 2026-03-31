package turboquant

import "testing"

func TestBlockMarshalRoundTrip(t *testing.T) {
	block := Block{
		Version:        BlockVersion,
		PresetID:       PresetTQ35.ID,
		Role:           uint8(roleKey),
		Objective:      uint8(objectiveProduct),
		OriginalDim:    8,
		PaddedDim:      8,
		BlockDim:       8,
		RegularBits:    3,
		RotationSeed:   77,
		CodebookID:     3,
		QJLRows:        4,
		AuxLayoutID:    1,
		Scale:          1,
		RegularIndices: []byte{1, 2, 3},
		Residual: ResidualSketch{
			Seed:      88,
			Scale:     0.5,
			SketchDim: 4,
			Signs:     []byte{0x0f},
		},
	}

	data, err := block.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	var decoded Block
	if err := decoded.UnmarshalBinary(data); err != nil {
		t.Fatal(err)
	}

	if decoded.Version != block.Version ||
		decoded.PresetID != block.PresetID ||
		decoded.Role != block.Role ||
		decoded.Objective != block.Objective ||
		decoded.OriginalDim != block.OriginalDim ||
		decoded.PaddedDim != block.PaddedDim ||
		decoded.BlockDim != block.BlockDim ||
		decoded.RegularBits != block.RegularBits ||
		decoded.RotationSeed != block.RotationSeed ||
		decoded.CodebookID != block.CodebookID ||
		decoded.QJLRows != block.QJLRows ||
		decoded.AuxLayoutID != block.AuxLayoutID ||
		string(decoded.RegularIndices) != string(block.RegularIndices) ||
		decoded.Residual.Seed != block.Residual.Seed ||
		decoded.Residual.Scale != block.Residual.Scale ||
		decoded.Residual.SketchDim != block.Residual.SketchDim ||
		string(decoded.Residual.Signs) != string(block.Residual.Signs) {
		t.Fatalf("decoded block mismatch: %+v", decoded)
	}
}

func TestBlockUnmarshalRejectsBadVersion(t *testing.T) {
	block := Block{Version: BlockVersion, PresetID: PresetTQ25.ID, Role: uint8(roleValue), Objective: uint8(objectiveMSE), OriginalDim: 4, PaddedDim: 4, BlockDim: 4, RegularBits: 2}
	data, err := block.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	data[0] = 99

	var decoded Block
	if err := decoded.UnmarshalBinary(data); err == nil {
		t.Fatal("expected unsupported block version error")
	}
}

func TestPackBitsRoundTripMixedWidths(t *testing.T) {
	values2 := []uint8{1, 3, 0, 2}
	roundTrip2 := unpackBits(packBits(values2, 2), 2, len(values2))
	for i := range values2 {
		if roundTrip2[i] != values2[i] {
			t.Fatalf("2-bit round trip mismatch at %d: got %d want %d", i, roundTrip2[i], values2[i])
		}
	}

	values3 := []uint8{3, 7, 1, 5}
	roundTrip3 := unpackBits(packBits(values3, 3), 3, len(values3))
	for i := range values3 {
		if roundTrip3[i] != values3[i] {
			t.Fatalf("3-bit round trip mismatch at %d: got %d want %d", i, roundTrip3[i], values3[i])
		}
	}
}

func TestEncodeUsesSinglePaperBlock(t *testing.T) {
	encoded, err := EncodeVector(pseudoRandomVector(70, 0x55), PresetTQ35)
	if err != nil {
		t.Fatal(err)
	}
	if len(encoded.Blocks) != 1 {
		t.Fatalf("block count = %d, want 1", len(encoded.Blocks))
	}
	if encoded.Blocks[0].OriginalDim != 70 {
		t.Fatalf("original dim = %d, want 70", encoded.Blocks[0].OriginalDim)
	}
	if encoded.Blocks[0].RegularBits != uint8(PresetTQ35.ValueBits) {
		t.Fatalf("regular bits = %d, want %d", encoded.Blocks[0].RegularBits, PresetTQ35.ValueBits)
	}
}
